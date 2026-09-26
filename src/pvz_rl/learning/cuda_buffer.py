"""Bounded host trajectories and exact raw-memory reconstruction for complete games."""

import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.policy.event_memory import MemoryContext


def trajectory_dtype(observations, capacity):
    return np.dtype(
        [
            ("observation", "<f4", observations),
            ("mask", "u1", (A.size + 7) // 8),
            ("action", "<u2"),
            ("previous", "<u2"),
            ("tick", "<u4"),
            ("reset", "?"),
            ("active", "?"),
            ("env", "<u2"),
            ("duration", "<u2"),
            ("reward", "<f8"),
            ("log_prob", "<f4"),
            ("baseline", "<f4"),
            ("target", "<f4"),
            ("advantage", "<f4"),
            ("ids", "<u4", capacity),
            ("counts", "<u2", capacity),
            ("starts", "<u4", capacity),
        ]
    )


class CompleteGameBuffer:
    """Blocks use RAM up to a budget, then disk-backed arrays without GPU growth.

    Memory entries reference earlier raw tokens plus compression metadata. This
    is a lossless checkpoint of the deterministic token bank, not hidden neural
    activations. A bounded minibatch reconstructs precisely its causal context.
    """

    def __init__(
        self, path, observation_size, capacity, n_envs, *, ram_bytes=6 * 1024**3, block_rows=32768
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dtype = trajectory_dtype(observation_size, capacity)
        self.observation_size, self.capacity, self.n_envs = observation_size, capacity, n_envs
        self.ram_bytes, self.block_rows = int(ram_bytes), int(block_rows)
        self.blocks = []
        self.size = self.ram_used = 0
        self.finalized = False
        self.cache = None
        self.transport_metrics = dict(
            preparation_seconds=0.0,
            transfer_wait_seconds=0.0,
            transfer_stream_seconds=0.0,
            cache_hits=0,
            cache_requests=0,
        )

    def configure_cache(self, device, gib):
        from pvz_rl.learning.transfers import TokenCache

        if torch.device(device).type != "cuda":
            return
        self.cache = TokenCache(device, self.block_rows, self.observation_size + 3, gib * 1024**3)
        # Resume rebuilds only raw inputs. CPU/disk storage remains authoritative.
        for start in range(0, self.size, self.block_rows):
            if self.cache.full:
                break
            tokens = self.raw_tokens(np.arange(start, min(self.size, start + self.block_rows)))
            self.cache.append(torch.from_numpy(tokens).to(device))

    def _allocate(self):
        amount = self.block_rows * self.dtype.itemsize
        if self.ram_used + amount <= self.ram_bytes:
            block = np.zeros(self.block_rows, dtype=self.dtype)
            self.ram_used += amount
        else:
            block = np.lib.format.open_memmap(
                self.path / f"block-{len(self.blocks):06d}.npy",
                mode="w+",
                dtype=self.dtype,
                shape=(self.block_rows,),
            )
        self.blocks.append(block)

    def append(self, records, raw_tokens=None):
        if self.finalized:
            raise RuntimeError("Cannot append after complete-return finalization")
        records = np.asarray(records, dtype=self.dtype)
        if self.cache is not None and not self.cache.full:
            if raw_tokens is None:
                raw_tokens = torch.from_numpy(self._tokens_from_rows(records)).to(self.cache.device)
            self.cache.append(raw_tokens)
        if self.size + len(records) >= 2**32:
            raise OverflowError("Raw memory reference capacity exceeded")
        at = 0
        while at < len(records):
            b, offset = divmod(self.size, self.block_rows)
            if b == len(self.blocks):
                self._allocate()
            take = min(len(records) - at, self.block_rows - offset)
            self.blocks[b][offset : offset + take] = records[at : at + take]
            at += take
            self.size += take

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Trajectory index outside collected history")
        result = np.empty(indices.shape, dtype=self.dtype)
        block_ids, offsets = np.divmod(indices, self.block_rows)
        for b in np.unique(block_ids):
            selected = block_ids == b
            result[selected] = self.blocks[b][offsets[selected]]
        return result

    def _tokens_from_rows(self, rows):
        result = np.empty((*rows.shape, self.observation_size + 3), np.float32)
        result[..., : self.observation_size] = rows["observation"]
        for i, key in enumerate(("previous", "reset", "tick"), self.observation_size):
            result[..., i] = rows[key]
        return result

    def raw_tokens(self, indices):
        """Gather fields before rows: never copy masks, returns or history metadata."""
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Trajectory index outside collected history")
        result = np.empty((*indices.shape, self.observation_size + 3), np.float32)
        blocks, offsets = np.divmod(indices, self.block_rows)
        for b in np.unique(blocks):
            selected = blocks == b
            rows = offsets[selected]
            result[selected, : self.observation_size] = self.blocks[b]["observation"][rows]
            for i, key in enumerate(("previous", "reset", "tick"), self.observation_size):
                result[..., i][selected] = self.blocks[b][key][rows]
        return result

    def valid_indices(self, planting=False):
        parts = []
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            keep = rows["active"].copy()
            if planting:
                keep &= (rows["action"] > 0) & (rows["action"] < A.dig_start)
            parts.append(np.flatnonzero(keep) + b * self.block_rows)
        return np.concatenate(parts) if parts else np.empty(0, np.int64)

    def finalize(self):
        """Independent episodes, gamma one, no truncation or rollout bootstrap."""
        running = np.zeros(self.n_envs, np.float64)
        coverage = np.zeros(3, np.int64)
        squared_error = np.zeros(3, np.float64)
        signed_error = np.zeros(3, np.float64)
        positive_advantages = np.zeros(3, np.int64)
        for b in reversed(range(len(self.blocks))):
            rows = self.blocks[b][: min(self.block_rows, self.size - b * self.block_rows)]
            # Returns are accumulated per environment, including instantaneous actions.
            for env in range(self.n_envs):
                ix = np.flatnonzero(rows["active"] & (rows["env"] == env))
                rewards = rows["reward"][ix]
                values = np.cumsum(rewards[::-1], dtype=np.float64)[::-1] + running[env]
                if len(ix):
                    running[env] = values[0]
                rows["target"][ix] = values
                rows["advantage"][ix] = values - rows["baseline"][ix]
            for group in range(3):
                kinds = np.where(
                    rows["action"] == 0, 0, np.where(rows["action"] < A.dig_start, 1, 2)
                )
                ix = rows["active"] & (kinds == group)
                errors = rows["baseline"][ix] - rows["target"][ix]
                coverage[group] += len(errors)
                squared_error[group] += np.square(errors.astype(np.float64)).sum()
                signed_error[group] += errors.astype(np.float64).sum()
                positive_advantages[group] += np.count_nonzero(errors < 0)
        self.finalized = True
        self.group_counts = coverage
        planting = self.valid_indices(planting=True)
        # Bounded reductions; no full context or computation graph is retained.
        total = squares = 0.0
        for start in range(0, len(planting), self.block_rows):
            a = self.take(planting[start : start + self.block_rows])["advantage"].astype(np.float64)
            total += a.sum()
            squares += np.square(a).sum()
        self.advantage_mean = total / max(1, len(planting))
        self.advantage_std = (
            max(0.0, squares / max(1, len(planting)) - self.advantage_mean**2) ** 0.5
        )
        return {
            name: {
                "count": int(coverage[i]),
                "mse": float(squared_error[i] / coverage[i]) if coverage[i] else None,
                "signed_error": float(signed_error[i] / coverage[i]) if coverage[i] else None,
                "positive_advantage_fraction": float(positive_advantages[i] / coverage[i])
                if coverage[i]
                else None,
            }
            for i, name in enumerate(("wait", "plant", "dig"))
        }

    def batch(self, indices, device, *, staging=None):
        started = perf_counter()
        rows = self.take(indices)
        ids = rows["ids"].astype(np.int64)
        valid = rows["counts"] > 0
        if np.any(ids[valid] > np.asarray(indices)[:, None].repeat(self.capacity, axis=1)[valid]):
            raise RuntimeError("Future token in causal memory")

        def tensor(x, key):
            if staging is not None:
                return staging.tensor(key, x, device)
            host = torch.from_numpy(np.ascontiguousarray(x))
            return (
                host.pin_memory().to(device, non_blocking=True)
                if str(device).startswith("cuda")
                else host.to(device)
            )

        cached_size = (
            self.cache.size if self.cache is not None and torch.device(device).type == "cuda" else 0
        )
        hit = valid & (ids < cached_size)
        self.transport_metrics["cache_hits"] += int(hit.sum())
        self.transport_metrics["cache_requests"] += int(valid.sum())
        if hit.any():
            tokens = torch.zeros(
                (*ids.shape, self.observation_size + 3), device=device, dtype=torch.float32
            )
            flat = tokens.view(-1, self.observation_size + 3)
            blocks = ids // self.block_rows
            for b in np.unique(blocks[hit]):
                positions = np.flatnonzero(hit & (blocks == b))
                ix = tensor(positions, f"positions-{b}")
                refs = tensor(ids.flat[positions] % self.block_rows, f"refs-{b}")
                flat.index_copy_(0, ix, self.cache.blocks[b].index_select(0, refs))
            misses = np.flatnonzero(valid & ~hit)
            if len(misses):
                flat.index_copy_(
                    0,
                    tensor(misses, "miss-ids"),
                    tensor(self.raw_tokens(ids.flat[misses]), "miss-tokens"),
                )
        else:
            raw = self.raw_tokens(np.where(valid, ids, 0))
            raw[~valid] = 0
            tokens = tensor(raw, "tokens")
        context = MemoryContext(
            tokens,
            tensor(valid, "valid"),
            tensor(rows["counts"].astype(np.float32), "counts"),
            tensor(rows["starts"].astype(np.float32), "starts"),
        )
        masks = np.unpackbits(rows["mask"], axis=-1, count=A.size, bitorder="little").astype(bool)
        data = {
            name: tensor(
                rows[name].astype(np.float32 if name not in ("action", "env") else np.int64), name
            )
            for name in (
                "observation",
                "action",
                "reward",
                "log_prob",
                "baseline",
                "target",
                "advantage",
                "env",
            )
        }
        data.update(masks=tensor(masks, "masks"), context=context)
        self.transport_metrics["preparation_seconds"] += perf_counter() - started
        return data

    def save(self, destination):
        """Publish a self-contained checkpoint sidecar, streaming one block at a time."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            target = destination / f"block-{b:06d}.npy"
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, rows, allow_pickle=False)
            temporary.replace(target)
        return self.metadata()

    def metadata(self):
        return {
            "observation_size": self.observation_size,
            "capacity": self.capacity,
            "n_envs": self.n_envs,
            "ram_bytes": self.ram_bytes,
            "block_rows": self.block_rows,
            "size": self.size,
            "finalized": self.finalized,
            "group_counts": getattr(self, "group_counts", None),
            "advantage_mean": getattr(self, "advantage_mean", 0.0),
            "advantage_std": getattr(self, "advantage_std", 0.0),
            "transport_metrics": dict(self.transport_metrics),
        }

    def write_archive(self, archive):
        """Stream each block into the checkpoint; never assemble the whole cohort."""
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            with archive.open(f"cohort/block-{b:06d}.npy", "w", force_zip64=True) as stream:
                np.save(stream, rows, allow_pickle=False)

    @classmethod
    def restore_archive(cls, archive, state, workspace):
        source = Path(workspace) / "restore"
        source.mkdir()
        try:
            for b in range((state["size"] + state["block_rows"] - 1) // state["block_rows"]):
                name = f"block-{b:06d}.npy"
                with archive.open("cohort/" + name) as inp, (source / name).open("wb") as out:
                    shutil.copyfileobj(inp, out, length=1024 * 1024)
            return cls.restore(source, state, workspace)
        finally:
            for path in source.glob("block-*.npy"):
                path.unlink()
            source.rmdir()

    @classmethod
    def restore(cls, source, state, workspace):
        source = Path(source)
        obj = cls(
            workspace,
            **{
                k: state[k]
                for k in ("observation_size", "capacity", "n_envs", "ram_bytes", "block_rows")
            },
        )
        # Copy into new writable workspace; original checkpoint remains immutable.
        for b in range((state["size"] + obj.block_rows - 1) // obj.block_rows):
            block = np.load(source / f"block-{b:06d}.npy", mmap_mode="r", allow_pickle=False)
            obj.append(block)
        for k in ("finalized", "group_counts", "advantage_mean", "advantage_std"):
            setattr(obj, k, state[k])
        obj.transport_metrics.update(state.get("transport_metrics", {}))
        return obj

    def close(self):
        self.cache = None
        for block in self.blocks:
            if isinstance(block, np.memmap):
                block.flush()
                block._mmap.close()
        self.blocks.clear()
        # Only transient files owned by this buffer are removed.
        if self.path.is_dir():
            for path in self.path.glob("block-*.npy"):
                path.unlink()
            try:
                self.path.rmdir()
            except OSError:
                pass
