"""Bounded metadata and ragged entity storage for complete recurrent games."""

import shutil
from pathlib import Path

import numpy as np

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import ENTITY_WIDTH, GLOBAL_WIDTH, collate_observations

TRAJECTORY_PROTOCOL = "pvz-rl/entity-trajectory-v1"


def trajectory_dtype():
    return np.dtype(
        [
            ("entity_offset", "<u8"),
            ("entity_count", "<u4"),
            ("entity_omitted", "<u4", 3),
            ("globals", "<f4", GLOBAL_WIDTH),
            ("mask", "u1", (A.size + 7) // 8),
            ("action", "<u2"),
            ("previous", "<u2"),
            ("tick", "<u4"),
            ("reset", "?"),
            ("active", "?"),
            ("env", "<u2"),
            ("duration", "<u2"),
            ("reward", "<f8"),
            ("branch_value", "<f4"),
            ("tile_value", "<f4"),
            ("greedy_action", "<u2"),
            ("species_coin", "?"),
            ("tile_coin", "?"),
            ("target", "<f4"),
            ("previous_outcome", "<f4", 2),
            ("accepted", "?"),
            ("done", "?"),
            ("executed_action", "<u2"),
        ]
    )


class CompleteGameBuffer:
    def __init__(self, path, n_envs, *, ram_bytes=6 * 1024**3, block_rows=32768, schema=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dtype = trajectory_dtype()
        self.n_envs, self.ram_bytes, self.block_rows = n_envs, int(ram_bytes), int(block_rows)
        self.schema = schema
        self.blocks, self.entity_blocks = [], []
        self.size = self.entity_size = self.ram_used = 0
        self.finalized = False
        self.transport_metrics = dict(
            preparation_seconds=0.0, transfer_wait_seconds=0.0, transfer_stream_seconds=0.0
        )

    def _allocate(self, entity=False):
        blocks = self.entity_blocks if entity else self.blocks
        shape = (self.block_rows, ENTITY_WIDTH) if entity else (self.block_rows,)
        dtype = np.dtype("<i4") if entity else self.dtype
        amount = int(np.prod(shape)) * dtype.itemsize
        if self.ram_used + amount <= self.ram_bytes:
            block = np.zeros(shape, dtype=dtype)
            self.ram_used += amount
        else:
            name = "entities" if entity else "block"
            block = np.lib.format.open_memmap(
                self.path / f"{name}-{len(blocks):06d}.npy", mode="w+", dtype=dtype, shape=shape
            )
        blocks.append(block)

    def _append(self, values, entity=False):
        blocks = self.entity_blocks if entity else self.blocks
        key = "entity_size" if entity else "size"
        at = 0
        while at < len(values):
            b, offset = divmod(getattr(self, key), self.block_rows)
            if b == len(blocks):
                self._allocate(entity)
            take = min(len(values) - at, self.block_rows - offset)
            blocks[b][offset : offset + take] = values[at : at + take]
            setattr(self, key, getattr(self, key) + take)
            at += take

    def append(self, records, observations):
        if self.finalized:
            raise RuntimeError("Cannot append after complete-return finalization")
        records = np.array(records, dtype=self.dtype, copy=True)
        observations = collate_observations(observations).observations()
        if len(records) != len(observations):
            raise ValueError("Each transition needs one public observation")
        for row, obs in zip(records, observations, strict=True):
            row["entity_offset"], row["entity_count"] = self.entity_size, len(obs["entities"])
            row["globals"] = obs["globals"]
            self._append(obs["entities"], entity=True)
        self._append(records)

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Trajectory index outside collected history")
        result = np.empty(indices.shape, dtype=self.dtype)
        blocks, offsets = np.divmod(indices, self.block_rows)
        for b in np.unique(blocks):
            selected = blocks == b
            result[selected] = self.blocks[b][offsets[selected]]
        return result

    def observations(self, rows, device=None):
        observations = []
        for row in rows.reshape(-1):
            start, count = int(row["entity_offset"]), int(row["entity_count"])
            if start < 0 or start + count > self.entity_size:
                raise ValueError("Corrupt trajectory entity offset/count")
            entities = np.empty((count, ENTITY_WIDTH), np.int32)
            copied = 0
            while copied < count:
                b, offset = divmod(start + copied, self.block_rows)
                take = min(count - copied, self.block_rows - offset)
                entities[copied : copied + take] = self.entity_blocks[b][offset : offset + take]
                copied += take
            observations.append(dict(entities=entities, globals=row["globals"]))
        return collate_observations(observations, device).reshape(*rows.shape)

    def valid_indices(self):
        parts = []
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            parts.append(np.flatnonzero(rows["active"]) + b * self.block_rows)
        return np.concatenate(parts) if parts else np.empty(0, np.int64)

    def finalize(self):
        """Actual per-game returns and errors measured before fitting either head."""
        running = np.zeros(self.n_envs, np.float64)
        counts = np.zeros(3, np.int64)
        species_counts = np.zeros(A.plant_types, np.int64)
        sums = np.zeros((3, 4), np.float64)
        for b in reversed(range(len(self.blocks))):
            rows = self.blocks[b][: min(self.block_rows, self.size - b * self.block_rows)]
            for env in range(self.n_envs):
                ix = np.flatnonzero(rows["active"] & (rows["env"] == env))
                values = np.cumsum(rows["reward"][ix][::-1], dtype=np.float64)[::-1] + running[env]
                if len(ix):
                    running[env] = values[0]
                rows["target"][ix] = values
            kinds = np.where(rows["action"] == 0, 0, np.where(rows["action"] < A.dig_start, 1, 2))
            for group in range(3):
                ix = rows["active"] & (kinds == group)
                first = rows["branch_value"][ix].astype(np.float64) - rows["target"][ix]
                second = rows["tile_value"][ix].astype(np.float64) - rows["target"][ix]
                counts[group] += len(first)
                sums[group] += [
                    np.square(first).sum(),
                    first.sum(),
                    np.square(second).sum(),
                    second.sum(),
                ]
            planted = rows["active"] & (kinds == 1)
            species_counts += np.bincount(
                (rows["action"][planted] - 1) // A.tiles, minlength=A.plant_types
            )
        if not np.isfinite(sums).all() or not np.isfinite(running).all():
            raise FloatingPointError("Non-finite complete returns or collection Q values")
        self.finalized, self.group_counts, self.species_counts = True, counts, species_counts
        return {
            name: dict(
                count=int(counts[i]),
                mse=float(sums[i, 0] / counts[i]) if counts[i] else None,
                signed_error=float(sums[i, 1] / counts[i]) if counts[i] else None,
                tile_mse=float(sums[i, 2] / counts[i]) if i and counts[i] else None,
                tile_signed_error=float(sums[i, 3] / counts[i]) if i and counts[i] else None,
            )
            for i, name in enumerate(("wait", "plant", "dig"))
        }

    def _arrays(self):
        for prefix, blocks, size in (
            ("block", self.blocks, self.size),
            ("entities", self.entity_blocks, self.entity_size),
        ):
            for b, block in enumerate(blocks):
                yield (
                    f"{prefix}-{b:06d}.npy",
                    block[: min(self.block_rows, size - b * self.block_rows)],
                )

    def metadata(self):
        return dict(
            protocol=TRAJECTORY_PROTOCOL,
            schema=self.schema,
            n_envs=self.n_envs,
            ram_bytes=self.ram_bytes,
            block_rows=self.block_rows,
            size=self.size,
            entity_size=self.entity_size,
            finalized=self.finalized,
            group_counts=getattr(self, "group_counts", None),
            species_counts=getattr(self, "species_counts", None),
            transport_metrics=dict(self.transport_metrics),
        )

    def save(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name, array in self._arrays():
            target = destination / name
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(target)
        return self.metadata()

    def write_archive(self, archive):
        for name, array in self._arrays():
            with archive.open("cohort/" + name, "w", force_zip64=True) as stream:
                np.save(stream, array, allow_pickle=False)

    @staticmethod
    def validate_metadata(state):
        if state.get("protocol") != TRAJECTORY_PROTOCOL:
            raise ValueError("Retired trajectory schema; fresh initialization is required")
        for name, minimum in (
            ("n_envs", 1),
            ("block_rows", 1),
            ("size", 0),
            ("entity_size", 0),
            ("ram_bytes", 0),
        ):
            if type(state.get(name)) is not int or state[name] < minimum:
                raise ValueError(f"Corrupt trajectory metadata: {name}")

    @classmethod
    def restore_archive(cls, archive, state, workspace):
        cls.validate_metadata(state)
        source = Path(workspace) / "restore"
        source.mkdir()
        try:
            for prefix, size in (("block", state["size"]), ("entities", state["entity_size"])):
                for b in range((size + state["block_rows"] - 1) // state["block_rows"]):
                    name = f"{prefix}-{b:06d}.npy"
                    with archive.open("cohort/" + name) as inp, (source / name).open("wb") as out:
                        shutil.copyfileobj(inp, out, length=1024 * 1024)
            return cls.restore(source, state, workspace)
        finally:
            for path in source.glob("*.npy"):
                path.unlink()
            source.rmdir()

    @classmethod
    def restore(cls, source, state, workspace):
        cls.validate_metadata(state)
        obj = cls(
            workspace, **{k: state[k] for k in ("n_envs", "ram_bytes", "block_rows", "schema")}
        )
        try:
            for entity, prefix, size in (
                (False, "block", state["size"]),
                (True, "entities", state["entity_size"]),
            ):
                for b in range((size + obj.block_rows - 1) // obj.block_rows):
                    array = np.load(Path(source) / f"{prefix}-{b:06d}.npy", allow_pickle=False)
                    count = min(obj.block_rows, size - b * obj.block_rows)
                    shape = (count, ENTITY_WIDTH) if entity else (count,)
                    dtype = np.dtype("<i4") if entity else obj.dtype
                    if array.shape != shape or array.dtype != dtype:
                        raise ValueError("Corrupt trajectory block shape or dtype")
                    obj._append(array, entity)
            expected = 0
            for b, block in enumerate(obj.blocks):
                for row in block[: min(obj.block_rows, obj.size - b * obj.block_rows)]:
                    count = int(row["entity_count"])
                    if (
                        int(row["entity_offset"]) != expected
                        or expected + count > obj.entity_size
                        or (obj.schema and count > obj.schema["max_entities"])
                    ):
                        raise ValueError("Corrupt trajectory entity offset/count")
                    expected += count
            if expected != obj.entity_size:
                raise ValueError("Orphan entity records in trajectory")
            for key in ("finalized", "group_counts", "species_counts"):
                setattr(obj, key, state[key])
            obj.transport_metrics.update(state.get("transport_metrics", {}))
            return obj
        except Exception:
            obj.close()
            raise

    def close(self):
        for blocks in (self.blocks, self.entity_blocks):
            for block in blocks:
                if isinstance(block, np.memmap):
                    block.flush()
                    block._mmap.close()
            blocks.clear()
        if self.path.is_dir():
            for prefix in ("block", "entities"):
                for path in self.path.glob(prefix + "-*.npy"):
                    path.unlink()
            try:
                self.path.rmdir()
            except OSError:
                pass
