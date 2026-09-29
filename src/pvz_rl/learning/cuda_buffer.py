"""Bounded metadata and ragged entity storage for complete recurrent games."""

import shutil
from pathlib import Path

import numpy as np
import torch

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import (
    ENTITY_WIDTH,
    GLOBAL_WIDTH,
    EntityBatch,
    collate_observations,
    validate_entity_records,
)
from pvz_rl.learning.objective import COMPONENTS, MAX_PROBES, training_rewards, victory_time

TRAJECTORY_PROTOCOL = "pvz-rl/entity-probe-trajectory-v2"


def probe_dtype():
    return np.dtype(
        [
            ("valid", "?"),
            ("branch_role", "?"),
            ("action", "<u2"),
            ("executed_action", "<u2"),
            ("accepted", "?"),
            ("reason", "u1"),
            ("duration", "<u2"),
            ("tick", "<u4"),
            ("done", "?"),
            ("won", "?"),
            ("components", "<f8", len(COMPONENTS)),
            ("bootstrap", "<f4"),
            ("target", "<f4"),
            ("entity_offset", "<u8"),
            ("entity_count", "<u4"),
            ("globals", "<f4", GLOBAL_WIDTH),
        ]
    )


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
            ("components", "<f8", len(COMPONENTS)),
            ("components_valid", "?"),
            ("won", "?"),
            ("home_entries", "<u4", 2),
            ("probes", probe_dtype(), MAX_PROBES),
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
    def __init__(
        self, path, n_envs, *, ram_bytes=6 * 1024**3, block_rows=32768, schema=None, cfg=None
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dtype = trajectory_dtype()
        self.n_envs, self.ram_bytes, self.block_rows = n_envs, int(ram_bytes), int(block_rows)
        self.schema = schema
        self.cfg = cfg
        self.rewards_finalized = False
        self.reward_summary = {}
        self.probe_counts = np.zeros((2, 10), np.int64)
        self.blocks, self.entity_blocks = [], []
        self.size = self.entity_size = self.ram_used = 0
        self.staging_bytes = 0
        self.finalized = False
        self._position_cache = {}
        self._entity_gather_workspace = None
        self.transport_metrics = dict(
            preparation_seconds=0.0, transfer_wait_seconds=0.0, transfer_stream_seconds=0.0
        )

    def _allocate(self, entity=False):
        blocks = self.entity_blocks if entity else self.blocks
        shape = (self.block_rows, ENTITY_WIDTH) if entity else (self.block_rows,)
        dtype = np.dtype("<i4") if entity else self.dtype
        amount = int(np.prod(shape)) * dtype.itemsize
        if self.ram_used + self.staging_bytes + amount <= self.ram_bytes:
            block = np.zeros(shape, dtype=dtype)
            self.ram_used += amount
        else:
            name = "entities" if entity else "block"
            block = np.lib.format.open_memmap(
                self.path / f"{name}-{len(blocks):06d}.npy", mode="w+", dtype=dtype, shape=shape
            )
        blocks.append(block)

    def reserve_staging(self, amount):
        """Charge reusable pinned buffers to the same budget, spilling RAM if needed."""
        if amount > self.ram_bytes:
            return False
        self.staging_bytes = amount
        for prefix, blocks in (("entities", self.entity_blocks), ("block", self.blocks)):
            for i, block in enumerate(blocks):
                if self.ram_used + amount <= self.ram_bytes:
                    return True
                if isinstance(block, np.memmap):
                    continue
                mapped = np.lib.format.open_memmap(
                    self.path / f"{prefix}-{i:06d}.npy",
                    mode="w+",
                    dtype=block.dtype,
                    shape=block.shape,
                )
                mapped[:] = block
                blocks[i] = mapped
                self.ram_used -= block.nbytes
        return True

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

    def append(self, records, observations, probe_observations=None):
        if self.finalized or self.rewards_finalized:
            raise RuntimeError("Cannot append after complete-return finalization")
        records = np.array(records, dtype=self.dtype, copy=True)
        observations = collate_observations(observations).reshape(-1).cpu()
        if len(records) != len(observations):
            raise ValueError("Each transition needs one public observation")
        entities, mask, globals_ = (x.numpy() for x in observations.tensors())
        counts = mask.sum(-1)
        packed = entities[mask]
        validate_entity_records(packed)
        if not np.isfinite(globals_).all():
            raise ValueError("Non-finite trajectory globals")
        records["entity_offset"] = self.entity_size + np.cumsum(counts) - counts
        records["entity_count"], records["globals"] = counts, globals_
        self._append(packed, entity=True)
        if probe_observations is not None:
            previous = []
            for slot, observation in enumerate(probe_observations):
                host = observation.cpu()
                pe, pm, pg = (x.numpy() for x in host.tensors())
                pc = pm.sum(-1)
                valid = records["probes"][:, slot]["valid"]
                validate_entity_records(pe[pm & valid[:, None]])
                if not np.isfinite(pg[valid]).all():
                    raise ValueError("Non-finite probe globals")
                unique = valid.copy()
                dest = records["probes"][:, slot]
                dest["globals"] = pg
                for prior, (oe, oc, og) in enumerate(previous):
                    width = min(pe.shape[1], oe.shape[1])
                    equal = (
                        unique
                        & records["probes"][:, prior]["valid"]
                        & (pc == oc)
                        & (pg == og).all(-1)
                        & (pe[:, :width] == oe[:, :width]).all(axis=(1, 2))
                    )
                    dest["entity_offset"][equal] = records["probes"][:, prior]["entity_offset"][
                        equal
                    ]
                    dest["entity_count"][equal] = pc[equal]
                    unique[equal] = False
                count = np.where(unique, pc, 0)
                dest["entity_offset"][unique] = (self.entity_size + np.cumsum(count) - count)[
                    unique
                ]
                dest["entity_count"][unique] = count[unique]
                self._append(pe[pm & unique[:, None]], entity=True)
                previous.append((pe, pc, pg))
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

    def observations(self, rows, device=None, *, destination=None):
        flat = rows.reshape(-1)
        starts, counts = flat["entity_offset"], flat["entity_count"]
        if np.any(starts > self.entity_size) or np.any(counts > self.entity_size - starts):
            raise ValueError("Corrupt trajectory entity offset/count")
        width = int(counts.max(initial=0))
        if destination is None:
            result = EntityBatch(
                torch.empty(len(flat), width, ENTITY_WIDTH, dtype=torch.int32),
                torch.empty(len(flat), width, dtype=torch.bool),
                torch.empty(len(flat), GLOBAL_WIDTH),
                validated=True,
            )
        else:
            result = EntityBatch(
                destination.entities[: len(flat), :width],
                destination.entity_mask[: len(flat), :width],
                destination.globals[: len(flat)],
                validated=True,
            )
        entities, mask, globals_ = (x.numpy() for x in result.tensors())
        positions = self._position_cache.get(width)
        if positions is None:
            positions = np.arange(width, dtype=np.int64)
            self._position_cache[width] = positions
        mask[:] = positions[None] < counts[:, None]
        entities.fill(0)
        globals_[:] = flat["globals"]
        indices = (starts[:, None] + positions[None])[mask].astype(np.int64, copy=False)
        blocks, offsets = np.divmod(indices, self.block_rows)
        shape = (len(indices), ENTITY_WIDTH)
        if self._entity_gather_workspace is None or self._entity_gather_workspace.shape != shape:
            self._entity_gather_workspace = np.empty(shape, np.int32)
        packed = self._entity_gather_workspace
        for b in np.unique(blocks):
            selected = blocks == b
            packed[selected] = self.entity_blocks[b][offsets[selected]]
        entities[mask] = packed
        return result.reshape(*rows.shape).to(device) if device else result.reshape(*rows.shape)

    def valid_indices(self):
        parts = []
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            parts.append(np.flatnonzero(rows["active"]) + b * self.block_rows)
        return np.concatenate(parts) if parts else np.empty(0, np.int64)

    def finalize_rewards(self):
        if self.rewards_finalized:
            return self.reward_summary
        terminal = []
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            terminal.extend(rows[rows["active"] & rows["done"]])
        durations = np.asarray([int(r["tick"]) + int(r["duration"]) for r in terminal], np.float64)
        median = float(np.median(durations)) if len(durations) else 0.0
        weight = self.cfg["reward"].get("win_time_weight", 0) if self.cfg else 0
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            eligible = rows["active"] & rows["done"] & rows["components_valid"]
            rows["components"][:, 5] = victory_time(
                rows["tick"].astype(np.float64) + rows["duration"],
                median,
                eligible & rows["won"],
                weight,
            )
            real = rows["components_valid"]
            rows["reward"][real] = rows["components"][real].sum(-1)
            probes = rows["probes"]
            probes["components"][..., 5] = victory_time(
                probes["tick"], median, probes["valid"] & probes["done"] & probes["won"], weight
            )
            if self.cfg:
                probes["target"] = training_rewards(probes["components"], self.cfg) + np.where(
                    probes["done"], 0, probes["bootstrap"]
                )
        # The pinned simulator clock is the only clock used for the median.
        from pvz_game import Rules

        self.reward_summary = dict(
            duration_median_seconds=median / Rules().game["tick_rate"],
            duration_reference_count=len(durations),
            victory_time_adjustments=[float(r["components"][5]) for r in terminal if r["won"]],
            home_entries=np.sum([
                block[:min(self.block_rows, self.size - b * self.block_rows)]["home_entries"].sum(0)
                for b, block in enumerate(self.blocks)
            ], axis=0).tolist() if self.blocks else [0, 0],
        )
        self.rewards_finalized = True
        return self.reward_summary

    def finalize(self):
        """Actual per-game returns and errors measured before fitting either head."""
        self.finalize_rewards()
        running = np.zeros(self.n_envs, np.float64)
        contribution = np.zeros((self.n_envs, len(COMPONENTS)), np.float64)
        component_sum = np.zeros(len(COMPONENTS), np.float64)
        component_square = np.zeros(len(COMPONENTS), np.float64)
        target_sum = target_square = 0.0
        episode_sum = np.zeros(self.n_envs, np.float64)
        episode_square = np.zeros(self.n_envs, np.float64)
        episode_count = np.zeros(self.n_envs, np.int64)
        quantile_sample = []
        self.probe_counts[:] = 0
        counts = np.zeros(3, np.int64)
        species_counts = np.zeros(A.plant_types, np.int64)
        sums = np.zeros((3, 4), np.float64)
        for b in reversed(range(len(self.blocks))):
            rows = self.blocks[b][: min(self.block_rows, self.size - b * self.block_rows)]
            for env in range(self.n_envs):
                ix = np.flatnonzero(rows["active"] & (rows["env"] == env))
                rewards = rows["reward"][ix]
                if self.cfg:
                    rewards = np.where(
                        rows["components_valid"][ix],
                        training_rewards(rows["components"][ix], self.cfg),
                        rewards,
                    )
                values = np.cumsum(rewards[::-1], dtype=np.float64)[::-1] + running[env]
                if len(ix):
                    running[env] = values[0]
                rows["target"][ix] = values
                parts = rows["components"][ix].copy()
                if self.cfg:
                    parts[:, 1] *= (
                        self.cfg["training"].get("objective", {}).get("dense_multiplier", 1)
                    )
                returns = np.cumsum(parts[::-1], axis=0)[::-1] + contribution[env]
                if len(ix):
                    contribution[env] = returns[0]
                component_sum += returns.sum(0)
                component_square += np.square(returns).sum(0)
                target_sum += values.sum()
                target_square += np.square(values).sum()
                episode_sum[env] += values.sum()
                episode_square[env] += np.square(values).sum()
                episode_count[env] += len(values)
                # Deterministic bounded sample, separately labeled from exact moments.
                quantile_sample.extend(values[:: max(1, self.size // 8192)].tolist())
            probes = rows["probes"]
            branch = np.where(
                probes["action"] == 0, 0, 1 + (probes["action"].astype(np.int64) - 1) // A.tiles
            )
            for head, valid in enumerate(
                (probes["valid"] & probes["branch_role"], probes["valid"] & (branch > 0))
            ):
                self.probe_counts[head] += np.bincount(branch[valid], minlength=10)
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
        n = max(1, counts.sum())
        variance = max(0, target_square / n - (target_sum / n) ** 2)
        self.reward_summary.update(
            target_std=float(np.sqrt(variance)),
            constant_predictor_mse=float(variance),
            component_return_mean=dict(zip(COMPONENTS, (component_sum / n).tolist())),
            component_return_std=dict(
                zip(
                    COMPONENTS,
                    np.sqrt(
                        np.maximum(0, component_square / n - (component_sum / n) ** 2)
                    ).tolist(),
                )
            ),
            probe_counts=self.probe_counts.tolist(),
        )
        live = episode_count > 0
        self.reward_summary["within_episode_target_variance"] = (
            float(
                np.mean(
                    np.maximum(
                        0,
                        episode_square[live] / episode_count[live]
                        - (episode_sum[live] / episode_count[live]) ** 2,
                    )
                )
            )
            if live.any()
            else 0.0
        )
        self.reward_summary["sampled_target_quantiles"] = (
            np.quantile(quantile_sample, [0, 0.1, 0.5, 0.9, 1]).tolist() if quantile_sample else []
        )
        self.reward_summary["component_episode_mean"] = (
            dict(zip(COMPONENTS, contribution[live].mean(0).tolist())) if live.any() else {}
        )
        raw = contribution[live].copy()
        if self.cfg:
            raw[:, 1] /= self.cfg["training"].get("objective", {}).get("dense_multiplier", 1)
        self.reward_summary["raw_component_episode_mean"] = (
            dict(zip(COMPONENTS, raw.mean(0).tolist())) if live.any() else {}
        )
        import warnings

        dense_std = self.reward_summary["component_return_std"]["development"]
        if variance > 1e-20 and dense_std < 0.05 * np.sqrt(variance):
            warnings.warn(
                "Dense return variation remains below 5% of target variation",
                RuntimeWarning,
                stacklevel=2,
            )
        if live.any():
            penalty = np.abs(contribution[live, 2:4].sum(-1)).mean()
            terminal = np.abs(contribution[live, 0]).mean()
            if penalty > 0.25 * terminal:
                warnings.warn(
                    "Rejection penalties exceed 25% of mean absolute terminal reward",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
            cfg=self.cfg,
            rewards_finalized=self.rewards_finalized,
            reward_summary=self.reward_summary,
            probe_counts=self.probe_counts,
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
            workspace,
            **{k: state[k] for k in ("n_envs", "ram_bytes", "block_rows", "schema")},
            cfg=state.get("cfg"),
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
                    if entity:
                        validate_entity_records(array)
                    elif not np.isfinite(array["globals"]).all():
                        raise ValueError("Non-finite trajectory globals")
                    obj._append(array, entity)
            expected = 0
            # Align metadata reads to collection steps: each step appends source
            # records then its probe records. Sorting one bounded block suffices.
            width = max(1, obj.block_rows // obj.n_envs) * obj.n_envs
            for at in range(0, obj.size, width):
                rows = obj.take(np.arange(at, min(obj.size, at + width)))
                probes = rows["probes"][rows["probes"]["valid"]]
                starts = np.concatenate((rows["entity_offset"], probes["entity_offset"]))
                counts = np.concatenate((rows["entity_count"], probes["entity_count"]))
                if (
                    np.any(starts > obj.entity_size)
                    or np.any(counts > obj.entity_size - starts)
                    or (obj.schema and np.any(counts > obj.schema["max_entities"]))
                ):
                    raise ValueError("Corrupt trajectory entity offset/count")
                for field in ("globals", "components", "bootstrap", "target"):
                    if not np.isfinite(probes[field]).all():
                        raise ValueError("Non-finite probe metadata")
                intervals = np.unique(
                    np.stack((starts[counts > 0], (starts + counts)[counts > 0]), -1), axis=0
                )
                if len(intervals):
                    if intervals[0, 0] != expected or np.any(intervals[1:, 0] != intervals[:-1, 1]):
                        raise ValueError("Corrupt trajectory entity offset/count")
                    expected = int(intervals[-1, 1])
            if expected != obj.entity_size:
                raise ValueError("Orphan entity records in trajectory")
            for key in ("finalized", "group_counts", "species_counts"):
                setattr(obj, key, state[key])
            obj.transport_metrics.update(state.get("transport_metrics", {}))
            for key in ("rewards_finalized", "reward_summary", "probe_counts"):
                setattr(obj, key, state[key])
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
        self._position_cache.clear()
        self._entity_gather_workspace = None
