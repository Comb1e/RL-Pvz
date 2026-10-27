"""Bounded metadata and ragged entity storage for complete stateless games."""

import shutil
from pathlib import Path

import numpy as np
import torch

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import (
    ENTITY_WIDTH,
    GLOBAL_WIDTH,
    EntityBatch,
    PackedEntityBatch,
    collate_observations,
    validate_entity_records,
)
from pvz_rl.learning.action_routing import FitRoute, validate_routes
from pvz_rl.learning.objective import (
    COMPONENTS,
    endgame_terminal_rewards,
    raw_rewards,
    training_rewards,
)
from pvz_rl.learning.source_actions import AppendReceipt, SourceActionRef
from pvz_rl.monitoring.moments import RunningMoments

TRAJECTORY_PROTOCOL = "pvz-rl/action-global-trajectory-v4"


def trajectory_dtype(cfg=None):
    return np.dtype(
        [
            ("entity_offset", "<u8"),
            ("entity_count", "<u4"),
            ("entity_omitted", "<u4", 3),
            ("globals", "<f4", GLOBAL_WIDTH),
            ("action", "<u2"),
            ("tick", "<u4"),
            ("reset", "?"),
            ("active", "?"),
            ("env", "<u2"),
            ("episode", "<u8"),
            ("decision", "<u8"),
            ("run_break", "?"),
            ("duration", "<u2"),
            ("reward", "<f8"),
            ("components", "<f8", len(COMPONENTS)),
            ("components_valid", "?"),
            ("won", "?"),
            ("home_entries", "<u4", 2),
            ("branch_value", "<f4"),
            ("tile_value", "<f4"),
            ("placement_gap", "<f4"),
            ("greedy_action", "<u2"),
            ("policy_action", "<u2"),
            ("exploration_mode", "u1"),
            ("pending_species", "u1"),
            ("next_species", "u1"),
            ("species_coin", "?"),
            ("tile_coin", "?"),
            ("target", "<f4"),
            ("accepted", "?"),
            ("route", "u1"),
            ("branch_usable", "?"),
            ("rejection_reason", "u1"),
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
        self.dtype = trajectory_dtype(cfg)
        self.n_envs, self.ram_bytes, self.block_rows = n_envs, int(ram_bytes), int(block_rows)
        self.schema = schema
        self.cfg = cfg
        self.rewards_finalized = False
        self.reward_summary = {}
        self.exploration = {}
        self.outcome_counts = np.zeros((3, 2), np.int64)
        self.consistency_counts = np.zeros((2, 2), np.int64)
        self.blocks, self.entity_blocks = [], []
        self.size = self.entity_size = self.ram_used = 0
        self.staging_bytes = 0
        self.finalized = False
        self._position_cache = {}
        self._entity_gather_workspace = None
        self.transport_metrics = dict(
            preparation_seconds=0.0,
            transfer_wait_seconds=0.0,
            transfer_stream_seconds=0.0,
            producer_wait_seconds=0.0,
            transferred_bytes=0,
        )
        # Host-side causal source metadata is deliberately kept separate from
        # model observations and the fixed trajectory dtype.  A CUDA backend
        # may fill this table when it has detailed event provenance; CPU and
        # replay paths use the same mutation API.
        self.causal_sources = {}
        self._source_index = {}
        self._index_arrays = []
        self._source_episodes = np.zeros(n_envs, np.uint64)
        self._source_lengths = np.zeros(n_envs, np.int64)
        self._next_run_break = np.ones(n_envs, bool)

    def _index_records(self, records, base, *, validate_only=False):
        """Small per-episode directories; each decision maps through an integer slab."""
        # Collection supplies at most one row per original slot. Validate the
        # common case as arrays, avoiding per-slot NumPy searches every step.
        environments = records["env"]
        if len(records) and np.all(environments[1:] > environments[:-1]):
            episodes = records["episode"]
            prior = self._source_episodes[environments]
            lengths = np.where(episodes == prior, self._source_lengths[environments], 0)
            if np.any(episodes < prior) or not np.array_equal(records["decision"], lengths):
                raise ValueError("Duplicate, missing or out-of-order source action identity")
            if validate_only:
                return
            for index, (env, episode, length) in enumerate(
                zip(environments, episodes, lengths, strict=True)
            ):
                key = int(env), int(episode)
                blocks = self._source_index.get(key, ([], 0))[0]
                block, offset = divmod(int(length), self.block_rows)
                if block == len(blocks):
                    blocks.append(self._allocate_source_index())
                blocks[block][offset] = base + index
                self._source_index[key] = blocks, int(length) + 1
            self._source_episodes[environments] = episodes
            self._source_lengths[environments] = lengths + 1
            return
        for env in np.unique(records["env"]):
            positions = np.flatnonzero(records["env"] == env)
            episodes = records["episode"][positions]
            latest = max(
                (episode for slot, episode in self._source_index if slot == int(env)), default=0
            )
            if np.any(episodes[1:] < episodes[:-1]) or (len(episodes) and episodes[0] < latest):
                raise ValueError("Out-of-order source action episode identity")
            for episode in np.unique(records["episode"][positions]):
                selected = positions[records["episode"][positions] == episode]
                key = (int(env), int(episode))
                blocks, length = self._source_index.get(key, ([], 0))
                if not np.array_equal(
                    records["decision"][selected], np.arange(length, length + len(selected))
                ):
                    raise ValueError("Duplicate, missing or out-of-order source action identity")
                if validate_only:
                    continue
                for start in range(0, len(selected), self.block_rows):
                    part = selected[start : start + self.block_rows]
                    offset = 0
                    while offset < len(part):
                        block, at = divmod(length, self.block_rows)
                        if block == len(blocks):
                            blocks.append(self._allocate_source_index())
                        take = min(len(part) - offset, self.block_rows - at)
                        blocks[block][at : at + take] = base + part[offset : offset + take]
                        length += take
                        offset += take
                self._source_index[key] = blocks, length
                self._source_episodes[env], self._source_lengths[env] = episode, length

    def _allocate_source_index(self):
        amount = self.block_rows * 8
        if self.ram_used + self.staging_bytes + amount <= self.ram_bytes:
            value = np.empty(self.block_rows, dtype=np.uint64)
            self.ram_used += amount
        else:
            value = np.lib.format.open_memmap(
                self.path / f"source-index-{len(self._index_arrays):06d}.npy",
                mode="w+",
                dtype=np.uint64,
                shape=(self.block_rows,),
            )
        self._index_arrays.append(value)
        return value

    def source_row(self, reference):
        if not isinstance(reference, SourceActionRef):
            raise TypeError("Causal credit requires a SourceActionRef")
        blocks, length = self._source_index.get((reference.env, reference.episode), ([], 0))
        if reference.decision >= length:
            raise ValueError("Stale or unknown source action identity")
        block, offset = divmod(reference.decision, self.block_rows)
        return int(blocks[block][offset])

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
        for prefix, blocks in (
            ("entities", self.entity_blocks),
            ("block", self.blocks),
            ("source-index", self._index_arrays),
        ):
            for i, block in enumerate(blocks):
                if self.ram_used + amount <= self.ram_bytes:
                    self.staging_bytes = amount
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
                if prefix == "source-index":
                    for index_blocks, _ in self._source_index.values():
                        for j, existing in enumerate(index_blocks):
                            if existing is block:
                                index_blocks[j] = mapped
                self.ram_used -= block.nbytes
        if self.ram_used + amount > self.ram_bytes:
            return False
        self.staging_bytes = amount
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

    def append(self, records, observations, *, identities=None):
        if self.finalized or self.rewards_finalized:
            raise RuntimeError("Cannot append after complete-return finalization")
        records = np.array(records, dtype=self.dtype, copy=True)
        if not isinstance(observations, PackedEntityBatch):
            batch = collate_observations(observations).reshape(-1).cpu()
            entities, mask, globals_ = (value.numpy() for value in batch.tensors())
            counts = mask.sum(-1)
            observations = PackedEntityBatch(
                entities[mask], np.cumsum(counts) - counts, counts, globals_
            )
        validate_routes(records[records["active"]])
        if np.any(records["rejection_reason"] > 7):
            raise ValueError("Unknown trajectory rejection reason")
        observations.validate()
        if observations.counts.shape != (len(records),):
            raise ValueError("Each transition needs one public observation")
        if np.any(records["env"] >= self.n_envs):
            raise ValueError("Trajectory environment outside canonical slots")
        slots = np.flatnonzero(records["active"])
        next_run_break = self._next_run_break.copy()
        # Capture padding boundaries before removing inactive metadata slots.
        if np.all(records["env"][1:] > records["env"][:-1]):
            records["run_break"] |= next_run_break[records["env"]] | records["reset"]
            next_run_break[records["env"]] = ~records["active"]
        else:
            for env in np.unique(records["env"]):
                positions = np.flatnonzero(records["env"] == env)
                enabled = records["active"][positions]
                previous = np.r_[not next_run_break[env], enabled[:-1]]
                records["run_break"][positions] |= ~previous | records["reset"][positions]
                next_run_break[env] = not bool(enabled[-1])
        records = records[slots]
        counts = np.asarray(observations.counts)[slots].astype(np.int64)
        offsets = np.asarray(observations.offsets)[slots].astype(np.int64)
        starts = np.cumsum(counts) - counts
        positions = (
            np.repeat(offsets, counts) + np.arange(int(counts.sum())) - np.repeat(starts, counts)
        )
        observations = PackedEntityBatch(
            observations.entities[positions],
            starts,
            counts,
            np.asarray(observations.globals)[slots],
        )
        if identities is None:
            for env in np.unique(records["env"]):
                for episode in np.unique(records["episode"][records["env"] == env]):
                    selected = np.flatnonzero(
                        (records["env"] == env) & (records["episode"] == episode)
                    )
                    length = self._source_index.get((int(env), int(episode)), ([], 0))[1]
                    records["decision"][selected] = np.arange(length, length + len(selected))
        else:
            episodes, decisions = identities
            records["episode"] = np.asarray(episodes)[slots]
            records["decision"] = np.asarray(decisions)[slots]
        self._index_records(records, self.size, validate_only=True)
        records["entity_offset"] = self.entity_size + observations.offsets
        records["entity_count"], records["globals"] = observations.counts, observations.globals
        base = self.size
        self._append(observations.entities, entity=True)
        self._append(records)
        self._index_records(records, base)
        self._next_run_break = next_run_break
        return AppendReceipt(
            slots,
            np.arange(base, self.size),
            records["episode"].copy(),
            records["decision"].copy(),
            records["env"].copy(),
        )

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

    @staticmethod
    def observations_from_packed(
        packed, *, destination=None, device=None, validated=False, width=None
    ):
        """Pad a fitting-spool packed batch into the transport shape."""
        if not validated:
            packed.validate()
        counts = np.asarray(packed.counts, dtype=np.int64)
        offsets = np.asarray(packed.offsets, dtype=np.int64)
        if (
            np.any(counts < 0)
            or np.any(offsets < 0)
            or np.any(offsets + counts > len(packed.entities))
        ):
            raise ValueError("Corrupt packed entity offset/count")
        required = int(counts.max(initial=0))
        width = required if width is None else int(width)
        if width < required:
            raise ValueError("Fitting width cannot truncate entities")
        if destination is not None and (
            len(counts) > len(destination) or width > destination.entities.shape[-2]
        ):
            raise ValueError("Fitting batch exceeds pinned capacity")
        if destination is None:
            result = EntityBatch(
                torch.empty(len(counts), width, ENTITY_WIDTH, dtype=torch.int32),
                torch.empty(len(counts), width, dtype=torch.bool),
                torch.empty(len(counts), GLOBAL_WIDTH),
                validated=True,
            )
        else:
            result = EntityBatch(
                destination.entities[: len(counts), :width],
                destination.entity_mask[: len(counts), :width],
                destination.globals[: len(counts)],
                validated=True,
            )
        entities, mask, globals_ = result.tensors()
        host_entities, host_mask, host_globals = (value.numpy() for value in result.tensors())
        host_entities.fill(0)
        host_mask[:] = np.arange(width)[None, :] < counts[:, None]
        host_globals[:] = packed.globals
        positions = offsets[:, None] + np.arange(width)[None, :]
        host_entities[host_mask] = packed.entities[positions[host_mask]]
        return result.to(device) if device else result

    def valid_indices(self):
        parts = []
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            parts.append(np.flatnonzero(rows["active"]) + b * self.block_rows)
        return np.concatenate(parts) if parts else np.empty(0, np.int64)

    def _row_view(self, index):
        index = int(index)
        if index < 0 or index >= self.size:
            raise IndexError("Causal source row outside collected history")
        block, offset = divmod(index, self.block_rows)
        return self.blocks[block][offset]

    def apply_causal_credits(self, credits):
        """Apply delayed action credit to already-collected origin rows.

        ``credits`` accepts either ``{row_index: {field: value}}`` or an
        iterable of mappings containing ``row_index`` and ``credits``.  The
        source mapping may also include ``env`` and ``episode``; when present
        the row's environment is checked before mutation.  This small host API
        is shared by CPU replay and CUDA event adapters, and keeps provenance
        out of the public observation tensor.

        Credit fields are the ledger names (``damage``, ``blocking``, ``sun``,
        ``mower``) or their trajectory component names.  All updates are
        atomic with respect to validation: a malformed or stale batch raises
        before any row is changed.
        """
        if self.finalized or self.rewards_finalized:
            raise RuntimeError("Causal credits must be applied before return finalization")
        if isinstance(credits, dict):
            entries = [
                {
                    ("action_ref" if isinstance(index, SourceActionRef) else "row_index"): index,
                    "credits": values,
                }
                for index, values in credits.items()
            ]
        else:
            entries = list(credits)
        normalized = []
        field_map = {
            "damage": "action_damage",
            "blocking": "action_blocking",
            "sun": "action_sun",
            "mower": "action_mower",
        }
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("Causal credit entries must be mappings")
            reference = entry.get("action_ref")
            if reference is not None:
                index = self.source_row(reference)
            else:
                index = entry.get("row_index", entry.get("index"))
            if type(index) is not int:
                raise ValueError("Causal credit entry needs an integer row_index")
            row = self._row_view(index)
            if "env" in entry and int(row["env"]) != int(entry["env"]):
                raise ValueError("Causal credit environment does not match trajectory row")
            if "episode" in entry and int(row["episode"]) != int(entry["episode"]):
                raise ValueError("Causal credit episode does not match trajectory row")
            values = entry.get("credits", entry.get("values", entry))
            if not isinstance(values, dict):
                raise TypeError("Causal credit values must be a mapping")
            updates = {}
            for key, value in values.items():
                target = field_map.get(key, key)
                if target not in ("action_damage", "action_blocking", "action_sun", "action_mower"):
                    continue
                if type(value) not in (int, float) or not np.isfinite(value):
                    raise ValueError(f"Non-finite causal credit for {target}")
                updates[target] = updates.get(target, 0.0) + float(value)
            normalized.append((index, updates, entry.get("source")))
        # Validate every target before changing any row, then apply in one
        # deterministic pass.  Duplicate entries intentionally add together.
        for index, updates, source in normalized:
            row = self._row_view(index)
            for key, value in updates.items():
                row["components"][COMPONENTS.index(key)] += value
            if updates:
                config = self.cfg
                if config is None:
                    from pvz_rl.config import load_config

                    config = load_config()
                row["reward"] = raw_rewards(row["components"], config)
                self.causal_sources.setdefault(index, []).append(
                    {"source": source, "credits": dict(updates)}
                )
        return {"rows_updated": len({index for index, updates, _ in normalized if updates})}

    def finalize_rewards(self):
        if self.rewards_finalized:
            return self.reward_summary
        terminal = []
        for b, block in enumerate(self.blocks):
            rows = block[: min(self.block_rows, self.size - b * self.block_rows)]
            for index in np.flatnonzero(rows["active"] & rows["done"] & rows["components_valid"]):
                row = rows[index]
                terminal.append(
                    (
                        block,
                        int(index),
                        float(int(row["tick"]) + int(row["duration"])),
                        bool(row["won"]),
                    )
                )
        durations = np.asarray([item[2] for item in terminal], np.float64)
        wins = np.asarray([item[3] for item in terminal], dtype=bool)
        if self.cfg and len(terminal):
            terminal_rewards, adjustments = endgame_terminal_rewards(
                durations,
                wins,
                win_reward=self.cfg["reward"]["win_reward"],
                loss_penalty=self.cfg["reward"]["loss_penalty"],
                weight=self.cfg["reward"].get("win_time_weight", 0.5),
            )
            for (block, index, _, _won), value in zip(terminal, terminal_rewards, strict=True):
                row = block[index]
                row["components"][0] = value
                row["reward"] = raw_rewards(row["components"], self.cfg)
        else:
            terminal_rewards = np.asarray([], dtype=np.float64)
            adjustments = np.asarray([], dtype=np.float64)
        median = float(np.median(durations)) if len(durations) else 0.0
        # The pinned simulator clock is the only clock used for the median.
        from pvz_game import Rules

        self.reward_summary = dict(
            duration_median_seconds=median / Rules().game["tick_rate"],
            duration_reference_count=len(durations),
            endgame_terminal_rewards=terminal_rewards.tolist(),
            endgame_time_adjustments=adjustments.tolist(),
            endgame_time_weight=(
                self.cfg["reward"].get("win_time_weight", 0.5) if self.cfg else 0.0
            ),
            # Compatibility alias for existing viewers and reports.
            victory_time_adjustments=adjustments.tolist(),
            home_entries=np.sum(
                [
                    block[: min(self.block_rows, self.size - b * self.block_rows)][
                        "home_entries"
                    ].sum(0)
                    for b, block in enumerate(self.blocks)
                ],
                axis=0,
            ).tolist()
            if self.blocks
            else [0, 0],
        )
        self.rewards_finalized = True
        return self.reward_summary

    def finalize(self, *, cohort=None):
        """Actual per-game returns and errors measured before fitting either head."""
        self.finalize_rewards()
        running = np.zeros(self.n_envs, np.float64)
        running_episode = [None] * self.n_envs
        contribution = np.zeros((self.n_envs, len(COMPONENTS)), np.float64)
        component_sum = np.zeros(len(COMPONENTS), np.float64)
        component_square = np.zeros(len(COMPONENTS), np.float64)
        immediate_component_sum = np.zeros(len(COMPONENTS), np.float64)
        target_moments, dense_moments = RunningMoments(), RunningMoments()
        episode_sum = np.zeros(self.n_envs, np.float64)
        episode_square = np.zeros(self.n_envs, np.float64)
        episode_count = np.zeros(self.n_envs, np.int64)
        quantile_sample = []
        self.outcome_counts[:] = 0
        counts = np.zeros(3, np.int64)
        routed_counts = np.zeros(4, np.int64)
        from pvz_rl.config import load_config

        invalid_target = (self.cfg or load_config())["training"]["objective"][
            "invalid_action_target"
        ]
        species_counts = np.zeros(A.plant_types, np.int64)
        sums = np.zeros((3, 3), np.float64)
        for b in reversed(range(len(self.blocks))):
            rows = self.blocks[b][: min(self.block_rows, self.size - b * self.block_rows)]
            routed_counts += np.bincount(rows["route"][rows["active"]], minlength=4)
            for env in range(self.n_envs):
                ix = np.flatnonzero(rows["active"] & (rows["env"] == env))
                rewards = rows["reward"][ix]
                if self.cfg:
                    rewards = np.where(
                        rows["components_valid"][ix],
                        training_rewards(rows["components"][ix], self.cfg),
                        rewards,
                    )
                parts = rows["components"][ix].copy()
                immediate_component_sum += parts.sum(0)
                if self.cfg:
                    parts[:, 1:-1] *= (
                        self.cfg["training"].get("objective", {}).get("dense_multiplier", 1)
                    )
                values = np.empty(len(ix), np.float64)
                returns = np.empty_like(parts)
                episodes, terminals = rows["episode"][ix], rows["done"][ix]
                boundaries = np.r_[
                    0, np.flatnonzero((episodes[1:] != episodes[:-1]) | terminals[:-1]) + 1, len(ix)
                ]
                for low, high in reversed(list(zip(boundaries[:-1], boundaries[1:], strict=True))):
                    if high == low:
                        continue
                    if running_episode[env] != int(episodes[high - 1]) or terminals[high - 1]:
                        running[env] = 0
                        contribution[env] = 0
                    values[low:high] = (
                        np.cumsum(rewards[low:high][::-1], dtype=np.float64)[::-1] + running[env]
                    )
                    returns[low:high] = (
                        np.cumsum(parts[low:high][::-1], axis=0)[::-1] + contribution[env]
                    )
                    running[env], contribution[env] = values[low], returns[low]
                    running_episode[env] = int(episodes[low])
                rows["target"][ix] = values
                component_sum += returns.sum(0)
                component_square += np.square(returns).sum(0)
                target_moments.update(values)
                # Component returns already include the dense multiplier. Sum
                # before taking moments to retain covariance, then convert once.
                scale = (
                    self.cfg["reward"]["progress_weight"] / self.cfg["reward"]["value_scale"]
                    if self.cfg
                    else 1.0
                )
                dense_moments.update(scale * returns[:, 1:-1].sum(-1))
                episode_sum[env] += values.sum()
                episode_square[env] += np.square(values).sum()
                episode_count[env] += len(values)
                # Deterministic bounded sample, separately labeled from exact moments.
                quantile_sample.extend(values[:: max(1, self.size // 8192)].tolist())
            kinds = np.where(rows["action"] == 0, 0, np.where(rows["action"] < A.dig_start, 1, 2))
            for group in range(3):
                ix = rows["active"] & (kinds == group)
                species_only = (rows["route"][ix] == FitRoute.WAIT_RETURN) | (
                    rows["route"][ix] == FitRoute.SPECIES_INVALID
                )
                selected = np.where(species_only, rows["branch_value"][ix], rows["tile_value"][ix])
                target = np.where(
                    rows["route"][ix] >= FitRoute.SPECIES_INVALID,
                    invalid_target,
                    rows["target"][ix],
                )
                first = selected.astype(np.float64) - target
                second = rows["placement_gap"][ix].astype(np.float64)
                counts[group] += len(first)
                self.outcome_counts[group] += np.bincount(
                    rows["accepted"][ix].astype(np.int64), minlength=2
                )
                sums[group] += [
                    np.square(first).sum(),
                    first.sum(),
                    second.sum(),
                ]
            planted = rows["active"] & (kinds == 1)
            species_counts += np.bincount(
                (rows["action"][planted] - 1) // A.tiles, minlength=A.plant_types
            )
        if not np.isfinite(sums).all() or not np.isfinite(running).all():
            raise FloatingPointError("Non-finite complete returns or collection Q values")
        self.finalized, self.group_counts, self.species_counts = True, counts, species_counts
        self.consistency_counts[:] = self.outcome_counts[1:]
        self.consistency_counts[:, 0] = 0
        n = max(1, counts.sum())
        variance = target_moments.variance
        target_std = float(np.sqrt(variance))
        dense_std = float(np.sqrt(dense_moments.variance))
        ratio = dense_std / target_std if variance > 1e-20 else None
        low_variation = ratio is not None and ratio < 0.05
        self.reward_summary.update(
            cohort=cohort,
            target_std=target_std,
            dense_return_std=dense_std,
            dense_to_target_std_ratio=ratio,
            low_dense_return_variation=low_variation,
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
            outcome_counts=self.outcome_counts.tolist(),
            consistency_counts=self.consistency_counts.tolist(),
            routed_workload=dict(
                routes={route.name.lower(): int(routed_counts[route]) for route in FitRoute},
                online_species_rows=int(routed_counts.sum() - routed_counts[FitRoute.TILE_INVALID]),
                online_tile_rows=int(
                    routed_counts[FitRoute.ACCEPTED] + routed_counts[FitRoute.TILE_INVALID]
                ),
                ema_tile_rows=int(routed_counts[FitRoute.ACCEPTED]),
                skipped_ema_rows=int(
                    routed_counts[FitRoute.SPECIES_INVALID] + routed_counts[FitRoute.TILE_INVALID]
                ),
            ),
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
            raw[:, 1:-1] /= self.cfg["training"].get("objective", {}).get("dense_multiplier", 1)
        self.reward_summary["raw_component_episode_mean"] = (
            dict(zip(COMPONENTS, raw.mean(0).tolist())) if live.any() else {}
        )
        self.reward_summary["global_reward_totals"] = {
            key: float(immediate_component_sum[index])
            for index, key in enumerate(("global_terminal", "global_asset_value"))
        }
        self.reward_summary["action_reward_totals"] = {
            key: float(immediate_component_sum[index])
            for index, key in enumerate(COMPONENTS[2:], start=2)
        }
        self.reward_summary["source_action_credited_rows"] = len(self.causal_sources)
        self.reward_summary["causal_source_credits"] = self.causal_sources
        import warnings

        if low_variation:
            warnings.warn(
                f"Cohort {cohort if cohort is not None else self.path.name}: dense return "
                f"variation is below 5% of target variation "
                f"(dense_std={dense_std:.8g}, target_std={target_std:.8g}, "
                f"ratio={ratio:.4%}; fitted reward units)",
                RuntimeWarning,
                stacklevel=2,
            )
        if live.any():
            scale = (
                self.cfg["reward"]["progress_weight"] / self.cfg["reward"]["value_scale"]
                if self.cfg
                else 1.0
            )
            penalty = np.abs(
                scale * contribution[live, 2:-1].sum(-1) + contribution[live, -1]
            ).mean()
            terminal = np.abs(contribution[live, 0]).mean()
            if penalty > 0.25 * terminal:
                warnings.warn(
                    "Action shaping exceeds 25% of mean absolute terminal reward",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return {
            name: dict(
                count=int(counts[i]),
                mse=float(sums[i, 0] / counts[i]) if counts[i] else None,
                signed_error=float(sums[i, 1] / counts[i]) if counts[i] else None,
                placement_gap=float(sums[i, 2] / counts[i]) if i and counts[i] else None,
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
            causal_sources=self.causal_sources,
            outcome_counts=self.outcome_counts,
            consistency_counts=self.consistency_counts,
            exploration=dict(self.exploration),
            next_run_break=self._next_run_break.tolist(),
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
        if any(
            key in state for key in ("probes", "probe_pending", "pending_probes", "probe_endpoints")
        ):
            raise ValueError("Counterfactual trajectory metadata is no longer supported")
        exploration = state.get("exploration")
        if not isinstance(exploration, dict) or (
            exploration
            and (
                set(exploration) != {"plant_epsilon", "tile_epsilon"}
                or any(
                    type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1
                    for value in exploration.values()
                )
            )
        ):
            raise ValueError("Corrupt trajectory exploration probabilities")
        for name, minimum in (
            ("n_envs", 1),
            ("block_rows", 1),
            ("size", 0),
            ("entity_size", 0),
            ("ram_bytes", 0),
        ):
            if type(state.get(name)) is not int or state[name] < minimum:
                raise ValueError(f"Corrupt trajectory metadata: {name}")
        for name, shape in (
            ("outcome_counts", (3, 2)),
            ("consistency_counts", (2, 2)),
        ):
            counts = np.asarray(state.get(name))
            if counts.shape != shape or counts.dtype.kind not in "iu" or np.any(counts < 0):
                raise ValueError(f"Corrupt trajectory acceptance counts: {name}")

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
            entity_intervals = []
            actual_counts = np.zeros((3, 2), np.int64)
            width = max(1, obj.block_rows // obj.n_envs) * obj.n_envs
            for at in range(0, obj.size, width):
                rows = obj.take(np.arange(at, min(obj.size, at + width)))
                if np.any(rows["action"] >= A.size) or np.any(rows["executed_action"] >= A.size):
                    raise ValueError("Corrupt trajectory action")
                if not all(
                    np.isfinite(rows[name]).all()
                    for name in ("branch_value", "tile_value", "placement_gap")
                ):
                    raise ValueError("Non-finite trajectory Q values or placement gap")
                if not np.all(rows["active"]) or np.any(rows["env"] >= obj.n_envs):
                    raise ValueError("Compact trajectories contain only active canonical rows")
                obj._index_records(rows, at)
                actual = rows
                validate_routes(actual)
                if np.any(actual["rejection_reason"] > 7):
                    raise ValueError("Unknown trajectory rejection reason")
                if (
                    np.any(rows["policy_action"] >= A.size)
                    or np.any(rows["exploration_mode"] > 2)
                    or np.any(rows["pending_species"] > A.plant_types)
                    or np.any(rows["next_species"] > A.plant_types)
                    or np.any((actual["exploration_mode"] > 0) != (actual["pending_species"] > 0))
                    or np.any((actual["exploration_mode"] == 1) & (actual["action"] != 0))
                    or np.any(
                        (actual["exploration_mode"] == 2)
                        & (
                            (actual["action"] == 0)
                            | (actual["action"] >= A.dig_start)
                            | (
                                (actual["action"].astype(np.int64) - 1) // A.tiles + 1
                                != actual["pending_species"]
                            )
                        )
                    )
                ):
                    raise ValueError("Corrupt trajectory exploration provenance")
                groups = np.where(
                    actual["action"] == 0, 0, np.where(actual["action"] < A.dig_start, 1, 2)
                )
                np.add.at(actual_counts, (groups, actual["accepted"].astype(np.int64)), 1)
                starts = rows["entity_offset"]
                counts = rows["entity_count"]
                if (
                    np.any(starts > obj.entity_size)
                    or np.any(counts > obj.entity_size - starts)
                    or (obj.schema and np.any(counts > obj.schema["max_entities"]))
                ):
                    raise ValueError("Corrupt trajectory entity offset/count")
                intervals = np.unique(
                    np.stack((starts[counts > 0], (starts + counts)[counts > 0]), -1), axis=0
                )
                entity_intervals.extend(map(tuple, intervals.tolist()))
            for begin, end in sorted(set(entity_intervals)):
                if begin != expected:
                    raise ValueError("Corrupt trajectory entity offset/count")
                expected = end
            if expected != obj.entity_size:
                raise ValueError("Orphan entity records in trajectory")
            if state["finalized"] and (
                not np.array_equal(actual_counts, state["outcome_counts"])
                or not np.array_equal(
                    actual_counts[1:] * np.array([0, 1]), state["consistency_counts"]
                )
                or not np.array_equal(actual_counts.sum(-1), state["group_counts"])
            ):
                raise ValueError("Corrupt trajectory acceptance counts disagree with records")
            for key in ("finalized", "group_counts", "species_counts"):
                setattr(obj, key, state[key])
            obj.transport_metrics.update(state.get("transport_metrics", {}))
            obj.exploration = dict(state["exploration"])
            for key in (
                "rewards_finalized",
                "reward_summary",
                "outcome_counts",
                "consistency_counts",
            ):
                setattr(obj, key, state[key])
            obj.causal_sources = {
                int(index): list(values)
                for index, values in state.get("causal_sources", {}).items()
            }
            boundary = np.asarray(state.get("next_run_break"))
            if boundary.shape != (obj.n_envs,) or boundary.dtype != np.bool_:
                raise ValueError("Corrupt compaction boundary recovery")
            obj._next_run_break = boundary.copy()
            return obj
        except Exception:
            obj.close()
            raise

    def close(self):
        self._source_index.clear()
        for blocks in (self.blocks, self.entity_blocks, self._index_arrays):
            for block in blocks:
                if isinstance(block, np.memmap):
                    block.flush()
                    block._mmap.close()
            blocks.clear()
        if self.path.is_dir():
            for prefix in ("block", "entities", "source-index"):
                for path in self.path.glob(prefix + "-*.npy"):
                    path.unlink()
            try:
                self.path.rmdir()
            except OSError:
                pass
        self._position_cache.clear()
        self._entity_gather_workspace = None
