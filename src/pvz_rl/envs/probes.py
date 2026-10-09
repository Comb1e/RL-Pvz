"""Isolated CPU reference and device-native counterfactual simulator execution."""

import copy
import warnings

import numpy as np
import torch
from pvz_game import Wait

from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures
from pvz_rl.envs.encoding import ENTITY_WIDTH, GLOBAL_WIDTH, ObservationEncoder, PackedEntityBatch
from pvz_rl.envs.history import public_history_event
from pvz_rl.envs.rewards import reward_parts
from pvz_rl.learning.cuda_buffer import probe_dtype
from pvz_rl.learning.objective import COMPONENTS, MAX_PROBES
from pvz_rl.policy.active_batch import ActiveBatchPlan, active_tile_values, forward_active
from pvz_rl.policy.sequential_q import action_parts, assemble
from pvz_rl.policy.transformer_lstm import EventMemoryState


def cpu_probe(game, proximity, proposal, cfg):
    scratch = ActionPhaseGame(game.rules)
    scratch.restore(game.snapshot())
    before = scratch.observe()
    action = ActionCodec(cfg).decode(proposal)
    ticks = int(isinstance(action, Wait) or not scratch.validate_action(action).accepted)
    result = scratch.step(action, ticks=ticks)
    parts = reward_parts(
        before,
        result.observation,
        cfg,
        events=result.events,
        rules=game.rules,
        action=action,
        action_result=result.action_result,
        proximity=copy.deepcopy(proximity),
    )
    timeout = (
        result.status.value == "running"
        and result.observation.tick
        >= cfg["environment"]["cutoff_seconds"] * game.rules.game["tick_rate"]
    )
    if timeout:
        parts["terminal"] = -cfg["reward"]["loss_penalty"]
        parts["total"] += parts["terminal"]
    return dict(
        proposal=proposal,
        executed_action=proposal if result.action_result.accepted else 0,
        accepted=result.action_result.accepted,
        reason=result.action_result.reason,
        duration=result.ticks_advanced,
        tick=result.observation.tick,
        done=timeout or result.status.value != "running",
        won=result.status.value == "won",
        components=parts,
        observation=ObservationEncoder(cfg, game.rules).encode(result.observation),
        history_event=public_history_event(result.events, game.rules),
    )


class CounterfactualCollector:
    def __init__(self, env):
        self.env = env
        batch = env.batch
        self.profiler = env.profiler
        self.batch = self.features = None
        self._pending_bootstrap = None
        self.inference_metrics = dict(
            logical_rows=0,
            padded_rows=0,
            unique_rows=0,
            bucket_widths=(),
            scratch_rows=0,
            scratch_capacity_rows=MAX_PROBES * batch.n,
        )
        self._ensure_workspaces()
        self.copy_kernel = env.features.module.get_function("copy_probe_state")
        objective = env.cfg["training"]["objective"]
        self.branch_probes = objective["branch_probes"]
        self.tile_probes = objective["tile_probes"]
        self.probe_interval = objective["probe_interval_decisions"]
        self.branch_cursor = torch.zeros(batch.n, dtype=torch.long, device=batch.device)
        self.tile_cursor = torch.zeros(batch.n, 10, dtype=torch.long, device=batch.device)
        self.decision_cursor = torch.zeros(batch.n, dtype=torch.long, device=batch.device)

    def _ensure_workspaces(self):
        if self.features is not None:
            return
        batch = self.env.batch
        self.lanes = 2
        fallback = False
        try:
            self._allocate()
        except (batch.cp.cuda.memory.OutOfMemoryError, torch.cuda.OutOfMemoryError, MemoryError):
            self.batch = self.features = None
            fallback = True
        if fallback:
            batch.cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()
            self.lanes = 1
            self._allocate()
            warnings.warn(
                "Probe scratch memory: using one lane; all scheduled probes remain enabled",
                RuntimeWarning,
                stacklevel=2,
            )
        self.features.profiler = self.profiler
        self.packed = torch.empty(
            MAX_PROBES * batch.n * self.features.limit,
            ENTITY_WIDTH,
            dtype=torch.int32,
            device=batch.device,
        )
        self.metadata = torch.empty(
            MAX_PROBES * batch.n, 17, dtype=torch.float64, device=batch.device
        )
        self.source = torch.empty(MAX_PROBES * batch.n, dtype=torch.long, device=batch.device)
        self.slots = torch.arange(MAX_PROBES, device=batch.device).repeat_interleave(batch.n)
        self.games = torch.arange(batch.n, device=batch.device).repeat(MAX_PROBES)

    def release_workspaces(self):
        """Release scratch storage after bootstrap staging; retain recovery cursors."""
        if self._pending_bootstrap is not None:
            raise RuntimeError("Stage the pending probe bootstrap before releasing workspaces")
        for name in ("batch", "features", "packed", "metadata", "source", "slots", "games"):
            setattr(self, name, None)

    def _allocate(self):
        batch = self.env.batch
        self.batch = AccountingCudaBatch(
            batch.n * self.lanes,
            zombie_capacity=batch.zcap,
            projectile_capacity=batch.qcap,
            max_step_ticks=batch.max_step_ticks,
            rules=batch.rules,
            device=batch.device,
        )
        self.features = CudaFeatures(
            self.batch,
            self.env.cfg,
            self.env.condition,
            store_rows=MAX_PROBES * batch.n,
        )

    def schedule(self, policy, actions, details, masks, active, *, plan=None):
        count, device = len(actions), actions.device
        branch, tile = action_parts(actions)
        all_branches = torch.arange(10, device=device).expand(count, -1)
        alternatives = all_branches[all_branches != branch[:, None]].reshape(count, 9)
        chosen = alternatives.gather(
            1,
            (self.branch_cursor[:, None] + torch.arange(self.branch_probes, device=device)) % 9,
        )
        geometry = masks[:, 1:].reshape(count, 9, A.tiles)
        proposals = torch.zeros(count, MAX_PROBES, dtype=torch.long, device=device)
        eligible = active & (self.decision_cursor % self.probe_interval == 0)
        valid = eligible[:, None].expand(-1, MAX_PROBES).clone()
        valid[:, self.branch_probes : 2] = False
        games = torch.arange(count, device=device)
        tile_values = policy.tile_values if plan is None else active_tile_values(policy, plan)
        for slot in range(self.branch_probes):
            alternative = chosen[:, slot]
            values = tile_values(details["tile_features"], details["context"], alternative)
            legal = geometry[games, (alternative - 1).clamp_min(0)]
            locations = values.masked_fill(~legal, -torch.inf).argmax(-1)
            proposals[:, slot] = assemble(alternative, locations)
        candidate = geometry[games, (branch - 1).clamp_min(0)].clone()
        candidate.scatter_(1, tile[:, None], False)
        indices = (
            torch.arange(A.tiles, device=device)
            .expand(count, -1)
            .masked_fill(~candidate, A.tiles)
            .sort(-1)
            .values
        )
        counts = candidate.sum(-1)
        cursor = self.tile_cursor.gather(1, branch[:, None])
        for slot in range(2):
            location = indices.gather(1, (cursor + slot) % counts[:, None].clamp_min(1)).flatten()
            valid[:, slot + 2] &= (branch != 0) & (counts > slot) & (slot < self.tile_probes)
            proposals[:, slot + 2] = assemble(branch, location.clamp_max(A.tiles - 1))
        self.branch_cursor += eligible.long() * self.branch_probes
        self.tile_cursor.scatter_add_(
            1,
            branch[:, None],
            (eligible & (branch != 0)).long()[:, None]
            * counts.clamp_max(self.tile_probes)[:, None],
        )
        self.decision_cursor += active.long()
        return proposals, valid

    @torch.no_grad()
    def collect(self, policy, teacher_memory, obs, actions, details, masks, active, *, plan=None):
        """Stage evidence and exact representatives without bootstrap or a device wait."""
        if self._pending_bootstrap is not None:
            raise RuntimeError("Stage the previous probe bootstrap before collecting again")
        self._ensure_workspaces()
        if plan is None:
            proposals, valid = self.schedule(policy, actions, details, masks, active)
            active_host = np.asarray(
                getattr(self.env, "enabled_envs", np.ones(self.env.batch.n, dtype=bool)), dtype=bool
            )
        else:
            proposals, valid = self.schedule(policy, actions, details, masks, active, plan=plan)
            active_host = np.zeros(self.env.batch.n, dtype=bool)
            active_host[plan.original_slot_ids] = True
        width = self.env.probe_width()
        if hasattr(self.env, "_pending_spawns") and hasattr(self.env, "_plant_counts"):
            bounds = np.minimum(
                self.features.limit,
                self.env.features.summary_host[:, 0]
                + self.env._pending_spawns
                + 2 * self.env._plant_counts
                + 1,
            )
        else:
            bounds = np.full(self.env.batch.n, width, dtype=np.int64)
        transfer_capacity = MAX_PROBES * int(bounds[active_host].sum())
        active_ids = np.flatnonzero(active_host)
        selected = torch.as_tensor(active_ids, device=actions.device)
        active_count = len(active_ids)
        canonical_rows = (
            torch.arange(MAX_PROBES, device=actions.device)[:, None] * self.env.batch.n
            + selected[None, :]
        ).flatten()
        stores = (
            self.features.entity_tensor,
            self.features.entity_mask_tensor,
            self.features.globals_tensor,
            self.features.summary_tensor,
            self.features.history_tensor,
        )
        for store in stores[1:]:
            store.zero_()
        self.metadata.zero_()
        teacher = teacher_memory.policy
        with self.profiler.track("ema_inference"):
            output, _, _ = teacher_memory.advance(obs, active, plan=plan)
        source = self.env.batch
        passes = range(0, MAX_PROBES, self.lanes) if active_count else ()
        if active_count:
            scratch = self.batch.compact_view(active_count * self.lanes)
            features = self.features.compact_view(scratch)
            compact_actions = proposals.index_select(0, selected)
            compact_valid = valid.index_select(0, selected)
            source_ids = scratch.cp.from_dlpack(selected)
        for first in passes:
            with self.profiler.track("probe_simulation"):
                for lane, slot in enumerate(range(first, first + self.lanes)):
                    destination = lane * active_count
                    self.copy_kernel(
                        (active_count,),
                        (128,),
                        (
                            source.header,
                            source.plants,
                            source.zombies,
                            source.projectiles,
                            source.mowers,
                            source.cooldowns,
                            source._schedules,
                            self.env.features.home_ledger,
                            self.env.features.assets,
                            scratch.header[destination:],
                            scratch.plants[destination:],
                            scratch.zombies[destination:],
                            scratch.projectiles[destination:],
                            scratch.mowers[destination:],
                            scratch.cooldowns[destination:],
                            scratch._schedules[destination:],
                            features.home_ledger[destination:],
                            features.assets[destination:],
                            source_ids,
                            scratch.cp.from_dlpack(compact_valid[:, slot].contiguous()),
                        ),
                    )
            pair_actions = compact_actions[:, first : first + self.lanes].T.contiguous().flatten()
            pair_valid = compact_valid[:, first : first + self.lanes].T.contiguous().flatten()
            features.totals.fill(0)
            features.step(
                scratch.cp.from_dlpack(pair_actions),
                offset=first * source.n,
                publish=False,
                phase="probe_simulation",
            )
            header = torch.from_dlpack(scratch.header)
            accepted = header[:, 12].bool()
            executed = torch.where(accepted, pair_actions, 0)
            duration = header[:, 14]
            done = (header[:, 1] != 0) | (
                header[:, 0]
                >= self.env.cfg["environment"]["cutoff_seconds"] * scratch.rules.game["tick_rate"]
            )
            parts = torch.from_dlpack(features.parts)[
                :, [REWARD_FIELDS.index(name) for name in COMPONENTS]
            ]
            rows = canonical_rows[first * active_count : (first + self.lanes) * active_count]
            if active_count < source.n:
                temporary = slice(first * source.n, first * source.n + scratch.n)
                values = [store[temporary].clone() for store in stores]
                for store in stores[1:]:
                    store[temporary].zero_()
                for store, value in zip(stores, values, strict=True):
                    store.index_copy_(0, rows, value)
            evidence = torch.cat(
                (
                    torch.stack(
                        (
                            pair_valid,
                            self.slots.index_select(0, rows) < 2,
                            pair_actions,
                            executed,
                            accepted,
                            header[:, 13],
                            duration,
                            header[:, 0],
                            done,
                            header[:, 1] == 1,
                        ),
                        -1,
                    ).double(),
                    parts,
                    torch.zeros_like(duration)[:, None].double(),
                ),
                -1,
            )
            self.metadata.index_copy_(0, rows, torch.where(pair_valid[:, None], evidence, 0))
        duration = self.metadata[:, 6].long()
        valid_flat = valid.T.contiguous().flatten()
        next_state = EventMemoryState(
            output.state.hidden.repeat(1, MAX_PROBES, 1),
            output.state.cell.repeat(1, MAX_PROBES, 1),
            output.state.pending.repeat(MAX_PROBES, 1),
            output.state.elapsed_ticks.repeat(MAX_PROBES),
        ).observe(self.features.history_tensor, duration, valid_flat)
        history_inputs = torch.where(valid_flat[:, None], next_state.inputs(), 0)
        history_writes = history_inputs[:, :7].ne(0).any(-1)
        with self.profiler.track("encoding"):
            self.features.dedup(history_inputs, valid_flat, self.source, MAX_PROBES)
            unique = valid_flat & (self.source == self.slots)
            counts = torch.where(unique, self.features.summary_tensor[:, 0], 0)
            offsets = counts.cumsum(0) - counts
            self.features.pack(unique, offsets, self.packed)
            representative = self.source.clamp_min(0) * source.n + self.games
            observation_counts = torch.where(valid_flat, self.features.summary_tensor[:, 0], 0)
            observation_offsets = torch.where(valid_flat, offsets[representative], 0)
        with self.profiler.track("transfer"):
            host = self.env.handoff.enqueue_fields(
                "probe",
                dict(
                    metadata=self.metadata,
                    events=history_inputs,
                    memory_write=history_writes,
                    entities=self.packed[:transfer_capacity],
                    counts=observation_counts,
                    offsets=observation_offsets,
                    size=counts.sum()[None],
                    globals=self.features.globals_tensor.index_select(0, canonical_rows),
                    summary=self.features.summary_tensor,
                    source=self.source,
                    width=torch.tensor(width, device=source.device),
                ),
            )
        host["active_ids"] = active_ids
        self._pending_bootstrap = dict(
            host=host,
            teacher=teacher,
            state=next_state,
            events=history_inputs,
            memory_write=history_writes,
            duration=duration,
            valid=valid_flat,
            representative=representative,
            done=self.metadata[:, 8].bool(),
            entity_bounds=np.tile(bounds, MAX_PROBES),
            scratch_rows=active_count * MAX_PROBES,
            scratch_capacity_rows=source.n * MAX_PROBES,
        )
        return host

    @torch.no_grad()
    def bootstrap(self, host):
        """Queue unique nonterminal EMA targets after the caller's existing handoff."""
        pending = self._pending_bootstrap
        if pending is None or pending["host"] is not host:
            raise RuntimeError("Bootstrap requires the current staged probe handoff")
        evidence = host["metadata"].numpy()
        valid = evidence[:, 0].astype(bool)
        done = evidence[:, 8].astype(bool)
        source = host["source"].numpy()
        count = self.env.batch.n
        if np.any(valid & ((source < 0) | (source >= MAX_PROBES))):
            raise RuntimeError("Invalid counterfactual representative")
        representatives = source.clip(min=0) * count + np.tile(np.arange(count), MAX_PROBES)
        needed = np.zeros(len(valid), dtype=bool)
        needed[representatives[valid & ~done]] = True
        if np.any(host["counts"].numpy() > pending["entity_bounds"]):
            raise RuntimeError("Counterfactual entity growth exceeded its public-count bound")
        plan = ActiveBatchPlan.from_counts(
            needed,
            pending["entity_bounds"],
            self.features.limit,
            pending["teacher"].entity.microbatch,
            pending["teacher"].entity.token_budget,
        )
        self.inference_metrics = dict(
            logical_rows=int((valid & ~done).sum()),
            padded_rows=plan.padded_rows,
            unique_rows=plan.logical_rows,
            bucket_widths=tuple(bucket.entity_width for bucket in plan.buckets),
            scratch_rows=pending["scratch_rows"],
            scratch_capacity_rows=pending["scratch_capacity_rows"],
        )
        with self.profiler.track("ema_inference"):
            if plan.logical_rows:
                output = forward_active(
                    pending["teacher"],
                    self.features.view(0, MAX_PROBES * count, self.features.limit),
                    pending["state"],
                    plan,
                    events=pending["events"],
                    memory_write=pending["memory_write"],
                )
                values = output.branch_q.max(-1).values[pending["representative"]]
                values *= torch.pow(self.env.cfg["training"]["gamma"], pending["duration"].float())
                values = torch.where(pending["valid"] & ~pending["done"], values, 0)
            else:
                values = pending["state"].hidden.new_zeros(MAX_PROBES * count)
        with self.profiler.track("transfer"):
            staged = self.env.handoff.enqueue("probe_bootstrap", values)
        self._pending_bootstrap = None
        return staged

    @staticmethod
    def materialize(host):
        count = host["metadata"].shape[0] // MAX_PROBES
        values = host["metadata"].numpy().reshape(MAX_PROBES, count, 17).transpose(1, 0, 2)
        if not np.isfinite(values).all():
            raise RuntimeError("Invalid counterfactual accounting or exhausted proximity ledger")
        if np.any(host["counts"].numpy() > int(host["width"])):
            raise RuntimeError("Counterfactual entity growth exceeded its padding bound")
        if int(host["size"][0]) > len(host["entities"]):
            raise RuntimeError("Counterfactual entity growth exceeded its transfer bound")
        records = np.zeros(values.shape[:2], dtype=probe_dtype())
        fields = (
            "valid",
            "branch_role",
            "action",
            "executed_action",
            "accepted",
            "reason",
            "duration",
            "tick",
            "done",
            "won",
        )
        for column, field in enumerate(fields):
            records[field] = values[..., column]
        records["components"] = values[..., 10:16]
        records["bootstrap"] = values[..., 16]
        records["events"] = host["events"].numpy().reshape(MAX_PROBES, count, 8).transpose(1, 0, 2)
        records["memory_write"] = host["memory_write"].numpy().reshape(MAX_PROBES, count).T
        globals_host = np.zeros((count, MAX_PROBES, GLOBAL_WIDTH), dtype=np.float32)
        active_ids = host["active_ids"]
        globals_host[active_ids] = (
            host["globals"]
            .numpy()
            .reshape(MAX_PROBES, len(active_ids), GLOBAL_WIDTH)
            .transpose(1, 0, 2)
        )
        observations = PackedEntityBatch(
            host["entities"].numpy()[: int(host["size"][0])],
            host["offsets"].numpy().reshape(MAX_PROBES, count).T,
            host["counts"].numpy().reshape(MAX_PROBES, count).T,
            globals_host,
        )
        return records, observations

    def snapshot(self):
        return dict(
            branch_cursor=self.branch_cursor.cpu(),
            tile_cursor=self.tile_cursor.cpu(),
            decision_cursor=self.decision_cursor.cpu(),
            timing=self.profiler.flush(),
        )

    def restore(self, state):
        for name in ("branch_cursor", "tile_cursor", "decision_cursor"):
            cursor = state[name]
            if cursor.shape != getattr(self, name).shape or torch.any(cursor < 0):
                raise ValueError("Invalid counterfactual schedule cursor")
            getattr(self, name).copy_(cursor)
        self.profiler.seconds.clear()
        self.profiler.seconds.update(state["timing"])
