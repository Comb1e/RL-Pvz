"""Isolated, bounded greedy-EMA counterfactual rollouts and endpoint evidence."""

import copy
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from pvz_game import Wait

from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures
from pvz_rl.envs.encoding import (
    ObservationEncoder,
    PackedEntityBatch,
    collate_observations,
)
from pvz_rl.envs.history import public_history_event
from pvz_rl.envs.rewards import reward_parts
from pvz_rl.learning.cuda_buffer import probe_dtype
from pvz_rl.learning.objective import COMPONENTS, training_rewards
from pvz_rl.learning.probe_layout import ProbeLayout
from pvz_rl.policy.active_batch import ActiveBatchPlan, forward_active
from pvz_rl.policy.sequential_q import action_parts, assemble, select_q_actions
from pvz_rl.policy.transformer_lstm import EventMemoryState


class ProbeStop(IntEnum):
    RUNNING = 0
    TERMINAL = 1
    CUTOFF = 2
    HORIZON = 3
    DECISION_CAP = 4


@torch.no_grad()
def cpu_probe(game, proximity, proposal, cfg, policy, memory=None):
    """Independent serial reference; memory has already consumed the source board."""
    scratch = ActionPhaseGame(game.rules)
    scratch.restore(game.snapshot())
    ledger = copy.deepcopy(proximity)
    codec, encoder = ActionCodec(cfg), ObservationEncoder(cfg, game.rules)
    objective = cfg["training"]["objective"]
    tick_rate = game.rules.game["tick_rate"]
    start = scratch.observe().tick
    device = next(policy.parameters()).device
    memory = policy.initial_state(1, device=device) if memory is None else memory.clone()
    accumulated = {}
    initial = None
    writes = 0
    current = int(proposal)
    for transitions in range(1, objective["probe_rollout_max_decisions"] + 1):
        before = scratch.observe()
        action = codec.decode(current)
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
            proximity=ledger,
        )
        cutoff = result.observation.tick >= cfg["environment"]["cutoff_seconds"] * tick_rate
        terminal = result.status.value != "running"
        if cutoff and not terminal:
            parts["terminal"] = -cfg["reward"]["loss_penalty"]
            parts["total"] += parts["terminal"]
        for name, value in parts.items():
            accumulated[name] = accumulated.get(name, 0.0) + value
        if initial is None:
            initial = dict(
                proposal=proposal,
                executed_action=proposal if result.action_result.accepted else 0,
                accepted=result.action_result.accepted,
                reason=result.action_result.reason,
                initial_duration=result.ticks_advanced,
            )
        event = public_history_event(result.events, game.rules)
        writes += int(event.writes)
        facts = torch.as_tensor(event.array(), device=device)[None]
        memory = memory.observe(
            facts,
            torch.tensor([result.ticks_advanced], device=device),
            torch.ones(1, dtype=torch.bool, device=device),
        )
        elapsed = result.observation.tick - start
        stop = (
            ProbeStop.TERMINAL
            if terminal
            else ProbeStop.CUTOFF
            if cutoff
            else ProbeStop.HORIZON
            if elapsed >= objective["probe_rollout_seconds"] * tick_rate
            else ProbeStop.DECISION_CAP
            if transitions >= objective["probe_rollout_max_decisions"]
            else ProbeStop.RUNNING
        )
        observation = encoder.encode(result.observation)
        if stop:
            break
        actions, memory, _ = policy.decide(collate_observations([observation], device), memory)
        current = int(actions[0])
    bootstrap = 0.0
    if stop in (ProbeStop.HORIZON, ProbeStop.DECISION_CAP):
        output = policy.forward_step(collate_observations([observation], device), memory)
        bootstrap = float(policy.action_values(output, observations=[observation]).branches.max())
    return dict(
        **initial,
        duration=elapsed,
        tick=result.observation.tick,
        done=stop in (ProbeStop.TERMINAL, ProbeStop.CUTOFF),
        won=result.status.value == "won",
        components=accumulated,
        observation=observation,
        events=memory.inputs()[0].cpu().numpy(),
        memory_write=bool(memory.pending.any()),
        transitions=transitions,
        event_writes=writes,
        stop_reason=int(stop),
        bootstrap=bootstrap,
        target=float(
            training_rewards(np.array([accumulated.get(key, 0) for key in COMPONENTS]), cfg)
        )
        + bootstrap,
        state=memory,
    )


class ProbePhase(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    ENDPOINT = "endpoint"
    PATCHED = "patched"


@dataclass
class ProbeJob:
    source_id: int
    cohort: int
    original_slot: int
    episode: int
    decision: int
    trajectory_row: int
    proposals: tuple[int, ...]
    ema_version: int
    created_at: float
    source_count: int
    phase: ProbePhase = ProbePhase.QUEUED
    patched: set[int] = field(default_factory=set)


def gather_memory(state, indices):
    return EventMemoryState(
        state.hidden[:, indices].contiguous(),
        state.cell[:, indices].contiguous(),
        state.pending[indices].contiguous(),
        state.elapsed_ticks[indices].contiguous(),
    )


def scatter_memory(target, indices, source):
    for destination, value, dimension in zip(
        target.tensors(), source.tensors(), (1, 1, 0, 0), strict=True
    ):
        destination.index_copy_(dimension, indices, value)


class ProbePool:
    """Persistent isolated lanes, serviced without a collect-to-completion barrier."""

    def __init__(self, env):
        self.env, self.profiler = env, env.profiler
        self.layout = ProbeLayout.from_config(env.cfg)
        self.tile_cursor = torch.zeros(
            env.batch.n, A.plant_types, dtype=torch.long, device=env.batch.device
        )
        self.plant_cursor = torch.zeros(env.batch.n, dtype=torch.long, device=env.batch.device)
        objective, performance = (
            env.cfg["training"]["objective"],
            env.cfg["training"]["performance"],
        )
        self.horizon_ticks = objective["probe_rollout_seconds"] * env.batch.rules.game["tick_rate"]
        self.max_decisions = objective["probe_rollout_max_decisions"]
        self.control_interval = objective["probe_rollout_control_interval_decisions"]
        self.capacity = performance["probe_active_capacity"]
        self.source_capacity = performance["probe_pending_capacity"]
        self.steps_per_round = performance["probe_steps_per_round"]
        self.batch = self.features = self.source_batch = self.source_features = None
        self.state = self.source_state = self.teacher = None
        self.jobs, self.queue, self.lanes = {}, deque(), []
        self.free_sources = deque()
        self._prepared = self._control = self._endpoint = self._bootstrap = None
        self.rounds = 0
        self.inference_metrics = self._empty_metrics()

    def _empty_metrics(self):
        return dict(
            logical_rows=0,
            unique_rows=0,
            padded_rows=0,
            bucket_widths=(),
            rollout_transitions=0,
            rollout_event_writes=0,
            rollout_logical_rows=0,
            rollout_padded_rows=0,
            rollout_enqueue_seconds=0.0,
            stop_reasons={},
            accepted_plant_triggers=0,
            tile_probes=0,
            workspace_bytes=0,
        )

    @property
    def has_work(self):
        return bool(
            self.jobs or self._prepared or self._control or self._endpoint or self._bootstrap
        )

    @property
    def workspace_bytes(self):
        amount = 0
        seen = set()
        for owner in (self.batch, self.features, self.source_batch, self.source_features):
            if owner is not None:
                for value in vars(owner).values():
                    if isinstance(value, self.env.batch.cp.ndarray) and value.data.ptr not in seen:
                        seen.add(value.data.ptr)
                        amount += value.nbytes
        for value in (self.state, self.source_state):
            if value is not None:
                amount += sum(tensor.numel() * tensor.element_size() for tensor in value.tensors())
        for name in (
            "start",
            "accumulated",
            "counts",
            "writes",
            "initial",
            "reason",
            "tile_cursor",
            "plant_cursor",
        ):
            value = getattr(self, name, None)
            if value is not None:
                amount += value.numel() * value.element_size()
        for name in ("counts_host", "writes_host", "entity_counts"):
            value = getattr(self, name, None)
            if value is not None:
                amount += value.nbytes
        for packet in (self._endpoint, self._bootstrap):
            if packet is not None:
                amount += packet.get("owned_bytes", 0)
                amount += sum(
                    value.nbytes for value in packet.values() if isinstance(value, np.ndarray)
                )
                observations = packet.get("observations")
                if observations is not None:
                    amount += sum(value.nbytes for value in vars(observations).values())
        return amount

    def telemetry(self):
        now = perf_counter()
        return dict(
            active_auxiliary_lanes=len(self.lanes),
            queued_probes=len(self.queue),
            pending_probe_jobs=len(self.jobs),
            auxiliary_capacity=self.capacity,
            source_capacity=self.source_capacity,
            oldest_queue_seconds=max(
                (now - self.jobs[bank].created_at for bank, _ in self.queue), default=0.0
            ),
            oldest_pending_job_seconds=max(
                (now - job.created_at for job in self.jobs.values()), default=0.0
            ),
            workspace_bytes=self.workspace_bytes,
        )

    def _ensure(self, teacher):
        if self.batch is not None:
            if self.teacher is not teacher:
                raise RuntimeError("Cannot replace a probe pool's frozen teacher")
            return
        source = self.env.batch
        while True:
            try:
                self.source_batch = AccountingCudaBatch(
                    self.source_capacity,
                    zombie_capacity=source.zcap,
                    projectile_capacity=source.qcap,
                    max_step_ticks=1,
                    rules=source.rules,
                    device=source.device,
                )
                self.source_features = SimpleNamespace(
                    home_ledger=source.cp.zeros(
                        (self.source_capacity, source.zcap, 2), source.cp.int64
                    ),
                    assets=source.cp.zeros(self.source_capacity, source.cp.float64),
                )
                self.batch = AccountingCudaBatch(
                    self.capacity,
                    zombie_capacity=source.zcap,
                    projectile_capacity=source.qcap,
                    max_step_ticks=1,
                    rules=source.rules,
                    device=source.device,
                )
                self.features = CudaFeatures(self.batch, self.env.cfg, self.env.condition)
                self.features.profiler = self.profiler
                self.state = teacher.initial_state(self.capacity, device=source.device)
                self.source_state = teacher.initial_state(
                    self.source_capacity, device=source.device
                )
                self.start = torch.zeros(self.capacity, dtype=torch.long, device=source.device)
                self.counts = self.start.clone()
                self.writes = self.start.clone()
                self.reason = self.start.clone()
                self.accumulated = torch.zeros(
                    self.capacity, len(COMPONENTS), dtype=torch.float64, device=source.device
                )
                self.initial = torch.zeros(self.capacity, 5, dtype=torch.long, device=source.device)
                self.counts_host = np.zeros(self.capacity, np.int64)
                self.writes_host = np.zeros(self.capacity, np.int64)
                self.entity_counts = np.zeros(self.capacity, np.int64)
                if (
                    self.workspace_bytes + self.env.handoff.nbytes
                    > self.env.cfg["training"]["storage"]["ram_gib"] * 1024**3
                ):
                    raise MemoryError(
                        "Auxiliary allocations exceed the trajectory workspace budget"
                    )
                free, _ = torch.cuda.mem_get_info(source.device)
                if (
                    free
                    < self.env.cfg["training"]["performance"]["collection_vram_headroom_mib"]
                    * 1024**2
                ):
                    raise MemoryError(
                        "Auxiliary allocations would consume collection VRAM headroom"
                    )
                break
            except (
                source.cp.cuda.memory.OutOfMemoryError,
                torch.cuda.OutOfMemoryError,
                MemoryError,
            ):
                self.batch = self.features = self.source_batch = self.source_features = None
                self.state = self.source_state = None
                for name in (
                    "start",
                    "counts",
                    "writes",
                    "reason",
                    "accumulated",
                    "initial",
                    "counts_host",
                    "writes_host",
                    "entity_counts",
                ):
                    setattr(self, name, None)
                source.cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()
                if self.capacity == self.source_capacity == 1:
                    raise MemoryError("Cannot allocate the minimum isolated probe source and lane")
                self.capacity, self.source_capacity = (
                    max(1, self.capacity // 2),
                    max(1, self.source_capacity // 2),
                )
                warnings.warn(
                    f"Probe pool allocation fallback: {self.capacity} lanes, {self.source_capacity} sources; all probes retained",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.teacher = teacher
        self.copy_kernel = self.env.features.module.get_function("copy_probe_state")
        self.free_sources = deque(range(self.source_capacity))
        self.inference_metrics["workspace_bytes"] = self.workspace_bytes

    def _copy(
        self,
        source,
        source_features,
        destination,
        destination_features,
        source_ids,
        target_ids,
        active=None,
    ):
        if not len(source_ids):
            return
        active = torch.ones_like(source_ids, dtype=torch.bool) if active is None else active
        self.copy_kernel(
            (len(source_ids),),
            (128,),
            (
                source.header,
                source.plants,
                source.zombies,
                source.projectiles,
                source.mowers,
                source.cooldowns,
                source._schedules,
                source_features.home_ledger,
                source_features.assets,
                destination.header,
                destination.plants,
                destination.zombies,
                destination.projectiles,
                destination.mowers,
                destination.cooldowns,
                destination._schedules,
                destination_features.home_ledger,
                destination_features.assets,
                destination.cp.from_dlpack(source_ids),
                destination.cp.from_dlpack(target_ids),
                destination.cp.from_dlpack(active),
            ),
        )

    def schedule(self, actions, masks, eligible):
        branch, actual_tile = action_parts(actions)
        eligible = eligible & (branch > 0) & (branch <= A.plant_types)
        rows = torch.arange(len(actions), device=actions.device)
        candidates = A.tile_masks(masks)[rows, (branch - 1).clamp(0, A.plant_types - 1)].clone()
        candidates.scatter_(1, actual_tile[:, None], False)
        candidates &= eligible[:, None]
        tiles = (
            torch.arange(A.tiles, device=actions.device)
            .expand(len(actions), -1)
            .masked_fill(~candidates, A.tiles)
            .sort(-1)
            .values
        )
        counts = candidates.sum(-1)
        cursor = self.tile_cursor[rows, (branch - 1).clamp(0, A.plant_types - 1)]
        locations = tiles.gather(
            1,
            (cursor[:, None] + torch.arange(self.layout.count, device=actions.device))
            % counts[:, None].clamp_min(1),
        )
        valid = torch.arange(self.layout.count, device=actions.device)[None] < counts[:, None]
        return assemble(branch[:, None], locations.clamp_max(A.tiles - 1)), valid & eligible[
            :, None
        ]

    @torch.no_grad()
    def reserve(self, teacher, state, actions, masks, issued, order):
        self._ensure(teacher)
        if self._prepared is not None:
            raise RuntimeError("Commit the previous source reservation first")
        legal = torch.from_dlpack(self.env.batch.action_masks_device())
        eligible = (
            issued
            & (actions > 0)
            & (actions < A.dig_start)
            & legal.gather(1, actions[:, None]).flatten()
        )
        proposals, valid = self.schedule(actions, masks, eligible)
        needs_source = valid.any(-1)
        priority = torch.argsort(order)
        ranks = torch.empty_like(order)
        ranks[priority] = needs_source[priority].long().cumsum(0) - 1
        free = torch.as_tensor(list(self.free_sources), dtype=torch.long, device=actions.device)
        admitted = needs_source & (ranks < len(free))
        executing = issued & (~needs_source | admitted)
        source_ids = torch.arange(len(actions), device=actions.device)
        bank_ids = free[ranks.clamp(0, len(free) - 1)] if len(free) else torch.zeros_like(actions)
        if len(free):
            self._copy(
                self.env.batch,
                self.env.features,
                self.source_batch,
                self.source_features,
                source_ids,
                bank_ids,
                admitted,
            )
            selected = priority[
                needs_source[priority].long().argsort(descending=True, stable=True)
            ][: min(len(free), len(actions))]
            destinations = free[: len(selected)]
            selected_state = gather_memory(state, selected)
            previous = gather_memory(self.source_state, destinations)
            used = admitted[selected]
            selected_state = EventMemoryState(
                *[
                    torch.where(
                        used[None, :, None]
                        if dimension == 1
                        else used.reshape(-1, *([1] * (value.ndim - 1))),
                        value,
                        old,
                    )
                    for value, old, dimension in zip(
                        selected_state.tensors(), previous.tensors(), (1, 1, 0, 0), strict=True
                    )
                ]
            )
            scatter_memory(self.source_state, destinations, selected_state)
        host = self.env.handoff.enqueue_fields(
            "probe_admission",
            dict(
                executing=executing,
                admitted=admitted,
                bank_ids=bank_ids,
                proposals=proposals,
                valid=valid,
            ),
        )
        self._prepared = host
        return executing

    def commit(
        self, active, accepted, actions, source_counts, row_ids, *, cohort=0, decisions=None
    ):
        host, self._prepared = self._prepared, None
        if host is None:
            raise RuntimeError("Missing source reservation")
        expected = np.zeros(len(active), np.uint8)
        admitted = host["admitted"].numpy()
        valid, proposals = host["valid"].numpy(), host["proposals"].numpy()
        for original_slot in np.flatnonzero(
            active & accepted & (actions > 0) & (actions < A.dig_start)
        ):
            self.plant_cursor[original_slot] += 1
            selected = np.flatnonzero(valid[original_slot])
            species = (int(actions[original_slot]) - 1) // A.tiles
            self.tile_cursor[original_slot, species] += len(selected)
            self.inference_metrics["accepted_plant_triggers"] += 1
            if not len(selected):
                continue
            if not admitted[original_slot]:
                raise RuntimeError("Accepted planting lacks a retained source")
            bank = int(host["bank_ids"][original_slot])
            self.free_sources.remove(bank)
            job = ProbeJob(
                bank,
                int(cohort),
                int(original_slot),
                int(getattr(self.env, "_episode_serial", [0] * len(active))[original_slot]),
                int(decisions[original_slot])
                if decisions is not None
                else int(row_ids[original_slot]),
                int(row_ids[original_slot]),
                tuple(map(int, proposals[original_slot, selected])),
                int(getattr(self, "ema_version", 0)),
                perf_counter(),
                int(source_counts[original_slot]),
            )
            self.jobs[bank] = job
            self.queue.extend((bank, slot) for slot in range(len(job.proposals)))
            expected[original_slot] = len(job.proposals)
            self.inference_metrics["tile_probes"] += len(job.proposals)
        return expected

    def _start(self):
        available = min(self.capacity - len(self.lanes), len(self.queue))
        if not available:
            return
        starting = [self.queue.popleft() for _ in range(available)]
        first = len(self.lanes)
        targets = torch.arange(first, first + available, device=self.env.batch.device)
        sources = torch.tensor([bank for bank, _ in starting], device=targets.device)
        self._copy(
            self.source_batch, self.source_features, self.batch, self.features, sources, targets
        )
        scatter_memory(self.state, targets, gather_memory(self.source_state, sources))
        header = torch.from_dlpack(self.batch.header)
        self.start[targets] = header[targets, 0]
        self.counts[targets] = self.writes[targets] = self.reason[targets] = 0
        self.accumulated[targets] = self.initial[targets] = 0
        self.features.totals[self.env.batch.cp.from_dlpack(targets)] = 0
        for lane, (bank, slot) in enumerate(starting, first):
            self.jobs[bank].phase = ProbePhase.RUNNING
            self.counts_host[lane] = self.writes_host[lane] = 0
            self.entity_counts[lane] = min(self.features.limit, self.jobs[bank].source_count + 1)
        self.lanes.extend(starting)
        batch = self.batch.compact_view(len(self.lanes))
        self.features.compact_view(batch).encode(publish=False)

    @torch.no_grad()
    def advance(self):
        if self._control is not None:
            raise RuntimeError("Consume the auxiliary control handoff before another slice")
        self._start()
        if not self.lanes:
            return
        started = perf_counter()
        count = len(self.lanes)
        batch, features = (
            self.batch.compact_view(count),
            self.features.compact_view(self.batch.compact_view(count)),
        )
        header = torch.from_dlpack(batch.header)
        state = gather_memory(self.state, torch.arange(count, device=header.device))
        bounds = self.entity_counts[:count].copy()
        for step in range(self.steps_per_round):
            continuing = self.counts_host[:count] > 0 if not step else np.ones(count, bool)
            plan = ActiveBatchPlan.from_counts(
                continuing,
                bounds,
                features.limit,
                self.teacher.entity.microbatch,
                self.teacher.entity.token_budget,
            )
            live = self.reason[:count] == 0
            with self.profiler.track("probe_inference"):
                obs = features.view(0, count, features.limit)
                output = forward_active(
                    self.teacher, obs, state, plan, memory_write=state.pending.ne(0).any(-1) & live
                )
                values = self.teacher.action_values(output, features.mask_tensor, plan=plan)
                actions, _, _, _ = select_q_actions(
                    values, features.mask_tensor, deterministic=True, active=live
                )
            state = output.state
            initial_actions = torch.tensor(
                [self.jobs[bank].proposals[slot] for bank, slot in self.lanes], device=header.device
            )
            actions = torch.where(self.counts[:count] == 0, initial_actions, actions)
            header[:, 17].copy_(live.long())
            features.step(batch.cp.from_dlpack(actions), publish=False, phase="probe_simulation")
            first = self.counts[:count] == 0
            initial = torch.stack(
                (
                    actions,
                    torch.where(header[:, 12].bool(), actions, 0),
                    header[:, 12],
                    header[:, 13],
                    header[:, 14],
                ),
                -1,
            )
            self.initial[:count].copy_(torch.where(first[:, None], initial, self.initial[:count]))
            self.accumulated[:count] += torch.from_dlpack(features.parts)[
                :, [REWARD_FIELDS.index(name) for name in COMPONENTS]
            ]
            self.counts[:count] += live.long()
            facts = features.history_tensor[:count]
            self.writes[:count] += (facts.ne(0).any(-1) & live).long()
            state = state.observe(facts, header[:, 14], live)
            elapsed = header[:, 0] - self.start[:count]
            cutoff = (
                header[:, 0]
                >= self.env.cfg["environment"]["cutoff_seconds"] * batch.rules.game["tick_rate"]
            )
            reason = torch.where(
                header[:, 1] != 0,
                int(ProbeStop.TERMINAL),
                torch.where(
                    cutoff,
                    int(ProbeStop.CUTOFF),
                    torch.where(
                        elapsed >= self.horizon_ticks,
                        int(ProbeStop.HORIZON),
                        torch.where(
                            self.counts[:count] >= self.max_decisions,
                            int(ProbeStop.DECISION_CAP),
                            0,
                        ),
                    ),
                ),
            )
            self.reason[:count] = torch.where(live, reason, self.reason[:count])
            self.inference_metrics["rollout_logical_rows"] += plan.logical_rows
            self.inference_metrics["rollout_padded_rows"] += plan.padded_rows
            bounds[:] = features.limit
        scatter_memory(self.state, torch.arange(count, device=header.device), state)
        ema_versions = torch.tensor(
            [self.jobs[bank].ema_version for bank, _ in self.lanes], device=header.device
        )
        initial = self.initial[:count]
        metadata = torch.cat(
            (
                torch.stack(
                    (
                        torch.ones_like(elapsed),
                        initial[:, 0],
                        initial[:, 1],
                        initial[:, 2],
                        initial[:, 3],
                        elapsed,
                        header[:, 0],
                        self.reason[:count] <= 2,
                        header[:, 1] == 1,
                    ),
                    -1,
                ).double(),
                self.accumulated[:count],
                torch.stack(
                    (
                        elapsed * 0,
                        initial[:, 4],
                        self.counts[:count],
                        self.writes[:count],
                        self.reason[:count],
                        ema_versions,
                    ),
                    -1,
                ).double(),
            ),
            -1,
        )
        self._control = self.env.handoff.enqueue_fields(
            "auxiliary_control", dict(metadata=metadata, summary=features.summary_tensor[:count])
        )
        self.inference_metrics["rollout_enqueue_seconds"] += perf_counter() - started
        self.rounds += 1

    @torch.no_grad()
    def finish_slice(self):
        if self._control is None:
            return
        host, self._control = self._control, None
        metadata, summary = host["metadata"].numpy(), host["summary"].numpy()
        count = len(self.lanes)
        self.inference_metrics["rollout_transitions"] += int(
            (metadata[:, 17] - self.counts_host[:count]).sum()
        )
        self.inference_metrics["rollout_event_writes"] += int(
            (metadata[:, 18] - self.writes_host[:count]).sum()
        )
        self.counts_host[:count], self.writes_host[:count] = metadata[:, 17], metadata[:, 18]
        self.entity_counts[:count] = summary[:, 0]
        finished = np.flatnonzero(metadata[:, 19])
        if len(finished):
            if self._endpoint is not None:
                raise RuntimeError(
                    "Consume the previous owned endpoints before staging another batch"
                )
            indices = torch.as_tensor(finished, device=self.env.batch.device)
            width = int(summary[finished, 0].max(initial=0))
            from pvz_rl.envs.encoding import EntityBatch

            observation = EntityBatch(
                self.features.entity_tensor[indices, :width].clone(),
                self.features.entity_mask_tensor[indices, :width].clone(),
                self.features.globals_tensor[indices].clone(),
                validated=True,
            )
            state = gather_memory(self.state, indices)
            packets = [self.lanes[index] for index in finished]
            fields = dict(
                entities=observation.entities,
                globals=observation.globals,
                events=state.inputs(),
                hidden=state.hidden[0],
                cell=state.cell[0],
            )
            self._endpoint = dict(
                packets=packets,
                metadata=metadata[finished].copy(),
                counts=summary[finished, 0].copy(),
                observation=observation,
                state=state,
                host=self.env.handoff.enqueue_fields("auxiliary_endpoint", fields),
                owned_bytes=sum(
                    value.numel() * value.element_size()
                    for value in (*observation.tensors(), *state.tensors())
                ),
            )
            for bank, _ in packets:
                self.jobs[bank].phase = ProbePhase.ENDPOINT
            for reason in ProbeStop:
                if reason:
                    number = int(np.count_nonzero(metadata[finished, 19] == reason))
                    if number:
                        totals = self.inference_metrics["stop_reasons"]
                        totals[reason.name.lower()] = totals.get(reason.name.lower(), 0) + number
            self._compact(np.flatnonzero(metadata[:, 19] == 0))
        if self.rounds % self.control_interval == 0:
            self.profiler.flush()

    def _compact(self, surviving):
        indices = torch.as_tensor(surviving, device=self.env.batch.device)
        for owner, names in (
            (
                self.batch,
                (
                    "header",
                    "plants",
                    "zombies",
                    "projectiles",
                    "mowers",
                    "cooldowns",
                    "_schedules",
                    "masks",
                ),
            ),
            (
                self.features,
                (
                    "home_ledger",
                    "assets",
                    "totals",
                    "entities",
                    "entity_mask",
                    "globals",
                    "summary",
                ),
            ),
        ):
            for name in names:
                tensor = torch.from_dlpack(getattr(owner, name))
                tensor[: len(indices)].copy_(tensor.index_select(0, indices))
        scatter_memory(
            self.state,
            torch.arange(len(indices), device=indices.device),
            gather_memory(self.state, indices),
        )
        for name in ("start", "counts", "writes", "reason", "accumulated", "initial"):
            tensor = getattr(self, name)
            tensor[: len(indices)].copy_(tensor.index_select(0, indices))
        for name in ("counts_host", "writes_host", "entity_counts"):
            array = getattr(self, name)
            array[: len(surviving)] = array[surviving]
        self.lanes = [self.lanes[index] for index in surviving]

    @torch.no_grad()
    def consume_transfers(self, buffer):
        if self._bootstrap is not None:
            packet, self._bootstrap = self._bootstrap, None
            packet["records"]["bootstrap"] = packet["bootstrap"].numpy()
            buffer.patch_probe_endpoints(
                packet["indices"], packet["slots"], packet["records"], packet["observations"]
            )
            for bank, slot in packet["packets"]:
                job = self.jobs[bank]
                job.patched.add(slot)
                if len(job.patched) == len(job.proposals):
                    job.phase = ProbePhase.PATCHED
                    del self.jobs[bank]
                    self.free_sources.append(bank)
        if self._endpoint is None:
            return
        packet, self._endpoint = self._endpoint, None
        host, metadata, counts = packet["host"], packet["metadata"], packet["counts"]
        records = np.zeros(len(counts), dtype=probe_dtype())
        for column, field_name in enumerate(
            (
                "valid",
                "action",
                "executed_action",
                "accepted",
                "reason",
                "duration",
                "tick",
                "done",
                "won",
            )
        ):
            records[field_name] = metadata[:, column]
        records["components"] = metadata[:, 9:15]
        records["events"] = host["events"].numpy()
        records["memory_write"] = records["events"][:, :7].any(-1)
        for column, field_name in enumerate(
            ("initial_duration", "transitions", "event_writes", "stop_reason", "ema_version"), 16
        ):
            records[field_name] = metadata[:, column]
        if not np.isfinite(metadata).all():
            raise RuntimeError("Invalid auxiliary reward accounting")
        entities, globals_ = host["entities"].numpy(), host["globals"].numpy()
        representatives, seen = np.arange(len(counts)), {}
        for index, (bank, _) in enumerate(packet["packets"]):
            key = (
                bank,
                entities[index, : counts[index]].tobytes(),
                globals_[index].tobytes(),
                host["events"][index].numpy().tobytes(),
                host["hidden"][index].numpy().tobytes(),
                host["cell"][index].numpy().tobytes(),
                bool(records["done"][index]),
            )
            representatives[index] = seen.setdefault(key, index)
        unique = representatives == np.arange(len(counts))
        offsets = np.cumsum(np.where(unique, counts, 0)) - np.where(unique, counts, 0)
        observations = PackedEntityBatch(
            np.concatenate(
                [entities[index, : counts[index]].copy() for index in np.flatnonzero(unique)],
                axis=0,
            ),
            offsets[representatives],
            counts.copy(),
            globals_.copy(),
        )
        needed = unique & ~records["done"]
        plan = ActiveBatchPlan.from_counts(
            needed,
            counts,
            self.features.limit,
            self.teacher.entity.microbatch,
            self.teacher.entity.token_budget,
        )
        bootstrap = packet["state"].hidden.new_zeros(len(counts))
        if plan.logical_rows:
            with self.profiler.track("probe_bootstrap_inference"):
                output = forward_active(self.teacher, packet["observation"], packet["state"], plan)
                values = (
                    self.teacher.action_values(
                        output, observations=packet["observation"], plan=plan
                    )
                    .branches.max(-1)
                    .values
                )
                bootstrap = torch.where(
                    torch.as_tensor(records["done"], device=values.device),
                    0,
                    values[torch.as_tensor(representatives, device=values.device)],
                )
        packet.update(
            records=records,
            observations=observations,
            indices=np.array([self.jobs[bank].trajectory_row for bank, _ in packet["packets"]]),
            slots=np.array([slot for _, slot in packet["packets"]]),
            bootstrap=self.env.handoff.enqueue("auxiliary_bootstrap", bootstrap),
        )
        packet.pop("host")
        self._bootstrap = packet
        self.inference_metrics["logical_rows"] += int((~records["done"]).sum())
        self.inference_metrics["unique_rows"] += plan.logical_rows
        self.inference_metrics["padded_rows"] += plan.padded_rows
        self.inference_metrics["bucket_widths"] = tuple(
            bucket.entity_width for bucket in plan.buckets
        )

    def release_workspaces(self):
        if self.has_work:
            raise RuntimeError("Drain background probes before releasing their workspaces")
        self.batch = self.features = self.source_batch = self.source_features = None
        self.state = self.source_state = self.teacher = None
        for name in (
            "start",
            "counts",
            "writes",
            "initial",
            "reason",
            "accumulated",
            "counts_host",
            "writes_host",
            "entity_counts",
        ):
            setattr(self, name, None)
        self.free_sources.clear()

    def snapshot(self):
        if self.has_work:
            raise RuntimeError("Drain background probes before checkpointing")
        return dict(
            tile_cursor=self.tile_cursor.cpu(),
            plant_cursor=self.plant_cursor.cpu(),
            layout=self.layout.metadata(),
            timing=self.profiler.flush(),
        )

    def restore(self, state):
        if state.get("layout") != self.layout.metadata():
            raise ValueError("Tile probe layout differs from recovery state")
        for name in ("tile_cursor", "plant_cursor"):
            value = state[name]
            if (
                value.shape != getattr(self, name).shape
                or value.dtype != torch.long
                or torch.any(value < 0)
            ):
                raise ValueError("Invalid tile schedule recovery state")
            getattr(self, name).copy_(value)
        self.profiler.seconds.clear()
