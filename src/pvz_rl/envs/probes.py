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
from pvz_rl.envs.encoding import ENTITY_WIDTH, ObservationEncoder, PackedEntityBatch
from pvz_rl.envs.history import public_history_event
from pvz_rl.envs.rewards import reward_parts
from pvz_rl.learning.cuda_buffer import probe_dtype
from pvz_rl.learning.objective import COMPONENTS, MAX_PROBES
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
        self.profiler = env.profiler
        self.features.profiler = self.profiler
        self.copy_kernel = env.features.module.get_function("copy_probe_state")
        objective = env.cfg["training"]["objective"]
        self.branch_probes = objective["branch_probes"]
        self.tile_probes = objective["tile_probes"]
        self.probe_interval = objective["probe_interval_decisions"]
        self.branch_cursor = torch.zeros(batch.n, dtype=torch.long, device=batch.device)
        self.tile_cursor = torch.zeros(batch.n, 10, dtype=torch.long, device=batch.device)
        self.decision_cursor = torch.zeros(batch.n, dtype=torch.long, device=batch.device)
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

    def schedule(self, policy, actions, details, masks, active):
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
        for slot in range(self.branch_probes):
            alternative = chosen[:, slot]
            values = policy.tile_values(details["tile_features"], details["context"], alternative)
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
    def collect(self, policy, teacher_memory, obs, actions, details, masks, active):
        proposals, valid = self.schedule(policy, actions, details, masks, active)
        teacher = teacher_memory.policy
        with self.profiler.track("ema_inference"):
            output, _, _ = teacher_memory.advance(obs, active)
        source, scratch = self.env.batch, self.batch
        for first in range(0, MAX_PROBES, self.lanes):
            with self.profiler.track("probe_simulation"):
                for lane, slot in enumerate(range(first, first + self.lanes)):
                    destination = lane * source.n
                    self.copy_kernel(
                        (source.n,),
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
                            self.features.home_ledger[destination:],
                            self.features.assets[destination:],
                            scratch.cp.from_dlpack(valid[:, slot].contiguous()),
                        ),
                    )
            pair_actions = proposals[:, first : first + self.lanes].T.contiguous().flatten()
            pair_valid = valid[:, first : first + self.lanes].T.contiguous().flatten()
            self.features.totals.fill(0)
            self.features.step(
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
            parts = torch.from_dlpack(self.features.parts)[
                :, [REWARD_FIELDS.index(name) for name in COMPONENTS]
            ]
            rows = slice(first * source.n, (first + self.lanes) * source.n)
            self.metadata[rows].copy_(
                torch.cat(
                    (
                        torch.stack(
                            (
                                pair_valid,
                                self.slots[rows] < 2,
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
            )
        duration = self.metadata[:, 6].long()
        width = self.env.probe_width()
        next_obs = self.features.view(0, MAX_PROBES * source.n, width)
        next_state = EventMemoryState(
            output.state.hidden.repeat(1, MAX_PROBES, 1),
            output.state.cell.repeat(1, MAX_PROBES, 1),
            output.state.pending.repeat(MAX_PROBES, 1),
            output.state.elapsed_ticks.repeat(MAX_PROBES),
        ).observe(self.features.history_tensor, duration, valid.T.contiguous().flatten())
        history_inputs = torch.where(
            valid.T.contiguous().flatten()[:, None], next_state.inputs(), 0
        )
        history_writes = history_inputs[:, :7].ne(0).any(-1)
        with self.profiler.track("ema_inference"):
            target = (
                teacher.forward_step(
                    next_obs,
                    next_state,
                    events=history_inputs,
                    memory_write=history_writes,
                )
                .branch_q.max(-1)
                .values
            )
            target *= torch.pow(self.env.cfg["training"]["gamma"], duration.float())
            self.metadata[:, 16].copy_(torch.where(self.metadata[:, 8].bool(), 0, target))
        with self.profiler.track("encoding"):
            valid_flat = valid.T.contiguous().flatten()
            self.features.dedup(history_inputs, valid_flat, self.source, MAX_PROBES)
            unique = valid_flat & (self.source == self.slots)
            counts = torch.where(unique, self.features.summary_tensor[:, 0], 0)
            offsets = counts.cumsum(0) - counts
            self.features.pack(unique, offsets, self.packed)
            representative = self.source.clamp_min(0) * source.n + self.games
            observation_counts = torch.where(valid_flat, self.features.summary_tensor[:, 0], 0)
            observation_offsets = torch.where(valid_flat, offsets[representative], 0)
        with self.profiler.track("transfer"):
            return self.env.handoff.enqueue_fields(
                "probe",
                dict(
                    metadata=self.metadata,
                    events=history_inputs,
                    memory_write=history_writes,
                    entities=self.packed[: MAX_PROBES * source.n * width],
                    counts=observation_counts,
                    offsets=observation_offsets,
                    size=counts.sum()[None],
                    globals=self.features.globals_tensor,
                    summary=self.features.summary_tensor,
                    source=self.source,
                    width=torch.tensor(width, device=source.device),
                ),
            )

    @staticmethod
    def materialize(host):
        count = host["metadata"].shape[0] // MAX_PROBES
        values = host["metadata"].numpy().reshape(MAX_PROBES, count, 17).transpose(1, 0, 2)
        if not np.isfinite(values).all():
            raise RuntimeError("Invalid counterfactual accounting or exhausted proximity ledger")
        if np.any(host["counts"].numpy() > int(host["width"])):
            raise RuntimeError("Counterfactual entity growth exceeded its padding bound")
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
        observations = PackedEntityBatch(
            host["entities"].numpy()[: int(host["size"][0])],
            host["offsets"].numpy().reshape(MAX_PROBES, count).T,
            host["counts"].numpy().reshape(MAX_PROBES, count).T,
            host["globals"].numpy().reshape(MAX_PROBES, count, -1).transpose(1, 0, 2),
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
