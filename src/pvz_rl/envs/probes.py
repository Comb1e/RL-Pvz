"""Isolated CPU reference and device-native counterfactual simulator execution."""

import copy

import numpy as np
import torch
from pvz_game import Wait

from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures
from pvz_rl.envs.encoding import EntityBatch, ObservationEncoder
from pvz_rl.envs.rewards import reward_parts
from pvz_rl.learning.cuda_buffer import probe_dtype
from pvz_rl.learning.host_transfer import HostTransfer
from pvz_rl.learning.objective import COMPONENTS, MAX_PROBES
from pvz_rl.monitoring.cuda_diagnostics import DeviceProfiler
from pvz_rl.policy.sequential_q import action_parts, assemble
from pvz_rl.policy.transformer_lstm import RecurrentState


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
    )


class CounterfactualCollector:
    def __init__(self, env):
        self.env = env
        b = env.batch
        self.batch = AccountingCudaBatch(
            b.n,
            zombie_capacity=b.zcap,
            projectile_capacity=b.qcap,
            max_step_ticks=b.max_step_ticks,
            rules=b.rules,
            device=b.device,
        )
        self.batch._schedules = b._schedules  # The pinned step kernel only reads schedules.
        self.features = CudaFeatures(self.batch, env.cfg, env.condition)
        self.copy_kernel = env.features.module.get_function("copy_probe_state")
        objective = env.cfg["training"].get("objective", {})
        self.branch_probes = int(objective.get("branch_probes", 2))
        self.tile_probes = int(objective.get("tile_probes", 2))
        self.probe_interval = int(objective.get("probe_interval_decisions", 1))
        if not 0 <= self.branch_probes <= 2 or not 0 <= self.tile_probes <= 2:
            raise ValueError("Counterfactual probes must be between zero and two per role")
        if self.probe_interval < 1:
            raise ValueError("probe_interval_decisions must be positive")
        self.branch_cursor = torch.zeros(b.n, dtype=torch.long, device="cuda")
        self.tile_cursor = torch.zeros(b.n, 10, dtype=torch.long, device="cuda")
        self.decision_cursor = torch.zeros(b.n, dtype=torch.long, device="cuda")
        self.transfer = HostTransfer()
        self.profiler = DeviceProfiler(b.cp, env.cfg.get("simulation", {}).get("profile", False))

    def schedule(self, policy, actions, details, masks, active):
        n = len(actions)
        device = actions.device
        branch, tile = action_parts(actions)
        all_branches = torch.arange(10, device=device).expand(n, -1)
        alternatives = all_branches[all_branches != branch[:, None]].reshape(n, 9)
        chosen = (
            alternatives.gather(
                1,
                (self.branch_cursor[:, None] + torch.arange(self.branch_probes, device=device)) % 9,
            )
            if self.branch_probes
            else alternatives[:, :0]
        )
        geometry = masks[:, 1:].reshape(n, 9, A.tiles)
        proposals = torch.zeros(n, MAX_PROBES, dtype=torch.long, device=device)
        eligible = active & (self.decision_cursor % self.probe_interval == 0)
        valid = eligible[:, None].expand(-1, MAX_PROBES).clone()
        for slot in range(self.branch_probes):
            b = chosen[:, slot]
            values = policy.tile_values(details["tile_features"], details["context"], b)
            legal = geometry[torch.arange(n, device=device), (b - 1).clamp_min(0)]
            locations = values.masked_fill(~legal, -torch.inf).argmax(-1)
            proposals[:, slot] = assemble(b, locations)
        candidate = geometry[torch.arange(n, device=device), (branch - 1).clamp_min(0)].clone()
        candidate.scatter_(1, tile[:, None], False)
        indices = (
            torch.arange(A.tiles, device=device)
            .expand(n, -1)
            .masked_fill(~candidate, A.tiles)
            .sort(-1)
            .values
        )
        counts = candidate.sum(-1)
        cursor = self.tile_cursor.gather(1, branch[:, None])
        for j in range(2):
            location = indices.gather(1, ((cursor + j) % counts[:, None].clamp_min(1))).flatten()
            valid[:, j + 2] &= (branch != 0) & (counts > j) & (j < self.tile_probes)
            proposals[:, j + 2] = assemble(branch, location.clamp_max(A.tiles - 1))
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
        with self.profiler.track("teacher_inference"):
            out = teacher.forward_step(
                obs,
                teacher_memory.state,
                previous_action=teacher_memory.previous_actions,
                execution_outcome=teacher_memory.outcomes,
            )
        keep = active[None, :, None]
        teacher_memory.state = RecurrentState(
            torch.where(keep, out.state.hidden, teacher_memory.state.hidden),
            torch.where(keep, out.state.cell, teacher_memory.state.cell),
        )
        teacher_memory.resets[active] = False
        source, scratch = self.env.batch, self.batch
        columns, observations = [], []
        for slot in range(MAX_PROBES):
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
                    self.env.features.home_ledger,
                    self.env.features.assets,
                    scratch.header,
                    scratch.plants,
                    scratch.zombies,
                    scratch.projectiles,
                    scratch.mowers,
                    scratch.cooldowns,
                    self.features.home_ledger,
                    self.features.assets,
                    scratch.cp.from_dlpack(valid[:, slot].contiguous()),
                ),
            )
            self.features.totals.fill(0)
            with self.profiler.track("probe_simulation_encoding"):
                next_obs, _ = self.features.step(
                    scratch.cp.from_dlpack(proposals[:, slot].contiguous())
                )
            h = torch.from_dlpack(scratch.header)
            accepted = h[:, 12].bool()
            executed = torch.where(accepted, proposals[:, slot], 0)
            duration = h[:, 14]
            done = (h[:, 1] != 0) | (
                h[:, 0]
                >= self.env.cfg["environment"]["cutoff_seconds"] * scratch.rules.game["tick_rate"]
            )
            outcome = torch.stack((accepted.float(), duration.float()), -1)
            with self.profiler.track("teacher_inference"):
                target = (
                    teacher.forward_step(
                        next_obs, out.state, previous_action=executed, execution_outcome=outcome
                    )
                    .branch_q.max(-1)
                    .values
                )
            target = target * torch.pow(self.env.cfg["training"]["gamma"], duration.float())
            parts = torch.from_dlpack(self.features.parts)[
                :, [REWARD_FIELDS.index(k) for k in COMPONENTS]
            ]
            columns.append(
                torch.cat(
                    (
                        torch.stack(
                            (
                                valid[:, slot],
                                torch.full_like(valid[:, slot], slot < 2),
                                proposals[:, slot],
                                executed,
                                accepted,
                                h[:, 13],
                                duration,
                                h[:, 0],
                                done,
                                h[:, 1] == 1,
                            ),
                            -1,
                        ).double(),
                        parts,
                        torch.where(done, 0, target)[:, None].double(),
                    ),
                    -1,
                )
            )
            observations.append(next_obs)
        payload = {"metadata": torch.stack(columns, 1)}
        for index, observation in enumerate(observations):
            for name, tensor in zip(("entities", "mask", "globals"), observation.tensors()):
                payload[f"{index}_{name}"] = tensor
        return self.transfer.enqueue(payload)

    @staticmethod
    def materialize(host):
        values = host["metadata"].numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("Invalid counterfactual accounting or exhausted proximity ledger")
        records = np.zeros(values.shape[:2], dtype=probe_dtype())
        for index, field in enumerate(
            (
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
        ):
            records[field] = values[..., index]
        records["components"] = values[..., 10:16]
        records["bootstrap"] = values[..., 16]
        observations = [
            EntityBatch(
                host[f"{i}_entities"], host[f"{i}_mask"], host[f"{i}_globals"], validated=True
            )
            for i in range(MAX_PROBES)
        ]
        return records, observations

    def snapshot(self):
        return dict(
            branch_cursor=self.branch_cursor.cpu(),
            tile_cursor=self.tile_cursor.cpu(),
            decision_cursor=self.decision_cursor.cpu(),
            timing=self.profiler.flush(),
        )

    def restore(self, state):
        for name, shape in (
            ("branch_cursor", self.branch_cursor.shape),
            ("tile_cursor", self.tile_cursor.shape),
            ("decision_cursor", self.decision_cursor.shape),
        ):
            cursor = state.get(name, torch.zeros(shape, dtype=torch.long))
            if cursor.shape != shape or torch.any(cursor < 0):
                raise ValueError("Invalid counterfactual schedule cursor")
        self.branch_cursor.copy_(state["branch_cursor"])
        self.tile_cursor.copy_(state["tile_cursor"])
        self.decision_cursor.copy_(
            state.get("decision_cursor", torch.zeros_like(self.decision_cursor))
        )
        self.profiler.seconds.update(state.get("timing", {}))
