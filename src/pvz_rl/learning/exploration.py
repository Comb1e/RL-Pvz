"""Cohort-frozen tile exploration and accepted-plant species commitments."""

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.policy.sequential_q import action_parts, assemble, select_q_tiles

EXPLORATION_PROTOCOL = "committed_plant_tile_epsilon_v1"
CONTROLLER_PROTOCOL = "pvz-rl/committed-plant-state-v1"


class PlantExplorationMode(IntEnum):
    NORMAL = 0
    WAITING = 1
    PLANTING = 2


@dataclass(frozen=True)
class ExplorationState:
    phase: str
    progress: float
    epsilon: float
    at_floor: bool
    plant_progress: float
    plant_epsilon: float
    plant_at_floor: bool

    @property
    def tile_epsilon(self):
        return self.epsilon


def exploration_state(cfg, stage_games):
    settings = cfg["training"]["exploration"]
    progress = min(1.0, max(0, stage_games) / settings["decay_games"])
    plant_progress = min(1.0, max(0, stage_games) / settings["plant_decay_games"])
    start, floor = settings["epsilon_start"], settings["epsilon_floor"]
    plant_start, plant_floor = settings["plant_epsilon_start"], settings["plant_epsilon_floor"]
    return ExplorationState(
        "formal",
        progress,
        start * (floor / start) ** progress,
        progress >= 1,
        plant_progress,
        plant_start * (plant_floor / plant_start) ** plant_progress,
        plant_progress >= 1,
    )


def configure_exploration(model, cfg):
    model.exploration_settings = dict(cfg["training"]["exploration"])
    set_exploration_rate(
        model,
        getattr(model, "exploration_rate", model.exploration_settings["epsilon_start"]),
        plant_rate=getattr(
            model, "plant_exploration_rate", model.exploration_settings["plant_epsilon_start"]
        ),
    )


def set_exploration_rate(model, rate, *, plant_rate=None):
    model.exploration_rate = float(rate)
    model.policy.exploration_epsilon = 0.0
    model.policy.tile_exploration_epsilon = float(rate)
    model.policy_kwargs["exploration_epsilon"] = 0.0
    model.policy_kwargs["tile_exploration_epsilon"] = float(rate)
    if plant_rate is not None:
        model.plant_exploration_rate = float(plant_rate)


def apply_exploration_state(model, state):
    set_exploration_rate(model, state.epsilon, plant_rate=state.plant_epsilon)
    model.exploration_phase = state.phase
    model.exploration_progress = state.progress
    model.exploration_at_floor = state.at_floor
    model.plant_exploration_progress = state.plant_progress
    model.plant_exploration_at_floor = state.plant_at_floor


class CommittedPlantExploration:
    def __init__(self, count, device, plant_epsilon, tile_epsilon):
        self.pending = torch.zeros(count, dtype=torch.long, device=device)
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in (plant_epsilon, tile_epsilon)
        ):
            raise ValueError("Invalid committed exploration probabilities")
        self.plant_epsilon, self.tile_epsilon = float(plant_epsilon), float(tile_epsilon)

    @torch.no_grad()
    def apply(self, policy, actions, details, masks, sun, cooldowns, costs, active, *, plan=None):
        pending = self.pending.clone()
        committed = pending.ne(0) & active
        rows = torch.arange(len(actions), device=actions.device)
        species = (pending - 1).clamp_min(0)
        empty = A.tile_masks(masks)[rows, species].any(-1)
        ready = committed & empty & (sun >= costs[species]) & (cooldowns[rows, species] == 0)
        tile_values = policy.tile_values
        if plan is not None:
            from pvz_rl.policy.active_batch import active_tile_values

            tile_values = active_tile_values(policy, plan)
        tiles, _, fired, tile_q = select_q_tiles(
            details["tile_features"],
            details["context"],
            tile_values,
            masks,
            pending.clamp_min(1),
            self.tile_epsilon,
        )
        proposed = torch.where(ready, assemble(pending, tiles), 0)
        chosen = torch.where(committed, proposed, actions)
        branches, locations = action_parts(chosen)
        mode = torch.where(committed, int(PlantExplorationMode.WAITING), 0)
        mode = torch.where(ready, int(PlantExplorationMode.PLANTING), mode)
        details["policy_action"] = actions
        details["exploration_mode"] = mode
        details["pending_species"] = pending
        details["coins"] = details["coins"].clone()
        details["coins"][:, 1] = torch.where(committed, fired & ready, details["coins"][:, 1])
        details["branch_value"] = details["branch_q"].gather(1, branches[:, None]).flatten()
        selected_tile = tile_q.gather(1, locations[:, None]).flatten()
        details["tile_value"] = torch.where(
            committed, torch.where(ready, selected_tile, 0), details["tile_value"]
        )
        details["valid"] &= (torch.isfinite(tile_q) | ~ready[:, None]).all()
        return chosen, details["branch_value"], details["tile_value"], details

    @torch.no_grad()
    def observe(self, actions, accepted, mode, active, done):
        branches, _ = action_parts(actions)
        eligible = (
            active
            & ~done
            & accepted
            & (self.pending == 0)
            & (mode == PlantExplorationMode.NORMAL)
            & (branches > 0)
            & (branches <= A.plant_types)
        )
        fired = eligible & (torch.rand(len(actions), device=actions.device) < self.plant_epsilon)
        alternative = torch.randint(1, A.plant_types, actions.shape, device=actions.device)
        alternative += alternative >= branches
        completed = active & accepted & (mode == PlantExplorationMode.PLANTING)
        self.pending = torch.where(completed | done, 0, self.pending)
        self.pending = torch.where(fired, alternative, self.pending)
        return fired

    def snapshot(self):
        return dict(
            protocol=CONTROLLER_PROTOCOL,
            pending=self.pending.cpu(),
            plant_epsilon=self.plant_epsilon,
            tile_epsilon=self.tile_epsilon,
        )

    def restore(self, saved):
        pending = saved.get("pending")
        if (
            saved.get("protocol") != CONTROLLER_PROTOCOL
            or not isinstance(pending, torch.Tensor)
            or pending.shape != self.pending.shape
            or pending.dtype != torch.long
            or torch.any((pending < 0) | (pending > A.plant_types))
            or saved.get("plant_epsilon") != self.plant_epsilon
            or saved.get("tile_epsilon") != self.tile_epsilon
        ):
            raise ValueError("Invalid committed plant recovery state")
        self.pending = pending.to(self.pending.device)
