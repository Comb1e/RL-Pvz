"""Cohort-frozen tile exploration and planting/wait-triggered commitments."""

import copy
import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.policy.sequential_q import (
    action_parts,
    assemble,
    candidate_maxima,
    select_q_tiles,
)

EXPLORATION_PROTOCOL = "committed_plant_tile_epsilon_v1"
CONTROLLER_PROTOCOL = "pvz-rl/committed-plant-state-v2"
LEGACY_CONTROLLER_PROTOCOL = "pvz-rl/committed-plant-state-v1"


def refresh_wait_exploration(cfg, current):
    """Refresh only the requested wait interval; saved probability schedules stay fixed."""
    result = copy.deepcopy(cfg)
    source = current["training"]["exploration"]
    target = result["training"]["exploration"]
    if "plant_wait_check_seconds" in source:
        target["plant_wait_check_seconds"] = source["plant_wait_check_seconds"]
    else:
        target.pop("plant_wait_check_seconds", None)
    return result


def exploration_status(model):
    """Host-only reporting of requested settings and this cohort's active trigger."""
    from pvz_game import Rules

    interval = getattr(getattr(model, "_plant_exploration", None), "wait_check_ticks", None)
    return dict(
        plant_exploration_epsilon=model.plant_exploration_rate,
        plant_wait_check_seconds=interval / Rules().game["tick_rate"]
        if interval is not None
        else None,
        requested_plant_wait_check_seconds=model.cfg["training"]["exploration"].get(
            "plant_wait_check_seconds"
        ),
    )


@dataclass(frozen=True)
class PlantExplorationCheck:
    fired: torch.Tensor
    plant_checks: torch.Tensor
    wait_checks: torch.Tensor


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
    species_only = settings["objective"] == "committed_species_epsilon_v1"
    progress = min(1.0, max(0, stage_games) / settings.get("decay_games", 1))
    plant_progress = min(1.0, max(0, stage_games) / settings["plant_decay_games"])
    start, floor = settings.get("epsilon_start", 1.0), settings.get("epsilon_floor", 1.0)
    plant_start, plant_floor = settings["plant_epsilon_start"], settings["plant_epsilon_floor"]
    return ExplorationState(
        "formal",
        progress,
        0.0 if species_only else start * (floor / start) ** progress,
        progress >= 1,
        plant_progress,
        plant_start * (plant_floor / plant_start) ** plant_progress,
        plant_progress >= 1,
    )


def configure_exploration(model, cfg):
    model.exploration_settings = dict(cfg["training"]["exploration"])
    set_exploration_rate(
        model,
        getattr(model, "exploration_rate", model.exploration_settings.get("epsilon_start", 0.0)),
        plant_rate=getattr(
            model, "plant_exploration_rate", model.exploration_settings["plant_epsilon_start"]
        ),
    )


def set_exploration_rate(model, rate, *, plant_rate=None):
    if model.cfg["training"]["exploration"]["objective"] == "committed_species_epsilon_v1":
        rate = 0.0
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
    def __init__(self, count, device, plant_epsilon, tile_epsilon, *, wait_check_ticks=None):
        self.pending = torch.zeros(count, dtype=torch.long, device=device)
        self.wait_ticks = torch.zeros_like(self.pending)
        if wait_check_ticks is not None and (
            type(wait_check_ticks) is not int or not 1 <= wait_check_ticks < 2**63
        ):
            raise ValueError("Invalid plant wait check interval")
        self.wait_check_ticks = wait_check_ticks
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in (plant_epsilon, tile_epsilon)
        ):
            raise ValueError("Invalid committed exploration probabilities")
        self.plant_epsilon, self.tile_epsilon = float(plant_epsilon), float(tile_epsilon)

    @torch.no_grad()
    def apply(
        self,
        policy,
        actions,
        details,
        masks,
        sun,
        cooldowns,
        costs,
        active,
        *,
        branch_usable=None,
        plan=None,
    ):
        pending = self.pending.clone()
        committed = pending.ne(0) & active
        rows = torch.arange(len(actions), device=actions.device)
        species = (pending - 1).clamp_min(0)
        # Tile occupancy is validated after the all-45 tile choice. A committed
        # branch therefore becomes ready based only on card availability; an
        # occupied selection is retained as a tile-invalid proposal.
        ready = committed & (sun >= costs[species]) & (cooldowns[rows, species] == 0)
        if branch_usable is not None:
            branch_usable = torch.as_tensor(branch_usable, device=actions.device, dtype=torch.bool)
            if branch_usable.shape != (len(actions), A.plant_types + 2):
                raise ValueError("branch_usable must have shape [batch, 10]")
            ready &= branch_usable.gather(1, pending.clamp_min(1)[:, None]).flatten()
        policy_branches, _ = action_parts(actions)
        reuse = ready & (policy_branches == pending)
        requests = torch.where(ready & ~reuse, pending, 0)
        conditional, reach = policy.conditional_tiles(
            details["observations"],
            requests,
            plan=plan,
            memory=details["board_memory"],
            return_reach=True,
        )
        details["reach_logits"] = torch.where(
            (ready & ~reuse)[:, None, None], reach, details["reach_logits"]
        )
        conditional = torch.where(reuse[:, None], details["tile_q"], conditional)
        tiles, _, fired, tile_q = select_q_tiles(
            conditional,
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
        details["tile_q"] = torch.where(committed[:, None], conditional, details["tile_q"])
        details["valid"] &= (torch.isfinite(tile_q) | ~ready[:, None]).all()
        details["selected_value"] = torch.where(
            branches > 0, details["tile_value"], details["branch_q"][:, 0]
        )
        details["placement_gap"] = torch.where(
            branches > 0,
            candidate_maxima(details["tile_q"], masks, branches) - details["tile_value"],
            0,
        )
        return chosen, details["branch_value"], details["tile_value"], details

    @torch.no_grad()
    def observe(self, actions, accepted, mode, active, done, transition_ticks, *, draws=None):
        branches, _ = action_parts(actions)
        normal = active & ~done & (self.pending == 0) & (mode == PlantExplorationMode.NORMAL)
        plant_checks = normal & accepted & (branches > 0) & (branches <= A.plant_types)
        ordinary_wait = normal & accepted & (actions == 0)
        wait_checks = torch.zeros_like(active)
        if self.wait_check_ticks is not None:
            # Inactive tickets consume neither simulated time nor their saved draw.
            self.wait_ticks = torch.where(
                active,
                torch.where(ordinary_wait, self.wait_ticks + transition_ticks, 0),
                self.wait_ticks,
            )
            wait_checks = ordinary_wait & (self.wait_ticks >= self.wait_check_ticks)
            self.wait_ticks = torch.where(wait_checks, 0, self.wait_ticks)
        coin, alternative, wait_choice = (
            draws
            if draws is not None
            else (
                torch.rand(len(actions), device=actions.device),
                torch.randint(1, A.plant_types, actions.shape, device=actions.device),
                torch.randint(1, A.plant_types + 1, actions.shape, device=actions.device)
                if self.wait_check_ticks is not None
                else torch.ones_like(actions),
            )
        )
        fired = (plant_checks | wait_checks) & (coin < self.plant_epsilon)
        alternative = alternative.clone()
        alternative += alternative >= branches
        alternative = torch.where(wait_checks, wait_choice, alternative)
        completed = active & accepted & (mode == PlantExplorationMode.PLANTING)
        self.pending = torch.where(completed | (active & done), 0, self.pending)
        self.pending = torch.where(fired, alternative, self.pending)
        return PlantExplorationCheck(fired, plant_checks, wait_checks)

    def snapshot(self):
        return dict(
            protocol=CONTROLLER_PROTOCOL,
            pending=self.pending.cpu().clone(),
            wait_ticks=self.wait_ticks.cpu().clone(),
            wait_check_ticks=self.wait_check_ticks,
            plant_epsilon=self.plant_epsilon,
            tile_epsilon=self.tile_epsilon,
        )

    def restore(self, saved):
        pending = saved.get("pending")
        protocol = saved.get("protocol")
        if (
            protocol not in (CONTROLLER_PROTOCOL, LEGACY_CONTROLLER_PROTOCOL)
            or not isinstance(pending, torch.Tensor)
            or pending.shape != self.pending.shape
            or pending.dtype != torch.long
            or torch.any((pending < 0) | (pending > A.plant_types))
            or saved.get("plant_epsilon") != self.plant_epsilon
            or saved.get("tile_epsilon") != self.tile_epsilon
        ):
            raise ValueError("Invalid committed plant recovery state")
        if protocol == LEGACY_CONTROLLER_PROTOCOL:
            ticks, interval = torch.zeros_like(pending), None
        else:
            ticks, interval = saved.get("wait_ticks"), saved.get("wait_check_ticks")
            if (
                "wait_check_ticks" not in saved
                or (
                    interval is not None
                    and (type(interval) is not int or not 1 <= interval < 2**63)
                )
                or not isinstance(ticks, torch.Tensor)
                or ticks.shape != pending.shape
                or ticks.dtype != torch.long
                or torch.any(ticks < 0)
                or (interval is None and torch.any(ticks != 0))
                or (interval is not None and torch.any(ticks >= interval))
                or torch.any((pending != 0) & (ticks != 0))
            ):
                raise ValueError("Invalid committed plant waiting recovery state")
        self.pending = pending.to(self.pending.device, copy=True)
        self.wait_ticks = ticks.to(self.pending.device, copy=True)
        self.wait_check_ticks = interval
