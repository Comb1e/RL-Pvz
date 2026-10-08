"""Tile-only epsilon budget resolved once per complete-game cohort."""

from dataclasses import dataclass

EXPLORATION_PROTOCOL = "sequential_tile_epsilon_v1"


@dataclass(frozen=True)
class ExplorationState:
    phase: str
    progress: float
    epsilon: float
    at_floor: bool

    @property
    def tile_epsilon(self):
        """Direct tile-choice coin probability (validation sets it to zero)."""
        return self.epsilon


def exploration_state(cfg, stage_games, *, staged=True):
    settings = cfg["training"]["exploration"]
    progress = min(1.0, max(0, stage_games) / settings["decay_games"])
    start, floor = settings["epsilon_start"], settings["epsilon_floor"]
    return ExplorationState("formal", progress, start * (floor / start) ** progress, progress >= 1)


def exploration_rate(cfg, stage_games, *, staged=True):
    return exploration_state(cfg, stage_games, staged=staged).epsilon


def configure_exploration(model, cfg):
    model.exploration_settings = dict(cfg["training"]["exploration"])
    set_exploration_rate(
        model, getattr(model, "exploration_rate", model.exploration_settings["epsilon_start"])
    )


def set_exploration_rate(model, rate):
    model.exploration_rate = float(rate)
    # The first-level ten-way proposal is always greedy. The cohort budget is
    # spent by the selected branch's tile selector only.
    model.policy.exploration_epsilon = 0.0
    model.policy.tile_exploration_epsilon = float(rate)
    model.policy_kwargs["exploration_epsilon"] = 0.0
    model.policy_kwargs["tile_exploration_epsilon"] = float(rate)


def apply_exploration_state(model, cfg, state):
    set_exploration_rate(model, state.epsilon)
    model.exploration_phase = state.phase
    model.exploration_progress = state.progress
    model.exploration_at_floor = state.at_floor
