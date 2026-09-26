"""Plant-only epsilon budget resolved once per complete-game cohort."""

from dataclasses import dataclass

from pvz_rl.policy.sequential_q import per_head_epsilon

EXPLORATION_PROTOCOL = "sequential_plant_epsilon_v1"


@dataclass(frozen=True)
class ExplorationState:
    phase: str
    progress: float
    epsilon: float
    at_floor: bool

    @property
    def per_head_epsilon(self):
        return per_head_epsilon(self.epsilon)


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
    per_head_epsilon(rate)
    model.exploration_rate = float(rate)
    model.policy.exploration_epsilon = float(rate)
    model.policy_kwargs["exploration_epsilon"] = float(rate)


def apply_exploration_state(model, cfg, state):
    set_exploration_rate(model, state.epsilon)
    model.exploration_phase = state.phase
    model.exploration_progress = state.progress
    model.exploration_at_floor = state.at_floor
