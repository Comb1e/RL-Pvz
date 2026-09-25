"""Plant-only exploration resolved once per complete-game cohort."""

from dataclasses import dataclass

import torch

EXPLORATION_PROTOCOL = "cohort_plant_floor_v1"


@dataclass(frozen=True)
class ExplorationState:
    phase: str
    progress: float
    epsilon: float
    entropy_factor: float
    at_floor: bool


def exploration_state(cfg, stage_games, *, staged=True):
    settings = cfg["training"]["exploration"]
    progress = min(1.0, max(0, stage_games) / settings["decay_games"])
    start, floor = settings["epsilon_start"], settings["epsilon_floor"]
    return ExplorationState(
        "formal",
        progress,
        start * (floor / start) ** progress,
        settings["entropy_floor_fraction"] ** progress,
        progress >= 1,
    )


def exploration_rate(cfg, stage_games, *, staged=True):
    return exploration_state(cfg, stage_games, staged=staged).epsilon


def entropy_factor(cfg, stage_games, *, staged=True):
    return exploration_state(cfg, stage_games, staged=staged).entropy_factor


def exploration_loss(policy, entropy, log_prob):
    settings = policy.exploration_settings
    dist = policy.action_dist
    counts = dist.legal_tile_counts
    available = counts > 0
    normalized = dist.locations.entropy() / counts.clamp_min(2).float().log()
    tiles = (normalized * available).sum(-1) / available.sum(-1).clamp_min(1)
    plants = dist.plants.entropy() / dist.available_plants.sum(-1).clamp_min(2).float().log()
    plant_bonus = settings["plant_coef"] * plants.mean()
    tile_bonus = settings["tile_coef"] * tiles.mean()
    bonus = plant_bonus + tile_bonus
    return -bonus, {
        "joint_entropy": entropy.mean().detach(),
        "exploration_bonus": bonus.detach(),
        "plant_exploration_bonus": plant_bonus.detach(),
        "tile_exploration_bonus": tile_bonus.detach(),
    }


def configure_exploration(model, cfg):
    model.exploration_settings = dict(cfg["training"]["exploration"])
    set_exploration_rate(
        model, getattr(model, "exploration_rate", model.exploration_settings["epsilon_start"])
    )
    set_entropy_factor(model, cfg, getattr(model, "entropy_factor", 1.0))


def set_entropy_factor(model, cfg, factor):
    model.entropy_factor = factor
    model.policy.exploration_settings = dict(cfg["training"]["exploration"])
    for key in ("plant_coef", "tile_coef"):
        model.policy.exploration_settings[key] *= factor


def set_exploration_rate(model, rate):
    model.exploration_rate = float(rate)
    model.policy.action_dist.epsilon = float(rate)
    model.policy.exploration_epsilon = float(rate)
    model.policy_kwargs["exploration_epsilon"] = float(rate)


def apply_exploration_state(model, cfg, state):
    set_exploration_rate(model, state.epsilon)
    set_entropy_factor(model, cfg, state.entropy_factor)
    model.exploration_phase = state.phase
    model.exploration_progress = state.progress
    model.exploration_at_floor = state.at_floor


def pooled_spatial(features, channels, scalar_channels, rows=5, cols=9):
    spatial = features[:, scalar_channels:].reshape(-1, channels, rows, cols)
    pooled = torch.cat(
        (spatial.mean((2, 3)), spatial.amax((2, 3)), features[:, :scalar_channels]), dim=1
    )
    return spatial, pooled
