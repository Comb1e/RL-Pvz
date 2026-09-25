"""Exploration regularization and the stage-aware phase/floor schedule."""

from dataclasses import dataclass

import torch

EXPLORATION_PROTOCOL = "phase_floor_v1"


@dataclass(frozen=True)
class ExplorationState:
    """Resolved exploration values frozen for one rollout window."""

    phase: str
    progress: float
    epsilon: float
    entropy_factor: float
    at_floor: bool


def exploration_state(cfg, stage_games, *, staged=True):
    """Resolve warm-up or formal exploration without hidden schedule state."""
    settings = cfg["training"]["exploration"]
    warmup_games = cfg["training"].get("critic_warmup_games", 0) if staged else 0
    if staged and stage_games < warmup_games:
        return ExplorationState(
            phase="warmup",
            progress=0.0,
            epsilon=float(settings["warmup_epsilon"]),
            entropy_factor=float(settings["warmup_entropy_fraction"]),
            at_floor=False,
        )

    elapsed = max(0, int(stage_games) - warmup_games)
    decay_games = int(settings["formal_decay_games"])
    progress = min(1.0, elapsed / decay_games) if decay_games else 1.0
    epsilon_start = float(settings["formal_epsilon_start"])
    epsilon_floor = float(settings["formal_epsilon_floor"])
    entropy_start = float(settings["formal_entropy_start_fraction"])
    entropy_floor = float(settings["formal_entropy_floor_fraction"])
    epsilon = epsilon_start * (epsilon_floor / epsilon_start) ** progress
    entropy = entropy_start * (entropy_floor / entropy_start) ** progress
    return ExplorationState(
        phase="formal",
        progress=progress,
        epsilon=epsilon,
        entropy_factor=entropy,
        at_floor=progress >= 1.0,
    )


def exploration_rate(cfg, stage_games, *, staged=True):
    return exploration_state(cfg, stage_games, staged=staged).epsilon


def entropy_factor(cfg, stage_games, *, staged=True):
    return exploration_state(cfg, stage_games, staged=staged).entropy_factor


def exploration_loss(policy, entropy, log_prob):
    """Return a loss and detached metrics for CUDA PPO."""
    settings = getattr(policy, "exploration_settings", {})
    joint_values = -log_prob if entropy is None else entropy
    joint = joint_values.mean()
    distribution = policy.action_dist
    type_entropy = distribution.types.entropy()
    counts = distribution.legal_tile_counts
    available = counts > 0
    normalized = distribution.locations.entropy() / counts.clamp_min(2).float().log()
    tiles = (normalized * available).sum(-1) / available.sum(-1).clamp_min(1)
    plant_count = distribution.available_plants.sum(-1)
    plants = distribution.plants.entropy() / plant_count.clamp_min(2).to(joint.dtype).log()
    plants = plants * (plant_count > 0)
    parts = {
        name + "_exploration_bonus": coefficient * values.mean()
        for name, coefficient, values in (
            ("kind", settings["type_coef"], type_entropy),
            ("plant", settings["plant_coef"], plants),
            ("tile", settings["tile_coef"], tiles),
        )
    }
    bonus = sum(parts.values())
    return -bonus, {
        "joint_entropy": joint.detach(),
        "exploration_bonus": bonus.detach(),
        **{name: value.detach() for name, value in parts.items()},
    }


def configure_exploration(model, cfg):
    settings = cfg["training"]["exploration"]
    model.exploration_settings = dict(settings)
    warmup_epsilon = settings.get("warmup_epsilon", settings.get("epsilon", 0.0))
    set_exploration_rate(
        model,
        getattr(model, "exploration_rate", warmup_epsilon),
    )
    set_entropy_factor(model, cfg, getattr(model, "entropy_factor", 1.0))


def set_entropy_factor(model, cfg, factor):
    model.entropy_factor = factor
    model.policy.exploration_settings = dict(cfg["training"]["exploration"])
    for key in ("type_coef", "plant_coef", "tile_coef"):
        model.policy.exploration_settings[key] *= factor


def set_exploration_rate(model, rate):
    """Keep collection, PPO ratios and checkpoint interfaces aligned."""
    model.exploration_rate = float(rate)
    model.policy.action_dist.epsilon = float(rate)
    model.policy.exploration_epsilon = float(rate)
    model.policy_kwargs["exploration_epsilon"] = float(rate)


def apply_exploration_state(model, cfg, state):
    """Apply one resolved state immediately before a new rollout window."""
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
