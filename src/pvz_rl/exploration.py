"""Exploration regularization, separate from the policy's true joint entropy."""

import torch


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
    model.policy.exploration_settings = dict(settings)
    # Checkpoints retain the rate that actually collected their last rollout.
    set_exploration_rate(model, getattr(model, "exploration_rate", settings.get("epsilon", 0.0)))
    set_entropy_factor(model, cfg, getattr(model, "entropy_factor", 1.0))


def set_entropy_factor(model, cfg, factor):
    model.entropy_factor = factor
    model.policy.exploration_settings = dict(cfg["training"]["exploration"])
    for key in ("type_coef", "plant_coef", "tile_coef"):
        model.policy.exploration_settings[key] *= factor


def decay_progress(cfg, stage_games, staged):
    target = cfg["training"]["exploration"].get("epsilon_target_games", 0)
    if not target:
        return 0.0
    warmup = cfg["training"].get("critic_warmup_games", 0) if staged else 0
    return max(0, stage_games - warmup) / (target - warmup)


def entropy_factor(cfg, stage_games, *, staged=True):
    # Historical inference metadata has no entropy schedule; it remains readable.
    target = cfg["training"]["exploration"].get("entropy_target_fraction", 1.0)
    return target ** decay_progress(cfg, stage_games, staged)


def set_exploration_rate(model, rate):
    """Keep collection, PPO ratios and both checkpoint-loading interfaces aligned."""
    model.exploration_rate = rate
    model.policy.action_dist.epsilon = rate
    model.policy.exploration_epsilon = rate
    model.policy_kwargs["exploration_epsilon"] = rate


def exploration_rate(cfg, stage_games, *, staged=True):
    """Hold during critic adaptation, then exponentially approach zero by stage games."""
    settings = cfg["training"]["exploration"]
    start = settings.get("epsilon", 0.0)
    target_games = settings.get("epsilon_target_games", 0)
    if not start or not target_games:
        return start
    fraction = decay_progress(cfg, stage_games, staged)
    return start * (settings["epsilon_target"] / start) ** fraction


def pooled_spatial(features, channels, scalar_channels, rows=5, cols=9):
    spatial = features[:, scalar_channels:].reshape(-1, channels, rows, cols)
    pooled = torch.cat(
        (spatial.mean((2, 3)), spatial.amax((2, 3)), features[:, :scalar_channels]), dim=1
    )
    return spatial, pooled
