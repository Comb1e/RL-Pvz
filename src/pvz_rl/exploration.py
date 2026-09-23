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
    bonus = torch.mean(settings["type_coef"] * type_entropy + settings["tile_coef"] * tiles)
    return -bonus, {"joint_entropy": joint.detach(), "exploration_bonus": bonus.detach()}


def configure_exploration(model, cfg):
    settings = cfg["training"]["exploration"]
    model.policy.exploration_settings = dict(settings)
    # Checkpoints retain the rate that actually collected their last rollout.
    set_exploration_rate(model, getattr(model, "exploration_rate", settings.get("epsilon", 0.0)))


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
    warmup = cfg["training"].get("critic_warmup_games", 0) if staged else 0
    fraction = max(0, stage_games - warmup) / (target_games - warmup)
    return start * (settings["epsilon_target"] / start) ** fraction


def pooled_spatial(features, channels, scalar_channels, rows=5, cols=9):
    spatial = features[:, scalar_channels:].reshape(-1, channels, rows, cols)
    pooled = torch.cat(
        (spatial.mean((2, 3)), spatial.amax((2, 3)), features[:, :scalar_channels]), dim=1
    )
    return spatial, pooled
