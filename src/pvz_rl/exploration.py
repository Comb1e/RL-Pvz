"""Exploration regularization, separate from the policy's true joint entropy."""

import torch

from .optimization import weighted_mean


def exploration_loss(policy, entropy, log_prob, ent_coef, weights=None):
    """Return a loss and detached metrics for CUDA PPO."""
    settings = getattr(policy, "exploration_settings", {})
    joint_values = -log_prob if entropy is None else entropy
    joint = joint_values.mean()
    reduce = torch.mean if weights is None else lambda x: weighted_mean(x, weights)
    if settings.get("objective", "joint") == "joint":
        bonus = ent_coef * reduce(joint_values)
    else:
        distribution = policy.action_dist
        type_entropy = distribution.types.entropy()
        counts = distribution.legal_tile_counts
        available = counts > 0
        normalized = distribution.locations.entropy() / counts.clamp_min(2).float().log()
        tiles = (normalized * available).sum(-1) / available.sum(-1).clamp_min(1)
        bonus = reduce(settings["type_coef"] * type_entropy + settings["tile_coef"] * tiles)
    return -bonus, {"joint_entropy": joint.detach(), "exploration_bonus": bonus.detach()}


def configure_exploration(model, cfg):
    settings = cfg["training"].get("exploration", {"objective": "joint"})
    model.policy.exploration_settings = dict(settings)
    model.actor_objective = cfg["training"].get("actor_objective", "all_steps")


def pooled_spatial(features, channels, rows=5, cols=9):
    spatial = features[:, channels:].reshape(-1, channels, rows, cols)
    pooled = torch.cat((spatial.mean((2, 3)), spatial.amax((2, 3)), features[:, :channels]), dim=1)
    return spatial, pooled
