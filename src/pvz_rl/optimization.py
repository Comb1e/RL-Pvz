"""PPO actor reductions with an explicit treatment of forced legal actions."""

import torch


def actor_weights(action_masks, advantages, objective="all_steps"):
    if objective == "all_steps":
        return torch.ones_like(advantages)
    if objective != "choice_points_v1" or action_masks is None:
        raise ValueError("choice_points_v1 requires a masked policy")
    return (action_masks.sum(-1) > 1).to(advantages.dtype)


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1)


def policy_advantages(advantages, weights, normalize):
    if not normalize:
        return advantages
    count = weights.sum()
    mean = weighted_mean(advantages, weights)
    variance = ((advantages - mean).square() * weights).sum() / (count - 1).clamp_min(1)
    normalized = (advantages - mean) / (variance.sqrt() + 1e-8)
    # One choice supplies no variance estimate; retain its original advantage.
    return torch.where(count > 1, normalized, advantages)
