"""Shared legality, exploration and regression algebra for sequential Q control."""

import math

import torch

from pvz_rl.envs.actions import ActionSchema as A

POLICY_SIGNATURE = "event_sequential_q_v2"
ACTION_DISTRIBUTION = "sequential_q_unmasked_penalty_v1"


def observation_tile_masks(obs):
    """Default proposal geometry from public plant fields, without private state."""
    empty = obs[:, : A.tiles * 3 : 3] == 0
    available = torch.ones(len(obs), 1, dtype=torch.bool, device=obs.device)
    return torch.cat((available, empty.repeat(1, A.plant_types), available.expand(-1, A.tiles)), -1)


def branch_masks(masks):
    """Return the ten-way comparison mask used by the controller."""
    return selection_masks(masks)


def transport_branch_masks(masks):
    """Describe which transport branches have at least one candidate tile."""
    return torch.cat((masks[:, :1], A.tile_masks(masks).any(-1)), -1)


def selection_masks(masks):
    """Return the ten-way comparison mask used by the controller.

    Every active environment compares wait, all eight plants, and dig.  Tile
    masks still constrain the second-level selector; a full board therefore
    produces a rejected plant proposal instead of silently hiding its branch.
    """
    if masks.ndim != 2 or masks.shape[-1] != A.size or masks.dtype != torch.bool:
        raise ValueError("Action masks must be boolean [batch, action] tensors")
    active = masks.any(-1)
    return active[:, None].expand(-1, A.tile_groups + 1)


def validate_values(values, masks):
    if values.shape != masks.shape or masks.dtype != torch.bool:
        raise ValueError("Q values and boolean legal masks must have matching shapes")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Q values must be finite")
    if not bool(masks.any(-1).all()):
        raise ValueError("Each decision requires at least one legal action")


def greedy_choice(values, masks, *, validate=True):
    if validate:
        validate_values(values, masks)
    return values.masked_fill(~masks, -torch.inf).argmax(-1)


def per_head_epsilon(budget):
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Exploration budget must be finite and in [0, 1]")
    return 1 - math.sqrt(1 - budget)


def explore(greedy, legal, epsilon):
    """Return choices and coin firings; firing need not change the choice."""
    if epsilon == 0:
        return greedy, torch.zeros_like(greedy, dtype=torch.bool)
    coins = torch.rand(greedy.shape, device=greedy.device) < epsilon
    uniform = torch.multinomial(legal.float(), 1).flatten()
    return torch.where(coins, uniform, greedy), coins


def action_parts(actions):
    branches = torch.where(actions == 0, 0, 1 + (actions.long() - 1) // A.tiles)
    tiles = torch.where(actions == 0, 0, (actions.long() - 1) % A.tiles)
    return branches, tiles


def assemble(branches, tiles):
    return torch.where(branches == 0, 0, 1 + (branches - 1) * A.tiles + tiles).long()


def action_groups(actions):
    return torch.where(actions == 0, 0, torch.where(actions < A.dig_start, 1, 2))


def balanced_q_loss(branch, tile, targets, actions, counts, batch_size):
    """Whole-cohort group weights, retaining scale in a short final minibatch."""
    branch_error = (branch - targets).square()
    tile_error = (tile - targets).square()
    errors = torch.where(actions == 0, branch_error, (branch_error + tile_error) / 2)
    counts = torch.as_tensor(counts, device=branch.device, dtype=branch.dtype)
    groups = action_groups(actions)
    weights = counts.sum() / (counts.clamp_min(1)[groups] * (counts > 0).sum())
    loss = (errors * weights).sum() / counts.sum().clamp(max=batch_size)
    return loss, branch_error, tile_error
