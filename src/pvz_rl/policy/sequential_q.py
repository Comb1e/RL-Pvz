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


def selection_masks(masks):
    """Return the ten-way comparison mask used by the controller.

    Every active environment compares wait, all eight plants, and dig.  Tile
    masks still constrain the second-level selector; a full board therefore
    produces a rejected plant proposal instead of silently hiding its branch.
    """
    if masks.ndim != 2 or masks.shape[-1] != A.size or masks.dtype != torch.bool:
        raise ValueError("Action masks must be boolean [batch, action] tensors")
    active = masks.any(-1)
    # The transport mask describes candidate tiles, not first-level branch
    # legality.  An active row therefore compares wait, every one of the
    # eight plant species, and dig even when a card cannot currently execute.
    # Inactive rows stay illegal here; select_q_actions overlays wait only
    # when its caller explicitly marks a row inactive for batched collection.
    legal = torch.ones(
        (len(masks), A.tile_groups + 1), dtype=torch.bool, device=masks.device
    )
    return legal & active[:, None]


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


def select_q_actions(
    values,
    board,
    pooled,
    tile_values,
    action_masks,
    *,
    deterministic=False,
    exploration_epsilon=0.0,
    tile_exploration_epsilon=0.0,
    active=None,
    diagnostic_indices=None,
):
    """Shared ten-way Q comparison and tile-only exploration.

    The first-level branch is always the highest-Q proposal.  Exploration is
    applied only after that branch has been selected, when choosing its tile.
    Keeping the branch and tile decisions separate is important because the
    simulator must receive the highest-Q proposal even when it later rejects
    it for affordability or cooldown.
    """
    # Every active environment compares all ten first-level outputs.  The
    # transport mask is used only for the selected tile; affordability and
    # cooldown are simulator outcomes.  Empty rows stay illegal and are
    # rejected by the normal value validator unless the caller marks them
    # inactive for a batched collection step.
    legal = selection_masks(action_masks)
    if active is not None:
        legal = legal.clone()
        legal[~active] = False
        legal[~active, 0] = True
    greedy = greedy_choice(values, legal)
    branches = greedy.clone()
    coins = torch.zeros(len(values), 2, device=values.device, dtype=torch.bool)
    tile_epsilon = 0.0 if deterministic else tile_exploration_epsilon
    tiles = torch.zeros_like(branches)
    greedy_tiles = torch.zeros_like(branches)
    selected_tile_values = values.new_zeros(len(values))
    nonwait = (branches > 0).nonzero(as_tuple=True)[0]
    if len(nonwait):
        tile_q = tile_values(board[nonwait], pooled[nonwait], branches[nonwait])
        tile_legal = A.tile_masks(action_masks[nonwait])[
            torch.arange(len(nonwait), device=values.device), branches[nonwait] - 1
        ]
        # Full-board proposals deterministically target tile zero.  The
        # simulator rejects them and advances time.  Validate every value,
        # even when the geometry has no empty tile.
        has_tile = tile_legal.any(-1)
        tile_candidates = tile_legal.clone()
        tile_candidates[~has_tile, 0] = True
        preferred = greedy_choice(tile_q, tile_candidates)
        tiles[nonwait] = preferred
        greedy_tiles[nonwait] = preferred
        selected, fired = explore(preferred, tile_candidates, tile_epsilon)
        tiles[nonwait] = selected
        coins[nonwait, 1] = fired
        selected_tile_values[nonwait] = tile_q.gather(1, tiles[nonwait, None]).flatten()
    actions = assemble(branches, tiles)
    details = dict(greedy_actions=assemble(greedy, greedy_tiles), coins=coins)
    details["branch_q"] = values
    if diagnostic_indices is not None:
        ix = diagnostic_indices
        details["viewer"] = dict(
            q=values[ix],
            legal=legal[ix],
            actions=actions[ix],
            greedy_actions=details["greedy_actions"][ix],
            coins=coins[ix],
        )
    return actions, values.gather(1, branches[:, None]).flatten(), selected_tile_values, details
