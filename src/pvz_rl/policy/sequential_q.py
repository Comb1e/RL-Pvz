"""Shared legality, exploration and regression algebra for sequential Q control."""

import torch
from pvz_game import Rules

from pvz_rl.envs.actions import ActionSchema as A

UNITS_PER_TILE = Rules().game["units_per_tile"]

POLICY_SIGNATURE = "transformer_lstm_q_v2"
ACTION_DISTRIBUTION = "sequential_q_unmasked_penalty_v1"


def observation_tile_masks(obs):
    """Default proposal geometry from public plant fields, without private state."""
    from pvz_rl.envs.encoding import collate_observations

    obs = collate_observations(obs)
    entities = obs.entities
    plants = obs.entity_mask & (entities[..., 0] >= 1) & (entities[..., 0] <= A.plant_types)
    tiles = entities[..., 2].long() * A.cols + entities[..., 3].long() // UNITS_PER_TILE
    occupied = torch.zeros(len(obs), A.tiles, dtype=torch.long, device=obs.device)
    occupied.scatter_add_(1, tiles.clamp(0, A.tiles - 1), plants.long())
    empty = occupied == 0
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
    legal = torch.ones((len(masks), A.tile_groups + 1), dtype=torch.bool, device=masks.device)
    return legal & active[:, None]


def greedy_choice(values, masks):
    return values.masked_fill(~masks, -torch.inf).argmax(-1)


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
):
    """Shared ten-way Q comparison and tile-only exploration.

    The first-level branch is always the highest-Q proposal.  Exploration is
    applied only after that branch has been selected, when choosing its tile.
    Keeping the branch and tile decisions separate is important because the
    simulator must receive the highest-Q proposal even when it later rejects
    it for affordability or cooldown.

    Selection never waits for the device: tile values are evaluated for every
    row and waits discard theirs. ``details["valid"]`` is a device flag (finite
    values and a legal choice in every row) that callers check at their next
    host boundary.
    """
    # Every active environment compares all ten first-level outputs.  The
    # transport mask is used only for the selected tile; affordability and
    # cooldown are simulator outcomes.  Empty rows stay illegal and invalidate
    # the decision unless the caller marks them inactive for batched collection.
    legal = selection_masks(action_masks)
    if values.shape != legal.shape:
        raise ValueError("Q values and boolean legal masks must have matching shapes")
    if active is not None:
        wait_only = torch.zeros_like(legal)
        wait_only[:, 0] = True
        legal = torch.where(active[:, None], legal, wait_only)
    branches = greedy_choice(values, legal)
    nonwait = branches > 0
    rows = torch.arange(len(values), device=values.device)
    tile_q = tile_values(board, pooled, branches)
    tile_legal = A.tile_masks(action_masks)[rows, (branches - 1).clamp_min(0)]
    # Full-board proposals deterministically target tile zero.  The simulator
    # rejects them and advances time.
    first_tile = torch.zeros_like(tile_legal)
    first_tile[:, 0] = True
    tile_candidates = torch.where(tile_legal.any(-1, keepdim=True), tile_legal, first_tile)
    preferred = greedy_choice(tile_q, tile_candidates)
    tile_epsilon = 0.0 if deterministic else tile_exploration_epsilon
    selected, fired = explore(preferred, tile_candidates, tile_epsilon)
    tiles = torch.where(nonwait, selected, 0)
    coins = torch.stack((torch.zeros_like(nonwait), fired & nonwait), -1)
    actions = assemble(branches, tiles)
    details = dict(
        greedy_actions=assemble(branches, torch.where(nonwait, preferred, 0)),
        coins=coins,
        branch_q=values,
        valid=torch.isfinite(values).all()
        & legal.any(-1).all()
        & (torch.isfinite(tile_q) | ~nonwait[:, None]).all(),
    )
    selected_tile_values = torch.where(nonwait, tile_q.gather(1, tiles[:, None]).flatten(), 0)
    return actions, values.gather(1, branches[:, None]).flatten(), selected_tile_values, details
