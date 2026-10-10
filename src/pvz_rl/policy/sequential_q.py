"""Shared legality, exploration and regression algebra for sequential Q control."""

from dataclasses import dataclass

import torch
from pvz_game import Rules

from pvz_rl.envs.actions import ActionSchema as A

UNITS_PER_TILE = Rules().game["units_per_tile"]

POLICY_SIGNATURE = "transformer_lstm_q_v4"
ACTION_DISTRIBUTION = "joint_q_unmasked_penalty_v1"


@dataclass
class ActionQValues:
    wait: torch.Tensor
    tiles: torch.Tensor
    branches: torch.Tensor

    @classmethod
    def derive(cls, wait, tiles, masks):
        candidates = A.tile_masks(masks).clone()
        candidates[..., 0] |= ~candidates.any(-1)
        maxima = tiles.masked_fill(~candidates, -torch.inf).max(-1).values
        return cls(wait.reshape(-1), tiles, torch.cat((wait.reshape(-1, 1), maxima), -1))

    def selected(self, actions):
        branches, locations = action_parts(actions)
        rows = torch.arange(len(actions), device=actions.device)
        tile = self.tiles[rows, (branches - 1).clamp_min(0), locations]
        return torch.where(branches == 0, self.wait, tile)


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


def outcome_weights(counts, *, device, dtype, accepted_share=0.5):
    """Whole-cohort coefficients for equal groups and conditional outcome strata."""
    counts = torch.as_tensor(counts, device=device, dtype=dtype)
    present = counts > 0
    both = present.all(-1, keepdim=True)
    shares = counts.new_tensor([1 - accepted_share, accepted_share])
    shares = torch.where(both, shares, present.to(dtype))
    represented = present.any(-1).sum().clamp_min(1)
    return shares / counts.clamp_min(1) / represented


def balanced_q_loss(selected, targets, actions, counts, *, accepted, accepted_share=0.5):
    """Equal nonempty wait/plant/dig groups, then equal accepted/rejected outcomes."""
    errors = (selected - targets).square()
    weights = outcome_weights(
        counts, device=selected.device, dtype=selected.dtype, accepted_share=accepted_share
    )
    loss = (errors * weights[action_groups(actions), accepted.long()]).sum()
    return loss, errors


def select_q_tiles(values, action_masks, branches, epsilon):
    rows = torch.arange(len(branches), device=branches.device)
    tile_q = values.tiles[rows, (branches - 1).clamp_min(0)]
    tile_legal = A.tile_masks(action_masks)[rows, (branches - 1).clamp_min(0)]
    first_tile = torch.zeros_like(tile_legal)
    first_tile[:, 0] = True
    candidates = torch.where(tile_legal.any(-1, keepdim=True), tile_legal, first_tile)
    preferred = greedy_choice(tile_q, candidates)
    selected, fired = explore(preferred, candidates, epsilon)
    return selected, preferred, fired, tile_q


def select_q_actions(
    values,
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
    if values.branches.shape != legal.shape:
        raise ValueError("Q values and boolean legal masks must have matching shapes")
    if active is not None:
        wait_only = torch.zeros_like(legal)
        wait_only[:, 0] = True
        legal = torch.where(active[:, None], legal, wait_only)
    branches = greedy_choice(values.branches, legal)
    nonwait = branches > 0
    tile_epsilon = 0.0 if deterministic else tile_exploration_epsilon
    selected, preferred, fired, tile_q = select_q_tiles(
        values, action_masks, branches, tile_epsilon
    )
    tiles = torch.where(nonwait, selected, 0)
    coins = torch.stack((torch.zeros_like(nonwait), fired & nonwait), -1)
    actions = assemble(branches, tiles)
    details = dict(
        greedy_actions=assemble(branches, torch.where(nonwait, preferred, 0)),
        coins=coins,
        branch_q=values.branches,
        action_q=values,
        selected_value=values.selected(actions),
        placement_gap=values.branches.gather(1, branches[:, None]).flatten()
        - values.selected(actions),
        valid=torch.isfinite(values.branches).all()
        & legal.any(-1).all()
        & (torch.isfinite(tile_q) | ~nonwait[:, None]).all(),
    )
    selected_tile_values = torch.where(nonwait, tile_q.gather(1, tiles[:, None]).flatten(), 0)
    return (
        actions,
        values.branches.gather(1, branches[:, None]).flatten(),
        selected_tile_values,
        details,
    )
