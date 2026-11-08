"""Shared legality, exploration and regression algebra for sequential Q control."""

import torch

from pvz_rl.envs.actions import ActionSchema as A

POLICY_SIGNATURE = "shared_public_outcomes_q_v11"
ACTION_DISTRIBUTION = "sequential_q_unmasked_tiles_v2"


def observation_tile_masks(obs):
    """Return open transport slots. Selection legality is a separate hard mask."""
    from pvz_rl.envs.encoding import collate_observations

    obs = collate_observations(obs)
    active = torch.ones(len(obs), 1, dtype=torch.bool, device=obs.device)
    return torch.cat((active, active.expand(-1, A.plant_types * A.tiles + A.tiles)), -1)


def occupied_cells(entities, entity_mask, *, rows, cols, units_per_tile):
    """Mark cells that contain a plant. Zombies and other records do not occupy a cell."""
    kind, row, col = entities[..., 0], entities[..., 2], entities[..., 3] // units_per_tile
    plant = entity_mask & (kind >= 1) & (kind <= A.plant_types)
    inside = plant & (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    index = (row * cols + col).long()
    occupied = torch.zeros(entities.shape[0], rows * cols, dtype=torch.bool, device=entities.device)
    if bool(inside.any()):
        batch = torch.arange(entities.shape[0], device=entities.device)[:, None].expand_as(inside)
        occupied[batch[inside], index[inside]] = True
    return occupied


def card_ready(globals_, *, sun_index, cooldown_index, cost_scale, costs):
    """True when public sun meets the pinned cost and that card's cooldown is zero."""
    sun = globals_[:, sun_index] * cost_scale
    cooling = globals_[:, list(cooldown_index)] > 0
    affordable = sun[:, None] >= costs.to(device=globals_.device, dtype=sun.dtype) - 0.5
    return affordable & ~cooling


def action_branch_mask(occupied, cards, dig_permitted):
    """Wait stays. A plant card needs an empty cell. Dig needs a planted cell."""
    legal = torch.ones(
        occupied.shape[0], A.tile_groups + 1, dtype=torch.bool, device=occupied.device
    )
    empty = (~occupied).any(-1, keepdim=True)
    legal[:, 1:-1] = cards & empty
    legal[:, -1] = occupied.any(-1) & dig_permitted
    return legal


def tile_candidates(occupied, branches, *, active=None):
    """Plant cells are empty. Dig cells contain a plant. Inactive rows keep tile 0."""
    mask = torch.ones_like(occupied)
    plant = (branches >= 1) & (branches <= A.plant_types)
    dig = branches == A.plant_types + 1
    mask = torch.where(plant[:, None], ~occupied, mask)
    mask = torch.where(dig[:, None], occupied, mask)
    if active is not None:
        first = torch.zeros_like(mask)
        first[:, 0] = True
        mask = torch.where(active[:, None], mask, first)
    return mask


def greedy_choice(values, masks):
    return values.masked_fill(~masks, -torch.inf).argmax(-1)


def explore(greedy, legal, epsilon):
    """Return choices and coin firings; firing need not change the choice."""
    if epsilon == 0:
        return greedy, torch.zeros_like(greedy, dtype=torch.bool)
    coins = torch.rand(greedy.shape, device=greedy.device) < epsilon
    covered = legal.any(-1)
    weights = legal.float()
    if not bool(covered.all()):
        weights = weights.clone()
        weights[~covered, 0] = 1
    uniform = torch.multinomial(weights, 1).flatten()
    take = coins & covered
    return torch.where(take, uniform, greedy), take


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


def balanced_q_loss(
    selected, targets, actions, counts, *, accepted, accepted_share=0.5, weights=None
):
    """Equal nonempty wait/plant/dig groups, then equal accepted/rejected outcomes."""
    errors = (selected - targets).square()
    weights = (
        outcome_weights(
            counts, device=selected.device, dtype=selected.dtype, accepted_share=accepted_share
        )
        if weights is None
        else weights
    )
    loss = (errors * weights[action_groups(actions), accepted.long()]).sum()
    return loss, errors


def candidate_maxima(tile_q, occupied, branches, *, active=None):
    return (
        tile_q.masked_fill(~tile_candidates(occupied, branches, active=active), -torch.inf)
        .max(-1)
        .values
    )


def select_q_tiles(tile_q, occupied, branches, epsilon, *, active=None):
    candidates = tile_candidates(occupied, branches, active=active)
    preferred = greedy_choice(tile_q, candidates)
    selected, fired = explore(preferred, candidates, epsilon)
    return selected, preferred, fired, tile_q


def select_q_actions(
    species_q,
    tile_q,
    action_masks,
    occupied,
    branch_mask,
    *,
    deterministic=False,
    tile_exploration_epsilon=0.0,
    active=None,
    supplied_branches=None,
):
    """Pick a legal branch, then a legal cell. Scores passed in are not rewritten.

    ``-inf`` is applied to a copy. An excluded card or cell is never turned into
    Wait. An inactive collection row still has only Wait.
    """
    if (
        action_masks.shape[0] != len(species_q)
        or species_q.shape != branch_mask.shape
        or tile_q.shape != (len(species_q), A.tiles)
        or occupied.shape != tile_q.shape
    ):
        raise ValueError("Q values and boolean legal masks must have matching shapes")
    legal = branch_mask
    if active is not None:
        wait_only = torch.zeros_like(legal)
        wait_only[:, 0] = True
        legal = torch.where(active[:, None], legal, wait_only)
    if supplied_branches is None:
        branches = greedy_choice(species_q, legal)
        branch_valid = True
    else:
        branches = supplied_branches
        if branches.shape != (len(species_q),) or branches.dtype != torch.long:
            raise ValueError("Supplied branches must be an int64 vector matching the board batch")
        branch_valid = bool(((branches >= 0) & (branches < legal.shape[1])).all())
        branches = branches.clamp(0, legal.shape[1] - 1)
    if active is not None:
        branches = torch.where(active, branches, 0)
    nonwait = branches > 0
    tile_epsilon = 0.0 if deterministic else tile_exploration_epsilon
    selected, preferred, fired, tile_q = select_q_tiles(
        tile_q, occupied, branches, tile_epsilon, active=active
    )
    candidates = tile_candidates(occupied, branches, active=active)
    tile_ok = candidates.gather(1, selected[:, None]).flatten() | ~nonwait
    branch_ok = legal.gather(1, branches[:, None]).flatten()
    tiles = torch.where(nonwait, selected, 0)
    coins = torch.stack((torch.zeros_like(nonwait), fired & nonwait), -1)
    actions = assemble(branches, tiles)
    branch_value = species_q.gather(1, branches[:, None]).flatten()
    tile_value = torch.where(nonwait, tile_q.gather(1, tiles[:, None]).flatten(), 0)
    selected_value = torch.where(nonwait, tile_value, species_q[:, 0])
    details = dict(
        greedy_actions=assemble(branches, torch.where(nonwait, preferred, 0)),
        coins=coins,
        branch_q=species_q,
        tile_q=tile_q,
        branch_value=branch_value,
        tile_value=tile_value,
        selected_value=selected_value,
        placement_gap=torch.where(
            nonwait, candidate_maxima(tile_q, occupied, branches, active=active) - tile_value, 0
        ),
        valid=branch_valid
        & branch_ok.all()
        & tile_ok.all()
        & torch.isfinite(species_q).all()
        & legal.any(-1).all()
        & (torch.isfinite(tile_q) | ~nonwait[:, None]).all(),
    )
    return actions, branch_value, tile_value, details
