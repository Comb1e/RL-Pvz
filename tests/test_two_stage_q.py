"""Independent two-stage attribution, consistency, strata and ranking controls."""

import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.objective import (
    consistency_diagnostics,
    demonstration_rank_loss,
    ranking_counts,
    species_consistency_loss,
    species_consistency_targets,
)
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.policy.sequential_q import (
    action_parts,
    balanced_q_loss,
    candidate_maxima,
    select_q_actions,
)
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy


class TablePolicy(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.cfg = load_config()
        self.species = torch.nn.Parameter(torch.zeros(count, 10))
        self.tiles = torch.nn.Parameter(torch.zeros(count, 9, 45))

    def species_values(self, observations):
        return self.species

    def conditional_tiles(self, observations, branches, *, rows=None):
        rows = torch.arange(len(branches)) if rows is None else rows
        return self.tiles[rows, branches - 1]


@pytest.mark.parametrize(
    "species,expected", [([0, 8, 3], 1), ([0, 2, 9], 2), ([0, -9, -2], 0), ([0, 5, 5], 1)]
)
def test_species_selection_is_independent_of_sampled_tile_values(species, expected):
    species_q = torch.full((1, 10), -20.0)
    species_q[0, :3] = torch.tensor(species)
    tiles = torch.full((1, 45), -20.0)
    tiles[0, :2] = torch.tensor([8.0, -4.0])
    masks = torch.ones(1, A.size, dtype=torch.bool)
    actions, _, _, details = select_q_actions(species_q, tiles, masks, deterministic=True)
    assert action_parts(actions)[0].item() == expected
    assert details["branch_q"][0, expected] == species[expected]
    if expected:
        assert details["selected_value"].item() == 8
        assert details["placement_gap"].item() == 0


def test_bad_tile_return_does_not_blame_species_and_consistency_stops_target_gradient():
    policy = TablePolicy(1)
    with torch.no_grad():
        policy.species[0, 1] = 7
        policy.tiles[0, 0, :2] = torch.tensor([8.0, -4.0])
    action = torch.tensor([2])
    selected = policy.tiles[:, 0, 1]
    loss, errors = balanced_q_loss(
        selected,
        torch.tensor([-3.0]),
        action,
        [[0, 0], [0, 1], [0, 0]],
        accepted=torch.tensor([True]),
    )
    loss.backward()
    assert errors.item() == 1
    assert policy.species.grad is None
    assert policy.tiles.grad[0, 0, 0] == 0 and policy.tiles.grad[0, 0, 1] == -2
    policy.zero_grad(set_to_none=True)
    target, _, _ = species_consistency_targets(
        policy.tiles[:, 0],
        policy.tiles[:, 0],
        torch.ones(1, A.size, dtype=torch.bool),
        torch.tensor([1]),
    )
    consistency = species_consistency_loss(
        policy.species,
        target,
        action,
        torch.tensor([True]),
        [[0, 1], [0, 0]],
    )
    consistency.backward()
    assert policy.species.grad[0, 1] == -2
    assert policy.species.grad.count_nonzero() == 1
    assert policy.tiles.grad is None


def test_all_tile_candidates_and_unrestricted_dig_consistency():
    masks = torch.ones(2, A.size, dtype=torch.bool)
    masks[:, 1 : A.dig_start] = False
    tiles = torch.zeros(2, A.tiles)
    tiles[0, :2] = torch.tensor([7.0, 100.0])
    tiles[1, 44] = 6
    branches = torch.tensor([2, 9])
    torch.testing.assert_close(candidate_maxima(tiles, masks, branches), torch.tensor([100.0, 6.0]))
    species = torch.zeros(2, 10)
    species[0, 2] = 100
    species[1, 9] = 100
    assert select_q_actions(species, tiles, masks, deterministic=True)[0].tolist() == [47, 405]


@pytest.mark.parametrize(
    "online,ema,occupied,branch,selected,target,gap",
    [
        ([3, 8, 1], [9, -4, 2], [], 1, 1, -4, 13),
        ([1, 5, 2], [1, 5, 2], [], 2, 1, 5, 0),
        ([4, 4, 1], [0, 9, 1], [], 3, 0, 0, 9),
        ([100, 2, 7], [999, 12, -1], [0], 4, 0, 999, 0),
        ([1, 8, 1], [-2, 9, 1], list(range(45)), 5, 1, 9, 0),
        ([1, 2, 8], [9, 1, -3], list(range(45)), 9, 2, -3, 12),
    ],
    ids=["disagreement", "identical", "tie", "occupied", "full-board", "dig"],
)
def test_double_estimator_targets_use_online_geometry_selection_and_ema_evaluation(
    online, ema, occupied, branch, selected, target, gap
):
    online_tiles = torch.full((1, A.tiles), -100.0)
    ema_tiles = torch.full((1, A.tiles), -100.0)
    online_tiles[0, :3] = torch.tensor(online)
    ema_tiles[0, :3] = torch.tensor(ema)
    online_tiles.requires_grad_()
    ema_tiles.requires_grad_()
    masks = torch.ones(1, A.size, dtype=torch.bool)
    for tile in occupied:
        masks[0, 1 + torch.arange(A.plant_types) * A.tiles + tile] = False
    targets, selections, gaps = species_consistency_targets(
        online_tiles, ema_tiles, masks, torch.tensor([branch])
    )
    torch.testing.assert_close(targets, torch.tensor([float(target)]))
    assert selections.tolist() == [selected] and selections.dtype == torch.long
    torch.testing.assert_close(gaps, torch.tensor([float(gap)]))
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in (targets, gaps))
    species = torch.zeros(1, 10, requires_grad=True)
    actions = torch.tensor([1 + (branch - 1) * A.tiles + selected])
    counts = [[0, int(branch != 9)], [0, int(branch == 9)]]
    species_consistency_loss(species, targets, actions, torch.tensor([True]), counts).backward()
    assert online_tiles.grad is None and ema_tiles.grad is None
    assert species.grad[0, branch] == -2 * max(-1, min(1, target))
    assert species.grad.count_nonzero() == int(target != 0)


def test_double_estimator_target_helper_accepts_empty_nonwait_batch():
    tiles = torch.empty(0, A.tiles, requires_grad=True)
    targets, selected, gaps = species_consistency_targets(
        tiles, tiles, torch.empty(0, A.size, dtype=torch.bool), torch.empty(0, dtype=torch.long)
    )
    assert targets.shape == selected.shape == gaps.shape == (0,)
    assert selected.dtype == torch.long
    assert not targets.requires_grad and not gaps.requires_grad


@pytest.mark.parametrize("delta", [0.25, 1.0, 2.5])
def test_mse_matched_huber_values_and_derivatives_follow_independent_piecewise_control(delta):
    residuals = [0.0]
    residuals += [sign * ratio * delta for ratio in (0.25, 0.99, 1, 1.01, 20) for sign in (-1, 1)]
    residuals += [-1e8, 1e8]
    species = torch.zeros(len(residuals), 10, dtype=torch.float64)
    species[:, 1] = torch.tensor(residuals, dtype=torch.float64)
    species.requires_grad_()
    targets = torch.zeros(len(residuals), dtype=torch.float64, requires_grad=True)
    actions = torch.ones(len(residuals), dtype=torch.long)
    accepted = torch.ones(len(residuals), dtype=torch.bool)
    counts = [[0, len(residuals)], [0, 0]]
    expected_values = [
        residual**2 if abs(residual) <= delta else 2 * delta * abs(residual) - delta**2
        for residual in residuals
    ]
    expected_gradients = [
        2 * max(-delta, min(delta, residual)) / len(residuals) for residual in residuals
    ]
    loss = species_consistency_loss(species, targets, actions, accepted, counts, delta=delta)
    assert loss.item() == pytest.approx(sum(expected_values) / len(residuals), rel=1e-12)
    loss.backward()
    torch.testing.assert_close(
        species.grad[:, 1],
        torch.tensor(expected_gradients, dtype=torch.float64),
        atol=0,
        rtol=1e-12,
    )
    assert not species.grad[:, [0, *range(2, 10)]].any()
    assert targets.grad is None


@pytest.mark.parametrize("partition", [[slice(None)], [slice(0, 1), slice(1, 4)]])
def test_consistency_uses_complete_cohort_group_and_outcome_denominators(partition):
    species = torch.zeros(4, 10, requires_grad=True)
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    actions = torch.tensor([1, 1, 46, 361])
    accepted = torch.tensor([True, False, False, True])
    counts = [[2, 1], [0, 1]]
    loss = sum(
        species_consistency_loss(
            species[part], targets[part], actions[part], accepted[part], counts
        )
        for part in partition
    )
    assert loss.item() == pytest.approx(1 / 4 + 3 / 8 + 5 / 8 + 7 / 2)
    loss.backward()
    torch.testing.assert_close(
        species.grad[[0, 1, 2, 3], [1, 1, 2, 9]], torch.tensor([-0.5, -0.25, -0.25, -1.0])
    )
    assert targets.grad is None


@pytest.mark.parametrize("accepted_share", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("include_dig", [False, True])
def test_consistency_missing_groups_and_outcomes_receive_their_full_weight(
    accepted_share, include_dig
):
    residuals = [0.2, 4.0, -2.0] if include_dig else [0.2, 4.0]
    species = torch.zeros(len(residuals), 10, dtype=torch.float64, requires_grad=True)
    targets = -torch.tensor(residuals, dtype=torch.float64, requires_grad=True)
    targets.retain_grad()
    actions = torch.tensor([1, 46, 361] if include_dig else [1, 46])
    accepted = torch.tensor([True, True, False] if include_dig else [True, True])
    counts = [[0, 2], [int(include_dig), 0]]
    weights = [0.25, 0.25, 0.5] if include_dig else [0.5, 0.5]
    values = [0.04, 7, 3] if include_dig else [0.04, 7]
    loss = species_consistency_loss(
        species, targets, actions, accepted, counts, accepted_share=accepted_share
    )
    assert loss.item() == pytest.approx(
        sum(weight * value for weight, value in zip(weights, values))
    )
    loss.backward()
    branches = [1, 2, 9] if include_dig else [1, 2]
    expected = torch.tensor(
        [2 * max(-1, min(1, residual)) * weight for residual, weight in zip(residuals, weights)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(species.grad[torch.arange(len(residuals)), branches], expected)
    assert targets.grad is None


@pytest.mark.parametrize("totals", [[7.5, 2, 6, 3], [0, 0, 0, 0]])
def test_species_consistency_diagnostics_use_raw_totals_and_none_without_proposals(totals):
    metrics = consistency_diagnostics(3.25, totals)
    expected = {
        "species_consistency_loss": 3.25 if totals[-1] else None,
        "species_consistency_mae": 2.5 if totals[-1] else None,
        "species_consistency_saturation_fraction": 2 / 3 if totals[-1] else None,
        "species_selection_evaluation_gap": 2.0 if totals[-1] else None,
    }
    assert metrics == expected


def test_demo_ranking_separates_species_and_tile_preferences_and_excludes_rejections():
    policy = TablePolicy(2)
    actions = torch.tensor([46, 46])
    accepted = torch.tensor([True, False])
    masks = torch.ones(2, A.size, dtype=torch.bool)
    counts = ranking_counts(actions, accepted, masks)[0]
    loss, _ = demonstration_rank_loss(
        policy,
        None,
        actions,
        accepted,
        masks,
        counts,
        0.05,
        species=policy.species,
        tiles=policy.tiles[:, 1],
    )
    loss.backward()
    assert policy.species.grad[0, 2] < 0
    assert (policy.species.grad[0, [0, 1, 3, 4, 5, 6, 7, 8, 9]] > 0).all()
    assert policy.tiles.grad[0, 1, 0] < 0 and (policy.tiles.grad[0, 1, 1:] > 0).all()
    assert not policy.species.grad[1].any() and not policy.tiles.grad[1].any()


@pytest.mark.parametrize("microbatch", [1, 3, 128])
def test_requested_pairs_equal_direct_selected_values_and_update_only_their_stage(microbatch):
    cfg = load_config()
    cfg["policy"]["encoder_microbatch"] = microbatch
    policy = TwoStageTransformerPolicy(cfg).eval()
    control = TwoStageTransformerPolicy(cfg).eval()
    control.load_state_dict(policy.state_dict())
    observations = collate_observations([observation(cfg, 20)[0]] * 4)
    actions = torch.tensor([0, 1, 90, 405])
    branches, positions = action_parts(actions)
    selected = policy.selected_values(observations, actions)
    species = control.species_values(observations)
    tiles = control.conditional_tiles(observations, branches)
    expected = torch.where(
        branches == 0, species[:, 0], tiles.gather(1, positions[:, None]).flatten()
    )
    torch.testing.assert_close(selected, expected)
    selected.sum().backward()
    expected.sum().backward()
    for actual, reference in zip(policy.parameters(), control.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, atol=3e-5, rtol=1e-4)
    policy.zero_grad(set_to_none=True)
    policy.selected_values(observations, torch.tensor([1, 1, 90, 405])).sum().backward()
    assert all(parameter.grad is None for parameter in policy.species_encoder.parameters())
    assert all(parameter.grad is None for parameter in policy.species_head.parameters())
