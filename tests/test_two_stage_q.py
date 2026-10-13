"""Independent two-stage attribution, consistency, strata and ranking controls."""

import numpy as np
import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.objective import (
    demonstration_rank_loss,
    probe_loss,
    ranking_counts,
    species_consistency_loss,
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
    target = policy.tiles[:, 0].max(-1).values
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


def test_full_board_candidate_fallback_and_unrestricted_dig_consistency():
    masks = torch.ones(2, A.size, dtype=torch.bool)
    masks[:, 1 : A.dig_start] = False
    tiles = torch.zeros(2, A.tiles)
    tiles[0, :2] = torch.tensor([7.0, 100.0])
    tiles[1, 44] = 6
    branches = torch.tensor([2, 9])
    torch.testing.assert_close(candidate_maxima(tiles, masks, branches), torch.tensor([7.0, 6.0]))
    species = torch.zeros(2, 10)
    species[0, 2] = 100
    species[1, 9] = 100
    assert select_q_actions(species, tiles, masks, deterministic=True)[0].tolist() == [46, 405]


@pytest.mark.parametrize("partition", [[slice(None)], [slice(0, 1), slice(1, 4)]])
def test_probe_strata_and_gradients_are_independent_of_flat_batches(partition):
    counts = np.zeros((10, 2), np.int64)
    counts[2] = [3, 1]
    probes = torch.tensor(
        [[[1, 46, target, accepted]] for target, accepted in ((1, 1), (3, 0), (5, 0), (7, 0))],
        dtype=torch.float32,
    )
    policy = TablePolicy(4)
    predictions = policy.tiles[:, 1, 0]
    loss = sum(
        probe_loss(policy, None, probes[chunk], counts, 1, predictions=predictions[chunk])
        for chunk in partition
    )
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    torch.testing.assert_close(
        policy.tiles.grad[:, 1, 0], torch.tensor([-0.5, -1 / 6, -1 / 6, -1 / 6])
    )
    assert policy.species.grad is None and policy.tiles.grad.count_nonzero() == 4


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
    assert loss.item() == pytest.approx(9.875)
    loss.backward()
    torch.testing.assert_close(
        species.grad[[0, 1, 2, 3], [1, 1, 2, 9]], torch.tensor([-0.5, -0.5, -0.75, -4.0])
    )
    assert targets.grad is None


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
    assert cfg["training"]["n_epochs"] == 4 and cfg["training"]["demo"]["passes"] == 20
    assert cfg["training"]["objective"]["tile_probes"] == 4
    assert cfg["training"]["objective"]["probe_rollout_seconds"] == 30


@pytest.mark.parametrize("tiles", [1, 4, 44])
def test_configurable_probe_layout_is_shared_with_storage(tiles):
    from pvz_rl.config import validate_config
    from pvz_rl.learning.cuda_buffer import trajectory_dtype
    from pvz_rl.learning.probe_layout import ProbeLayout

    cfg = load_config()
    cfg["training"]["objective"].update(tile_probes=tiles)
    validate_config(cfg)
    layout = ProbeLayout.from_config(cfg)
    assert trajectory_dtype(cfg)["probes"].shape == (tiles,)
    assert layout.metadata_width == 7 + tiles * 4


@pytest.mark.parametrize(
    "changes",
    [
        {"branch_probes": 10},
        {"tile_probes": 45},
        {"branch_probes": -1},
        {"tile_probes": True},
        {"branch_probes": 0, "tile_probes": 0},
        {"probe_trigger": "periodic"},
        {"probe_interval_decisions": 512},
    ],
)
def test_retired_and_invalid_probe_settings_are_rejected(changes):
    from pvz_rl.config import validate_config

    cfg = load_config()
    cfg["training"]["objective"].update(changes)
    with pytest.raises(ValueError):
        validate_config(cfg)
