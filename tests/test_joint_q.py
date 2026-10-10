"""Independent complete-action attribution, maxima, strata and ranking controls."""

import numpy as np
import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.objective import demonstration_rank_loss, probe_loss, ranking_counts
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.policy.sequential_q import (
    ActionQValues,
    action_parts,
    balanced_q_loss,
    select_q_actions,
)
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy


class TablePolicy:
    cfg = load_config()

    def selected_values(self, wait, tiles, context, actions):
        branches, locations = action_parts(actions)
        rows = torch.arange(len(actions))
        return torch.where(
            branches == 0, wait.flatten(), tiles[rows, (branches - 1).clamp_min(0), locations]
        )

    def action_values(self, output, masks):
        return ActionQValues.derive(output.wait_q, output.tile_features, masks)


@pytest.mark.parametrize(
    "tiles, expected",
    [
        ([[8, -4], [3, 2]], 1),
        ([[2, 1], [9, -8]], 2),
        ([[-9, -10], [-2, -3]], 0),
        ([[5, 5], [5, 5]], 1),
    ],
)
def test_species_quality_is_best_placement_not_sampled_placement(tiles, expected):
    wait = torch.zeros(1)
    tables = torch.full((1, 9, 45), -20.0)
    tables[0, :2, :2] = torch.tensor(tiles)
    masks = torch.ones(1, A.size, dtype=torch.bool)
    masks[:, 1 : A.dig_start] = False
    masks[:, [1, 2, 46, 47]] = True
    values = ActionQValues.derive(wait, tables, masks)
    actions, _, _, details = select_q_actions(values, masks, deterministic=True)
    assert action_parts(actions)[0].item() == expected
    if expected:
        bad = torch.tensor([2 if expected == 1 else 47])
        assert values.branches[0, expected] >= values.selected(bad)[0]
        assert details["branch_q"][0, expected] == max(tiles[expected - 1])


def test_selected_regression_does_not_blame_best_tile_and_max_routes_only_to_best():
    wait = torch.tensor([1.0], requires_grad=True)
    tiles = torch.full((1, 9, 45), -10.0)
    tiles[0, 0, :2] = torch.tensor([8.0, -4.0])
    tiles.requires_grad_()
    values = ActionQValues.derive(wait, tiles, torch.ones(1, A.size, dtype=torch.bool))
    loss, error = balanced_q_loss(
        values.selected(torch.tensor([2])),
        torch.tensor([-3.0]),
        torch.tensor([2]),
        [[0, 0], [0, 1], [0, 0]],
        accepted=torch.tensor([True]),
    )
    assert error.item() == 1
    loss.backward(retain_graph=True)
    assert tiles.grad[0, 0, 0] == 0 and tiles.grad[0, 0, 1] == -2
    assert wait.grad.item() == 0
    tiles.grad.zero_()
    values.branches[0, 1].backward()
    assert tiles.grad[0, 0, 0] == 1 and tiles.grad.count_nonzero() == 1


def test_full_board_tile_zero_and_dig_candidates_do_not_hide_species():
    masks = torch.ones(1, A.size, dtype=torch.bool)
    masks[:, 1 : A.dig_start] = False
    tiles = torch.zeros(1, 9, 45)
    tiles[0, 1, 0] = 7
    tiles[0, 1, 1] = 100
    tiles[0, 8, 44] = 6
    values = ActionQValues.derive(torch.zeros(1), tiles, masks)
    assert values.branches[0, 2] == 7 and values.branches[0, 9] == 6
    assert select_q_actions(values, masks, deterministic=True)[0].item() == 46


@pytest.mark.parametrize("partition", [[slice(None)], [slice(0, 1), slice(1, 4)]])
def test_joint_probe_strata_and_gradients_are_independent_of_chunks(partition):
    counts = np.zeros((10, 2), np.int64)
    counts[2] = [3, 1]
    probes = torch.tensor(
        [[[1, 46, target, accepted]] for target, accepted in ((1, 1), (3, 0), (5, 0), (7, 0))],
        dtype=torch.float32,
    )
    wait = torch.zeros(4, 1, requires_grad=True)
    tiles = torch.zeros(4, 9, 45, requires_grad=True)
    loss = sum(
        probe_loss(
            TablePolicy(),
            wait[chunk],
            tiles[chunk],
            torch.empty(4, 0)[chunk],
            probes[chunk],
            counts,
            1,
        )
        for chunk in partition
    )
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    torch.testing.assert_close(tiles.grad[:, 1, 0], torch.tensor([-0.5, -1 / 6, -1 / 6, -1 / 6]))
    assert not wait.grad.any()
    assert tiles.grad.count_nonzero() == 4


def test_demo_ranking_anchors_sampled_action_and_excludes_rejections():
    wait = torch.zeros(2, 1, requires_grad=True)
    tiles = torch.zeros(2, 9, 45, requires_grad=True)
    actions = torch.tensor([46, 46])
    accepted = torch.tensor([True, False])
    masks = torch.ones(2, A.size, dtype=torch.bool)
    counts = ranking_counts(actions, accepted, masks)[0]
    loss, _ = demonstration_rank_loss(
        TablePolicy(), wait, tiles, torch.empty(2, 0), actions, accepted, masks, counts, 0.05
    )
    loss.backward()
    assert tiles.grad[0, 1, 0] < 0
    assert torch.all(tiles.grad[0, 1, 1:] > 0)
    assert wait.grad[0, 0] > 0
    assert not wait.grad[1].any() and not tiles.grad[1].any()


@pytest.mark.parametrize("microbatch", [1, 3, 128])
def test_requested_pairs_equal_shared_tables_and_wait_head_is_single_output(microbatch):
    torch.set_num_threads(1)
    cfg = load_config()
    cfg["policy"]["encoder_microbatch"] = microbatch
    policy = TransformerLSTMPolicy(cfg).eval()
    observations = collate_observations([observation(cfg, 20)[0]] * 4)
    output = policy.forward_step(observations)
    values = policy.action_values(output, observations=observations)
    actions = torch.tensor([0, 1, 90, 405])
    selected = policy.selected_values(output.wait_q, output.tile_features, output.context, actions)
    torch.testing.assert_close(selected, values.selected(actions))
    parameters = (
        *policy.wait_head.parameters(),
        *policy.tile_head.parameters(),
        policy.tile_offsets,
    )
    direct = torch.autograd.grad(selected.sum(), parameters, retain_graph=True)
    tables = torch.autograd.grad(values.selected(actions).sum(), parameters)
    for expected, actual in zip(direct, tables, strict=True):
        torch.testing.assert_close(actual, expected)
    assert output.wait_q.shape == (4, 1)
    assert not hasattr(policy, "branch_head")
    assert cfg["training"]["n_epochs"] == 4 and cfg["training"]["demo"]["passes"] == 20
    assert cfg["training"]["objective"]["probe_trigger"] == "accepted_plant"
    assert "branch_probes" not in cfg["training"]["objective"]
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
    assert layout.metadata_width == 15 + tiles * 4


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
