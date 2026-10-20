"""Independent flat-loss attribution, global denominators and owned transport."""

from contextlib import closing
from copy import deepcopy
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from pvz_rl.config import load_config, validate_config
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.action_routing import FitRoute, compact_invalid_rows, route_outcomes
from pvz_rl.learning.batch_transport import BatchPrefetch, FittingSpool, batch_rows
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.flat_q import batch_loss
from pvz_rl.learning.objective import (
    consistency_diagnostics,
    demonstration_rank_loss,
    ranking_counts,
)
from pvz_rl.policy.sequential_q import action_parts


class IndependentValues(torch.nn.Module):
    def __init__(self, cfg, boards=8):
        super().__init__()
        self.cfg = cfg
        self.species = torch.nn.Parameter(torch.zeros(boards, 10))
        self.tiles = torch.nn.Parameter(torch.zeros(boards, 9, 45))
        self.species_calls = 0
        self.tile_calls = []

    def species_values(self, observations):
        self.species_calls += 1
        return self.species[observations.globals[:, 0].long()]

    def conditional_tiles(self, observations, branches, *, rows=None):
        boards = observations.globals[:, 0].long()
        source = boards if rows is None else boards[rows]
        self.tile_calls.append((source.tolist(), branches.tolist(), torch.is_grad_enabled()))
        return self.tiles[source, branches - 1]

    def selected_values(self, observations, actions, *, species=None, tiles=None):
        branches, locations = action_parts(actions)
        return torch.where(
            branches == 0, species[:, 0], tiles.gather(1, locations[:, None]).flatten()
        )


@pytest.mark.parametrize(
    "length,expected", [(0, 0), (1, 1), (99, 1), (100, 1), (101, 2), (199, 2), (200, 2), (201, 3)]
)
def test_compact_invalid_rows_uses_first_row_of_each_hundred_window(length, expected):
    dtype = np.dtype(
        [
            ("active", "?"),
            ("action", "<i8"),
            ("accepted", "?"),
            ("route", "u1"),
            ("branch_usable", "?"),
            ("rejection_reason", "u1"),
            ("env", "<i2"),
            ("reset", "?"),
            ("done", "?"),
        ]
    )
    rows = np.zeros(length, dtype=dtype)
    rows["active"] = True
    rows["action"] = 1
    rows["route"] = FitRoute.SPECIES_INVALID
    rows["rejection_reason"] = 2
    if length:
        rows["reset"][0] = True
    selected, counts, stats = compact_invalid_rows(rows)
    np.testing.assert_array_equal(selected, np.arange(0, length, 100))
    assert len(selected) == expected
    assert counts.tolist() == [[0, 0], [expected, 0], [0, 0]]
    assert stats["raw_invalid_rows"] == length
    assert stats["retained_invalid_rows"] == expected


def test_compact_invalid_rows_keeps_episode_slots_and_operation_boundaries():
    dtype = np.dtype(
        [
            ("active", "?"),
            ("action", "<i8"),
            ("accepted", "?"),
            ("route", "u1"),
            ("branch_usable", "?"),
            ("rejection_reason", "u1"),
            ("env", "<i2"),
            ("reset", "?"),
            ("done", "?"),
        ]
    )
    rows = np.zeros(8, dtype=dtype)
    rows["active"] = True
    rows["env"] = [0, 1, 0, 1, 0, 1, 0, 1]
    rows["action"] = [1, 1, 1, 1, 0, 1, 1, 1]
    rows["route"] = [FitRoute.SPECIES_INVALID] * 8
    rows["route"][4] = FitRoute.WAIT_RETURN
    rows["rejection_reason"] = [2, 2, 2, 2, 0, 2, 3, 2]
    rows["reset"][[0, 1]] = True
    rows["done"][2] = True
    selected, _, _ = compact_invalid_rows(rows)
    # Each environment gets its own run; valid rows and reason changes remain.
    np.testing.assert_array_equal(selected, [0, 1, 4, 6])


def test_fitting_spool_compacts_invalid_rows_but_returns_keep_raw_counts(tmp_path):
    cfg = load_config()
    buffer = CompleteGameBuffer(tmp_path / "buffer", 1, block_rows=128, cfg=cfg)
    rows = np.zeros(101, dtype=buffer.dtype)
    rows["active"] = True
    rows["action"] = 1
    rows["accepted"] = False
    rows["branch_usable"] = False
    rows["rejection_reason"] = 2
    rows["reset"][0] = True
    rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
    buffer.append(rows, [{"entities": [], "globals": [0] * 18}] * len(rows))
    try:
        buffer.finalize()
        assert buffer.group_counts.tolist() == [0, 101, 0]
        spool = FittingSpool(buffer)
        try:
            assert len(spool) == 2
            assert spool.outcome_counts.tolist() == [[0, 0], [2, 0], [0, 0]]
            assert spool.compaction["invalid_windows"] == 2
        finally:
            spool.close()
    finally:
        buffer.close()


def test_causal_source_credit_updates_origin_row_before_returns(tmp_path):
    cfg = load_config()
    buffer = CompleteGameBuffer(tmp_path / "causal", 1, block_rows=2, cfg=cfg)
    rows = np.zeros(2, dtype=buffer.dtype)
    rows["active"] = True
    rows["reset"][0] = True
    rows["done"][1] = True
    rows["components_valid"] = True
    rows["route"] = FitRoute.WAIT_RETURN
    rows["components"][1, 0] = 1.0
    buffer.append(rows, [{"entities": [], "globals": [0] * 18}] * 2)
    try:
        buffer.apply_causal_credits({0: {"damage": 270 * 30 / 270}})
        buffer.finalize()
        actual = buffer.take([0, 1])
        assert actual[0]["components"][2] == pytest.approx(30)
        assert actual[0]["target"] > actual[1]["target"]
        assert buffer.reward_summary["source_action_credited_rows"] == 1
    finally:
        buffer.close()


def test_cohort_ranked_endgame_rewards_update_terminal_rows(tmp_path):
    cfg = load_config()
    buffer = CompleteGameBuffer(tmp_path / "ranked-endgame", 3, block_rows=8, cfg=cfg)
    rows = np.zeros(3, dtype=buffer.dtype)
    rows["active"] = True
    rows["done"] = True
    rows["components_valid"] = True
    rows["won"] = [True, False, True]
    rows["env"] = [0, 1, 2]
    rows["tick"] = [80, 160, 240]
    rows["route"] = FitRoute.WAIT_RETURN
    rows["components"][:, 0] = [1.0, -1.0, 1.0]
    rows["reset"] = True
    buffer.append(rows, [{"entities": [], "globals": [0] * 18}] * len(rows))
    try:
        buffer.finalize()
        actual = buffer.take([0, 1, 2])
        np.testing.assert_allclose(actual["components"][:, 0], [1.5, -1.0, 0.5])
        np.testing.assert_allclose(
            buffer.reward_summary["endgame_time_adjustments"], [0.5, 0.0, -0.5]
        )
    finally:
        buffer.close()


def boards(count):
    observations = [{"entities": [], "globals": [index] + [0] * 17} for index in range(count)]
    return collate_observations(observations)


def loss_rows(actions, accepted, targets):
    rows = np.zeros(
        len(actions),
        dtype=[
            ("active", "?"),
            ("action", "<i8"),
            ("accepted", "?"),
            ("target", "<f4"),
            ("route", "u1"),
            ("branch_usable", "?"),
        ],
    )
    rows["active"] = True
    rows["action"], rows["accepted"], rows["target"] = actions, accepted, targets
    rows["branch_usable"] = True
    rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
    return rows


@pytest.mark.parametrize(
    "route,action,accepted,usable,head",
    [
        (0, 0, True, True, "species"),
        (1, 2, True, True, "both"),
        (2, 2, False, False, "species"),
        (3, 2, False, True, "tiles"),
        (2, 361, False, False, "species"),
        (3, 361, False, True, "tiles"),
    ],
)
def test_routed_head_gradients_and_exact_forward_counts(route, action, accepted, usable, head):
    cfg = load_config()
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(False)
    rows = loss_rows([action], [accepted], [4])
    rows["route"], rows["branch_usable"] = route, usable
    counts = np.zeros((3, 2), np.int64)
    group = 0 if action == 0 else 1 if action < 361 else 2
    counts[group, int(accepted)] = 1
    loss, errors, _ = batch_loss(model, rows, counts, boards(1), teacher=teacher)
    loss.backward()
    assert rows["target"][0] == 4  # fixed -1 never replaces chronological returns
    assert errors.item() == (1 if route in (2, 3) else 16)
    assert model.species_calls == int(head in ("species", "both"))
    assert len(model.tile_calls) == int(head in ("tiles", "both"))
    assert len(teacher.tile_calls) == int(route == 1)
    assert teacher.species_calls == 0
    if head == "species":
        assert model.tiles.grad is None
        branch = 0 if action == 0 else 1 + (action - 1) // 45
        assert model.species.grad[0, branch] == (-8 if route == 0 else 2)
        assert model.species.grad.count_nonzero() == 1
    if head == "tiles":
        assert model.species.grad is None
        assert model.tiles.grad.count_nonzero() == 1
        assert model.tiles.grad[0, (action - 1) // 45, (action - 1) % 45] == 2
    assert all(p.grad is None for p in teacher.parameters())


def test_public_classifier_branch_precedence_and_boundaries():
    from pvz_game import Game, LevelSpec, Place, Spawn
    from pvz_game.config import InitialPlant

    from pvz_rl.envs.actions import ActionCodec
    from pvz_rl.learning.action_routing import classify_actions

    codec = ActionCodec()
    game = Game()
    game.reset(
        LevelSpec(
            "routing",
            (Spawn(9999, "basic", 0),),
            initial_sun=100,
            plants=(InitialPlant("peashooter", 0, 0),),
        ),
        101,
    )
    action = codec.encode(Place("peashooter", 0, 0))
    snapshot = game.snapshot()
    snapshot["cooldowns"]["peashooter"] = 1
    game.restore(snapshot)
    # Branch-level failure wins even on an occupied tile.
    route, usable = classify_actions([action], [False], codec.mask(game)[None], ["occupied_tile"])
    assert route.tolist() == [2] and not usable[0]
    snapshot = game.snapshot()
    snapshot["cooldowns"]["peashooter"] = 0
    game.restore(snapshot)
    # Exactly the purchase cost with cooldown zero is usable elsewhere.
    route, usable = classify_actions([action], [False], codec.mask(game)[None], ["occupied_tile"])
    assert route.tolist() == [3] and usable[0]
    masks = codec.mask(game)
    route, _ = classify_actions([action], [False], masks[None], ["task_restriction"])
    assert route.tolist() == [2]
    route, _ = classify_actions([362], [False], masks[None], ["empty_tile"])
    assert route.tolist() == [3]
    masks[361:] = False
    route, _ = classify_actions([362], [False], masks[None], ["empty_tile"])
    assert route.tolist() == [2]


@pytest.mark.parametrize("consistency", [0.0, 1.0])
def test_poor_tile_return_never_directly_trains_species(consistency):
    cfg = load_config()
    cfg["training"]["objective"]["species_consistency_weight"] = consistency
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        model.tiles[0, 0, 0] = 4
        model.tiles[0, 0, 1] = -2
        teacher.tiles[0, 0, 0] = 7
        teacher.tiles[0, 0, 1] = 9
    rows = loss_rows([2], [True], [-4])
    loss, error, gap = batch_loss(model, rows, [[0, 0], [0, 1], [0, 0]], boards(1), teacher=teacher)
    loss.backward()
    assert error.tolist() == [4]
    assert gap.tolist() == [6]
    assert model.tiles.grad[0, 0, 1] == 4
    assert model.tiles.grad[0, 0, 0] == 0
    assert model.species.grad[0, 1] == -2 * consistency
    assert model.species.grad[:, [0, *range(2, 10)]].count_nonzero() == 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert model.species_calls == 1 and model.tile_calls == [([0], [1], True)]
    assert teacher.species_calls == 0
    assert teacher.tile_calls == ([([0], [1], False)] if consistency else [])
    if consistency:
        assert model._last_species_consistency_loss.item() == 13
        torch.testing.assert_close(
            model._last_species_consistency_metrics,
            torch.tensor([7, 1, 2, 1], dtype=torch.float64),
        )


@pytest.mark.parametrize("partition", [[slice(None)], [slice(0, 2), slice(2, 5), slice(5, 6)]])
def test_selected_and_consistency_losses_match_independent_global_control(partition):
    cfg = load_config()
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(False)
    rows = loss_rows([0, 0, 1, 46, 361, 362], [1, 0, 1, 0, 1, 0], [1, -1, 2, -2, 3, -3])
    with torch.no_grad():
        for index, branch, target in zip(
            range(2, 6), [1, 2, 9, 9], [0.25, -1.0, 3.0, -4.0], strict=True
        ):
            model.tiles[index, branch - 1, index + 1] = 6
            teacher.tiles[index, branch - 1] = -100
            teacher.tiles[index, branch - 1, index + 1] = target
            teacher.tiles[index, branch - 1, index + 2] = target + 2
    observations = boards(6)
    reference_model = deepcopy(model)
    reference_teacher = deepcopy(teacher)
    reference_loss, _, _ = batch_loss(
        reference_model,
        rows,
        [[1, 1], [1, 1], [1, 1]],
        observations,
        teacher=reference_teacher,
        consistency_counts=[[1, 1], [1, 1]],
    )
    losses, metric_totals, consistency_losses = [], [], []
    for part in partition:
        losses.append(
            batch_loss(
                model,
                rows[part],
                [[1, 1], [1, 1], [1, 1]],
                observations[part],
                teacher=teacher,
                consistency_counts=[[1, 1], [1, 1]],
            )[0]
        )
        metric_totals.append(model._last_species_consistency_metrics.clone())
        consistency_losses.append(model._last_species_consistency_loss)
    loss = sum(losses)
    torch.testing.assert_close(loss, reference_loss)
    totals = torch.stack(metric_totals).sum(0)
    assert totals[3] == 2
    assert all(not total.requires_grad for total in metric_totals)
    diagnostics = consistency_diagnostics(sum(consistency_losses).item(), totals.tolist())
    assert diagnostics["species_consistency_loss"] is not None
    assert diagnostics["species_consistency_mae"] == pytest.approx(totals[0] / totals[3])
    assert model.species_calls == sum(bool((rows["route"][part] != 3).any()) for part in partition)
    assert len(model.tile_calls) == sum(
        bool((rows["action"][part] != 0).any()) for part in partition
    )
    assert len(teacher.tile_calls) == sum(
        bool(((rows["action"][part] != 0) & rows["accepted"][part]).any()) for part in partition
    )


def test_candidate_geometry_full_board_and_missing_outcome_strata():
    cfg = load_config()
    cfg["training"]["objective"]["species_consistency_weight"] = 2
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(False)
    observations = boards(3).observations()

    def plant(tile):
        return [1, 1, tile // 9, (tile % 9) * 1000, 300, 0, 0, 0, 0, 0, 0]

    observations[0]["entities"] = [plant(0)]
    observations[1]["entities"] = [plant(tile) for tile in range(45)]
    observations[2]["entities"] = [plant(44)]
    with torch.no_grad():
        model.tiles[0, 0, 0] = 99
        model.tiles[0, 0, 1] = 4
        teacher.tiles[0, 0, 0] = 99
        teacher.tiles[0, 0, 1] = 3
        teacher.tiles[0, 0, 2] = 9
        model.tiles[1, 0, 1] = 99
        teacher.tiles[1, 0, 0] = 5
        teacher.tiles[1, 0, 1] = 99
        model.tiles[2, 8, 44] = 8
        teacher.tiles[2, 8, 0] = 99
        teacher.tiles[2, 8, 44] = 7
    rows = loss_rows([2, 1, 361], [1, 0, 0], [4, 0, 0])
    rows["branch_usable"][1] = False  # full board: species error
    rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
    loss, *_ = batch_loss(
        model,
        rows,
        [[0, 0], [1, 1], [1, 0]],
        observations,
        teacher=teacher,
        consistency_counts=[[1, 1], [1, 0]],
    )
    assert float(model._last_species_consistency_loss) == pytest.approx(5)
    assert loss.item() == pytest.approx(10.75)
    torch.testing.assert_close(
        model._last_species_consistency_metrics, torch.tensor([3, 1, 6, 1], dtype=torch.float64)
    )
    loss.backward()
    assert model.species.grad[2, 9] == 0
    assert model.species.grad[1, 1] == 0.5
    assert model.tiles.grad[2, 8, 0] == 1


def test_actual_commitment_and_rejection_branches_reuse_online_tables_without_extra_forwards():
    cfg = load_config()
    cfg["training"]["objective"]["species_consistency_huber_delta"] = 0.5
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(True)
    rows = loss_rows([0, 47, 183], [True, False, True], [1, -2, -3])
    with torch.no_grad():
        model.tiles[1, 1, 2] = 8
        model.tiles[2, 4, 3] = 7
        teacher.tiles[1, 1, 2] = 4
        teacher.tiles[1, 1, 3] = 10
        teacher.tiles[2, 4, 3] = -2
        teacher.tiles[2, 4, 4] = 3
    loss, _, _ = batch_loss(
        model,
        rows,
        [[0, 1], [1, 1], [0, 0]],
        boards(3),
        teacher=teacher,
        consistency_counts=[[1, 1], [0, 0]],
    )
    assert model.species_calls == 1
    assert model.tile_calls == [([1, 2], [2, 5], True)]
    assert teacher.species_calls == 0 and teacher.tile_calls == [([2], [5], False)]
    assert model._last_species_consistency_loss.item() > 0
    assert model._last_species_consistency_metrics[3] == 1
    loss.backward()
    assert model.species.grad[1, 2] == 0 and model.species.grad[2, 5] == 1
    assert model.species.grad[1, 1] == model.species.grad[2, 9] == 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


@pytest.mark.parametrize("active", [[False, False, False], [True, False, True]])
def test_inactive_or_wait_only_batches_have_no_consistency_or_tile_forwards(active):
    cfg = load_config()
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(False)
    rows = loss_rows([0, 46, 0], [True, False, False], [1, 100, -2])
    rows["active"] = active
    loss, errors, gaps = batch_loss(
        model, rows, [[1, 1], [0, 0], [0, 0]], boards(3), teacher=teacher
    )
    assert loss.item() == (2.5 if any(active) else 0)
    assert errors.tolist() == ([1, 4] if any(active) else []) and gaps.numel() == 0
    assert model.species_calls == int(any(active))
    assert model.tile_calls == teacher.tile_calls == [] and teacher.species_calls == 0
    assert model._last_species_consistency_loss.item() == 0
    metrics = model._last_species_consistency_metrics
    torch.testing.assert_close(metrics, torch.zeros(4, dtype=torch.float64))
    assert not metrics.requires_grad
    assert all(value is None for value in consistency_diagnostics(0, metrics.tolist()).values())
    loss.backward()
    assert model.tiles.grad is None
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_demo_ranking_uses_independent_branch_and_actual_tile_anchors():
    cfg = load_config()
    model = IndependentValues(cfg)
    actions = torch.tensor([2, 46, 0])
    accepted = torch.tensor([True, False, True])
    masks = torch.zeros(3, 406, dtype=torch.bool)
    masks[:, 0] = True
    masks[:, [1, 2, 46]] = True
    with torch.no_grad():
        model.species[0, 1] = 4
        model.species[0, 2] = 1
        model.tiles[0, 0, 0] = 5
        model.tiles[0, 0, 1] = -2
        model.species[2, 0] = 2
    counts, _, _ = ranking_counts(actions, accepted, masks)
    loss, accuracy = demonstration_rank_loss(
        model, boards(3), actions, accepted, masks, counts, 0.05
    )
    branch = (
        (
            torch.nn.functional.softplus(torch.tensor(0.05 - 4))
            + torch.nn.functional.softplus(torch.tensor(0.05 - 4 + 1))
        )
        / 2
        + torch.nn.functional.softplus(torch.tensor(0.05 - 2))
    ) / 2
    tile = torch.nn.functional.softplus(torch.tensor(0.05 + 2 + 5))
    torch.testing.assert_close(loss, branch + tile)
    loss.backward()
    assert model.species.grad[0, 1] < 0
    assert model.tiles.grad[0, 0, 1] < 0
    assert model.tiles.grad[0, 0, 0] > 0
    assert model.species.grad[1].count_nonzero() == model.tiles.grad[1].count_nonzero() == 0
    assert accuracy.tolist() == [4, 4, 0, 1]


@pytest.mark.parametrize("ram_bytes", [1, 1000000])
def test_flat_batches_exclude_capacity_gaps_preserve_returns_and_release_maps(tmp_path, ram_bytes):
    buffer = CompleteGameBuffer(tmp_path / "buffer", 2, ram_bytes=ram_bytes, block_rows=3)
    try:
        rows = np.zeros(8, buffer.dtype)
        rows["env"] = [0, 1, 0, 1, 0, 1, 0, 1]
        rows["active"] = [True, True, False, True, True, False, True, False]
        rows["reward"] = [1, 2, 999, 3, 4, 999, 5, 999]
        rows["done"][[3, 6]] = True
        rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
        buffer.append(rows, [{"entities": [], "globals": [0] * 18}] * len(rows))
        buffer.finalize()
        with closing(batch_rows(buffer, 2)) as source:
            actual = list(source)
        assert [len(batch) for batch in actual] == [2, 2, 1]
        merged = np.concatenate(actual)
        assert merged["active"].all()
        np.testing.assert_array_equal(merged["env"], [0, 1, 1, 0, 0])
        np.testing.assert_array_equal(merged["target"], [10, 5, 3, 9, 5])
        assert buffer.staging_bytes == 0
        assert not list(buffer.path.glob("flat-batches-*"))
        with closing(batch_rows(buffer, 1)) as source:
            next(source)
        assert buffer.staging_bytes == 0
        assert not list(buffer.path.glob("flat-batches-*"))
    finally:
        buffer.close()


@pytest.mark.parametrize("ram_bytes", [1, 1000000])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_ram_and_spilled_fitting_match_independent_losses_gradients_and_metrics(
    tmp_path, ram_bytes, batch_size
):
    cfg = load_config()
    buffer = CompleteGameBuffer(tmp_path / "buffer", 1, ram_bytes=ram_bytes, block_rows=3, cfg=cfg)
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        for index, branch, tile, target in ((2, 1, 4, 0.25), (3, 9, 10, 3), (5, 2, 7, -4)):
            model.tiles[index, branch - 1, tile] = 6
            teacher.tiles[index, branch - 1] = -100
            teacher.tiles[index, branch - 1, tile] = target
            teacher.tiles[index, branch - 1, tile + 1] = target + 2
    try:
        rows = np.zeros(6, buffer.dtype)
        rows["active"] = [True, False, True, True, False, True]
        rows["action"] = [0, 1, 1, 361, 46, 46]
        rows["accepted"] = [True, False, False, True, False, True]
        rows["reward"] = [0.5, 999, 1, -0.5, 999, 2]
        rows["done"][-1] = True
        rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
        buffer.append(rows, boards(6).observations())
        buffer.finalize()
        np.testing.assert_array_equal(buffer.outcome_counts, [[0, 1], [1, 1], [0, 1]])
        np.testing.assert_array_equal(buffer.consistency_counts, [[0, 1], [0, 1]])
        assert any(isinstance(block, np.memmap) for block in buffer.blocks) == (ram_bytes == 1)
        losses, metrics, consistency_losses, selected_errors, gaps = [], [], [], [], []
        with closing(batch_rows(buffer, batch_size)) as source:
            for current in source:
                loss, errors, gap = batch_loss(
                    model,
                    current,
                    buffer.outcome_counts,
                    buffer.observations(current),
                    teacher=teacher,
                    consistency_counts=buffer.consistency_counts,
                )
                losses.append(loss)
                metrics.append(model._last_species_consistency_metrics.clone())
                consistency_losses.append(model._last_species_consistency_loss)
                selected_errors.append(errors)
                gaps.append(gap)
        loss = sum(losses)
        expected_selected = 3**2 / 3 + 1**2 / 6 + 1.5**2 / 3 + 2**2 / 6
        expected_consistency = 5 / 2 + 7 / 2
        assert loss.item() == pytest.approx(expected_selected + expected_consistency)
        torch.testing.assert_close(torch.cat(selected_errors), torch.tensor([9, 1, 2.25, 4]))
        torch.testing.assert_close(torch.cat(gaps), torch.tensor([6.0, 6.0]))
        totals = torch.stack(metrics).sum(0)
        torch.testing.assert_close(totals, torch.tensor([7, 2, 4, 2], dtype=torch.float64))
        assert consistency_diagnostics(
            sum(consistency_losses).item(), totals.tolist()
        ) == pytest.approx(
            {
                "species_consistency_loss": expected_consistency,
                "species_consistency_mae": 7 / 2,
                "species_consistency_saturation_fraction": 1,
                "species_selection_evaluation_gap": 2,
            }
        )
        loss.backward()
        expected_species = torch.zeros_like(model.species)
        expected_species[[0, 2, 3, 5], [0, 1, 9, 2]] = torch.tensor([-2.0, 1 / 3, -1.0, 1.0])
        expected_tiles = torch.zeros_like(model.tiles)
        expected_tiles[[3, 5], [8, 1], 0] = torch.tensor([-1, -2 / 3])
        torch.testing.assert_close(model.species.grad, expected_species)
        torch.testing.assert_close(model.tiles.grad, expected_tiles)
        assert all(parameter.grad is None for parameter in teacher.parameters())
        assert buffer.staging_bytes == 0
        assert not list(buffer.path.glob("flat-batches-*"))
    finally:
        buffer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_double_buffer_transport_matches_serial_rows_and_owns_device_outputs(tmp_path):
    buffer = CompleteGameBuffer(tmp_path / "buffer", 1, ram_bytes=1000000, block_rows=2)
    try:
        rows = np.zeros(5, buffer.dtype)
        rows["active"] = True
        rows["action"] = [0, 1, 46, 361, 0]
        rows["reward"] = [1, 2, 3, 4, 5]
        rows["done"][-1] = True
        rows["placement_gap"] = np.arange(5)
        rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
        buffer.append(rows, boards(5).observations())
        buffer.finalize()
        snapshots = []
        with closing(BatchPrefetch(buffer, 2, "cuda", 256)) as source:
            for actual, obs, fields in source:
                snapshots.append((actual.copy(), obs.clone(), fields.target.clone()))
        torch.cuda.synchronize()
        assert [len(item[0]) for item in snapshots] == [2, 2, 1]
        for rows, obs, targets in snapshots:
            torch.testing.assert_close(targets.cpu(), torch.from_numpy(rows["target"].copy()))
            torch.testing.assert_close(obs.globals.cpu(), torch.from_numpy(rows["globals"].copy()))
        assert buffer.staging_bytes == 0
    finally:
        buffer.close()


@pytest.mark.parametrize("suffix", [".pt", ".zip"])
@pytest.mark.parametrize("mode", ["init_from", "resume"])
@pytest.mark.parametrize("family", ["recurrent", "stateless"])
def test_previous_model_rejected_before_cuda_probe_or_simulator(
    tmp_path, monkeypatch, suffix, mode, family
):
    import json

    from pvz_rl.learning.training import train

    path = tmp_path / ("previous" + suffix)
    if suffix == ".pt":
        version = 6 if family == "recurrent" else 7
        torch.save({"protocol": f"pvz-rl/demo-initialization-checkpoint-v{version}"}, path)
    else:
        with ZipFile(path, "w") as archive:
            archive.writestr(
                "protocol.json",
                json.dumps(
                    {
                        "policy": (
                            "transformer_lstm_q_v4"
                            if family == "recurrent"
                            else "two_stage_transformer_q_v1"
                        ),
                        "optimizer": (
                            "complete_return_joint_tile_probe_v1"
                            if family == "recurrent"
                            else "complete_return_sequential_tile_probe_v1"
                        ),
                    }
                ),
            )

    def forbidden(*args, **kwargs):
        pytest.fail("Retired checkpoints must fail before simulator/CUDA allocation")

    monkeypatch.setattr("pvz_rl.learning.training_requirements._cuda_probe", forbidden)
    monkeypatch.setattr("pvz_rl.learning.training.vector_env", forbidden)
    with pytest.raises(ValueError, match="protocol|family"):
        train(load_config(), "masked", 101, tmp_path / "absent", **{mode: path})
    assert not (tmp_path / "absent").exists()


def _write_contract_checkpoint(path, cfg, *, runtime_protocol=None):
    import io
    import json

    from pvz_game import Rules

    from pvz_rl.config import digest
    from pvz_rl.envs.encoding import ObservationEncoder
    from pvz_rl.learning.checkpoints import (
        AUTONOMOUS_STATE_PROTOCOL,
        DEMO_PROTOCOL,
        MODEL_STATE_PROTOCOL,
        protocol_for,
    )

    schema = ObservationEncoder(cfg, Rules()).schema()
    if path.suffix == ".pt":
        torch.save(
            {
                "protocol": DEMO_PROTOCOL,
                "model_storage_protocol": MODEL_STATE_PROTOCOL,
                "precision_contract": "bf16_master",
                "config": cfg,
                "config_digest": digest(cfg),
                "passes_completed": 1,
                "model": {"weight": torch.zeros(1)},
                "observation_schema": schema,
                "learner_seed": 101,
            },
            path,
        )
        return
    runtime = {
        "protocol": runtime_protocol or AUTONOMOUS_STATE_PROTOCOL,
        "model_storage_protocol": MODEL_STATE_PROTOCOL,
        "precision_contract": "bf16_master",
        "buffer": None,
    }
    stream = io.BytesIO()
    torch.save(runtime, stream)
    with ZipFile(path, "w") as archive:
        archive.writestr("protocol.json", json.dumps(protocol_for(cfg["policy"]["kind"])))
        archive.writestr(
            "run.json",
            json.dumps(
                {
                    "config": cfg,
                    "condition": "masked",
                    "family": "preset",
                    "learner_seed": 101,
                    "validation_limit": None,
                }
            ),
        )
        archive.writestr("data", "{}")
        archive.writestr("observation-schema.json", json.dumps(schema))
        archive.writestr("cohort-state.pt", stream.getvalue())
        for name in ("policy.pth", "policy.optimizer.pth"):
            archive.writestr(name, b"unread weights")


@pytest.mark.parametrize("suffix", [".pt", ".zip"])
@pytest.mark.parametrize("mode", ["init_from", "resume"])
@pytest.mark.parametrize("fault", ["old_method", "old_objective", "missing_delta"])
def test_current_checkpoint_header_cannot_hide_retired_learning_config(
    tmp_path, monkeypatch, suffix, mode, fault
):
    from pvz_rl.learning.training import train

    cfg = load_config()
    source_cfg = deepcopy(cfg)
    if fault == "old_method":
        source_cfg["training"]["method"] = "complete_return_sequential_tile_probe_v1"
    elif fault == "old_objective":
        source_cfg["training"]["objective"]["protocol"] = "complete_return_sequential_tile_probe_v1"
    else:
        source_cfg["training"]["objective"].pop("species_consistency_huber_delta")
    path = tmp_path / ("invalid" + suffix)
    _write_contract_checkpoint(path, source_cfg)

    def forbidden(*args, **kwargs):
        pytest.fail("Old learning configuration reached simulator/CUDA allocation")

    monkeypatch.setattr("pvz_rl.learning.training_requirements._cuda_probe", forbidden)
    monkeypatch.setattr("pvz_rl.learning.training.vector_env", forbidden)
    with pytest.raises(
        ValueError,
        match="objective protocol|complete-return method|species_consistency_huber_delta|storage",
    ):
        train(cfg, "masked", 101, tmp_path / "absent", **{mode: path})
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("mode", ["init_from", "resume"])
def test_previous_stateless_recovery_rejected_for_resume_and_weight_initialization(
    tmp_path, monkeypatch, mode
):
    from pvz_rl.learning.training import train

    cfg = load_config()
    path = tmp_path / "previous-recovery.zip"
    _write_contract_checkpoint(path, cfg, runtime_protocol="pvz-rl/stateless-tile-probe-state-v6")

    def forbidden(*args, **kwargs):
        pytest.fail("Old recovery state reached simulator/CUDA allocation")

    monkeypatch.setattr("pvz_rl.learning.training_requirements._cuda_probe", forbidden)
    monkeypatch.setattr("pvz_rl.learning.training.vector_env", forbidden)
    with pytest.raises(ValueError, match="recovery protocol"):
        train(cfg, "masked", 101, tmp_path / "absent", **{mode: path})
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("suffix", [".pt", ".zip"])
@pytest.mark.parametrize("weights_only", [False, True])
def test_current_checkpoint_contract_metadata_reloads(tmp_path, suffix, weights_only):
    from pvz_rl.learning.checkpoints import inspect_checkpoint

    cfg = load_config()
    path = tmp_path / ("current" + suffix)
    _write_contract_checkpoint(path, cfg)
    inspected = inspect_checkpoint(path, weights_only=weights_only)
    assert inspected["config"] == cfg
    assert inspected["initialization_type"] == (
        "demonstration" if suffix == ".pt" else "autonomous"
    )


def test_flat_batch_sizes_no_longer_require_recurrent_multiples():
    cfg = load_config()
    cfg["training"]["batch_size"] = 257
    validate_config(cfg)
    assert "chunk_length" not in cfg["policy"]
    assert cfg["training"]["performance"]["fit_batch_groups"] == 4


@pytest.mark.parametrize("fault", ["nonfinite", "memory", "compilation"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_failed_flat_pass_restarts_without_skipping_updates(tmp_path, monkeypatch, fault):
    from stable_baselines3.common.logger import configure

    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.training import build_model, vector_env
    from pvz_rl.policy.cudagraph_backend import EncoderCompilationError

    cfg = load_config()
    cfg["training"].update(n_envs=1, n_epochs=2)
    cfg["visualization"]["live_enabled"] = False
    cfg["training"]["performance"].update(compile_kernels=False, prefetch=False)
    env = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", env, 101)
    model.trajectory_root = tmp_path
    model._buffer = model._new_buffer()
    model.set_logger(configure(format_strings=[]))
    rows = np.zeros(3, model._buffer.dtype)
    rows["active"] = True
    rows["reward"][-1] = 1
    rows["done"][-1] = True
    rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
    model._buffer.append(rows, boards(3).observations())
    model._buffer.finalize()
    model._stats = {"q_optimizer_steps": 0}
    model.phase = CohortPhase.FIT
    original = model.policy.species_values
    attempts = []

    def failing_once(*args, **kwargs):
        attempts.append(model.policy.fit_precision)
        if len(attempts) == 1 and fault == "memory":
            raise torch.cuda.OutOfMemoryError("injected allocation failure")
        output = original(*args, **kwargs)
        if len(attempts) == 1 and fault == "compilation":

            def fail_backward(gradient):
                raise EncoderCompilationError("injected backward capture failure")

            output.register_hook(fail_backward)
        elif len(attempts) == 1:
            output = output * float("nan")
        return output

    monkeypatch.setattr(model.policy, "species_values", failing_once)
    callback = object()
    try:
        if fault == "nonfinite":
            with pytest.raises(FloatingPointError, match="Non-finite"):
                for _ in range(15):
                    model._fit_step(callback)
            assert model._fit_epoch == model._stats["q_optimizer_steps"] == 0
            assert model.execution_state["fallbacks"] == ["nonfinite_bf16"]
            return
        for _ in range(15):
            model._fit_step(callback)
            if fault == "compilation" and len(attempts) == 1:
                assert model._fit_epoch == model._stats["q_optimizer_steps"] == 0
                assert all(
                    encoder.compilation_status == "fallback" for encoder in model.policy.encoders
                )
                assert all(parameter.grad is None for parameter in model.policy.parameters())
            if model._fit_epoch == 2:
                break
        assert model._fit_epoch == model._stats["q_optimizer_steps"] == model._teacher_version == 2
        assert len(attempts) == 3
        assert model.execution_state["fallbacks"] == [
            {
                "nonfinite": "nonfinite_bf16",
                "memory": "memory",
                "compilation": "encoder_compilation",
            }[fault]
        ]
        assert attempts[1:] == ["bf16_master"] * 2
        assert model.execution_state["microbatch"] == (96 if fault == "memory" else 192)
        assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    finally:
        model._drain()
        model._close_batches()
        model._buffer.close()
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ema_tile_allocation_failure_reduces_both_network_microbatches(tmp_path, monkeypatch):
    from stable_baselines3.common.logger import configure

    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.training import build_model, vector_env

    cfg = load_config()
    cfg["training"].update(n_envs=1, n_epochs=2)
    cfg["visualization"]["live_enabled"] = False
    cfg["training"]["performance"].update(compile_kernels=False, prefetch=False)
    env = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", env, 101)
    model.trajectory_root = tmp_path
    model._buffer = model._new_buffer()
    model.set_logger(configure(format_strings=[]))
    rows = np.zeros(3, model._buffer.dtype)
    rows["active"] = rows["accepted"] = True
    rows["action"] = 1
    rows["reward"][-1] = 1
    rows["done"][-1] = True
    rows["route"] = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
    model._buffer.append(rows, boards(3).observations())
    model._buffer.finalize()
    model._stats = {"q_optimizer_steps": 0}
    model.phase = CohortPhase.FIT
    model._ensure_teacher()
    original = model._teacher.fitting_tiles
    attempts = []

    def failing_once(*args, **kwargs):
        attempts.append(
            tuple(
                encoder.microbatch for encoder in (*model.policy.encoders, *model._teacher.encoders)
            )
        )
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("injected EMA tile allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model._teacher, "fitting_tiles", failing_once)
    try:
        for _ in range(15):
            model._fit_step(object())
            if model._fit_epoch == 2:
                break
        assert model._fit_epoch == model._stats["q_optimizer_steps"] == model._teacher_version == 2
        assert attempts == [(192, 192, 192, 192), (96, 96, 96, 96), (96, 96, 96, 96)]
        assert model.execution_state["fallbacks"] == ["memory"]
        assert model.execution_state["microbatch"] == 96
        assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    finally:
        model._drain()
        model._close_batches()
        model._buffer.close()
        env.close()
