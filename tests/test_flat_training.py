"""Independent flat-loss attribution, global denominators and owned transport."""

from contextlib import closing
from copy import deepcopy
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from pvz_rl.config import load_config, validate_config
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.batch_transport import BatchPrefetch, batch_rows
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.flat_q import batch_loss
from pvz_rl.learning.objective import demonstration_rank_loss, ranking_counts
from pvz_rl.policy.sequential_q import action_parts


class IndependentValues(torch.nn.Module):
    def __init__(self, cfg, boards=8):
        super().__init__()
        self.cfg = cfg
        self.species = torch.nn.Parameter(torch.zeros(boards, 10))
        self.tiles = torch.nn.Parameter(torch.zeros(boards, 9, 45))

    def species_values(self, observations):
        return self.species[observations.globals[:, 0].long()]

    def conditional_tiles(self, observations, branches, *, rows=None):
        boards = observations.globals[:, 0].long()
        return self.tiles[boards if rows is None else boards[rows], branches - 1]

    def selected_values(self, observations, actions, *, species=None, tiles=None):
        branches, locations = action_parts(actions)
        return torch.where(
            branches == 0, species[:, 0], tiles.gather(1, locations[:, None]).flatten()
        )


def boards(count):
    observations = [{"entities": [], "globals": [index] + [0] * 17} for index in range(count)]
    return collate_observations(observations)


def loss_rows(actions, accepted, targets):
    probe_dtype = np.dtype(
        [("valid", "?"), ("action", "<i8"), ("target", "<f4"), ("accepted", "?")]
    )
    rows = np.zeros(
        len(actions),
        dtype=[
            ("active", "?"),
            ("action", "<i8"),
            ("accepted", "?"),
            ("target", "<f4"),
            ("probes", probe_dtype, (2,)),
        ],
    )
    rows["active"] = True
    rows["action"], rows["accepted"], rows["target"] = actions, accepted, targets
    return rows


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
    rows = loss_rows([2], [True], [-4])
    loss, error, gap = batch_loss(model, rows, [[0, 0], [0, 1], [0, 0]], boards(1), teacher=teacher)
    loss.backward()
    assert error.tolist() == [4]
    assert gap.tolist() == [6]
    assert model.tiles.grad[0, 0, 1] == 4
    assert model.tiles.grad[0, 0, 0] == 0
    assert model.species.grad[0, 1] == -14 * consistency
    assert model.species.grad[:, [0, *range(2, 10)]].count_nonzero() == 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


@pytest.mark.parametrize("partition", [[slice(None)], [slice(0, 2), slice(2, 5), slice(5, 6)]])
def test_selected_consistency_and_probe_losses_match_independent_global_control(partition):
    cfg = load_config()
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(False)
    rows = loss_rows([0, 0, 1, 46, 361, 362], [1, 0, 1, 0, 1, 0], [1, -1, 2, -2, 3, -3])
    rows["probes"][2, 0] = (True, 2, 4, True)
    rows["probes"][3, 0] = (True, 47, -4, False)
    with torch.no_grad():
        teacher.tiles[2, 0, 3] = 5
        teacher.tiles[3, 1, 4] = 6
        teacher.tiles[4, 8, 5] = 7
        teacher.tiles[5, 8, 6] = 8
    observations = boards(6)
    probe_counts = np.zeros((10, 2), np.int64)
    probe_counts[1, 1] = probe_counts[2, 0] = 1
    loss = sum(
        batch_loss(
            model,
            rows[part],
            [[1, 1], [1, 1], [1, 1]],
            observations[part],
            teacher=teacher,
            probe_counts=probe_counts,
            consistency_counts=[[1, 1], [1, 1]],
        )[0]
        for part in partition
    )
    reference_species = model.species.detach().clone().requires_grad_()
    reference_tiles = model.tiles.detach().clone().requires_grad_()
    selected_terms = []
    consistency_terms = []
    for index, (action, target) in enumerate(zip(rows["action"], rows["target"], strict=True)):
        branch = 0 if action == 0 else 1 + (action - 1) // 45
        tile = 0 if action == 0 else (action - 1) % 45
        chosen = (
            reference_species[index, 0] if branch == 0 else reference_tiles[index, branch - 1, tile]
        )
        selected_terms.append((chosen - float(target)).square() / 6)
        if branch:
            consistency_terms.append((reference_species[index, branch] - (index + 3)).square() / 4)
    probe_terms = []
    for index, branch, target in ((2, 1, 4), (3, 2, -4)):
        distance = (reference_tiles[index, branch - 1, 1] - target).abs()
        probe_terms.append(torch.where(distance < 1, distance.square() / 2, distance - 0.5) / 2)
    expected = sum(selected_terms) + sum(consistency_terms) + 0.25 * sum(probe_terms)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.species.grad, reference_species.grad)
    torch.testing.assert_close(model.tiles.grad, reference_tiles.grad)


def test_candidate_geometry_full_board_and_missing_outcome_strata():
    cfg = load_config()
    model = IndependentValues(cfg)
    teacher = deepcopy(model).requires_grad_(False)
    observations = boards(3).observations()

    def plant(tile):
        return [1, 1, tile // 9, (tile % 9) * 1000, 300, 0, 0, 0, 0, 0, 0]

    observations[0]["entities"] = [plant(0)]
    observations[1]["entities"] = [plant(tile) for tile in range(45)]
    with torch.no_grad():
        teacher.tiles[0, 0, 0] = 99
        teacher.tiles[0, 0, 1] = 3
        teacher.tiles[1, 0, 0] = 5
        teacher.tiles[1, 0, 1] = 99
        teacher.tiles[2, 8, 0] = 7
    rows = loss_rows([2, 1, 361], [1, 0, 0], [0, 0, 0])
    loss, *_ = batch_loss(
        model,
        rows,
        [[0, 0], [1, 1], [1, 0]],
        observations,
        teacher=teacher,
        consistency_counts=[[1, 1], [1, 0]],
    )
    assert float(model._last_species_consistency_loss) == pytest.approx(
        (3**2 / 2 + 5**2 / 2 + 7**2) / 2
    )
    loss.backward()
    assert model.species.grad[2, 9] == -7


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
def test_previous_model_rejected_before_cuda_probe_or_simulator(
    tmp_path, monkeypatch, suffix, mode
):
    from pvz_rl.learning.training import train

    path = tmp_path / ("previous" + suffix)
    if suffix == ".pt":
        torch.save({"protocol": "pvz-rl/demo-initialization-checkpoint-v6"}, path)
    else:
        with ZipFile(path, "w") as archive:
            archive.writestr(
                "protocol.json",
                '{"policy":"transformer_lstm_q_v4","optimizer":"complete_return_joint_tile_probe_v1"}',
            )

    def forbidden(*args, **kwargs):
        pytest.fail("Retired checkpoints must fail before simulator/CUDA allocation")

    monkeypatch.setattr("pvz_rl.learning.training_requirements._cuda_probe", forbidden)
    monkeypatch.setattr("pvz_rl.learning.training.vector_env", forbidden)
    with pytest.raises(ValueError, match="protocol|family"):
        train(load_config(), "masked", 101, tmp_path / "absent", **{mode: path})
    assert not (tmp_path / "absent").exists()


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
        assert attempts[1:] == (["fp32"] * 2 if fault == "nonfinite" else ["features_bf16"] * 2)
        assert model.execution_state["microbatch"] == (64 if fault == "memory" else 128)
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
    model._buffer.append(rows, boards(3).observations())
    model._buffer.finalize()
    model._stats = {"q_optimizer_steps": 0}
    model.phase = CohortPhase.FIT
    model._ensure_teacher()
    original = model._teacher.conditional_tiles
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

    monkeypatch.setattr(model._teacher, "conditional_tiles", failing_once)
    try:
        for _ in range(15):
            model._fit_step(object())
            if model._fit_epoch == 2:
                break
        assert model._fit_epoch == model._stats["q_optimizer_steps"] == model._teacher_version == 2
        assert attempts == [(128, 128, 128, 128), (64, 64, 64, 64), (64, 64, 64, 64)]
        assert model.execution_state["fallbacks"] == ["memory"]
        assert model.execution_state["microbatch"] == 64
        assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    finally:
        model._drain()
        model._close_batches()
        model._buffer.close()
        env.close()
