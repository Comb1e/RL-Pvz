"""Independent selected-Q math and atomic checkpoint interruption controls."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from pvz_rl.config import load_config
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.recurrent_q import CudaRecurrentQ
from pvz_rl.learning.training import build_model, vector_env


def tiny_config():
    c = load_config()
    c["training"].update(n_envs=2, n_epochs=1, batch_size=128)
    c["policy"]["chunk_length"] = 16
    c["environment"]["cutoff_seconds"] = 1
    c["visualization"].update(live_enabled=False, enabled=False, demos=False)
    c["training"]["performance"].update(compile_kernels=False, telemetry=False)
    return c


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_state_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_interrupted_optimizer_phase_resumes_exactly(tmp_path, monkeypatch):
    import signal

    cfg = tiny_config()

    def make():
        env = vector_env(cfg, "masked", 103)
        m = build_model(cfg, "masked", env, 103)
        with torch.no_grad():
            m.policy.branch_head[-1].bias[1] = 0.1
        return env, m

    env, full = make()
    full.learn(1)
    expected = deepcopy(full.policy.state_dict())
    optimizer = deepcopy(full.policy.optimizer.state_dict())
    env.close()
    env, partial = make()
    name = "_fit_chunk"
    original = getattr(partial, name)
    triggered = []

    def interrupt(data):
        original(data)
        if not triggered:
            triggered.append(True)
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(partial, name, interrupt)
    with pytest.raises(KeyboardInterrupt):
        partial.learn(1)
    monkeypatch.delattr(partial, name)
    partial.save(tmp_path / "partial.zip")
    env.close()
    env = vector_env(cfg, "masked", 103)
    m = CudaRecurrentQ.load(tmp_path / "partial.zip", env=env, device="cuda")
    m.learn(0, reset_num_timesteps=False)
    assert_state_equal(expected, m.policy.state_dict())
    assert_state_equal(optimizer, m.policy.optimizer.state_dict())
    env.close()


def test_atomic_checkpoint_failure_preserves_previous_archive(tmp_path, monkeypatch):
    from zipfile import ZipFile

    env = vector_env(tiny_config(), "masked", 101)
    model = build_model(tiny_config(), "masked", env, 101)
    path = tmp_path / "checkpoint.zip"
    model.save(path)
    before = path.read_bytes()
    original = ZipFile.writestr

    def fail(self, name, *args, **kwargs):
        if name == "cohort-state.pt":
            raise OSError("controlled disk failure")
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="controlled disk failure"):
        model.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.zip")) == [path]
    monkeypatch.undo()
    CudaRecurrentQ.load(path, device="cuda")
    # Incomplete/corrupt archives cannot silently start a different RNG stream.
    damaged = tmp_path / "damaged.zip"
    with ZipFile(path) as inp, ZipFile(damaged, "w") as out:
        for name in inp.namelist():
            if name != "cohort-state.pt":
                out.writestr(name, inp.read(name))
    with pytest.raises(ValueError, match="runtime state"):
        CudaRecurrentQ.load(damaged, device="cuda")
    env.close()


def test_resume_at_game_ceiling_finishes_pending_optimization(tmp_path, monkeypatch):
    import signal

    from pvz_rl.learning.training import load_policy, train

    cfg = tiny_config()
    cfg["training"]["total_games"] = 2
    train(cfg, "masked", 101, tmp_path / "full", validation_limit=1)
    expected, _ = load_policy(tmp_path / "full/final.zip", "cuda")
    original = CudaRecurrentQ._fit_chunk
    triggered = []

    def interrupt(self, data):
        original(self, data)
        if not triggered:
            triggered.append(True)
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(CudaRecurrentQ, "_fit_chunk", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, tmp_path / "partial", validation_limit=1)
    monkeypatch.setattr(CudaRecurrentQ, "_fit_chunk", original)
    train(
        cfg,
        "masked",
        101,
        tmp_path / "resumed",
        validation_limit=1,
        resume=tmp_path / "partial/interrupted.zip",
    )
    actual, _ = load_policy(tmp_path / "resumed/final.zip", "cuda")
    assert actual._n_updates == expected._n_updates == 1
    assert actual.training_games == 2
    assert_state_equal(actual.policy.state_dict(), expected.policy.state_dict())
    assert_state_equal(actual.policy.optimizer.state_dict(), expected.policy.optimizer.state_dict())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sequential_balanced_loss_gradients_adam_and_nonfinite_returns(device, tmp_path):
    from pvz_rl.policy.sequential_q import balanced_q_loss

    actions = torch.tensor([0, 0, 0, 1, 46, 361], device=device)
    targets = torch.arange(1, 7, device=device, dtype=torch.float64)
    actual = torch.nn.Parameter(torch.zeros(6, 2, device=device, dtype=torch.float64))
    reference = torch.nn.Parameter(actual.detach().clone())
    opt = torch.optim.Adam([actual], lr=3e-4, eps=1e-5)
    ref_opt = torch.optim.Adam([reference], lr=3e-4, eps=1e-5)
    for _ in range(2):
        # Independently enumerate group means and the non-wait head average.
        per_row = [
            (reference[i, 0] - targets[i]) ** 2
            if i < 3
            else ((reference[i, 0] - targets[i]) ** 2 + (reference[i, 1] - targets[i]) ** 2) / 2
            for i in range(6)
        ]
        expected = (sum(per_row[:3]) / 3 + sum(per_row[3:5]) / 2 + per_row[5]) / 3
        loss, _, _ = balanced_q_loss(
            actual[:, 0],
            actual[:, 1],
            targets,
            actions,
            [[0, 3], [0, 2], [0, 1]],
            accepted=torch.ones(6, dtype=torch.bool, device=device),
        )
        torch.testing.assert_close(loss, expected)
        opt.zero_grad()
        ref_opt.zero_grad()
        loss.backward()
        expected.backward()
        torch.testing.assert_close(actual.grad, reference.grad, atol=1e-12, rtol=1e-12)
        assert actual.grad[:3, 1].count_nonzero() == 0
        for param in (actual, reference):
            torch.nn.utils.clip_grad_norm_([param], 0.5)
        opt.step()
        ref_opt.step()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
        assert_state_equal(opt.state_dict(), ref_opt.state_dict())
    with pytest.raises(FloatingPointError, match="Non-finite"):
        buffer = CompleteGameBuffer(tmp_path / "nonfinite", 1)
        rows = np.zeros(1, buffer.dtype)
        rows["active"] = True
        rows["reward"] = np.nan
        buffer.append(rows, [{"entities": [], "globals": [0] * 18}])
        try:
            buffer.finalize()
        finally:
            buffer.close()


@pytest.mark.parametrize("partition", [[slice(None)], [slice(0, 2), slice(2, 5), slice(5, 8)]])
def test_outcome_balancing_independent_hierarchical_control(partition):
    from pvz_rl.policy.sequential_q import balanced_q_loss

    actions = torch.tensor([0, 0, 1, 46, 91, 136, 361, 362])
    accepted = torch.tensor([1, 1, 1, 0, 0, 0, 1, 0], dtype=torch.bool)
    counts = [[0, 2], [3, 1], [1, 1]]
    predictions = torch.arange(16, dtype=torch.float64).reshape(8, 2).requires_grad_()
    reference = predictions.detach().clone().requires_grad_()
    targets = torch.arange(8, dtype=torch.float64)
    row_errors = [
        (reference[index, 0] - targets[index]).square()
        if index < 2
        else (
            (reference[index, 0] - targets[index]).square()
            + (reference[index, 1] - targets[index]).square()
        )
        / 2
        for index in range(8)
    ]
    expected = (
        (row_errors[0] + row_errors[1]) / 2
        + row_errors[2] / 2
        + sum(row_errors[3:6]) / 6
        + (row_errors[6] + row_errors[7]) / 2
    ) / 3
    loss = sum(
        balanced_q_loss(
            predictions[chunk, 0],
            predictions[chunk, 1],
            targets[chunk],
            actions[chunk],
            counts,
            accepted=accepted[chunk],
        )[0]
        for chunk in partition
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(predictions.grad, reference.grad)


def test_probe_outcomes_share_half_even_with_three_rejections():
    from pvz_rl.learning.objective import probe_loss

    class TablePolicy:
        cfg = load_config()

        def tile_values(self, tiles, context, branches):
            return tiles

    counts = np.zeros((2, 10, 2), dtype=np.int64)
    counts[:, 2] = [3, 1]
    probes = torch.tensor(
        [[[1, 1, 46, target, accepted]] for target, accepted in ((1, 1), (3, 0), (5, 0), (7, 0))],
        dtype=torch.float32,
    )
    first = torch.zeros(4, 10, requires_grad=True)
    second = torch.zeros(4, 45, requires_grad=True)
    total = 0
    for start, stop in ((0, 1), (1, 4)):
        loss = probe_loss(
            TablePolicy(),
            first[start:stop],
            second[start:stop],
            torch.empty(stop - start, 0),
            probes[start:stop],
            counts,
            1,
        )
        loss.backward()
        total += loss.detach()
    torch.testing.assert_close(total, torch.tensor(2.5))
    torch.testing.assert_close(first.grad[:, 2], torch.tensor([-0.25, -1 / 12, -1 / 12, -1 / 12]))
    torch.testing.assert_close(second.grad[:, 0], first.grad[:, 2])


def test_counterfactual_and_ranking_gradients_match_partitioned_cohort():
    from pvz_rl.learning.objective import demonstration_rank_loss, probe_loss, ranking_counts

    class TablePolicy:
        cfg = load_config()

        def tile_values(self, tiles, context, branches):
            return tiles[torch.arange(len(branches)), branches - 1]

    policy = TablePolicy()
    probes = torch.tensor(
        [
            [[1, 1, 46, 1.0, 1], [1, 1, 361, -1.0, 0], [1, 0, 2, 0.5, 1]],
            [[1, 1, 0, -1.0, 1], [1, 1, 91, 1.0, 0], [0, 0, 0, 0, 0]],
        ]
    )
    counts = np.zeros((2, 10, 2), np.int64)
    counts[0, [0, 2], 1] = 1
    counts[0, [3, 9], 0] = 1
    counts[1, [1, 2], 1] = 1
    counts[1, [3, 9], 0] = 1
    outcomes = []
    for partition in ([slice(None)], [slice(0, 1), slice(1, 2)]):
        q = torch.zeros(2, 10, requires_grad=True)
        tiles = torch.zeros(2, 9, 45, requires_grad=True)
        total = 0
        for ix in partition:
            loss = probe_loss(
                policy, q[ix], tiles[ix], torch.empty(2, 0)[ix], probes[ix], counts, 1.0
            )
            loss.backward()
            total += float(loss.detach())
        outcomes.append((total, q.grad.clone(), tiles.grad.clone()))
    for a, b in zip(*outcomes):
        torch.testing.assert_close(a, b)
    qg, tg = outcomes[0][1:]
    assert torch.count_nonzero(qg) == 4
    assert torch.count_nonzero(tg) == 4
    assert qg[0, 2] < 0 and qg[0, 9] > 0
    assert tg[0, 0, 1] < 0  # Alternative tile gets its own gradient.
    actions = torch.tensor([46, 46])
    accepted = torch.tensor([True, False])
    masks = torch.ones(2, 406, dtype=torch.bool)
    q = torch.zeros(2, 10, requires_grad=True)
    tiles = torch.zeros(2, 9, 45, requires_grad=True)
    rank_counts = ranking_counts(actions, accepted, masks)[0]
    rank, _ = demonstration_rank_loss(
        policy, q, tiles, torch.empty(2, 0), actions, accepted, masks, rank_counts, 0.05
    )
    rank.backward()
    assert q.grad[0, 2] < 0 and tiles.grad[0, 1, 0] < 0
    assert q.grad[1].count_nonzero() == tiles.grad[1].count_nonzero() == 0
    assert torch.all(q.grad[0, torch.arange(10) != 2] > 0)
