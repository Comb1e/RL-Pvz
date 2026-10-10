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
            m.policy.tile_offsets[0] = 0.1
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
        per_row = [(reference[index, 0] - targets[index]) ** 2 for index in range(6)]
        expected = (sum(per_row[:3]) / 3 + sum(per_row[3:5]) / 2 + per_row[5]) / 3
        loss, _ = balanced_q_loss(
            actual[:, 0],
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
    row_errors = [(reference[index, 0] - targets[index]).square() for index in range(8)]
    expected = (
        (row_errors[0] + row_errors[1]) / 2
        + row_errors[2] / 2
        + sum(row_errors[3:6]) / 6
        + (row_errors[6] + row_errors[7]) / 2
    ) / 3
    loss = sum(
        balanced_q_loss(
            predictions[chunk, 0],
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
