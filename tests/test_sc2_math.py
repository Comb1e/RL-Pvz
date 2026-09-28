"""Independent sequential-Q math, causal storage and exact interruption controls."""

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
    critic = deepcopy(full.policy.optimizer.state_dict())
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
    assert_state_equal(critic, m.policy.optimizer.state_dict())
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


def test_sequential_q_independent_math_controls():
    import math

    for budget in (0, 0.001, 0.1, 1):
        alpha = 1 - math.sqrt(1 - budget)
        assert 1 - (1 - alpha) ** 2 == pytest.approx(budget)
        probabilities = []
        for k, tiles in enumerate((2, 3, 7)):
            species = alpha / 3 + (1 - alpha) * (k == 1)
            probabilities.extend(
                species * (alpha / tiles + (1 - alpha) * (t == 0)) for t in range(tiles)
            )
        assert sum(probabilities) == pytest.approx(1)
    errors = [4] * 5 + [1] * 2 + [0]
    weights = [8 / 15] * 5 + [8 / 6] * 2 + [8 / 3]
    assert sum(e * w for e, w in zip(errors, weights)) / 8 == pytest.approx(5 / 3)
    alpha = 1 - math.sqrt(0.9)
    assert (1 - alpha) + alpha * (-2 + 1 + 0.25) / 3 == pytest.approx(0.935854122563)
    rewards = [0.01, -0.02, 1.0]
    assert [sum(rewards[i:]) for i in range(3)] == pytest.approx([0.99, 0.98, 1.0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sequential_balanced_loss_gradients_adam_and_partial_batches(device):
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
        loss, _, _ = balanced_q_loss(actual[:, 0], actual[:, 1], targets, actions, [3, 2, 1], 6)
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
    # Last short batch retains denominator four, not its own length two.
    full = balanced_q_loss(actual[:, 0], actual[:, 1], targets, actions, [3, 2, 1], 4)[0]
    pieces = sum(
        balanced_q_loss(actual[a:b, 0], actual[a:b, 1], targets[a:b], actions[a:b], [3, 2, 1], 4)[0]
        for a, b in ((0, 4), (4, 6))
    )
    torch.testing.assert_close(full, pieces)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        buffer = CompleteGameBuffer(__import__("tempfile").mkdtemp(), 1)
        rows = np.zeros(1, buffer.dtype)
        rows["active"] = True
        rows["reward"] = np.nan
        buffer.append(rows, [{"entities": [], "globals": [0] * 18}])
        try:
            buffer.finalize()
        finally:
            buffer.close()
