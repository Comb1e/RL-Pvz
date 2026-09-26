"""Independent sequential-Q math, causal storage and exact interruption controls."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.learning.cohort import CohortPhase
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.cuda_q import CudaSequentialQ
from pvz_rl.learning.training import build_model, vector_env
from pvz_rl.policy.event_memory import EventMemory


@pytest.mark.parametrize("ram", [0, 1 << 20])
def test_host_storage_actual_returns_context_causality_and_disk_restore(tmp_path, ram):
    cfg = load_config()
    cfg["training"]["n_envs"] = 2
    from pvz_game import Rules

    memory = EventMemory(cfg, Rules(), 2, "cpu")
    buffer = CompleteGameBuffer(tmp_path / "buffer", 286, 48, 2, ram_bytes=ram, block_rows=4)
    contexts = []
    for step in range(5):
        obs = torch.zeros(2, 286)
        obs[:, 270] = step
        masks = torch.zeros(2, 406, dtype=torch.bool)
        masks[:, 0] = True
        ids = torch.arange(2 * step, 2 * step + 2)
        c = memory.observe(
            obs, masks, torch.zeros(2), torch.full((2,), step == 0), torch.full((2,), step), ids
        )
        contexts.append(c.clone())
        rows = np.zeros(2, dtype=buffer.dtype)
        rows["observation"] = obs
        rows["mask"] = np.packbits(masks.numpy(), axis=-1, bitorder="little")
        rows["tick"] = step
        rows["reset"] = step == 0
        rows["env"] = [0, 1]
        rows["active"] = [True, step < 3]
        rows["reward"] = [step + 1, 10 * (step + 1)]
        rows["branch_value"] = 0.5
        rows["ids"] = memory.ids
        rows["counts"] = memory.counts
        rows["starts"] = memory.starts
        buffer.append(rows)
    errors = buffer.finalize()
    data = buffer.take(np.arange(10))
    np.testing.assert_array_equal(data["target"][::2], [15, 14, 12, 9, 5])
    np.testing.assert_array_equal(data["target"][1:6:2], [60, 50, 30])
    assert errors["wait"]["count"] == 8 and errors["plant"]["mse"] is None
    context = buffer.batch([8, 9], "cpu")["context"]
    for name in ("tokens", "valid", "counts", "starts"):
        torch.testing.assert_close(getattr(context, name), getattr(contexts[-1], name))
    state = buffer.save(tmp_path / "saved")
    restored = CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "restore")
    np.testing.assert_array_equal(restored.take(np.arange(10)), data)
    assert restored.ram_used <= ram
    restored.close()
    buffer.close()


def tiny_config():
    c = load_config()
    c["training"].update(n_envs=2, n_epochs=1, batch_size=128)
    c["environment"]["cutoff_seconds"] = 1
    c["visualization"].update(live_enabled=False, enabled=False, demos=False)
    c["training"]["performance"].update(compile_kernels=False, telemetry=False)
    return c


class InterruptOnce(BaseCallback):
    def __init__(self, after):
        super().__init__()
        self.after = after

    def _on_step(self):
        return self.n_calls < self.after


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


@pytest.mark.parametrize("interrupt_after", [20, None])
def test_interrupted_collection_resume_matches_uninterrupted_optimizer_and_rng(
    tmp_path, interrupt_after
):
    c = tiny_config()
    e = vector_env(c, "masked", 102)
    full = build_model(c, "masked", e, 102)
    full.learn(1)
    if interrupt_after is None:
        full.learn(1, reset_num_timesteps=False)
    expected = deepcopy(full.policy.state_dict())
    optim = deepcopy(full.policy.optimizer.state_dict())
    hashes = [e.batch.state_hash(i) for i in range(2)]
    e.close()
    e = vector_env(c, "masked", 102)
    partial = build_model(c, "masked", e, 102)
    if interrupt_after is None:
        partial.learn(1)
        assert partial.phase == CohortPhase.IDLE
    else:
        with pytest.raises(KeyboardInterrupt):
            partial.learn(1, callback=InterruptOnce(interrupt_after))
        assert partial.phase == CohortPhase.COLLECT
    partial.save(tmp_path / "interrupted.zip")
    e.close()
    resume_cfg = deepcopy(c)
    resume_cfg["training"]["performance"]["token_cache_gib"] = 0
    e = vector_env(resume_cfg, "masked", 102)
    resumed = CudaSequentialQ.load(tmp_path / "interrupted.zip", env=e, device="cuda")
    resumed.learn(1, reset_num_timesteps=False)
    assert_state_equal(expected, resumed.policy.state_dict())
    assert_state_equal(optim, resumed.policy.optimizer.state_dict())
    assert hashes == [e.batch.state_hash(i) for i in range(2)]
    assert resumed.training_games == (4 if interrupt_after is None else 2)
    assert resumed.cohort_metrics["cache_hits"] == 0
    e.close()


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
    name = "_q_step"
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
    m = CudaSequentialQ.load(tmp_path / "partial.zip", env=env, device="cuda")
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
    CudaSequentialQ.load(path, device="cuda")
    # Incomplete/corrupt archives cannot silently start a different RNG stream.
    damaged = tmp_path / "damaged.zip"
    with ZipFile(path) as inp, ZipFile(damaged, "w") as out:
        for name in inp.namelist():
            if name != "cohort-state.pt":
                out.writestr(name, inp.read(name))
    with pytest.raises(ValueError, match="runtime state"):
        CudaSequentialQ.load(damaged, device="cuda")
    env.close()


def test_resume_at_game_ceiling_finishes_pending_optimization(tmp_path, monkeypatch):
    import signal

    from pvz_rl.learning.training import load_policy, train

    cfg = tiny_config()
    cfg["training"]["total_games"] = 2
    train(cfg, "masked", 101, tmp_path / "full", validation_limit=1)
    expected, _ = load_policy(tmp_path / "full/final.zip", "cuda")
    original = CudaSequentialQ._q_step
    triggered = []

    def interrupt(self, data):
        original(self, data)
        if not triggered:
            triggered.append(True)
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(CudaSequentialQ, "_q_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, tmp_path / "partial", validation_limit=1)
    monkeypatch.setattr(CudaSequentialQ, "_q_step", original)
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


@pytest.mark.parametrize("cache_blocks", [0, 1, 8])
@pytest.mark.parametrize("ram", [0, 1 << 20])
def test_raw_token_cache_prefetch_and_partial_batches_match_independent_control(
    tmp_path, cache_blocks, ram
):
    from pvz_rl.learning.transfers import BatchPrefetch

    buffer = CompleteGameBuffer(tmp_path / "trajectory", 286, 48, 2, block_rows=8, ram_bytes=ram)
    rng = np.random.default_rng(141)
    rows = np.zeros(42, dtype=buffer.dtype)
    rows["observation"] = rng.normal(size=(42, 286))
    rows["previous"] = rng.integers(0, 406, 42)
    rows["tick"] = np.arange(42) * 3001
    rows["reset"][:2] = True
    rows["env"] = np.arange(42) % 2
    rows["active"] = True
    rows["mask"] = 255
    rows["action"] = 1
    rows["reward"] = rng.normal(size=42)
    for i in range(42):
        valid = min(i // 2 + 1, 48)
        rows["ids"][i, :valid] = np.arange(i, i - valid * 2, -2)
        rows["counts"][i, :valid] = 1
        rows["starts"][i, :valid] = rows["tick"][rows["ids"][i, :valid]]
    buffer.append(rows)
    buffer.finalize()
    buffer.configure_cache("cuda", cache_blocks * 8 * 289 * 4 / 1024**3)
    assert buffer.cache.size == min(42, 8 * cache_blocks)
    order = rng.permutation(42)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    prefetch = BatchPrefetch(buffer, order, 11, "cuda")
    try:
        for at, batch in enumerate(prefetch):
            ids = order[at * 11 : (at + 1) * 11]
            expected = np.zeros((len(ids), 48, 289), np.float32)
            for j, row in enumerate(ids):
                for k in range(48):
                    if rows["counts"][row, k]:
                        source = rows[rows["ids"][row, k]]
                        expected[j, k, :286] = source["observation"]
                        expected[j, k, 286:] = [source["previous"], source["reset"], source["tick"]]
            np.testing.assert_array_equal(batch["context"].tokens.cpu(), expected)
            control = buffer.batch(ids, "cpu")
            # Compare authoritative rows without using batch/context reconstruction.
            torch.testing.assert_close(
                batch["target"].cpu(), torch.from_numpy(buffer.take(ids)["target"].copy())
            )
            if control:
                torch.testing.assert_close(batch["context"].tokens.cpu(), control["context"].tokens)
    finally:
        prefetch.close()
        buffer.close()
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
        cuda_rng, torch.cuda.get_rng_state()
    )


def test_token_cache_reserves_two_gib_and_falls_back(tmp_path, monkeypatch):
    from pvz_rl.learning.transfers import VRAM_RESERVE, TokenCache

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (VRAM_RESERVE, 8 * 1024**3))
    cache = TokenCache("cuda", 8, 289, 2 * 1024**3)
    cache.append(torch.zeros(2, 289, device="cuda"))
    assert cache.full and not cache.blocks and cache.size == 0


def test_token_storage_math_and_transport_settings_do_not_change_resume_protocol():
    from pvz_rl.learning.training_requirements import resume_protocol

    cfg = load_config()
    changed = deepcopy(cfg)
    changed["training"]["performance"]["token_cache_gib"] = 0
    assert resume_protocol(cfg, "masked") == resume_protocol(changed, "masked")
    from pvz_game import Rules

    width = EventMemory(cfg, Rules(), 1, "cpu").width
    amount = 32768 * width * np.dtype(np.float32).itemsize
    assert 2 * 1024**3 // amount == 56
    assert amount * 56 == 2121269248
    assert width * 4 * 1024 * 48 == 56819712


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
        buffer = CompleteGameBuffer(__import__("tempfile").mkdtemp(), 286, 48, 1)
        rows = np.zeros(1, buffer.dtype)
        rows["active"] = True
        rows["reward"] = np.nan
        buffer.append(rows)
        try:
            buffer.finalize()
        finally:
            buffer.close()


def test_wait_losses_enable_planting_and_both_heads_fit(tmp_path):
    c = tiny_config()
    env = vector_env(c, "masked", 101)
    try:
        model = build_model(c, "masked", env, 101)
        model.set_logger(configure(str(tmp_path), []))
        model.learn(1)
        assert model.num_timesteps == 200 and model.training_games == 2
        assert model.prefit_errors["wait"]["mse"] == 4
        assert model.prefit_errors["plant"]["count"] == 0
        assert model.policy.optimizer.state
        actions, _ = model.policy.sample_actions(
            env.reset(), env.action_masks(), deterministic=True
        )
        assert bool(((actions > 0) & (actions < A.dig_start)).all())
        model.learn(1, reset_num_timesteps=False)
        assert model.prefit_errors["plant"]["count"] > 0
        assert model.prefit_errors["plant"]["tile_mse"] is not None
        assert model._stats["q_optimizer_steps"] > 0
    finally:
        env.close()


def test_old_checkpoint_rejected_before_any_deserialization(tmp_path, monkeypatch):
    from zipfile import ZipFile

    import stable_baselines3.common.base_class as base

    path = tmp_path / "retired.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("data", '{"policy_class": "retired"}')
    monkeypatch.setattr(
        base, "load_from_zip_file", lambda *a, **kw: pytest.fail("retired class loaded")
    )
    with pytest.raises(ValueError, match="fresh models"):
        CudaSequentialQ.load(path)


def test_cache_prefetch_optimizer_and_rng_match_uncached(tmp_path):
    results = []
    for cache in (0, 2):
        cfg = tiny_config()
        cfg["training"]["performance"]["token_cache_gib"] = cache
        env = vector_env(cfg, "masked", 102)
        try:
            model = build_model(cfg, "masked", env, 102)
            with torch.no_grad():
                model.policy.branch_head[-1].bias[1] = 0.1
            model.learn(1)
            results.append(
                (
                    deepcopy(model.policy.state_dict()),
                    deepcopy(model.policy.optimizer.state_dict()),
                    torch.cuda.get_rng_state(),
                    torch.get_rng_state(),
                )
            )
        finally:
            env.close()
    assert_state_equal(results[0], results[1])
