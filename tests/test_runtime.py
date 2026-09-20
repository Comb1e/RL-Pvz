import copy
import json

import numpy as np
import pytest
import torch
from gymnasium import spaces
from pvz_game import LevelSpec, Place, Spawn
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.logger import configure

from pvz_rl.config import research_config, runtime_settings, validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.runtime import MASK_INFO_KEY, configure_rollout_buffer, rollout_buffer_class
from pvz_rl.timing import TrainingTimings
from pvz_rl.training import build_model, vector_env


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_minibatches_match_stock_shapes_values_rng_and_resets(masked, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    buffers = [
        rollout_buffer_class(masked, enabled)(
            3,
            spaces.Box(-1, 1, (7,), dtype=np.float32),
            spaces.Discrete(5),
            n_envs=2,
            device=device,
        )
        for enabled in (False, True)
    ]
    rng = np.random.default_rng(18)
    for cycle in range(2):
        for buffer in buffers:
            buffer.reset()
        assert buffers[1]._device_samples is None
        for name in ("observations", "actions", "values", "log_probs", "advantages", "returns"):
            values = rng.normal(size=getattr(buffers[0], name).shape)
            for buffer in buffers:
                getattr(buffer, name)[:] = values
        if masked:
            masks = rng.integers(2, size=buffers[0].action_masks.shape)
            for buffer in buffers:
                buffer.action_masks[:] = masks
        for buffer in buffers:
            buffer.full = True
        # Multiple epochs and an incomplete last batch exercise ordering and reuse.
        for batch_size in (4, None, 1):
            batches = []
            next_random = []
            for buffer in buffers:
                np.random.seed(900 + cycle)
                batches.append(list(buffer.get(batch_size)))
                next_random.append(np.random.rand())
            assert next_random[0] == next_random[1]
            for reference, cached in zip(*batches, strict=True):
                for a, b in zip(reference, cached, strict=True):
                    assert a.dtype == b.dtype and a.shape == b.shape
                    torch.testing.assert_close(a, b, atol=0, rtol=0)


def plain(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["runtime"] = dict.fromkeys(runtime_settings(cfg), False)
    return cfg


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("condition", ["masked", "hybrid"])
def test_transport_preserves_trajectories_curriculum_and_reset_masks(smoke_cfg, workers, condition):
    smoke_cfg["training"].update(n_envs=workers, total_steps=16)
    smoke_cfg["environment"]["cutoff_seconds"] = 1
    configs = [plain(smoke_cfg), smoke_cfg]
    traces = []
    for cfg in configs:
        env = vector_env(cfg, condition, 101)
        try:
            env.seed(400)
            trace = [env.reset()]
            for _ in range(12):
                masks = get_action_masks(env)
                actions = np.array([np.flatnonzero(mask)[-1] for mask in masks])
                obs, rewards, dones, infos = env.step(actions)
                trace.extend([masks, obs, rewards, dones])
                for info in infos:
                    assert MASK_INFO_KEY not in info
                    if "episode_metrics" in info:
                        trace.append(info["terminal_observation"])
                        trace.append(info["episode_metrics"])
                        assert info["TimeLimit.truncated"]
            assert env.get_attr("progress") == [12 * workers] * workers
            traces.append(trace)
        finally:
            env.close()
    for a, b in zip(*traces, strict=True):
        if isinstance(a, dict):
            assert a == b
        else:
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "scenario,status",
    [
        (LevelSpec("empty"), "won"),
        (LevelSpec("breach", (Spawn(1, "basic", 0, x=0),), mowers=False), "lost"),
    ],
)
def test_transport_natural_terminal_uses_reset_mask(smoke_cfg, scenario, status, monkeypatch):
    smoke_cfg["environment"]["cutoff_seconds"] = 60
    env = vector_env(smoke_cfg, "masked", 101)
    try:
        env.set_options({"scenario": scenario})
        env.reset()
        for _ in range(120):
            _, _, dones, infos = env.step([0])
            if dones[0]:
                break
        assert dones[0] and infos[0]["status"] == status
        assert not infos[0]["TimeLimit.truncated"]
        assert "terminal_observation" in infos[0]
        expected = env.venv.env_method("action_masks")[0]

        # Fetches must use the cached reset observation, with no mask RPC at all.
        def reject(*args, **kwargs):
            raise AssertionError("unexpected worker RPC")

        monkeypatch.setattr(env.venv, "env_method", reject)
        np.testing.assert_array_equal(get_action_masks(env)[0], expected)
        returned = env.env_method("action_masks", indices=0)[0]
        returned[:] = False
        np.testing.assert_array_equal(get_action_masks(env)[0], expected)
    finally:
        env.close()


@pytest.mark.learning
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("condition", ["masked", "unmasked", "sparse", "mixed", "hybrid"])
def test_transport_preserves_learned_weights_and_optimizer(smoke_cfg, tmp_path, condition, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    smoke_cfg["training"].update(device=device, n_epochs=2)
    torch.set_num_threads(1)
    states = []
    for cfg in (plain(smoke_cfg), smoke_cfg):
        env = vector_env(cfg, condition, 103)
        try:
            model = build_model(cfg, condition, env, 103)
            model.set_logger(configure(str(tmp_path), []))
            model.learn(128)
            states.append(
                (
                    copy.deepcopy(model.policy.state_dict()),
                    copy.deepcopy(model.policy.optimizer.state_dict()),
                )
            )
            policy, optimizer = model.policy, model.policy.optimizer
            configure_rollout_buffer(model, cfg["conditions"][condition]["masked"], False)
            assert model.policy is policy and model.policy.optimizer is optimizer
        finally:
            env.close()
    for name, value in states[0][0].items():
        torch.testing.assert_close(value, states[1][0][name], atol=0, rtol=0)
    assert states[0][1]["param_groups"] == states[1][1]["param_groups"]
    for key, state in states[0][1]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, states[1][1]["state"][key][name], atol=0, rtol=0)


def test_legacy_runtime_defaults_and_compatibility(cfg):
    old = copy.deepcopy(cfg)
    old.pop("runtime")
    validate_config(old)
    assert runtime_settings(old) == runtime_settings(cfg)
    assert research_config(old) == research_config(cfg) == research_config(plain(cfg))
    cfg["runtime"]["coalesce_masks"] = "false"
    with pytest.raises(ValueError, match="Runtime"):
        validate_config(cfg)


def test_phase_timing_excludes_validation_and_captures_final_update():
    ticks = iter([1, 3, 6, 100, 104, 105])
    timing = TrainingTimings(clock=lambda: next(ticks))
    timing.end_update()  # No preceding update at startup.
    timing.begin_collection()
    timing.end_collection()
    timing.end_update()
    timing.begin_collection()  # Long validation gap must not enter either phase.
    timing.end_collection()
    timing.end_update()
    timing.end_update()  # Idempotent.
    assert timing.snapshot() == {
        "collection_seconds": 6,
        "optimization_seconds": 4,
        "last_collection_seconds": 4,
        "last_optimization_seconds": 1,
    }


def test_legality_cache_requeries_only_after_relevant_public_change(cfg, monkeypatch):
    env = PvZEnv(cfg)
    env.reset(
        seed=42,
        options={"scenario": LevelSpec("quiet", (Spawn(2000, "basic", 0),), initial_sun=200)},
    )
    calls = []
    original = env.game.legal_actions

    def query():
        calls.append(env.public.tick)
        return original()

    monkeypatch.setattr(env.game, "legal_actions", query)
    expected = [env.game.validate_action(a).accepted for a in env.codec.actions]
    np.testing.assert_array_equal(env.action_masks(), expected)
    for _ in range(5):
        env.step(0)
        np.testing.assert_array_equal(env.action_masks(), expected)
    assert calls == [0]  # Time passes without changing legality.
    # Plant occupancy, sun thresholds, and cooldown change. Then the bomb destroys itself.
    env.step(env.codec.encode(Place("cherry_bomb", 2, 0)))
    for _ in range(4):
        expected = [env.game.validate_action(a).accepted for a in env.codec.actions]
        np.testing.assert_array_equal(env.action_masks(), expected)
        env.step(0)
    assert len(calls) >= 3
    env.reset(seed=43)
    np.testing.assert_array_equal(
        env.action_masks(), [env.game.validate_action(a).accepted for a in env.codec.actions]
    )
    assert calls[-1] == 0


@pytest.mark.parametrize("level", ["easy", "standard", "hard"])
def test_cached_and_reference_game_hashes_rewards_and_masks_match(cfg, level):
    from pvz_rl.frozen_baseline import choose_action

    reference, cached = PvZEnv(plain(cfg), level=level), PvZEnv(cfg, level=level)
    left, _ = reference.reset(seed=42)
    right, _ = cached.reset(seed=42)
    while reference.state == "running":
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(reference.action_masks(), cached.action_masks())
        action = reference.codec.encode(choose_action(reference.public_board()))
        left, r1, term1, trunc1, info1 = reference.step(action)
        right, r2, term2, trunc2, info2 = cached.step(action)
        assert (r1, term1, trunc1, info1) == (r2, term2, trunc2, info2)
        assert reference.game.state_hash() == cached.game.state_hash()
    assert reference.state == "won"  # Independently known seed-42 control.


@pytest.mark.learning
def test_paired_benchmark_records_phases_and_preserves_protocol(smoke_cfg, tmp_path):
    from pvz_rl.benchmark import benchmark

    result = benchmark(
        smoke_cfg,
        tmp_path / "bench",
        steps=64,
        workers=[1],
        devices=["cpu"],
        repeats=1,
        compare_runtime=True,
    )
    assert result["n_envs"] == 1 and result["device"] == "cpu"
    rows = json.loads((tmp_path / "bench/measurements.json").read_text())
    complete = [r for r in rows if r["state"] == "complete"]
    assert len(complete) == 2
    for row in complete:
        assert row["steps"] == row["warmup_steps"] == 64
        assert 0 < row["collection_seconds"] < row["seconds"]
        assert 0 < row["optimization_seconds"] < row["seconds"]
    paired = json.loads((tmp_path / "bench/runtime-comparison.json").read_text())
    assert paired[0]["identical_policy_weights"]
