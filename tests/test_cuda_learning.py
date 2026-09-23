"""Independent CPU/GPU encoder, reward, storage and optimization controls."""

import copy

import numpy as np
import pytest
import torch
from gymnasium import spaces
from pvz_game import LevelSpec, Spawn
from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES, InitialPlant
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config
from pvz_rl.cuda_buffer import TensorRolloutBuffer
from pvz_rl.env import PvZEnv
from pvz_rl.training import build_model, vector_env


@pytest.fixture
def gpu_cfg():
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = load_config()
    cfg["simulation"] = {"backend": "cuda"}
    cfg["training"].update(
        n_envs=2,
        rollout_size=256,
        device="cuda",
        rollout_steps_per_env=128,
        batch_size=64,
        target_kl=0,
    )
    cfg["visualization"].update(enabled=False, demos=False, videos=False)
    return cfg


@pytest.mark.parametrize("condition", ["masked"])
def test_observations_rewards_and_metrics_against_cpu(gpu_cfg, condition):
    from pvz_rl.cuda_features import REWARD_FIELDS, CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch as CudaBatch

    cfg = copy.deepcopy(gpu_cfg)
    scenario = LevelSpec(
        "mixed",
        tuple(Spawn(1 + (j % 3), kind, j % 5, x=2800) for j, kind in enumerate(ZOMBIE_TYPES * 3)),
        initial_sun=9990,
        plants=tuple(InitialPlant(kind, j % 5, j // 5) for j, kind in enumerate(PLANT_TYPES)),
    )
    env = PvZEnv(cfg, condition=condition)
    env.reset(seed=15, options={"scenario": scenario})
    batch = CudaBatch(1, zombie_capacity=15, max_step_ticks=1)
    batch.reset([scenario], [15])
    cp = batch.cp
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        feature = CudaFeatures(batch, cfg, condition)
        feature.encode()
        for tick in range(600):
            np.testing.assert_allclose(
                feature.observations.get()[0], env.encoder.encode(env.public), atol=1e-7, rtol=1e-6
            )
            action = 0
            obs, reward, done, truncated, info = env.step(action)
            feature.step(cp.asarray([action], dtype=cp.int64))
            assert float(feature.rewards.get()[0]) == pytest.approx(reward, abs=2e-6)
            for key, value in zip(REWARD_FIELDS, feature.parts.get()[0]):
                assert value == pytest.approx(info["reward_parts"][key], abs=1e-10), key
            if done or truncated:
                break


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("gamma", [0.999, 0.9999])
def test_gae_and_shuffle_match_sb3(device, gamma):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    obs = spaces.Box(-1, 1, (5,), dtype=np.float32)
    action = spaces.Discrete(4)
    reference = RolloutBuffer(8, obs, action, n_envs=3, gamma=gamma, gae_lambda=0.98, device=device)
    tensor = TensorRolloutBuffer(
        8, obs, action, n_envs=3, gamma=gamma, gae_lambda=0.98, device=device
    )
    rng = np.random.default_rng(15)
    for _ in range(8):
        o = rng.normal(size=(3, 5)).astype(np.float32)
        a = rng.integers(0, 4, size=(3, 1)).astype(np.float32)
        r = rng.normal(size=3).astype(np.float32)
        starts = rng.integers(0, 2, size=3).astype(np.float32)
        v, lp = [
            torch.tensor(rng.normal(size=3), device=device, dtype=torch.float32) for _ in range(2)
        ]
        reference.add(o, a, r, starts, v, lp)
        tensor.add(
            *(torch.as_tensor(x, device=device) for x in (o, a, r, starts)),
            v,
            lp,
            torch.ones(3, 4, dtype=torch.bool, device=device),
        )
    last = torch.tensor([0.5, 0.1, 0.2], device=device)
    done = np.array([False, True, False])
    reference.compute_returns_and_advantage(last, done)
    tensor.compute_returns_and_advantage(last, torch.as_tensor(done, device=device))
    np.testing.assert_allclose(tensor.advantages.cpu(), reference.advantages, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(tensor.returns.cpu(), reference.returns, atol=2e-6, rtol=1e-6)
    np.random.seed(102)
    expected = list(reference.get(6))
    np.random.seed(102)
    actual = list(tensor.get(6))
    for left, right in zip(expected, actual):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("condition", ["masked"])
def test_cuda_collect_update_and_timeout(gpu_cfg, condition):
    cfg = copy.deepcopy(gpu_cfg)
    cfg["environment"]["cutoff_seconds"] = 1
    env = vector_env(cfg, condition, 101)
    try:
        model = build_model(cfg, condition, env, 101)
        model.set_logger(configure(format_strings=[]))
        model.learn(256)
        assert model.num_timesteps == 256
        assert model._n_updates == cfg["training"]["n_epochs"]
        assert model.rollout_buffer.observations.is_cuda
        assert torch.isfinite(model.rollout_buffer.returns).all()
        assert np.isfinite(model.logger.name_to_value["train/loss"])
    finally:
        env.close()


@pytest.mark.parametrize("condition", ["masked"])
def test_fixed_rollout_losses_and_optimizer_match_stock(gpu_cfg, condition, monkeypatch):
    configs = [copy.deepcopy(gpu_cfg), copy.deepcopy(gpu_cfg)]
    configs[0]["training"]["n_envs"] = 1
    configs[0]["training"]["rollout_steps_per_env"] = 256
    envs = [vector_env(c, condition, 12) for c in configs]
    try:
        from sb3_contrib import MaskablePPO

        from pvz_rl.spatial_policy import SpatialFeatures, SpatialGroupedPolicy

        # Independent upstream optimizer on supplied data; no CPU collection/training run.
        t = configs[0]["training"]
        for c in configs:
            c["training"]["exploration"].update(type_coef=0, tile_coef=0)
        reference = MaskablePPO(
            SpatialGroupedPolicy,
            PvZEnv(configs[0]),
            device="cuda",
            seed=12,
            n_steps=256,
            batch_size=t["batch_size"],
            n_epochs=t["n_epochs"],
            learning_rate=t["learning_rate"],
            gamma=configs[0]["training"]["gamma"],
            gae_lambda=t["gae_lambda"],
            clip_range=t["clip_range"],
            ent_coef=0,
            policy_kwargs={
                "net_arch": {"pi": t["hidden_sizes"], "vf": t["hidden_sizes"]},
                "features_extractor_class": SpatialFeatures,
                "features_extractor_kwargs": {"layout_cfg": configs[0]},
            },
        )
        reference.policy.initialize_dig_logit(configs[0]["policy"]["initial_dig_logit"])
        models = [reference, build_model(configs[1], condition, envs[1], 12)]
        for m in models:
            m.set_logger(configure(format_strings=[]))
            m._current_progress_remaining = 1.0
        # Fill flattened env-major data identically even though vector shapes differ.
        rng = np.random.default_rng(14)
        observation, _ = PvZEnv(configs[0]).reset(seed=4)
        fields = {
            "observations": np.repeat(observation[None], 256, axis=0),
            "actions": rng.integers(0, 406, size=(256, 1)).astype(np.float32),
        }
        for key in ("values", "log_probs", "advantages", "returns"):
            fields[key] = rng.normal(size=256).astype(np.float32)
        for model in models:
            buffer = model.rollout_buffer
            for key, values in fields.items():
                # get() flattens by environment, then timestep.
                shaped = values.reshape(
                    buffer.n_envs, buffer.buffer_size, *values.shape[1:]
                ).swapaxes(0, 1)
                if isinstance(getattr(buffer, key), torch.Tensor):
                    getattr(buffer, key).copy_(torch.as_tensor(shaped.copy(), device="cuda"))
                else:
                    getattr(buffer, key)[:] = shaped
            if condition == "masked":
                if isinstance(buffer.action_masks, torch.Tensor):
                    buffer.action_masks.fill_(True)
                else:
                    buffer.action_masks[:] = True
            buffer.full = True
        # Keep upstream PPO's loss, reduction, shuffle and Adam calculations.
        # Only adapt its optimizer/clipping interface to two disjoint groups.
        actor_optimizer = reference.policy.optimizer
        critic_optimizer = reference.policy.critic_optimizer

        class IndependentOptimizers:
            param_groups = actor_optimizer.param_groups + critic_optimizer.param_groups

            def zero_grad(self):
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()

            def step(self):
                actor_optimizer.step()
                critic_optimizer.step()

        clip = torch.nn.utils.clip_grad_norm_

        def independent_clip(parameters, limit):
            assert {id(p) for p in parameters} == {id(p) for p in reference.policy.parameters()}
            a = clip(reference.policy.actor_parameters(), limit)
            b = clip(reference.policy.critic_parameters(), limit)
            return torch.maximum(a, b)

        reference.policy.optimizer = IndependentOptimizers()
        np.random.seed(92)
        with monkeypatch.context() as patch:
            patch.setattr(torch.nn.utils, "clip_grad_norm_", independent_clip)
            reference.train()
        np.random.seed(92)
        models[1].train()
        for key, value in models[0].policy.state_dict().items():
            torch.testing.assert_close(
                value, models[1].policy.state_dict()[key], atol=2e-7, rtol=2e-6
            )
        for name in (
            "train/policy_gradient_loss",
            "train/value_loss",
            "train/entropy_loss",
            "train/explained_variance",
        ):
            assert float(models[0].logger.name_to_value[name]) == pytest.approx(
                float(models[1].logger.name_to_value[name]), abs=2e-6
            )
        for a, b in zip(models[0].policy.parameters(), models[1].policy.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-6)
        for expected, actual in (
            (actor_optimizer, models[1].policy.optimizer),
            (critic_optimizer, models[1].policy.critic_optimizer),
        ):
            left, right = expected.state_dict(), actual.state_dict()
            assert left["param_groups"] == right["param_groups"]
            assert left["state"].keys() == right["state"].keys()
            for index, state in left["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        value, right["state"][index][name], atol=2e-7, rtol=2e-6
                    )
    finally:
        for env in envs:
            env.close()


def test_grouped_shared_policy_identity_and_reload(gpu_cfg, tmp_path):
    from pvz_rl.cuda_ppo import CudaMaskablePPO

    cfg = gpu_cfg
    cfg["environment"]["cutoff_seconds"] = 1
    env = vector_env(cfg, "masked", 23)
    try:
        model = build_model(cfg, "masked", env, 23)
        model.set_logger(configure(format_strings=[]))
        policy_id, optimizer_id = id(model.policy), id(model.policy.optimizer)
        critic_id = id(model.policy.critic_optimizer)
        model.learn(256)
        env.env_method("set_curriculum_stage", 4)
        model.learn(256, reset_num_timesteps=False)
        assert (id(model.policy), id(model.policy.optimizer)) == (policy_id, optimizer_id)
        assert id(model.policy.critic_optimizer) == critic_id
        checkpoint = tmp_path / "model.zip"
        model.training_games = 7
        model.curriculum_state = {"stage": 4}
        model.save(checkpoint)
        loaded = CudaMaskablePPO.load(checkpoint, device="cuda")
        assert loaded.training_games == 7
        assert loaded.curriculum_state == {"stage": 4}
        obs = env.reset().clone()
        with torch.no_grad():
            a = model.policy(obs, deterministic=True, action_masks=env.action_masks())[0]
            b = loaded.policy(obs, deterministic=True, action_masks=env.action_masks())[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    finally:
        env.close()


@pytest.mark.learning
def test_cuda_game_budget_report_demos_and_resume(gpu_cfg, tmp_path):
    import json

    from pvz_rl.provenance import file_hash
    from pvz_rl.recordings import open_playback
    from pvz_rl.training import load_policy, train

    cfg = copy.deepcopy(gpu_cfg)
    cfg["training"].update(total_games=2, eval_interval_games=2, validation_schedule="periodic")
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["visualization"].update(enabled=True, demos=True, videos=False)
    run = train(cfg, "masked", 101, tmp_path / "gpu-run", validation_limit=1)
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "complete" and status["training_games"] >= 2
    assert (run / "visualizations/index.html").exists()
    demos = list(run.rglob("*.pvzdemo"))
    assert len(demos) == 3
    for demo in demos:
        playback = open_playback(demo)
        playback.verify()
        assert playback.metadata["checkpoint_sha256"] == file_hash(run / "best.zip")
        assert playback.display_outcome == "truncated"
    model, data = load_policy(run / "final.zip", "cuda")
    assert model.training_games == status["training_games"]
    resumed = train(
        cfg, "masked", 101, tmp_path / "resume", resume=run / "final.zip", validation_limit=1
    )
    after = json.loads((resumed / "status.json").read_text())
    assert after["training_games"] == status["training_games"]
    assert after["steps"] == status["steps"]
    html = (run / "visualizations/index.html").read_text("utf-8")
    assert "cuda" in html.lower() or "shared" in html.lower()


def test_rollout_configuration_and_legacy_cpu_defaults(gpu_cfg):
    from pvz_rl.config import resolve_rollout, simulator, validate_config

    legacy = load_config()
    assert simulator(legacy) == "cuda"
    legacy.pop("simulation")
    assert simulator(legacy) == "cpu"
    for count in (32, 64, 128):
        cfg = copy.deepcopy(gpu_cfg)
        cfg["training"]["n_envs"] = count
        resolve_rollout(cfg, per_env=128)
        assert cfg["training"]["rollout_size"] == count * 128
        validate_config(cfg)
        with pytest.raises(ValueError, match="conflict"):
            resolve_rollout(cfg, per_env=128, total=count * 128 + 1)


def test_gpu_lesson_and_changed_scenarios_match_public_encodings(gpu_cfg):
    from pvz_game import Game, Rules

    from pvz_rl.cuda_features import CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch as CudaBatch
    from pvz_rl.scenarios import scenario

    cfg = copy.deepcopy(gpu_cfg)
    families = ["placement", "saving", "redistributed", "faster", "concentrated"]
    # Development seeds only; formal test/changed-distribution seeds stay untouched.
    levels = [scenario("standard", family, 9, Rules(), cfg) for family in families]
    games = [Game() for _ in levels]
    for game, level in zip(games, levels):
        game.reset(level, 9)
        game.step(ticks=200)
    batch = CudaBatch(len(games), zombie_capacity=200)
    batch.restore([g.snapshot() for g in games])
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        expected = np.stack([features.encoder.encode(g.observe()) for g in games])
        np.testing.assert_allclose(features.observations.get(), expected, atol=1e-7, rtol=1e-6)


def test_gpu_encoding_crowds_order_and_private_schedule(gpu_cfg):
    from pvz_game import Game

    from pvz_rl.cuda_features import CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch as CudaBatch

    cfg = copy.deepcopy(gpu_cfg)
    games = []
    for seed in (8, 9):
        game = Game()
        game.reset(
            LevelSpec(
                f"private-{seed}",
                tuple(
                    [Spawn(1, "basic", 0, x=3000)] * 90 + [Spawn(1000 + seed, "basic", seed % 5)]
                ),
            ),
            seed,
        )
        game.step()
        games.append(game)
    batch = CudaBatch(2, zombie_capacity=91)
    batch.restore([g.snapshot() for g in games])
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        expected = features.encoder.encode(games[0].observe())
        np.testing.assert_array_equal(expected, features.encoder.encode(games[1].observe()))
        before = features.observations.get().copy()
        np.testing.assert_allclose(before, np.stack([expected, expected]), atol=1e-7, rtol=1e-6)
        batch.zombies[:, :90] = batch.zombies[:, :90][:, ::-1].copy()
        features.encode()
        np.testing.assert_array_equal(features.observations.get(), before)
