"""Independent CPU/GPU encoder, reward, storage and optimization controls."""

import copy

import numpy as np
import pytest
import torch
from pvz_game import LevelSpec, Spawn
from pvz_game.config import PLANT_TYPES, ZOMBIE_TYPES, InitialPlant

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv


@pytest.fixture
def gpu_cfg():
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = load_config()
    cfg["simulation"] = {"backend": "cuda"}
    cfg["training"].update(
        n_envs=2,
        device="cuda",
        batch_size=64,
        target_kl=0,
    )
    cfg["visualization"].update(enabled=False, demos=False, videos=False)
    return cfg


@pytest.mark.parametrize("condition", ["masked"])
def test_observations_rewards_and_metrics_against_cpu(gpu_cfg, condition):
    from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures
    from pvz_rl.envs.cuda_lessons import LessonCudaBatch as CudaBatch

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


def test_gpu_lesson_and_changed_scenarios_match_public_encodings(gpu_cfg):
    from pvz_game import Game, Rules

    from pvz_rl.envs.cuda_features import CudaFeatures
    from pvz_rl.envs.cuda_lessons import LessonCudaBatch as CudaBatch
    from pvz_rl.envs.scenarios import scenario

    cfg = copy.deepcopy(gpu_cfg)
    families = ["saving", "redistributed", "faster", "concentrated"]
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

    from pvz_rl.envs.cuda_features import CudaFeatures
    from pvz_rl.envs.cuda_lessons import LessonCudaBatch as CudaBatch

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
