"""Rollout length is not an episode limit; action diagnostics exclude waiting."""

import numpy as np
import pytest
import torch
from pvz_game import Dig, LevelSpec, Place, Spawn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from pvz_rl.config import gpu_defaults, load_config
from pvz_rl.env import PvZEnv
from pvz_rl.metrics import agent_action_count, mean_agent_actions
from pvz_rl.training import build_model, vector_env


def test_defaults_and_archived_action_counts():
    cfg = load_config()
    assert cfg == load_config("configs/train.toml")
    assert cfg["training"]["n_envs"] == gpu_defaults()["n_envs"] == 128
    assert cfg["training"]["rollout_size"] == 16384
    assert agent_action_count({"agent_actions": 2, "decisions": 1000}) == 2
    assert agent_action_count({"action_timing": "per_tick", "instant_actions": 3}) == 3
    assert agent_action_count({"decisions": 1000}) is None
    assert mean_agent_actions([]) is None
    assert mean_agent_actions([{"agent_actions": 2}, {}]) is None
    assert mean_agent_actions([{"agent_actions": 2}, {"agent_actions": 0}]) == 1


@pytest.mark.parametrize("timing,ticks", [("per_tick", 1), ("fixed", 10)])
def test_accepted_actions_exclude_waits_and_rejections(timing, ticks):
    cfg = load_config()
    cfg["environment"].update(action_timing=timing, decision_ticks=ticks)
    env = PvZEnv(cfg)
    env.reset(options={"scenario": LevelSpec("long", (Spawn(20000, "basic", 0),))})
    for _ in range(129):
        assert env.step(0)[2:4] == (False, False)
    assert env.public.tick == 129 * ticks
    assert env.episode_metrics()["agent_actions"] == 0
    for action, count in (
        (Place("sunflower", 0, 0), 1),
        (Place("sunflower", 0, 1), 1),  # Cooldown rejection.
        (Dig(0, 0), 2),
        (Dig(0, 0), 2),  # Empty tile rejection.
    ):
        assert env.step(env.codec.encode(action))[2:4] == (False, False)
        assert env.episode_metrics()["agent_actions"] == count
    row = env.episode_metrics()
    assert row["decisions"] == 133 and row["invalid_actions"] == 2
    env.reset()
    assert env.episode_metrics()["agent_actions"] == 0


def test_cuda_counts_and_games_survive_128_waits():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = load_config()
    cfg["training"].update(n_envs=1, rollout_size=128, batch_size=128)
    env = vector_env(cfg, "masked", 101, family="saving")
    try:
        env.reset()
        actions = torch.zeros(1, dtype=torch.long, device="cuda")
        for _ in range(129):
            _, _, done, timeout, _, _ = env.step_tensors(actions)
            assert not done.item() and not timeout.item()
        assert env.episode_metrics(0)["agent_actions"] == 0
        assert env.episode_metrics(0)["tick"] == 129
        for action, count in ((1, 1), (2, 1), (361, 2), (361, 2)):
            actions.fill_(action)
            env.step_tensors(actions)
            assert env.episode_metrics(0)["agent_actions"] == count
        row = env.episode_metrics(0)
        assert row["decisions"] == 133 and row["invalid_actions"] == 2
        assert row["tick"] == 131
        env.reset()
        assert env.episode_metrics(0)["agent_actions"] == 0
    finally:
        env.close()


@pytest.mark.learning
def test_rollout_update_preserves_unfinished_games():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = load_config()
    cfg["training"].update(n_envs=2, rollout_size=256, batch_size=128, n_epochs=1)
    env = vector_env(cfg, "masked", 101, family="placement")

    class Boundaries(BaseCallback):
        def __init__(self):
            super().__init__()
            self.states = []

        def _on_step(self):
            assert not any("episode_metrics" in info for info in self.locals["infos"])
            return True

        def _on_rollout_start(self):
            if self.states:
                np.testing.assert_array_equal(env.batch.header.get(), self.states[-1])
                assert env.task_counts()["placement"]["started_games"] == 2

        def _on_rollout_end(self):
            self.states.append(env.batch.header.get().copy())

    try:
        torch.set_num_threads(1)
        model = build_model(cfg, "masked", env, 101)
        model.set_logger(configure(format_strings=[]))
        callback = Boundaries()
        model.learn(512, callback=callback)
        assert len(callback.states) == 2
        assert model._n_updates == 2
        assert all(env.episode_metrics(i)["decisions"] == 256 for i in range(2))
        assert env.task_counts()["placement"]["completed_games"] == 0
    finally:
        env.close()
