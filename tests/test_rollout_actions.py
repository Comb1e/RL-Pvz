"""Rollout length is not an episode limit; action diagnostics exclude waiting."""

import pytest
import torch
from pvz_game import Dig, LevelSpec, Place, Spawn

from pvz_rl.config import gpu_defaults, load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.learning.training import vector_env
from pvz_rl.monitoring.metrics import agent_action_count, mean_agent_actions


def test_defaults_and_archived_action_counts():
    cfg = load_config()
    assert cfg == load_config("configs/train.toml")
    assert cfg["training"]["n_envs"] == gpu_defaults()["n_envs"] == 128
    assert cfg["training"]["method"] == "sequential_q_mc_v2"
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
    cfg["training"].update(n_envs=1, batch_size=128)
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
