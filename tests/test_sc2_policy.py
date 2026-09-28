"""Independent controls for structured exploration and spatial placement."""

import pytest
import torch
from pvz_game import Dig, Place

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.learning.curriculum import CurriculumState
from pvz_rl.learning.deadline import RunBudget


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def test_stage_residency_counts_only_matching_episode_starts():
    cfg = load_config()
    state = CurriculumState(stage=0)
    for _ in range(100):
        state.completed_episode(1)
    assert not state.observe({"easy": 100}, 500, cfg)
    assert not state.observe({"easy": 100}, 1000, cfg)
    for _ in range(99):
        state.completed_episode(0)
    restored = CurriculumState(**state.to_dict())
    assert not restored.observe({"easy": 100}, 1500, cfg)
    restored.completed_episode(0)
    assert restored.observe({"easy": 100}, 2000, cfg)
    assert restored.name == "standard" and restored.completed_stage_games == 0


def test_wall_budget_reserves_cleanup_and_restores_consumption():
    cfg = load_config()
    now = [100.0]
    budget = RunBudget(cfg, elapsed=6000, clock=lambda: now[0])
    assert budget.deadline == 1300
    assert budget.learning_deadline == 400
    assert budget.can_collect(200)
    assert not budget.can_collect(251)
    now[0] = 250
    restored = RunBudget(cfg, elapsed=budget.state()["elapsed_seconds"], clock=lambda: 500)
    assert restored.state()["remaining_seconds"] == 1050
    assert restored.state()["reserve_seconds"] == 900


def test_early_dig_metric_tracks_actual_accepted_actions(per_tick_cfg):
    env = PvZEnv(per_tick_cfg)
    env.reset(seed=1)
    env.step(env.codec.encode(Place("sunflower", 0, 0)))
    env.step(env.codec.encode(Dig(0, 0)))
    assert env.episode_metrics()["early_voluntary_digs"] == 1
    env.step(env.codec.encode(Dig(0, 0)))
    assert env.episode_metrics()["early_voluntary_digs"] == 1


@pytest.mark.parametrize("budget", [0.0, 0.1, 1.0])
def test_exploration_unequal_tiles_matches_independent_enumeration(budget):
    from pvz_rl.policy.sequential_q import explore, per_head_epsilon

    torch.manual_seed(103)
    count = 50000
    alpha = per_head_epsilon(budget)
    legal = torch.tensor([[True, False, True]]).expand(count, -1)
    species, first_coin = explore(torch.zeros(count, dtype=torch.long), legal, alpha)
    tile_mask = torch.arange(5)[None] < torch.where(species == 0, 2, 5)[:, None]
    tile, second_coin = explore(torch.zeros(count, dtype=torch.long), tile_mask, alpha)
    for k, n in ((0, 2), (2, 5)):
        for t in range(n):
            expected = ((1 - alpha) * (k == 0) + alpha / 2) * ((1 - alpha) * (t == 0) + alpha / n)
            measured = ((species == k) & (tile == t)).float().mean().item()
            assert measured == pytest.approx(expected, abs=0.007)
    assert (first_coin | second_coin).float().mean().item() == pytest.approx(budget, abs=0.007)
    assert not (species == 1).any()
