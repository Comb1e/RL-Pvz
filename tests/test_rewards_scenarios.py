import random
from collections import Counter
from dataclasses import replace

import pytest
from pvz_game import Game, Place, Rules, Status
from pvz_game.config import resolve_level

from pvz_rl.config import validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.rewards import potential, reward_parts
from pvz_rl.scenarios import difficulty_weights, scenario


def test_discounted_shaping_telescopes_and_plant_dig_is_not_free_reward(cfg):
    env = PvZEnv(cfg)
    env.reset(seed=10)
    phi0 = potential(env.public, cfg)
    gamma = cfg["reward"]["gamma"]
    total = 0
    actions = [env.codec.encode(Place("sunflower", 0, 0)), 361, 0, 0, 0]
    for t, action in enumerate(actions):
        _, reward, _, _, _ = env.step(action)
        total += gamma**t * reward
    expected = -phi0 + gamma ** len(actions) * potential(env.public, cfg)
    assert total == pytest.approx(expected, abs=1e-12)
    assert total < 0


def test_shaping_true_terminal_independent_algebra(cfg):
    obs = Game().reset("easy", 1)
    phi = potential(obs, cfg)
    for status, terminal in ((Status.WON, 1), (Status.LOST, -1)):
        end = replace(obs, status=status, sun=9990)
        parts = reward_parts(obs, end, cfg, True)
        assert parts["shaping"] == pytest.approx(-phi)
        assert parts["total"] == pytest.approx(terminal - phi)
        assert reward_parts(obs, end, cfg, False)["total"] == terminal


def test_concurrent_spawn_and_defeat_uses_explicit_count(cfg):
    before = Game().reset("easy", 1)
    before = replace(
        before, counts=replace(before.counts, spawned=1, alive=1, remaining=15, not_yet_spawned=14)
    )
    after = replace(
        before,
        counts=replace(
            before.counts, spawned=2, alive=1, defeated=1, remaining=14, not_yet_spawned=13
        ),
    )
    assert after.counts.alive - before.counts.alive == 0
    assert potential(after, cfg) - potential(before, cfg) == pytest.approx(0.5 / 15)


@pytest.mark.parametrize("level", ["standard", "hard"])
def test_changed_scenarios_preserve_required_controls(level):
    rules, seed = Rules(), 300001
    base = resolve_level(level, rules, random.Random(seed))
    shuffled = scenario(level, "redistributed", seed, rules)
    assert Counter(s.zombie_type for s in base.spawns) == Counter(
        s.zombie_type for s in shuffled.spawns
    )
    assert Counter(s.wave for s in base.spawns) == Counter(s.wave for s in shuffled.spawns)
    assert [s for s in base.spawns if s.wave <= 3] == [s for s in shuffled.spawns if s.wave <= 3]
    fast = scenario(level, "faster", seed, rules)
    # Compare by wave and original generation ordering: jitter and rows are unchanged.
    reference = sorted((s.wave, s.zombie_type, s.row, s.tick) for s in base.spawns)
    transformed = sorted(
        (s.wave, s.zombie_type, s.row, s.tick + (s.wave - 1) * 100) for s in fast.spawns
    )
    assert transformed == reference
    concentrated = scenario(level, "concentrated", seed, rules)
    assert len({s.row for s in concentrated.spawns}) <= 3
    assert [(s.tick, s.zombie_type, s.wave) for s in concentrated.spawns] == [
        (s.tick, s.zombie_type, s.wave) for s in base.spawns
    ]
    assert scenario(level, "concentrated", seed, rules) == concentrated


def test_curriculum_boundaries_are_exact(cfg):
    assert difficulty_weights(cfg, 9, 100, True) == [1, 0, 0]
    assert difficulty_weights(cfg, 10, 100, True) == [0.5, 0.5, 0]
    assert difficulty_weights(cfg, 39, 100, True) == [0.5, 0.5, 0]
    assert difficulty_weights(cfg, 40, 100, True) == [0.2, 0.4, 0.4]
    assert difficulty_weights(cfg, 0, 100, False) == [0.2, 0.4, 0.4]


def test_overlapping_splits_are_rejected(cfg):
    cfg["splits"]["test"] = [99999, 200299]
    with pytest.raises(ValueError, match="disjoint"):
        validate_config(cfg)


def test_known_winning_diagnostic_control(cfg):
    env = PvZEnv(cfg, family="diagnostic")
    env.reset(seed=7)
    env.step(env.codec.encode(Place("peashooter", 2, 2)))
    while env.state == "running":
        env.step(0)
    assert env.state == "won" and env.metrics["MowerActivated"] == 0
