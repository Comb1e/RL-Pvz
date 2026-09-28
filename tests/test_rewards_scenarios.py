import random
from collections import Counter
from dataclasses import replace

import pytest
from pvz_game import Game, Place, Rules
from pvz_game.config import resolve_level

from pvz_rl.config import validate_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.envs.rewards import reward_parts
from pvz_rl.envs.scenarios import scenario


def test_defeat_count_without_effective_damage_does_not_earn_credit(cfg):
    before = Game().reset("easy", 1)
    after = replace(before, counts=replace(before.counts, spawned=2, alive=1, defeated=1))
    assert reward_parts(before, after, cfg)["total"] == 0


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
        (s.wave, s.zombie_type, s.row, s.tick + (s.wave - 1) * 5 * rules.game["tick_rate"])
        for s in fast.spawns
    )
    assert transformed == reference
    concentrated = scenario(level, "concentrated", seed, rules)
    assert len({s.row for s in concentrated.spawns}) <= 3
    assert [(s.tick, s.zombie_type, s.wave) for s in concentrated.spawns] == [
        (s.tick, s.zombie_type, s.wave) for s in base.spawns
    ]
    assert scenario(level, "concentrated", seed, rules) == concentrated


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
