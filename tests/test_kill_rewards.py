from dataclasses import replace

import pytest
from pvz_game import Game, LevelSpec, Place, Spawn, Status
from pvz_game.types import Event

from pvz_rl.config import validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.rewards import defeat_counts, reward_parts


def damage(zombie, source, tick=1):
    return Event("DamageApplied", tick, zombie, (("source", source), ("health_damage", 20)))


def death(zombie, tick=1):
    return Event("ZombieDefeated", tick, zombie)


def test_killing_blow_not_damage_or_mower_activation():
    events = [
        damage(10, 5),
        damage(10, -1),
        death(10),  # Plant wounded; mower killed.
        Event("MowerActivated", 1, details=(("row", 0),)),
        damage(11, 8),
        death(11),  # Plant killed while mower active.
        damage(12, 9),  # Surviving zombie earns no kill reward.
        damage(13, -3, 4),
        death(13, 4),
        damage(14, 123, 5),
        death(14, 5),
    ]
    assert defeat_counts(events) == {"plant_kills": 2, "mower_kills": 2}
    assert defeat_counts([Event("MowerActivated", 1)]) == {"plant_kills": 0, "mower_kills": 0}
    with pytest.raises(RuntimeError, match="attribute"):
        defeat_counts([death(1)])
    with pytest.raises(RuntimeError, match="attribute"):
        defeat_counts([damage(1, 4), death(1), damage(1, 4), death(1)])


def test_kills_are_diagnostics_not_extra_event_rewards(cfg):
    before = Game().reset("easy", 1)
    after = replace(before, counts=replace(before.counts, defeated=1))
    for source, key in ((1, "plant_kills"), (-1, "mower_kills")):
        parts = reward_parts(before, after, cfg, True, events=[damage(2, source), death(2)])
        assert parts[key] == 1
        assert parts["total"] == parts["shaping"]
    empty = replace(before, counts=replace(before.counts, initial_total=0), status=Status.WON)
    assert reward_parts(before, empty, cfg, False)["total"] == 1


@pytest.mark.parametrize(
    "plant", ["peashooter", "snow_pea", "repeater", "cherry_bomb", "potato_mine", "chomper"]
)
def test_real_engine_credits_all_plant_damage_types(cfg, plant):
    env = PvZEnv(cfg)
    spawn_x = 9500 if plant in ("peashooter", "snow_pea", "repeater") else 1400
    env.reset(
        seed=4,
        options={
            "scenario": LevelSpec(
                "plant-kill", (Spawn(300, "basic", 2, x=spawn_x),), initial_sun=500, mowers=False
            )
        },
    )
    col = 1 if plant == "cherry_bomb" else 0
    placed = False
    parts = []
    while env.state == "running":
        action = 0
        # Bomb fuse must coincide with the spawn; other plants can wait for it.
        if not placed and (plant != "cherry_bomb" or env.public.tick >= 290):
            action = env.codec.encode(Place(plant, 2, col))
            placed = True
        _, _, _, _, info = env.step(action)
        parts.append(info["reward_parts"])
    assert env.state == "won"
    assert env.episode_metrics()["plant_kills"] == 1
    assert env.episode_metrics()["mower_kills"] == 0
    assert sum(p["plant_kills"] for p in parts) == 1
    assert env.episode_metrics()["defeated"] == 1


def test_real_batched_mower_kills_count_each_zombie_once_and_terminal(cfg):
    env = PvZEnv(cfg)
    env.reset(
        seed=8,
        options={
            "scenario": LevelSpec(
                "mower", tuple(Spawn(1, "basic", 0, x=0) for _ in range(3)), mowers=True
            )
        },
    )
    _, total, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and env.state == "won"
    assert info["reward_parts"]["terminal"] == 1
    assert info["reward_parts"]["mower_activation_penalty"] == -0.2
    assert info["episode_metrics"]["mower_kills"] == 3
    assert info["episode_metrics"]["mowers_used"] == 1
    assert total == pytest.approx(info["reward_parts"]["shaping"] + 0.8)
    env.reset(seed=8)
    assert env.episode_metrics()["mower_kills"] == 0


def test_real_simultaneous_batch_keeps_both_sources(cfg):
    env = PvZEnv(cfg)
    env.reset(
        seed=2,
        options={
            "scenario": LevelSpec(
                "mixed", (Spawn(1, "basic", 1, x=1400), Spawn(25, "basic", 0, x=0)), initial_sun=500
            )
        },
    )
    env.step(env.codec.encode(Place("cherry_bomb", 1, 1)))
    env.step(0)
    _, _, terminated, _, info = env.step(0)
    assert terminated
    parts = info["reward_parts"]
    assert parts["plant_kills"] == parts["mower_kills"] == 1
    assert parts["mower_activation_penalty"] == -0.2
    assert info["episode_metrics"]["defeated"] == 2


@pytest.mark.parametrize(
    "plant,col", [("peashooter", 0), ("snow_pea", 0), ("repeater", 0), ("chomper", 1)]
)
def test_failed_close_threat_controls_do_not_earn_kill_credit(cfg, plant, col):
    env = PvZEnv(cfg)
    env.reset(
        seed=4,
        options={
            "scenario": LevelSpec(
                "too-close", (Spawn(300, "basic", 2, x=1400),), initial_sun=500, mowers=False
            )
        },
    )
    env.step(env.codec.encode(Place(plant, 2, col)))
    while env.state == "running":
        env.step(0)
    assert env.state == "lost"
    assert env.episode_metrics()["plant_kills"] == env.episode_metrics()["mower_kills"] == 0


@pytest.mark.parametrize(
    "key,value",
    [("defeated_weight", -1), ("mower_activation_cost", float("nan")), ("economy_scale", 0)],
)
def test_invalid_reward_configuration(cfg, key, value):
    cfg["reward"][key] = value
    with pytest.raises(ValueError, match="reward"):
        validate_config(cfg)
