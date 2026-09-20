"""Independent numerical controls for reward version 0.4.1."""

import copy
from dataclasses import replace

import pytest
from pvz_game import Dig, Game, LevelSpec, Place, Rules, Spawn, Status
from pvz_game.config import PLANT_TYPES
from pvz_game.types import Event

from pvz_rl.config import validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.rewards import REWARD_METRICS, potential, reward_parts


def hit(zombie, hp, *, source=10, armor=0, tick=1):
    return Event(
        "DamageApplied",
        tick,
        zombie,
        (("source", source), ("health_damage", hp), ("armor_damage", armor)),
    )


def dead(zombie, tick=1):
    return Event("ZombieDefeated", tick, zombie)


def activated(row=0, tick=1):
    return Event("MowerActivated", tick, details=(("row", row),))


def public_zombie(kind="basic", n=4, rules=None):
    game = Game(rules)
    game.reset(LevelSpec("damage", tuple(Spawn(1, kind, 0) for _ in range(n))), 1)
    return game.step().observation


@pytest.mark.parametrize("kind,hp", [("basic", 200), ("pole_vaulting", 400)])
@pytest.mark.parametrize("remaining", [200, 25])
def test_nonlethal_damage_uses_full_starting_health(cfg, kind, hp, remaining):
    obs = public_zombie(kind)
    zombie = replace(obs.zombies[0], health=remaining)
    before = replace(obs, zombies=(zombie, *obs.zombies[1:]))
    after = replace(before, zombies=(replace(zombie, health=remaining - 20), *obs.zombies[1:]))
    parts = reward_parts(before, after, cfg, False, events=[hit(zombie.id, 20)])
    assert parts["damage_reward"] == pytest.approx(20 / (hp * 4))
    assert parts["total"] == parts["damage_reward"]
    assert parts["nonlethal_health_damage"] == 20


@pytest.mark.parametrize("hp,armor,expected", [(0, 20, 0), (5, 15, 5 / 800)])
def test_armor_is_not_base_health_damage(cfg, hp, armor, expected):
    obs = public_zombie("buckethead")
    parts = reward_parts(obs, obs, cfg, False, events=[hit(obs.zombies[0].id, hp, armor=armor)])
    assert parts["total"] == pytest.approx(expected)
    assert parts["nonlethal_health_damage"] == hp


@pytest.mark.parametrize("source,kill_reward", [(10, 0.25), (-1, -0.5)])
def test_only_the_lethal_hit_is_excluded_even_when_death_is_in_same_batch(cfg, source, kill_reward):
    obs = public_zombie()
    zombie = obs.zombies[0].id
    events = [
        hit(zombie, 20),
        hit(zombie, 20, tick=2),
        hit(zombie, 160, source=source, tick=3),
        dead(zombie, 3),
    ]
    batch = reward_parts(obs, obs, cfg, False, events=events)
    assert batch["damage_reward"] == pytest.approx(40 / 800)
    assert batch["total"] == pytest.approx(40 / 800 + kill_reward)
    individual = [
        reward_parts(obs, obs, cfg, False, events=[e for e in events if e.tick == tick])
        for tick in (1, 2, 3)
    ]
    assert sum(p["total"] for p in individual) == pytest.approx(batch["total"])


def test_spawn_damage_and_death_within_batch_uses_public_type(cfg):
    obs = Game().reset("easy", 1)
    events = [
        Event("ZombieSpawned", 1, 500, (("zombie_type", "pole_vaulting"), ("row", 0))),
        hit(500, 20),
        hit(500, 380, tick=2),
        dead(500, 2),
    ]
    parts = reward_parts(obs, obs, cfg, False, events=events)
    assert parts["damage_reward"] == pytest.approx(20 / (400 * obs.counts.initial_total))
    assert parts["plant_kills"] == 1
    with pytest.raises(RuntimeError, match="zombie type"):
        reward_parts(obs, obs, cfg, False, events=[hit(500, 20)])


@pytest.mark.parametrize(
    "events,empty,kills",
    [
        ([activated()], 1, 0),
        ([activated(), hit(5, 200, source=-1), dead(5)], 0, 1),
        ([activated(), hit(5, 200, source=-2), dead(5)], 1, 1),
        ([activated(), hit(5, 200, source=-1, tick=2), dead(5, 2)], 1, 1),
        ([activated(), activated(1), hit(5, 200, source=-1), dead(5)], 1, 1),
        ([activated(), hit(5, 200), dead(5)], 1, 0),
    ],
)
def test_empty_activation_is_matched_to_same_row_and_tick(cfg, events, empty, kills):
    obs = public_zombie()
    parts = reward_parts(obs, obs, cfg, False, events=events)
    assert parts["empty_mower_activations"] == empty
    assert parts["mower_activation_penalty"] == pytest.approx(-empty / 5)
    assert parts["mower_kill_penalty"] == pytest.approx(-2 * kills / 4)
    assert parts["damage_reward"] == 0


def test_real_mower_activation_is_not_empty_and_terminal_takes_precedence(cfg):
    cfg["environment"]["cutoff_seconds"] = 1
    env = PvZEnv(cfg)
    env.reset(options={"scenario": LevelSpec("mower", (Spawn(20, "basic", 0, x=0),))})
    env.step(0)
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["reward_parts"]["terminal"] == 1
    assert info["episode_metrics"]["mowers_used"] == 1
    assert info["episode_metrics"]["empty_mower_activations"] == 0
    assert info["episode_metrics"]["mower_activation_penalty"] == 0
    assert info["episode_metrics"]["mower_kill_penalty"] == -2


@pytest.mark.parametrize("plant", PLANT_TYPES)
def test_buying_preserves_value_and_digging_removes_it_above_scale(cfg, plant):
    env = PvZEnv(cfg)
    env.reset(
        options={"scenario": LevelSpec("wealth", (Spawn(1000, "basic", 0),), initial_sun=1000)}
    )
    before = potential(env.public, cfg)
    _, _, _, _, info = env.step(env.codec.encode(Place(plant, 0, 0)))
    assert potential(env.public, cfg) == pytest.approx(before)
    assert info["reward_parts"]["shaping"] == pytest.approx((0.999 - 1) * before)
    damaged = replace(env.public, plants=(replace(env.public.plants[0], health=1, id=999),))
    assert potential(damaged, cfg) == pytest.approx(before)
    env.step(env.codec.encode(Dig(0, 0)))
    expected = before - 0.5 * env.rules.plants[plant]["cost"] / 300
    assert potential(env.public, cfg) == pytest.approx(expected)


def test_potential_sums_all_living_plants_without_capping_or_order_dependence(cfg):
    game = Game()
    game.reset(LevelSpec("wealth", (Spawn(1000, "basic", 0),), initial_sun=1000), 1)
    game.step(Place("peashooter", 0, 0))
    obs = game.step(Place("sunflower", 0, 1)).observation
    obs = replace(obs, counts=replace(obs.counts, initial_total=4, defeated=2))
    expected = 0.5 * 2 / 4 + 0.5 * (850 + 100 + 50) / 300
    assert potential(obs, cfg) == pytest.approx(expected)
    assert potential(replace(obs, plants=tuple(reversed(obs.plants))), cfg) == pytest.approx(
        expected
    )
    assert potential(replace(obs, plants=()), cfg) == pytest.approx(expected - 0.5 * 150 / 300)
    assert potential(replace(obs, status=Status.WON), cfg) == 0
    assert potential(replace(obs, status=Status.LOST), cfg) == 0


def test_active_custom_rules_supply_cost_and_maximum_health(cfg):
    raw = Rules().to_dict()
    raw["plants"]["peashooter"]["cost"] = 75
    raw["zombies"]["basic"]["health"] = 400
    rules = Rules(raw)
    env = PvZEnv(cfg, rules=rules)
    env.reset(
        options={
            "scenario": LevelSpec("custom", (Spawn(1, "basic", 0),), initial_sun=200, mowers=False)
        }
    )
    env.step(env.codec.encode(Place("peashooter", 0, 0)))
    assert env.public.sun == 125
    assert potential(env.public, cfg) == pytest.approx(0.5 * 200 / 300)
    while env.state == "running":
        env.step(0)
    # 19 nonlethal 20-HP hits, then one lethal hit. N = 1.
    assert env.episode_metrics()["damage_reward"] == pytest.approx(380 / 400)
    assert env.episode_metrics()["plant_kill_reward"] == 1


def test_event_components_are_invariant_to_real_engine_batch_size(cfg):
    totals, hashes = [], []
    for ticks in (1, 10):
        settings = copy.deepcopy(cfg)
        settings["environment"]["decision_ticks"] = ticks
        env = PvZEnv(settings)
        env.reset(
            seed=8,
            options={
                "scenario": LevelSpec(
                    "batch", (Spawn(100, "basic", 0),), initial_sun=100, mowers=False
                )
            },
        )
        env.step(env.codec.encode(Place("peashooter", 0, 0)))
        while env.state == "running":
            env.step(0)
        totals.append({key: env.episode_metrics()[key] for key in REWARD_METRICS})
        hashes.append(env.game.state_hash())
    assert hashes[0] == hashes[1]
    assert totals[0] == pytest.approx(totals[1])
    assert totals[0]["damage_reward"] == pytest.approx(0.9)


def test_truncation_retains_potential_and_reset_clears_reward_metrics(cfg):
    cfg["environment"]["cutoff_seconds"] = 1
    env = PvZEnv(cfg)
    env.reset(
        options={"scenario": LevelSpec("cutoff", (Spawn(1000, "basic", 0),), initial_sun=100)}
    )
    env.step(env.codec.encode(Place("peashooter", 0, 0)))
    before = potential(env.public, cfg)
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and before > 0
    assert info["reward_parts"]["shaping"] == pytest.approx((0.999 - 1) * before)
    assert info["reward_parts"]["terminal"] == 0
    env.reset(seed=1)
    assert all(env.episode_metrics()[key] == 0 for key in REWARD_METRICS)


def test_legacy_reward_config_preserves_original_formula_and_event_defaults(cfg):
    cfg["reward"] = dict(
        gamma=0.999,
        defeated_weight=0.5,
        sun_weight=0.3,
        flower_weight=0.2,
        sun_target=300,
        flower_target=8,
    )
    validate_config(cfg)
    game = Game()
    game.reset(LevelSpec("legacy", (Spawn(1, "basic", 0),), initial_sun=200), 1)
    obs = game.step(Place("sunflower", 1, 0)).observation
    expected = 0.3 * 150 / 300 + 0.2 / 8
    assert potential(obs, cfg) == pytest.approx(expected)
    zombie = obs.zombies[0].id
    events = [
        hit(zombie, 20),
        activated(tick=2),
        hit(zombie, 180, source=-1, tick=3),
        dead(zombie, 3),
    ]
    parts = reward_parts(obs, obs, cfg, True, events=events)
    assert parts["total"] == pytest.approx((0.999 - 1) * expected)
    cfg["reward"].update(plant_kill_weight=1, mower_kill_weight=1, normalize_kills=True)
    assert reward_parts(obs, obs, cfg, False, events=events)["total"] == -1
    cfg["reward"]["sun_target"] = 0
    with pytest.raises(ValueError, match="sun_target"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("potential_mode", "unknown"),
        ("damage_weight", -1),
        ("damage_weight", True),
        ("damage_weight", None),
        ("damage_weight", "1"),
        ("economy_weight", float("nan")),
        ("economy_scale", 0),
        ("economy_scale", -10),
        ("economy_scale", float("inf")),
        ("empty_mower_activation_penalty", -0.2),
        ("empty_mower_activation_penalty", float("nan")),
    ],
)
def test_invalid_new_reward_settings(cfg, key, value):
    cfg["reward"][key] = value
    with pytest.raises(ValueError, match="reward"):
        validate_config(cfg)
