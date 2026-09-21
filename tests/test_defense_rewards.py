"""Independent arithmetic and actual-engine controls for defensive rewards."""

from dataclasses import replace

import pytest
from pvz_game import Game, InitialPlant, LevelSpec, Place, Rules, Spawn, Wait
from pvz_game.types import Event

from pvz_rl.config import validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.rewards import REWARD_METRICS, reward_parts


def event(kind, tick=1, entity=None, **details):
    return Event(kind, tick, entity, tuple(details.items()))


def initial(sun=300):
    return Game().reset(LevelSpec("defense", (Spawn(1000, "basic", 0),), initial_sun=sun), 1)


@pytest.mark.parametrize("sun", [0, 50, 300, 600, 9990])
def test_sun_penalty_increases_without_clipping_and_activates_once(per_tick_cfg, sun):
    obs = initial(sun)
    activation = event("MowerActivated", row=0)
    parts = reward_parts(obs, obs, per_tick_cfg, False, events=[activation])
    assert parts["mower_sun_penalty"] == pytest.approx(-sun / 300)
    assert parts["mower_activation_penalty"] == -0.2  # No kill in this control.
    assert parts["total"] == pytest.approx(-sun / 300 - 0.2)
    assert reward_parts(obs, obs, per_tick_cfg, False)["mower_sun_penalty"] == 0


def test_activation_uses_event_time_sun_after_spending_and_capped_income(per_tick_cfg):
    obs = initial(300)
    events = [
        event("PlantPlaced", entity=11, plant_type="peashooter"),
        event("SunProduced", amount=25, produced=25),
        event("MowerActivated", row=0),
        event("SunProduced", tick=2, amount=5, produced=25),
        event("MowerActivated", tick=2, row=1),
        event("SunProduced", tick=3, amount=25, produced=25),
    ]
    parts = reward_parts(obs, replace(obs, sun=255), per_tick_cfg, False, events=events)
    assert parts["mower_activations"] == 2
    assert parts["mower_activation_sun"] == 225 + 230
    assert parts["mower_sun_penalty"] == pytest.approx(-(225 + 230) / 300)


def test_real_activation_and_multiple_kills_at_income_tick(per_tick_cfg):
    game = Game()
    before = game.reset(
        LevelSpec("sun", tuple(Spawn(200, "basic", 0, x=0) for _ in range(3)), initial_sun=300), 1
    )
    result = game.step(Place("sunflower", 4, 0), ticks=200)
    # Paid 50, first flower payout 25 at tick 120, sky payout 25 at tick 200.
    assert result.observation.sun == 300
    parts = reward_parts(before, result.observation, per_tick_cfg, False, events=result.events)
    assert parts["mower_sun_penalty"] == -1
    assert parts["mower_activations"] == 1 and parts["mower_kills"] == 3
    assert parts["mower_kill_penalty"] == -2 and parts["mower_activation_penalty"] == 0
    assert parts["total"] == -2  # +1 win, -2 kills, -1 activation.


@pytest.mark.parametrize("kind", ["wall_nut", "sunflower", "peashooter"])
def test_bite_only_rewards_wall_nut_actual_damage_including_final_partial_hit(per_tick_cfg, kind):
    game = Game()
    game.reset(
        LevelSpec(
            "bite", (Spawn(1, "basic", 0, x=800),), plants=(InitialPlant(kind, 0, 0),), mowers=False
        ),
        1,
    )
    snapshot = game.snapshot()
    snapshot["plants"][0]["health"] = 25
    game.restore(snapshot)
    obs = game.observe()
    result = game.step(ticks=20)
    bites = [e for e in result.events if e.kind == "PlantDamaged"]
    assert len(bites) == 1 and bites[0].get("damage") == 25
    assert not result.observation.plants
    parts = reward_parts(obs, result.observation, per_tick_cfg, False, events=result.events)
    assert parts["wall_nut_damage"] == (25 if kind == "wall_nut" else 0)
    assert parts["wall_nut_reward"] == pytest.approx(0.2 * 25 / 4000 if kind == "wall_nut" else 0)


def test_newly_placed_nut_bites_custom_health_and_no_reward_for_presence(per_tick_cfg):
    raw = Rules().to_dict()
    raw["plants"]["wall_nut"]["health"] = 1000
    rules = Rules(raw)
    obs = initial()
    events = [event("PlantPlaced", entity=11, plant_type="wall_nut")]
    assert (
        reward_parts(obs, obs, per_tick_cfg, False, events=events, rules=rules)["wall_nut_reward"]
        == 0
    )
    events += [
        event("PlantDamaged", tick=t, entity=11, source=12, damage=100) for t in range(1, 11)
    ]
    events.append(event("PlantRemoved", tick=10, entity=11, reason="eaten"))
    parts = reward_parts(obs, obs, per_tick_cfg, False, events=events, rules=rules)
    assert parts["wall_nut_reward"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "hp,armor,source,tick,miss",
    [
        (0, 0, 11, 1, True),
        (0, 20, 11, 1, False),
        (20, 0, 11, 1, False),
        (20, 0, 12, 1, True),
        (20, 0, 11, 0, True),
    ],
)
def test_explosion_requires_its_own_same_tick_damage(per_tick_cfg, hp, armor, source, tick, miss):
    obs = initial()
    events = [
        event("ZombieSpawned", tick=0, entity=22, zombie_type="basic"),
        event(
            "DamageApplied",
            tick=tick,
            entity=22,
            source=source,
            health_damage=hp,
            armor_damage=armor,
        ),
        event("PlantExploded", entity=11),
    ]
    parts = reward_parts(obs, obs, per_tick_cfg, False, events=events)
    assert parts["empty_explosions"] == int(miss)
    assert parts["empty_explosion_penalty"] == pytest.approx(-0.2 if miss else 0)


@pytest.mark.parametrize(
    "kind,spawn_tick,x,expected",
    [
        ("cherry_bomb", 1000, 9500, 1),  # Future enemies don't make this a hit.
        ("cherry_bomb", 1, 800, 0),
        ("potato_mine", 300, 800, 0),
        ("potato_mine", 1000, 9500, 0),  # Arming without detonation is not waste.
    ],
)
def test_real_explosions_and_untriggered_mine(per_tick_cfg, kind, spawn_tick, x, expected):
    game = Game()
    before = game.reset(
        LevelSpec("blast", (Spawn(spawn_tick, "basic", 0, x=x),), initial_sun=300), 1
    )
    result = game.step(Place(kind, 0, 0), ticks=350)
    parts = reward_parts(before, result.observation, per_tick_cfg, False, events=result.events)
    assert parts["empty_explosions"] == expected
    assert parts["empty_explosion_penalty"] == pytest.approx(-0.2 * expected)
    if spawn_tick < 1000:
        assert parts["plant_kills"] == 1 and parts["damage_reward"] == 0


def test_two_bombs_cannot_share_credit_and_digging_is_not_an_explosion(per_tick_cfg):
    game = Game()
    obs = game.reset(
        LevelSpec(
            "double",
            (Spawn(1, "basic", 0, x=1800),),
            plants=(InitialPlant("cherry_bomb", 0, 0), InitialPlant("cherry_bomb", 0, 1)),
        ),
        1,
    )
    result = game.step(ticks=24)
    parts = reward_parts(obs, result.observation, per_tick_cfg, False, events=result.events)
    assert parts["plant_kills"] == 1 and parts["empty_explosions"] == 1
    assert parts["empty_explosion_penalty"] == -0.2
    events = [event("PlantRemoved", entity=obs.plants[0].id, reason="dug")]
    assert reward_parts(obs, obs, per_tick_cfg, False, events=events)["empty_explosions"] == 0


def test_event_totals_are_independent_of_batching_and_reset(per_tick_cfg):
    scenario = LevelSpec(
        "batch",
        (Spawn(200, "basic", 1, x=0), Spawn(1, "basic", 0, x=800)),
        plants=(InitialPlant("wall_nut", 0, 0), InitialPlant("cherry_bomb", 4, 8)),
        initial_sun=300,
    )
    sums, hashes = [], []
    for batch in (1, 10):
        game = Game()
        before = game.reset(scenario, 1)
        total = dict.fromkeys(REWARD_METRICS, 0)
        for _ in range(400 // batch):
            result = game.step(Wait(), ticks=batch)
            parts = reward_parts(
                before, result.observation, per_tick_cfg, False, events=result.events
            )
            for key in total:
                total[key] += parts[key]
            before = result.observation
        sums.append(total)
        hashes.append(game.state_hash())
    assert sums[0] == pytest.approx(sums[1]) and hashes[0] == hashes[1]
    assert sums[0]["wall_nut_reward"] > 0 and sums[0]["empty_explosions"] == 1
    assert sums[0]["mower_sun_penalty"] < 0
    env = PvZEnv(per_tick_cfg)
    env.reset(options={"scenario": scenario})
    for _ in range(400):
        env.step(0)
    assert env.episode_metrics()["wall_nut_reward"] == pytest.approx(sums[0]["wall_nut_reward"])
    env.reset(seed=1)
    assert all(env.episode_metrics()[key] == 0 for key in REWARD_METRICS)


def test_missing_weights_preserve_legacy_reward(per_tick_cfg):
    obs = initial()
    events = [event("MowerActivated", row=0), event("PlantExploded", entity=22)]
    for key in (
        "mower_sun_weight",
        "wall_nut_damage_weight",
        "empty_explosion_penalty",
        "mower_sun_scale",
    ):
        per_tick_cfg["reward"].pop(key)
    validate_config(per_tick_cfg)
    parts = reward_parts(obs, obs, per_tick_cfg, False, events=events)
    assert parts["total"] == -0.2
    assert (
        parts["mower_sun_penalty"]
        == parts["wall_nut_reward"]
        == parts["empty_explosion_penalty"]
        == 0
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("mower_sun_weight", -1),
        ("mower_sun_weight", True),
        ("mower_sun_scale", 0),
        ("mower_sun_scale", float("inf")),
        ("wall_nut_damage_weight", float("nan")),
        ("empty_explosion_penalty", -0.2),
    ],
)
def test_invalid_weights(per_tick_cfg, key, value):
    per_tick_cfg["reward"][key] = value
    with pytest.raises(ValueError, match="reward"):
        validate_config(per_tick_cfg)
