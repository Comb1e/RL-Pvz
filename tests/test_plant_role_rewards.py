"""Independent placement/removal arithmetic and CPU/CUDA event attribution."""

from dataclasses import replace

import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Place, Spawn, Status
from pvz_game.types import Event

from pvz_rl.config import load_config, validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.plant_rewards import offensive_potential
from pvz_rl.rewards import reward_parts


@pytest.fixture
def role_cfg():
    return load_config("configs/sc2-plant-rewards.toml")


def test_plant_then_dig_has_no_discounted_placement_profit(role_cfg):
    env = PvZEnv(role_cfg)
    env.reset(
        seed=7,
        options={"scenario": LevelSpec("cycle", (Spawn(1000, "basic", 0),), initial_sun=200)},
    )
    _, _, _, _, place = env.step(env.codec.encode(Place("peashooter", 0, 0)))
    _, _, _, _, dig = env.step(361)
    a, b = place["reward_parts"], dig["reward_parts"]
    assert a["offensive_plantings"] == 1
    assert a["offensive_shaping"] == pytest.approx(0.0999)
    assert b["offensive_shaping"] == pytest.approx(-0.1)
    assert a["offensive_shaping"] + 0.999 * b["offensive_shaping"] == pytest.approx(0)
    assert b["plant_eaten_penalty"] == 0


@pytest.mark.parametrize(
    "kind", ["peashooter", "snow_pea", "repeater", "chomper", "sunflower", "wall_nut"]
)
def test_role_taxonomy_and_terminal_timeout_potential(role_cfg, kind):
    obs = Game().reset(
        LevelSpec("plants", (Spawn(1000, "basic", 0),), plants=(InitialPlant(kind, 0, 0),)), 1
    )
    expected = 0 if kind in ("sunflower", "wall_nut") else 0.1
    assert offensive_potential(obs, role_cfg) == expected
    for outcome in (Status.WON, Status.LOST):
        parts = reward_parts(obs, replace(obs, status=outcome), role_cfg, True)
        assert parts["offensive_shaping"] == -expected
    # External truncation leaves the public engine running and keeps its potential.
    parts = reward_parts(obs, obs, role_cfg, True)
    assert parts["offensive_shaping"] == pytest.approx(-0.001 * expected)


@pytest.mark.parametrize(
    "kind", ["sunflower", "peashooter", "wall_nut", "potato_mine", "cherry_bomb"]
)
def test_eaten_only_penalizes_non_wall_nuts_cpu_and_cuda(role_cfg, kind):
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
    before = game.observe()
    result = game.step(ticks=20)
    assert any(e.kind == "PlantRemoved" and e.get("reason") == "eaten" for e in result.events)
    parts = reward_parts(before, result.observation, role_cfg, True, events=result.events)
    assert parts["plants_eaten"] == int(kind != "wall_nut")
    assert parts["plant_eaten_penalty"] == (-0.02 if kind != "wall_nut" else 0)
    assert parts["empty_explosion_penalty"] == 0  # Eaten ash plants never exploded.
    if not torch.cuda.is_available():
        return
    from pvz_game.cuda import CudaBatch

    from pvz_rl.cuda_features import REWARD_FIELDS, CudaFeatures

    batch = CudaBatch(1, zombie_capacity=1, diagnostic=True, max_step_ticks=20)
    batch.restore([snapshot])
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, role_cfg, "masked")
        features.encode()
        features.step(batch.cp.asarray([0], dtype=batch.cp.int64), ticks=20)
        for key, value in zip(REWARD_FIELDS, features.parts.get()[0]):
            assert value == pytest.approx(parts[key], abs=1e-10), key
        assert batch.state_hash(0) == game.state_hash()


def test_detonation_is_not_eating_and_empty_explosions_are_still_penalized(role_cfg):
    env = PvZEnv(role_cfg)
    env.reset(
        seed=1,
        options={"scenario": LevelSpec("empty", (Spawn(1000, "basic", 0),), initial_sun=500)},
    )
    env.step(env.codec.encode(Place("cherry_bomb", 0, 0)))
    while env.public.tick < 25:
        env.step(0)
    metrics = env.episode_metrics()
    assert metrics["empty_explosions"] == 1
    assert metrics["empty_explosion_penalty"] == -0.2
    assert metrics["plants_eaten"] == 0


def test_legacy_settings_and_ambiguous_events(role_cfg):
    obs = Game().reset("easy", 1)
    missing = Event("PlantRemoved", 1, 999, (("reason", "eaten"),))
    with pytest.raises(RuntimeError, match="attribute"):
        reward_parts(obs, obs, role_cfg, True, events=[missing])
    role_cfg["reward"].pop("offensive_plant_weight")
    role_cfg["reward"].pop("eaten_plant_penalty")
    legacy = reward_parts(obs, obs, role_cfg, True, events=[missing])
    assert legacy["offensive_shaping"] == legacy["plant_eaten_penalty"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("eaten_plant_penalty", -1),
        ("offensive_plant_weight", float("nan")),
        ("offensive_plants", ["sunflower"]),
        ("offensive_plants", []),
    ],
)
def test_invalid_role_settings(role_cfg, key, value):
    role_cfg["reward"][key] = value
    with pytest.raises(ValueError, match="reward"):
        validate_config(role_cfg)
