"""Independent information, truncation, public-boundary and CPU/CUDA controls."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Rules, Spawn
from pvz_game.cuda.schema import MOWER_STATES, PLANT_STATES, ZOMBIE_STATES
from pvz_game.types import ProjectileView, ZombieView

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ObservationEncoder, collate_observations, observations_equal


def public_board():
    return Game().reset(LevelSpec("public", (Spawn(10000, "basic", 0),)))


def zombie(kind="basic", row=0, x=1000, **changes):
    values = dict(
        id=1,
        zombie_type=kind,
        row=row,
        x=x,
        health=270,
        armor=0,
        state="walking",
        slow_ticks=0,
        has_pole=False,
        timer_ticks=0,
        headless=False,
    )
    return ZombieView(**(values | changes))


def test_old_region_alias_is_separated_and_ids_order_are_irrelevant():
    encoder = ObservationEncoder(load_config(), Rules())
    board = public_board()
    a = replace(board, zombies=(zombie(x=1000, health=100), zombie(x=2000, health=200)))
    b = replace(board, zombies=(zombie(x=1000, health=200), zombie(x=2000, health=100)))

    # Independent old aggregate: same type count, total HP/armor and nearest x.
    def aggregate(obs):
        return len(obs.zombies), sum(z.health for z in obs.zombies), min(z.x for z in obs.zombies)

    assert aggregate(a) == aggregate(b)
    assert not observations_equal(encoder.encode(a), encoder.encode(b))
    changed = replace(a, level="hidden", zombies=tuple(replace(z, id=987) for z in a.zombies[::-1]))
    assert observations_equal(encoder.encode(a), encoder.encode(changed))
    encoded = encoder.encode(a)
    np.testing.assert_array_equal(encoded["entities"][:5, 0], 16)
    np.testing.assert_array_equal(encoded["entities"][5:, [3, 4]], [[1000, 100], [2000, 200]])
    assert encoder.space.contains(encoded)


@pytest.mark.parametrize("state", ZOMBIE_STATES)
@pytest.mark.parametrize("kind", list(Rules().zombies))
def test_zombie_fields_types_states_and_boundaries(kind, state):
    e = ObservationEncoder(load_config(), Rules())
    z = zombie(
        kind,
        4,
        -500,
        state=state,
        armor=700,
        health=70,
        slow_ticks=300,
        timer_ticks=11,
        has_pole=True,
        headless=True,
    )
    raw = e.encode(replace(public_board(), zombies=(z,)))["entities"][-1]
    np.testing.assert_array_equal(
        raw, [e.zombies[kind], e.states["zombie:" + state], 4, -500, 70, 700, 11, 300, 1, 1, 0]
    )
    batch = collate_observations(dict(entities=[raw], globals=[0] * 18))
    _, _, numeric = e.normalize(batch)
    np.testing.assert_allclose(
        numeric[0, 0].numpy(),
        [1, 0, 70 / Rules().zombies[kind]["health"], 700 / 1100, 0.11, 3, 1, 1, 0],
        atol=1e-7,
    )


@pytest.mark.parametrize("state", PLANT_STATES)
def test_every_plant_phase_is_preserved(state):
    e = ObservationEncoder(load_config(), Rules())
    obs = Game().reset(
        LevelSpec(
            "plants",
            plants=tuple(InitialPlant(k, i % 5, i // 5) for i, k in enumerate(Rules().plants)),
        )
    )
    obs = replace(obs, plants=tuple(replace(p, state=state, timer_ticks=123) for p in obs.plants))
    raw = e.encode(obs)["entities"][5:]
    assert set(raw[:, 0]) == set(e.plants.values())
    assert (raw[:, 1] == e.states["plant:" + state]).all()
    assert (raw[:, 6] == 123).all()


@pytest.mark.parametrize("state", MOWER_STATES)
def test_projectiles_and_mower_details(state):
    e = ObservationEncoder(load_config(), Rules())
    obs = public_board()
    obs = replace(
        obs,
        mowers=tuple(replace(m, state=state, x=500) for m in obs.mowers),
        projectiles=(ProjectileView(1, 2, 4000, 20, False), ProjectileView(2, 2, 4100, 1800, True)),
    )
    raw = e.encode(obs)["entities"]
    assert (raw[:5, 1] == e.states["mower:" + state]).all()
    np.testing.assert_array_equal(raw[-2:, [0, 3, 10]], [[14, 4000, 20], [15, 4100, 1800]])


@pytest.mark.parametrize("limit", [5, 7, 10, 256, 512])
def test_cap_retains_mowers_then_plants_then_nearest_zombies_then_projectiles(limit):
    cfg = load_config()
    cfg["encoding"]["max_entities"] = limit
    e = ObservationEncoder(cfg, Rules())
    base = Game().reset(
        LevelSpec("crowd", plants=(InitialPlant("sunflower", 0, 0), InitialPlant("wall_nut", 4, 8)))
    )
    zombies = tuple(zombie(row=i % 5, x=9500 - i) for i in range(300))
    projectiles = tuple(ProjectileView(i, 0, i, 20, False) for i in range(300))
    obs = replace(base, zombies=zombies, projectiles=projectiles)
    encoded = e.encode(obs)
    raw = encoded["entities"]
    assert len(raw) == limit and (raw[:5, 0] == 16).all()
    assert ((raw[:, 0] >= 1) & (raw[:, 0] <= 8)).sum() == min(2, max(0, limit - 5))
    kept_z = raw[(raw[:, 0] >= 9) & (raw[:, 0] <= 13)]
    assert kept_z[:, 3].tolist() == sorted(z.x for z in zombies)[: max(0, limit - 7)]
    assert e.last_truncation["total"] == 607 - limit
    np.testing.assert_allclose(encoded["globals"][14:], np.array([2, 300, 300, 5]) / 75)
    assert observations_equal(
        encoded, e.encode(replace(obs, zombies=zombies[::-1], projectiles=projectiles[::-1]))
    )


def test_unknown_category_rejected_and_duplicates_retained():
    e = ObservationEncoder(load_config(), Rules())
    base = public_board()
    encoded = e.encode(replace(base, zombies=(zombie(),) * 12))
    assert len(encoded["entities"]) == 17
    with pytest.raises(ValueError, match="Unknown"):
        e.encode(replace(base, zombies=(zombie("unknown"),)))
    with pytest.raises(ValueError, match="Unknown"):
        e.encode(replace(base, zombies=(zombie(state="unknown"),)))


@pytest.mark.parametrize("limit", [5, 9, 256, 512])
def test_cuda_cpu_cap_and_public_countdowns(limit):
    from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
    from pvz_rl.envs.cuda_features import CudaFeatures

    cfg = load_config()
    cfg["encoding"]["max_entities"] = limit
    spec = LevelSpec(
        "mixed",
        tuple(
            Spawn(1, k, i % 5, x=1800 + i * 10) for i, k in enumerate(tuple(Rules().zombies) * 3)
        ),
        plants=tuple(InitialPlant(k, i % 5, i // 5) for i, k in enumerate(Rules().plants)),
    )
    game = Game()
    game.reset(spec, 4)
    batch = AccountingCudaBatch(1, zombie_capacity=15, max_step_ticks=1)
    batch.reset([spec], [4])
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        for _ in range(180):
            cpu = features.encoder.encode(game.observe())
            actual = features.obs_tensor.observations()[0]
            np.testing.assert_array_equal(actual["entities"], cpu["entities"])
            np.testing.assert_allclose(actual["globals"], cpu["globals"], atol=1e-7, rtol=1e-6)
            assert features.truncation_counts.get()[0].tolist() == [
                features.encoder.last_truncation[k] for k in ("plants", "zombies", "projectiles")
            ]
            game.step()
            features.step(batch.cp.zeros(1, batch.cp.int64))
        assert batch.state_hash(0) == game.state_hash()


@pytest.mark.parametrize("bad", [0, 100, 17])
def test_invalid_type_state_pair_is_rejected(bad):
    encoder = ObservationEncoder(load_config(), Rules())
    batch = collate_observations(
        dict(entities=[[1, bad, 0, 500, 10, 0, 0, 0, 0, 0, 0]], globals=[0] * 18)
    )
    with pytest.raises(ValueError, match="state"):
        encoder.normalize(batch)


def test_int32_overflow_is_rejected_and_valid_coordinates_are_not_clipped():
    encoder = ObservationEncoder(load_config(), Rules())
    with pytest.raises(ValueError, match="int32"):
        encoder.encode(replace(public_board(), zombies=(zombie(x=2**31),)))
    encoded = encoder.encode(replace(public_board(), zombies=(zombie(x=10500),)))
    _, _, values = encoder.normalize(collate_observations(encoded))
    assert values[0, -1, 1] > 1


def test_mower_lane_order_survives_positions_beyond_adjacent_lane_width():
    encoder = ObservationEncoder(load_config(), Rules())
    public = public_board()
    public = replace(
        public, mowers=tuple(replace(m, x=10500 if m.row == 0 else 0) for m in public.mowers)
    )
    records = encoder.encode(public)["entities"]
    assert records[:, 2].tolist() == list(range(5))
