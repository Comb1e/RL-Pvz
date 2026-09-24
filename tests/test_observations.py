"""Independent controls for the single event_v5 public observation layout."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Rules, Spawn
from pvz_game.types import ZombieView

from pvz_rl.config import load_config
from pvz_rl.encoding import ObservationEncoder


def public_board():
    game = Game()
    game.reset(LevelSpec("public", (Spawn(10000, "basic", 0),)))
    return game.observe()


def zombie(kind="basic", row=0, x=1000, **changes):
    values = dict(
        id=1,
        zombie_type=kind,
        row=row,
        x=x,
        health=200,
        armor=0,
        state="walking",
        slow_ticks=0,
        has_pole=False,
        timer_ticks=0,
    )
    return ZombieView(**(values | changes))


def test_layout_counts_assets_distances_and_empty_regions():
    encoder = ObservationEncoder(load_config(), Rules())
    assert encoder.size == 342 and encoder.zombie_width == 9
    assert [(s.start, s.stop) for s in encoder.slices.values()] == [
        (0, 135),
        (135, 270),
        (270, 315),
        (315, 342),
    ]
    # Literal values use the pinned house/spawn positions -500/9500. A nearer
    # spent pole must not hide the next carrier whose pole is still unused.
    zombies = (
        zombie(x=1000),
        zombie("flag", x=2000),
        zombie("conehead", x=1500, armor=400),
        zombie("buckethead", x=2200, armor=1100),
        zombie("pole_vaulting", x=1200, health=400),
        zombie("pole_vaulting", x=1700, health=400, has_pole=True),
        zombie("pole_vaulting", x=2300, health=400, has_pole=True),
    )
    obs = replace(public_board(), zombies=zombies)
    encoded = encoder.encode(obs)
    grid = encoded[135:270].reshape(5, 3, 9)
    np.testing.assert_allclose(
        grid[0, 0],
        [0.2, 0.2, 0.2, 0.2, 0.6, 1, 1500 / 5500, 0.15, 0.22],
        atol=1e-7,
    )
    np.testing.assert_array_equal(grid[1:, :, :7], 0)
    np.testing.assert_array_equal(grid[1:, :, 7:], -1)
    np.testing.assert_array_equal(
        encoded, encoder.encode(replace(obs, zombies=tuple(reversed(zombies))))
    )
    # Crowds are sums, never clipped to one, even when IDs repeat.
    crowded = encoder.encode(replace(obs, zombies=zombies * 20))[135:270].reshape(5, 3, 9)
    np.testing.assert_allclose(crowded[0, 0, :7], grid[0, 0, :7] * 20, atol=1e-6)
    np.testing.assert_array_equal(crowded[0, 0, 7:], grid[0, 0, 7:])
    spent = tuple(replace(z, has_pole=False) for z in zombies)
    after = encoder.encode(replace(obs, zombies=spent))
    assert np.flatnonzero(encoded != after).tolist() == [143]
    assert after[143] == -1


@pytest.mark.parametrize("row", range(5))
@pytest.mark.parametrize(
    "x,region", [(-500, 0), (2833, 0), (2834, 1), (6166, 1), (6167, 2), (9500, 2)]
)
def test_distance_and_region_boundaries(row, x, region):
    encoder = ObservationEncoder(load_config(), Rules())
    obs = replace(public_board(), zombies=(zombie("pole_vaulting", row, x, has_pole=True),))
    grid = encoder.encode(obs)[135:270].reshape(5, 3, 9)
    assert grid[:, :, 4].sum() == pytest.approx(0.2)
    assert grid[row, region, 4] == pytest.approx(0.2)
    np.testing.assert_allclose(grid[row, region, 7:], (x + 500) / 10000, atol=1e-7)
    # Presence remains distinguishable at both physical endpoints.
    assert np.count_nonzero(grid[:, :, 8] != -1) == 1


def test_no_states_countdowns_or_private_fields_in_cpu_and_cuda_inputs():
    from pvz_game.cuda.schema import PLANT, ZOMBIE, ZOMBIE_STATES

    from pvz_rl.cuda_features import CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch

    cfg = load_config()
    specs = [
        LevelSpec(
            "public",
            tuple(Spawn(1, "pole_vaulting", r, x=2000 + 1000 * r) for r in range(5)),
            plants=(InitialPlant("potato_mine", 1, 1),),
        )
    ]
    batch = LessonCudaBatch(1, zombie_capacity=5, max_step_ticks=1)
    batch.reset(specs, [0])
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, cfg, "masked")
        features.step(batch.cp.zeros(1, dtype=batch.cp.int64))
        public = batch.observe(0)
        expected = features.encoder.encode(public)
        np.testing.assert_allclose(features.observations.get()[0], expected, atol=1e-7, rtol=1e-6)
        for field in ("due", "burst_due"):
            batch.plants[:, :, PLANT.index(field)] += 777
        for field in ("slow_until", "vault_until", "bite_progress", "id"):
            batch.zombies[:, :, ZOMBIE.index(field)] += 777
        batch.cooldowns[:] += 777
        for code, state in enumerate(ZOMBIE_STATES):
            # Synthetic state-only changes establish that no behavior label or
            # countdown leaks into the features. These states are not simulated.
            batch.zombies[:, :, ZOMBIE.index("state")] = code
            changed = replace(
                public,
                level="private-label",
                zombies=tuple(
                    replace(z, state=state, slow_ticks=123, timer_ticks=456, id=999)
                    for z in reversed(public.zombies)
                ),
                plants=tuple(replace(p, timer_ticks=321) for p in public.plants),
                cards=tuple(replace(c, cooldown_ticks=654) for c in public.cards),
            )
            np.testing.assert_array_equal(features.encoder.encode(changed), expected)
            features.encode()
            np.testing.assert_allclose(
                features.observations.get()[0], expected, atol=1e-7, rtol=1e-6
            )
        batch.zombies[:, :, ZOMBIE.index("has_pole")] = 0
        features.encode()
        changed = replace(public, zombies=tuple(replace(z, has_pole=False) for z in public.zombies))
        np.testing.assert_allclose(
            features.observations.get()[0], features.encoder.encode(changed), atol=1e-7, rtol=1e-6
        )


def test_actual_vault_consumes_pole_in_cpu_and_cuda():
    from pvz_rl.cuda_features import CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch

    cfg = load_config()
    spec = LevelSpec(
        "vault",
        (Spawn(1, "pole_vaulting", 0, x=2000),),
        plants=(InitialPlant("wall_nut", 0, 0),),
        mowers=False,
    )
    game = Game()
    game.reset(spec, 0)
    batch = LessonCudaBatch(1, zombie_capacity=1, max_step_ticks=1)
    batch.reset([spec], [0])
    seen = set()
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        for _ in range(1000):
            game.step()
            features.step(batch.cp.zeros(1, dtype=batch.cp.int64))
            obs = game.observe()
            encoded = features.encoder.encode(obs)
            np.testing.assert_allclose(
                features.observations.get()[0], encoded, atol=1e-7, rtol=1e-6
            )
            z = obs.zombies[0]
            seen.add(z.state)
            assert (encoded[143] != -1) == z.has_pole
            if "vaulting" in seen and z.state != "vaulting":
                break
    assert {"carrying_pole", "vaulting", "walking"} <= seen
    assert batch.state_hash(0) == game.state_hash()
