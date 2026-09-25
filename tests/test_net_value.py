"""Independent net-value arithmetic and physical-time return controls."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pvz_game import Dig, InitialPlant, LevelSpec, Place, Spawn
from pvz_game.types import Event

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.envs.rewards import reward_parts


def test_shipped_objective_bounds_independent_of_reward_implementation():
    from collections import Counter

    from pvz_game import Rules
    from pvz_game.config import bundled

    rules = Rules()
    cfg = load_config()
    assert cfg["training"]["gamma"] == 1
    assert cfg["reward"]["progress_weight"] / cfg["reward"]["value_scale"] == 1 / 30000
    assert cfg["reward"]["mower_value"] == 200
    assert cfg["reward"]["basic_zombie_value"] == 50
    assert cfg["reward"]["win_reward"] == 1 and cfg["reward"]["loss_penalty"] == 2
    assets = rules.game["sun_cap"] + 45 * max(p["cost"] for p in rules.plants.values())
    tick, drops = 425, 0
    while tick <= 120000:
        drops += 1
        tick += min(950, 425 + 10 * drops)
    sky = drops * 25
    damage = []
    for level in ("easy", "standard", "hard"):
        counts = Counter(k for wave in bundled("levels.toml")[level]["waves"] for k in wave)
        damage.append(
            sum(
                n * (rules.zombies[k]["health"] + rules.zombies[k]["armor"])
                for k, n in counts.items()
            )
        )
    assert assets == 18990 and sky == 3525
    assert damage == [4050, 17240, 38130]
    lower, upper = -50 - sky - 5 * 200, assets - 50 + max(damage) * 50 / 270
    assert (lower, upper) == pytest.approx((-4575, 26001.111111111))
    assert 1 + lower / 30000 == pytest.approx(0.8475)
    assert -2 + upper / 30000 == pytest.approx(-1.1332962963)
    assert 1 + lower / 30000 > -2 + upper / 30000


def event(kind, source=1, hp=0, armor=0, amount=0):
    return Event(
        kind,
        1,
        99,
        (("source", source), ("health_damage", hp), ("armor_damage", armor), ("amount", amount)),
    )


def test_independent_accounting_orders_and_late_projectile_credit():
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        options={
            "scenario": LevelSpec(
                "value",
                (Spawn(999, "basic", 0),),
                initial_sun=500,
                plants=(InitialPlant("peashooter", 0, 0),),
            )
        }
    )
    full = env.public
    gone = replace(full, plants=())
    # A source ID remains valid when its firing plant no longer exists.
    hit = event("DamageApplied", hp=200, armor=400)
    events = [hit, Event("ZombieDefeated", 1, 99)]
    late = (
        reward_parts(full, gone, cfg)["net_value"]
        + reward_parts(gone, gone, cfg, events=events)["net_value"]
    )
    early = (
        reward_parts(full, full, cfg, events=events)["net_value"]
        + reward_parts(full, gone, cfg)["net_value"]
    )
    assert (
        late == early == pytest.approx(600 * 50 / 270 - 100)
    )  # 150 damage value minus 100 lost asset.
    parts = reward_parts(full, gone, cfg, events=events)
    assert parts["effective_damage"] == 600 and parts["plant_value_loss"] == 100
    assert parts["total"] == pytest.approx((600 * 50 / 270 - 100) / 30000)


@pytest.mark.parametrize("health_fraction", [1, 0.5, 0.01])
def test_damage_then_removal_never_double_charges(health_fraction):
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        options={
            "scenario": LevelSpec(
                "asset", (Spawn(999, "basic", 0),), plants=(InitialPlant("peashooter", 0, 0),)
            )
        }
    )
    start = env.public
    damaged = replace(
        start,
        plants=(
            replace(start.plants[0], health=int(start.plants[0].max_health * health_fraction)),
        ),
    )
    removed = replace(damaged, plants=())
    first = reward_parts(start, damaged, cfg)
    second = reward_parts(damaged, removed, cfg)
    assert first["net_value"] + second["net_value"] == pytest.approx(-100)
    assert second["net_value"] == pytest.approx(-100 * health_fraction)


@pytest.mark.parametrize("sky,flower", [(0, 0), (25, 25), (10, 0), (0, 10)])
def test_actual_capped_income_not_requested_income(sky, flower):
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(seed=3)
    before = replace(env.public, sun=9990 - sky - flower)
    after = replace(before, sun=9990)
    events = [
        event("SunProduced", source="sky", amount=sky),
        event("SunProduced", source="sunflower", amount=flower),
    ]
    parts = reward_parts(before, after, cfg, events=events)
    assert parts["net_value"] == flower
    assert parts["plant_value_loss"] == 0


@pytest.mark.parametrize(
    "kind,count,expected",
    [
        ("basic", 1, -100),
        ("basic", 3, 0),
        ("conehead", 1, 640 * 50 / 270 - 150),
        ("buckethead", 1, 1370 * 50 / 270 - 150),
        ("basic", 0, -150),
    ],
)
def test_real_explosions_break_even_and_empty_loss_on_cpu_and_cuda(kind, count, expected):
    from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures
    from pvz_rl.envs.cuda_lessons import LessonCudaBatch

    cfg = load_config()
    # A remote future zombie keeps the control running after this detonation.
    case = LevelSpec(
        "bomb",
        tuple([Spawn(3502, kind, 2, x=1400)] * count + [Spawn(9999, "basic", 4)]),
        initial_sun=300,
        mowers=False,
    )
    env = PvZEnv(cfg)
    env.reset(seed=4, options={"scenario": case})
    batch = LessonCudaBatch(1, zombie_capacity=count + 1)
    cp = batch.cp
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        batch.reset([case], [4])
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        actions = [0] * 3501 + [env.codec.encode(Place("cherry_bomb", 2, 1))] + [0] * 125
        for action in actions:
            info = env.step(action)[4]
            features.step(cp.asarray([action], cp.int64))
            for key, value in zip(REWARD_FIELDS, features.parts.get()[0]):
                assert value == pytest.approx(info["reward_parts"][key], abs=1e-9), key
        assert batch.state_hash(0) == env.game.state_hash()
    assert env.episode_metrics()["net_value"] == pytest.approx(expected)
    assert (
        env.episode_metrics()["effective_damage"]
        == {"basic": 270, "conehead": 640, "buckethead": 1370}[kind] * count
    )
    assert -100 > -200  # One-basic bomb costs less than consuming a mower.


def test_projectiles_keep_credit_after_voluntary_dig():
    from pvz_rl.envs.cuda_features import CudaFeatures
    from pvz_rl.envs.cuda_lessons import LessonCudaBatch

    case = LevelSpec(
        "late-shot",
        (Spawn(1, "basic", 0, x=5000), Spawn(9999, "basic", 4)),
        initial_sun=100,
        mowers=False,
    )
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(seed=4, options={"scenario": case})
    batch = LessonCudaBatch(1, zombie_capacity=2)
    cp = batch.cp
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        batch.reset([case], [4])
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        dug = False
        for step in range(400):
            action = env.codec.encode(Place("peashooter", 0, 0)) if step == 0 else 0
            if env.public.projectiles and not dug:
                action = env.codec.encode(Dig(0, 0))
                dug = True
            reward = env.step(action)[1]
            features.step(cp.asarray([action], cp.int64))
            assert features.rewards.get()[0] == pytest.approx(reward, abs=1e-8)
        assert dug
        assert batch.state_hash(0) == env.game.state_hash()
    assert env.episode_metrics()["effective_damage"] == 20
    assert env.episode_metrics()["net_value"] == pytest.approx(-100 + 20 * 50 / 270)


def test_mixed_cuda_resets_preserve_duration_and_episode_ledgers():
    from pvz_rl.envs.cuda_env import CudaVecEnv
    from pvz_rl.envs.rewards import LEDGER_METRICS

    cfg = load_config()
    cfg["training"].update(n_envs=2, batch_size=128)
    cfg["environment"]["cutoff_seconds"] = 1
    cases = [("easy", "saving", 4), ("easy", "saving", 5)]
    gpu = CudaVecEnv(cfg, "masked", 0, training=False, cases=cases)
    cpus = [PvZEnv(cfg, family=family, level=level) for level, family, _ in cases]
    for env, (_, _, seed) in zip(cpus, cases):
        env.reset(seed=seed)
    try:
        gpu.reset()
        for step in range(23):
            actions = [0, 0]
            if step == 0:
                actions = [
                    cpus[0].codec.encode(Place("sunflower", 0, 0)),
                    cpus[1].codec.encode(Place("peashooter", 0, 0)),
                ]
            elif step == 1:
                actions = [cpus[0].codec.encode(Dig(0, 0)), 0]
            reference = [env.step(a) for env, a in zip(cpus, actions)]
            *_, infos = gpu.step_tensors(torch.tensor(actions, device="cuda"))
            np.testing.assert_array_equal(
                gpu.transition_ticks.cpu(), [r[4]["ticks_advanced"] for r in reference]
            )
            for i, (r, info) in enumerate(zip(reference, infos)):
                if r[2] or r[3]:
                    for key in LEDGER_METRICS:
                        assert info["episode_metrics"][key] == pytest.approx(
                            r[4]["episode_metrics"][key], abs=1e-9
                        )
                    level, family, seed = gpu._episode[i]
                    cpus[i] = PvZEnv(cfg, family=family, level=level)
                    cpus[i].reset(seed=seed)
                    assert all(gpu.episode_metrics(i)[key] == 0 for key in LEDGER_METRICS)
    finally:
        gpu.close()


def test_partial_plant_damage_then_mower_credit_once():
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(seed=3)
    events = [
        Event("ZombieSpawned", 1, 99, (("zombie_type", "basic"),)),
        event("DamageApplied", hp=20),
        Event("MowerActivated", 1, details=(("row", 0),)),
        event("DamageApplied", source=-1, hp=180),
        Event("ZombieDefeated", 1, 99),
    ]
    parts = reward_parts(env.public, env.public, cfg, events=events)
    assert parts["effective_damage"] == 20 and parts["combat_value"] == pytest.approx(20 * 50 / 270)
    assert parts["mower_expenditure"] == 200 and parts["mower_kills"] == 1
    assert parts["net_value"] == pytest.approx(20 * 50 / 270 - 200)
    assert parts["total"] == pytest.approx((20 * 50 / 270 - 200) / 30000)


@pytest.mark.parametrize("retired", ["reward", "clock"])
def test_retired_method_rejected_before_weights_are_read(tmp_path, retired):
    from copy import deepcopy

    from pvz_rl.learning.training import initial_weights, load_policy
    from pvz_rl.provenance import write_json

    cfg = load_config()
    old = deepcopy(cfg)
    if retired == "reward":
        old["reward"]["version"] = "potential_mower_v1"
    else:
        old["training"].pop("discount_clock")
    write_json(
        tmp_path / "metadata.json", {"config": old, "condition": "masked", "learner_seed": 101}
    )
    missing = tmp_path / "does-not-exist.zip"
    with pytest.raises(ValueError, match="fresh"):
        initial_weights(missing, cfg)
    with pytest.raises(ValueError, match="fresh"):
        load_policy(missing)
