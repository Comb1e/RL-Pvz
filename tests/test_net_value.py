"""Independent net-value arithmetic and physical-time return controls."""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from pvz_game import Dig, InitialPlant, LevelSpec, Place, Spawn
from pvz_game.types import Event

from pvz_rl.config import load_config
from pvz_rl.cuda_buffer import TensorRolloutBuffer
from pvz_rl.env import PvZEnv
from pvz_rl.rewards import reward_parts


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
    assert late == early == 50  # 150 damage value minus 100 lost asset.
    parts = reward_parts(full, gone, cfg, events=events)
    assert parts["effective_damage"] == 600 and parts["plant_value_loss"] == 100
    assert parts["total"] == pytest.approx(50 / 3000)


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
        ("conehead", 1, 0),
        ("buckethead", 1, 175),
        ("basic", 0, -150),
    ],
)
def test_real_explosions_break_even_and_empty_loss_on_cpu_and_cuda(kind, count, expected):
    from pvz_rl.cuda_features import REWARD_FIELDS, CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch

    cfg = load_config()
    # A remote future zombie keeps the control running after this detonation.
    case = LevelSpec(
        "bomb",
        tuple([Spawn(1, kind, 2, x=1400)] * count + [Spawn(9999, "basic", 4)]),
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
        actions = [env.codec.encode(Place("cherry_bomb", 2, 1))] + [0] * 25
        for action in actions:
            info = env.step(action)[4]
            features.step(cp.asarray([action], cp.int64))
            for key, value in zip(REWARD_FIELDS, features.parts.get()[0]):
                assert value == pytest.approx(info["reward_parts"][key], abs=1e-9), key
        assert batch.state_hash(0) == env.game.state_hash()
    assert env.episode_metrics()["net_value"] == expected
    assert (
        env.episode_metrics()["effective_damage"]
        == {"basic": 200, "conehead": 600, "buckethead": 1300}[kind] * count
    )
    assert -100 > -600  # One-basic bomb costs less than consuming a mower.


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("gamma,lam", [(0.999, 0.999), (0.0, 0.0), (1.0, 1.0)])
def test_variable_duration_gae_timeout_and_zero_time_boundaries(device, gamma, lam):
    buffer = TensorRolloutBuffer(
        4,
        spaces.Box(-100, 100, (1,), dtype=np.float32),
        spaces.Discrete(2),
        device=device,
        n_envs=2,
        gamma=gamma,
        gae_lambda=lam,
    )
    duration = np.array([[0, 1], [1, 0], [7, 2], [0, 9]], dtype=np.int64)
    values = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    rewards = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    starts = np.array([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.float32)
    for step in range(4):
        if step == 1:
            buffer.add_timeouts(
                torch.tensor([0], device=device), torch.tensor([[10.0]], device=device)
            )
        buffer.add(
            torch.tensor(values[step, :, None], device=device),
            torch.zeros(2, device=device),
            torch.tensor(rewards[step], device=device),
            torch.tensor(starts[step], device=device),
            None,
            torch.zeros(2, device=device),
            torch.ones(2, 2, dtype=torch.bool, device=device),
            durations=torch.tensor(duration[step], device=device),
        )

    class Critic:
        def predict_values(self, obs):
            return obs

    last = buffer.evaluate_values(Critic(), torch.tensor([[9.0], [10.0]], device=device), 3)
    rewards[1, 0] += gamma ** duration[1, 0] * 10
    done = torch.tensor([False, True], device=device)
    buffer.compute_returns_and_advantage(last, done)
    expected = np.zeros((4, 2))
    carry = np.zeros(2)
    for step in reversed(range(4)):
        live = 1 - (done.cpu().numpy() if step == 3 else starts[step + 1])
        next_values = np.array([9, 10]) if step == 3 else values[step + 1]
        delta = rewards[step] + gamma ** duration[step] * next_values * live - values[step]
        carry = delta + (gamma * lam) ** duration[step] * live * carry
        expected[step] = carry
    np.testing.assert_allclose(buffer.advantages.cpu(), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(buffer.returns.cpu(), expected + values, rtol=2e-6, atol=2e-6)


def test_projectiles_keep_credit_after_voluntary_dig():
    from pvz_rl.cuda_features import CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch

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
        actions = [env.codec.encode(Place("peashooter", 0, 0)), 0, env.codec.encode(Dig(0, 0))] + [
            0
        ] * 80
        for step, action in enumerate(actions):
            reward = env.step(action)[1]
            features.step(cp.asarray([action], cp.int64))
            assert features.rewards.get()[0] == pytest.approx(reward, abs=1e-8)
            if step == 1:
                assert env.public.projectiles
        assert batch.state_hash(0) == env.game.state_hash()
    assert env.episode_metrics()["effective_damage"] == 20
    assert env.episode_metrics()["net_value"] == -95


@pytest.mark.parametrize("lanes", list(combinations(range(5), 2)))
@pytest.mark.parametrize("mode", ["flowers", "dig"])
def test_all_saving_pairs_rank_unproductive_controls_below_winner(lanes, mode):
    env = PvZEnv(load_config(), family="saving")
    env.reset(
        options={
            "scenario": LevelSpec(
                "saving",
                tuple(Spawn(t, "basic", r) for t in (860, 1100, 1340) for r in lanes),
                initial_sun=150,
                mowers=False,
            )
        }
    )
    if mode == "dig":
        for row, kind in enumerate(("sunflower", "peashooter")):
            action = env.codec.encode(Place(kind, row, 0))
            assert env.action_masks()[action]
            env.step(action)
            env.step(env.codec.encode(Dig(row, 0)))
    else:
        # Fixed two investments without attackers; no learner ever sees this control.
        for row in range(2):
            action = env.codec.encode(Place("sunflower", row, 0))
            while not env.action_masks()[action]:
                env.step(0)
            env.step(action)
    while env.state == "running":
        env.step(0)
    metrics = env.episode_metrics()
    assert not metrics["win"] and metrics["discounted_return"] < 0.177950
    if mode == "dig":
        assert metrics["net_value"] == -150
        assert metrics["early_voluntary_digs"] == 2
        assert metrics["discounted_return"] == pytest.approx(-0.361679, abs=1e-6)
    else:
        assert metrics["produced_sun"] > 0
        assert metrics["attacker_purchases"] == 0


def test_mixed_cuda_resets_preserve_duration_and_episode_ledgers():
    from pvz_rl.cuda_env import CudaVecEnv
    from pvz_rl.rewards import LEDGER_METRICS

    cfg = load_config()
    cfg["training"].update(n_envs=2, rollout_size=256, batch_size=128)
    cfg["environment"]["cutoff_seconds"] = 1
    cases = [("easy", "saving", 4), ("easy", "placement", 5)]
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
    assert parts["effective_damage"] == 20 and parts["combat_value"] == 5
    assert parts["mower_expenditure"] == 600 and parts["mower_kills"] == 1
    assert parts["net_value"] == -595
    assert parts["total"] == pytest.approx(-595 / 3000)


@pytest.mark.parametrize("retired", ["reward", "clock"])
def test_retired_method_rejected_before_weights_are_read(tmp_path, retired):
    from copy import deepcopy

    from pvz_rl.provenance import write_json
    from pvz_rl.training import initial_weights, load_policy

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
