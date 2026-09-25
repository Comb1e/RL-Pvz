"""Independent lesson controls; never used as learner actions or demonstrations."""

import copy
from itertools import combinations

import numpy as np
import pytest
import torch
from pvz_game import Dig, LevelSpec, Place, Rules, Spawn

from pvz_rl.config import load_config, validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.recordings import open_playback, verify_replay
from pvz_rl.rewards import asset_value
from pvz_rl.scenarios import scenario
from pvz_rl.training_requirements import resume_protocol, transfer_protocol

LANE_PAIRS = list(combinations(range(5), 2))


def saving_case(lanes):
    return LevelSpec(
        "saving",
        tuple(Spawn(t, "basic", r) for t in (4300, 5500, 6700) for r in lanes),
        initial_sun=150,
        mowers=False,
    )


def control_action(env, flowers, invest=True, delay=0):
    obs = env.public
    if obs.tick < delay:
        return 0, flowers
    if invest and flowers < 2:
        candidate = Place("sunflower", flowers, 0)
    else:
        defended = {p.row for p in obs.plants if p.plant_type == "peashooter"}
        exposed = sorted({z.row for z in obs.zombies} - defended)
        if not exposed:
            return 0, flowers
        candidate = Place("peashooter", exposed[0], 1)
    action = env.codec.encode(candidate)
    if not env.action_masks()[action]:
        return 0, flowers
    return action, flowers + int(candidate.plant_type == "sunflower")


def run_control(env, *, invest=True, delay=0, wait=False):
    flowers, purchases = 0, []
    while env.state == "running":
        action, flowers = (0, flowers) if wait else control_action(env, flowers, invest, delay)
        if action:
            purchases.append((env.public.tick, env.codec.decode(action)))
        env.step(action)
    return purchases


def test_lesson_economics_and_pressure_independent_calculations():
    cfg, rules = load_config(), Rules()
    placement, saving = (cfg["curriculum"]["lessons"][k] for k in ("placement", "saving"))
    assert placement == dict(
        spawn_ticks=[5, 105, 205], initial_sun=100, natural_sun=False, allowed_plants=["peashooter"]
    )
    assert saving == dict(
        spawn_ticks=[4300, 5500, 6700],
        initial_sun=150,
        natural_sun=False,
        lanes_per_spawn=2,
        allowed_plants=["sunflower", "peashooter"],
    )
    g, pea, flower = rules.game, rules.plants["peashooter"], rules.plants["sunflower"]
    assert pea["cost"] == 100 and flower["cost"] == 50
    # No income without flowers: at most one shooter can ever be purchased.
    # No refunds/relocation/cross-lane shots; at least one lane stays unblocked.
    assert saving["initial_sun"] < saving["lanes_per_spawn"] * pea["cost"]
    travel = (g["spawn_x"] - g["house_x"]) // (rules.zombies["basic"]["speed"] // 100)
    assert travel == 5000 and 4300 + travel - 1 == 9299
    assert rules.zombies["basic"]["health"] // pea["damage"] == 10
    assert pea["interval_ticks"] == 150
    # One shooter needs 1500 ticks of sustained fire per basic zombie; saving
    # arrivals every 1200 ticks exceed this rate for a finite burst.
    assert 10 * 150 > saving["spawn_ticks"][1] - saving["spawn_ticks"][0] == 1200
    assert flower["first_ticks"] == 600 and flower["interval_ticks"] == 2400
    assert flower["recharge_ticks"] == 750
    # Flowers at 0/750 make six 25-sun payments. After buying both flowers and
    # the first shooter, the second shooter becomes affordable at tick 6150.
    incomes = sorted(t + 600 + k * 2400 for t in (0, 750) for k in range(3))
    assert incomes == [600, 1350, 3000, 3750, 5400, 6150]
    assert 150 - 2 * 50 + len(incomes) * 25 == 2 * 100


@pytest.mark.parametrize("lanes", LANE_PAIRS)
@pytest.mark.parametrize("mode", ["invest", "no_flowers", "wait", "flowers3", "flowers_all", "dig"])
def test_saving_success_and_necessary_income(lanes, mode):
    env = PvZEnv(load_config(), family="saving")
    env.reset(seed=4, options={"scenario": saving_case(lanes)})
    if mode in ("invest", "no_flowers", "wait"):
        purchases = run_control(env, invest=mode == "invest", wait=mode == "wait")
    elif mode == "dig":
        env.step(env.codec.encode(Place("peashooter", lanes[0], 0)))
        env.step(env.codec.encode(Dig(lanes[0], 0)))
        run_control(env, wait=True)
    else:
        flowers = 0
        while env.state == "running":
            legal = [int(x) for x in np.flatnonzero(env.action_masks()) if 1 <= x <= 45]
            action = legal[0] if legal and flowers < (3 if mode == "flowers3" else 45) else 0
            flowers += bool(action)
            env.step(action)
    result = env.episode_metrics()
    # Derived independently from six basics and nine actual sunflower payments.
    scale = env.cfg["reward"]["progress_weight"] / env.cfg["reward"]["value_scale"]
    winning_return = env.cfg["reward"]["win_reward"] + scale * (225 + 6 * 50)
    assert result["return"] == pytest.approx(result["discounted_return"])
    assert result["discounted_outcome_return"] + result[
        "discounted_development_return"
    ] == pytest.approx(result["return"])
    if mode not in ("invest", "no_flowers", "wait"):
        assert not result["win"] and result["return"] < winning_return
        assert result["return"] == pytest.approx(-2 + scale * result["net_value"])
        if mode == "dig":
            assert result["net_value"] == -100 and result["early_voluntary_digs"] == 1
        else:
            assert result["produced_sun"] > 0 and result["attacker_purchases"] == 0
        return
    assert env.state == ("won" if mode == "invest" else "lost")
    assert env.public.tick == (10501 if mode == "invest" else 9299)
    assert (
        env.episode_metrics()["attacker_purchases"]
        == {"invest": 2, "no_flowers": 1, "wait": 0}[mode]
    )
    if mode == "invest":
        assert [t for t, _ in purchases] == [0, 750, 4300, 6150]
        assert len(env.public.plants) == 4  # No sacrificial blockers/replacements.
        assert env.episode_metrics()["net_value"] == 525
        assert env.episode_metrics()["discounted_return"] == pytest.approx(winning_return, abs=1e-9)
    elif mode == "wait":
        assert env.episode_metrics()["discounted_return"] == pytest.approx(
            -2 * env.cfg["training"]["gamma"] ** 9298, abs=1e-10
        )
    env.close()


@pytest.mark.parametrize("delay,outcome,tick", [(347, "won", 10848), (348, "lost", 11998)])
def test_saving_witness_timing_boundary(delay, outcome, tick):
    env = PvZEnv(load_config(), family="saving")
    env.reset(options={"scenario": saving_case((2, 4))})
    run_control(env, delay=delay)
    assert env.state == outcome and env.public.tick == tick


@pytest.mark.parametrize("lane", range(5))
@pytest.mark.parametrize(
    "delay,outcome,tick", [(5, "won", 4359), (502, "won", 4853), (503, "lost", 5503)]
)
def test_placement_timing_boundary(lane, delay, outcome, tick):
    env = PvZEnv(load_config(), family="placement")
    env.reset(
        options={
            "scenario": LevelSpec(
                "placement",
                tuple(Spawn(t, "basic", lane) for t in (5, 105, 205)),
                initial_sun=100,
                mowers=False,
            )
        }
    )
    for _ in range(delay):
        env.step(0)
    assert env.step(env.codec.encode(Place("peashooter", lane, 0)))[4]["accepted"]
    while env.state == "running":
        env.step(0)
    assert env.state == outcome and env.public.tick == tick


@pytest.mark.parametrize("family", ["placement", "saving"])
def test_early_dig_discards_irreplaceable_budget(family):
    env = PvZEnv(load_config(), family=family)
    env.reset(seed=4)
    kind = "peashooter" if family == "placement" else "sunflower"
    env.step(env.codec.encode(Place(kind, 0, 0)))
    assert env.step(env.codec.encode(Dig(0, 0)))[4]["accepted"]
    assert env.episode_metrics()["early_voluntary_digs"] == 1
    if family == "saving":
        run_control(env)
    else:
        while env.state == "running":
            env.step(0)
    assert env.state == "lost"


def test_seeded_cases_cover_all_pairs_and_one_current_definition():
    cfg, rules = load_config(), Rules()
    cases = [scenario("easy", "saving", s, rules, cfg) for s in range(100050, 100150)]
    assert {tuple(sorted({s.row for s in case.spawns})) for case in cases} == set(LANE_PAIRS)
    assert all(len(case.spawns) == 6 and not case.mowers and not case.plants for case in cases)
    assert cfg == load_config("configs/train.toml")
    changed = copy.deepcopy(cfg)
    changed["curriculum"]["lessons"]["saving"]["spawn_ticks"] = [1000]
    assert transfer_protocol(changed) == transfer_protocol(cfg)
    assert resume_protocol(changed, "masked") != resume_protocol(cfg, "masked")


@pytest.mark.parametrize("count", [0, 6, 1.5, True])
def test_invalid_lane_count_rejected(count):
    cfg = load_config()
    cfg["curriculum"]["lessons"]["saving"]["lanes_per_spawn"] = count
    with pytest.raises(ValueError, match="lanes_per_spawn"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_explicit_natural_sun_boolean_required(value):
    cfg = load_config()
    cfg["curriculum"]["lessons"]["saving"]["natural_sun"] = value
    with pytest.raises(ValueError, match="natural_sun"):
        validate_config(cfg)


def test_saving_conserves_purchase_and_credits_only_actual_production():
    env = PvZEnv(load_config(), family="saving")
    env.reset(seed=0)
    assert asset_value(env.public) == 150
    assert env.step(env.codec.encode(Place("sunflower", 0, 0)))[1] == 0
    assert asset_value(env.public) == 150
    reward = sum(env.step(0)[1] for _ in range(1000))
    assert env.public.sun == 125 and asset_value(env.public) == 175
    assert reward == pytest.approx(25 / 30000)


@pytest.mark.parametrize("invest", [False, True])
def test_saving_cuda_states_observations_rewards_match_reference(invest):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pvz_rl.cuda_features import CudaFeatures
    from pvz_rl.cuda_lessons import LessonCudaBatch

    cfg = load_config()
    envs = [PvZEnv(cfg, family="saving") for _ in LANE_PAIRS]
    cases = [saving_case(lanes) for lanes in LANE_PAIRS]
    for env, case in zip(envs, cases):
        env.reset(seed=4, options={"scenario": case})
    batch = LessonCudaBatch(len(cases), zombie_capacity=6, max_step_ticks=1)
    cp = batch.cp
    flowers = [0] * len(cases)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        batch.reset(
            cases, [4] * len(cases), allowed=[3] * len(cases), natural_sun=[False] * len(cases)
        )
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        while envs[0].state == "running":
            actions, rewards = [], []
            for i, env in enumerate(envs):
                action, flowers[i] = control_action(env, flowers[i], invest)
                actions.append(action)
                rewards.append(env.step(action)[1])
            features.step(cp.asarray(actions, cp.int64))
            np.testing.assert_allclose(features.rewards.get(), rewards, atol=2e-7, rtol=1e-6)
            if any(actions) or envs[0].public.tick % 120 == 0 or envs[0].state != "running":
                np.testing.assert_allclose(
                    features.observations.get(),
                    np.stack([e.encoder.encode(e.public) for e in envs]),
                    atol=1e-7,
                    rtol=1e-6,
                )
                np.testing.assert_array_equal(
                    batch.masks.get(), np.stack([e.action_masks() for e in envs])
                )
                for i, env in enumerate(envs):
                    assert batch.state_hash(i) == env.game.state_hash()
                    assert batch.observe(i) == env.public
    assert all(e.state == ("won" if invest else "lost") for e in envs)


def test_sunless_recordings_verify_and_seek_without_external_settings(tmp_path):
    env = PvZEnv(load_config(), family="saving", record=True)
    env.reset(seed=4, options={"scenario": saving_case((2, 4))})
    run_control(env)
    path = tmp_path / "saving.pvzdemo"
    env.recorder.save(path)
    assert verify_replay(path).state_hash() == env.game.state_hash()
    playback = open_playback(path)
    assert playback.game.rules.game["sky_sun_amount"] == 0
    playback.seek(1000)
    assert playback.game.observe().sun == 75
    playback.seek(playback.end_tick)
    assert playback.game.state_hash() == env.game.state_hash()


def test_cuda_mixed_income_events_cap_restore_and_atomic_failure():
    from pvz_rl.action_timing import ActionPhaseGame
    from pvz_rl.cuda_lessons import LessonCudaBatch, lesson_kernel_source
    from pvz_rl.lesson_rules import sky_rules

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    case = LevelSpec("cap", (Spawn(2000, "basic", 4),), initial_sun=9990)
    games = [ActionPhaseGame(sky_rules(Rules(), flag)) for flag in (False, True)]
    for game in games:
        game.reset(case, 4)
    batch = LessonCudaBatch(2, zombie_capacity=1, diagnostic=True, max_step_ticks=10)
    batch.reset([case] * 2, [4] * 2, natural_sun=[False, True])
    cp = batch.cp
    for _ in range(100):
        # A sunflower first pays at the same tick as sky sun, including at cap.
        action = Place("sunflower", 0, 0) if games[0].observe().tick == 400 else None
        if action:
            expected = [g.step(action, ticks=0) for g in games]
            batch.step_device(cp.ones(2, cp.int64))
            for i, result in enumerate(expected):
                assert tuple(batch.events(i)) == result.events
        expected = [g.step(ticks=10) for g in games]
        batch.step_device(cp.zeros(2, cp.int64), ticks=10)
        for i, result in enumerate(expected):
            assert tuple(batch.events(i)) == result.events
            expected = [
                sum(
                    e.get("health_damage", 0) + e.get("armor_damage", 0)
                    for e in result.events
                    if e.kind == "DamageApplied" and e.get("source", 0) > 0
                ),
                sum(
                    e.get("amount", 0)
                    for e in result.events
                    if e.kind == "SunProduced" and e.get("source") == "sky"
                ),
                sum(
                    e.get("amount", 0)
                    for e in result.events
                    if e.kind == "SunProduced" and e.get("source") == "sunflower"
                ),
            ]
            assert batch.accounting[i].get().tolist() == expected
            assert batch.state_hash(i) == games[i].state_hash()
    assert [g.observe().sun for g in games] == [9965, 9990]
    snapshots = [batch.snapshot(i) for i in range(2)]
    batch.reset([case] * 2, [4] * 2, natural_sun=[True, False])
    batch.restore(snapshots)
    assert [batch.state_hash(i) for i in range(2)] == [g.state_hash() for g in games]
    altered = copy.deepcopy(snapshots)
    altered[1]["rules"]["game"]["sky_sun_amount"] = 1
    with pytest.raises(ValueError, match="only disable sky sun"):
        batch.restore(altered)
    assert [batch.state_hash(i) for i in range(2)] == [g.state_hash() for g in games]
    with pytest.raises(ValueError, match="natural_sun boolean"):
        batch.reset([case] * 2, [4] * 2, natural_sun=[False, 1])
    assert [batch.state_hash(i) for i in range(2)] == [g.state_hash() for g in games]
    with pytest.raises(RuntimeError, match="hook changed"):
        lesson_kernel_source("unexpected engine source")


def test_mixed_vector_reset_uses_episode_family_and_leaves_normal_rules_unchanged():
    from pvz_rl.cuda_env import CudaVecEnv

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = load_config()
    cfg["training"].update(n_envs=3, rollout_size=384, batch_size=128)
    cases = [("easy", family, 4) for family in ("placement", "saving", "preset")]
    env = CudaVecEnv(cfg, "masked", 101, training=False, cases=cases)
    references = [PvZEnv(cfg, level=level, family=family) for level, family, _ in cases]
    try:
        env.reset()
        for ref, (_, _, seed) in zip(references, cases):
            ref.reset(seed=seed)
        for _ in range(1000):
            env.step_tensors(torch.zeros(3, dtype=torch.long, device="cuda"), autoreset=False)
            for ref in references:
                ref.step(0)
        assert [ref.public.sun for ref in references] == [100, 150, 75]
        for i, ref in enumerate(references):
            assert env.batch.state_hash(i) == ref.game.state_hash()
        untouched = env.batch.state_hash(1)
        with env.device_context():
            env.reset_indices([0, 2], [cases[2], cases[0]])
        references[0].reset(seed=4, options={"family": "preset"})
        references[2].reset(seed=4, options={"family": "placement"})
        assert env.batch.state_hash(1) == untouched
        for _ in range(1000):
            env.step_tensors(torch.zeros(3, dtype=torch.long, device="cuda"), autoreset=False)
            for ref in references:
                ref.step(0)
        for i, ref in enumerate(references):
            assert env.batch.state_hash(i) == ref.game.state_hash()
    finally:
        env.close()
