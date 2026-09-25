"""Independent lesson controls; never used as learner actions or demonstrations."""

import copy
from itertools import combinations

import numpy as np
import pytest
import torch
from pvz_game import Dig, InitialPlant, LevelSpec, Place, Rules, Spawn

from pvz_rl.config import load_config, validate_config
from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.env import PvZEnv
from pvz_rl.envs.rewards import asset_value
from pvz_rl.envs.scenarios import scenario
from pvz_rl.learning.training_requirements import resume_protocol, transfer_protocol
from pvz_rl.presentation.recordings import open_playback, verify_replay

LANE_TRIPLES = list(combinations(range(5), 3))


def saving_case(lanes):
    return LevelSpec(
        "saving",
        tuple(Spawn(t, "basic", r) for t in (7500, 8700, 9900) for r in lanes),
        initial_sun=100,
        mowers=False,
    )


def control_action(env, flowers, invest=True, delay=0):
    obs = env.public
    if obs.tick < delay:
        return 0, flowers
    if invest and flowers < 5:
        candidate = Place("sunflower", flowers, 0)
    else:
        lanes = sorted({z.row for z in obs.zombies if not z.headless})
        defended = {p.row for p in obs.plants if p.plant_type == "peashooter"}
        exposed = sorted(set(lanes) - defended)
        if not exposed:
            return 0, flowers
        candidate = Place("peashooter", exposed[0], 1)
        if invest and not env.game.validate_action(candidate).accepted:
            mined = {p.row for p in obs.plants if p.plant_type == "potato_mine"}
            for row in reversed(exposed):
                mine = Place("potato_mine", row, 3)
                if row not in mined and env.game.validate_action(mine).accepted:
                    candidate = mine
                    break
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
    saving = cfg["curriculum"]["lessons"]["saving"]
    assert set(cfg["curriculum"]["lessons"]) == {"saving"}
    assert saving["spawn_ticks"] == [7500, 8700, 9900]
    assert saving["initial_sun"] == 100 and saving["lanes_per_spawn"] == 3
    assert not saving["natural_sun"]
    assert set(saving["allowed_plants"]) == set(rules.plants)
    g, pea, flower, mine = (
        rules.game,
        rules.plants["peashooter"],
        rules.plants["sunflower"],
        rules.plants["potato_mine"],
    )
    assert all(
        rules.plants[k]["cost"] > 100 for k in ("cherry_bomb", "chomper", "snow_pea", "repeater")
    )
    assert pea["cost"] == 100 and flower["cost"] == rules.plants["wall_nut"]["cost"] == 50
    assert mine["cost"] == 25 and 100 // mine["cost"] == 4
    assert (100 - flower["cost"]) // mine["cost"] == 2 < 3
    # A four-mine budget leaves one of three lanes with at most one mine.
    # Even the conservative three-second eating delay cannot close the 2.4-tile gap
    # into the mine's one-tile explosion interval.
    spacing = (8700 - 7500) / g["tick_rate"] * rules.zombies["basic"]["speed"]
    delay = mine["health"] / g["bite_damage"] * g["bite_ticks"] / g["tick_rate"]
    assert spacing == 1620 and delay == 3
    assert spacing - delay * rules.zombies["basic"]["speed"] > g["units_per_tile"]
    assert flower["first_ticks"] == 300 and flower["interval_ticks"] == 2350
    assert flower["recharge_ticks"] == 750
    # Worst production delays still finance five flowers before first spawn.
    purchases = [0, 751, 2001, 3750, 5000]
    for i, tick in enumerate(purchases):
        payments = sum(max(0, (tick - t - 1250) // 2500 + 1) for t in purchases[:i])
        assert 100 - i * 50 + payments * 25 >= 50


@pytest.mark.parametrize("lanes", LANE_TRIPLES)
@pytest.mark.parametrize("mode", ["invest", "no_flowers", "wait", "dig", "mines"])
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
        purchases = 0
        while env.state == "running":
            candidate = env.codec.encode(
                Place("potato_mine", lanes[purchases % 3], 1 + purchases // 3)
            )
            action = candidate if purchases < 4 and env.action_masks()[candidate] else 0
            purchases += bool(action)
            env.step(action)
    result = env.episode_metrics()
    # Nine full basic HP pools, plus actual production; no plant loss in the witness.
    scale = env.cfg["reward"]["progress_weight"] / env.cfg["reward"]["value_scale"]
    winning_return = env.cfg["reward"]["win_reward"] + scale * (
        env.episode_metrics()["produced_sun"]
        + env.episode_metrics()["effective_damage"] * 50 / 270
        - env.episode_metrics()["plant_value_loss"]
    )
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
            assert result["produced_sun"] == 0 and result["plant_usage"] == {"potato_mine": 4}
        return
    assert env.state == ("won" if mode == "invest" else "lost")
    assert 9900 < env.public.tick < 20000
    assert (
        env.episode_metrics()["attacker_purchases"]
        == {"invest": 3, "no_flowers": 1, "wait": 0}[mode]
    )
    if mode == "invest":
        assert len(purchases) == 9 and purchases[4][0] < 7500
        assert len(env.public.plants) == 8
        assert result["produced_sun"] > 0
        assert result["net_value"] == pytest.approx(
            result["produced_sun"] + result["effective_damage"] * 50 / 270 - 25
        )
        assert result["discounted_return"] == pytest.approx(winning_return, abs=1e-9)
    elif mode == "wait":
        assert env.episode_metrics()["discounted_return"] == pytest.approx(
            -2 * env.cfg["training"]["gamma"] ** 12498, abs=1e-10
        )
    env.close()


def test_all_species_legal_and_mine_arming_and_blast_boundaries():
    env = PvZEnv(load_config(), family="saving")
    env.reset(
        options={
            "scenario": LevelSpec(
                "affordable", (Spawn(5000, "basic", 0),), initial_sun=1000, mowers=False
            )
        }
    )
    for _ in range(3501):
        env.step(0)
    for kind in env.cfg["environment"]["plants"]:
        assert env.action_masks()[env.codec.encode(Place(kind, 0, 0))]
    case = LevelSpec(
        "mine-boundary",
        tuple(
            Spawn(1607, "basic", r, x=x) for r, x in [(0, 1000), (0, 1999), (0, 2000), (1, 1500)]
        ),
        plants=(InitialPlant("potato_mine", 0, 1),),
        mowers=False,
    )
    game = ActionPhaseGame()
    game.reset(case, 4)
    game.step(ticks=1605)
    assert game.observe().plants[0].state == "rising"
    game.step()
    assert game.observe().plants[0].state == "armed"
    result = game.step()
    assert len([e for e in result.events if e.kind == "ZombieDefeated"]) == 2
    assert {(z.row, z.health) for z in game.observe().zombies} == {(0, 270), (1, 270)}


def test_seeded_cases_cover_all_triples_and_one_current_definition():
    cfg, rules = load_config(), Rules()
    cases = [scenario("easy", "saving", s, rules, cfg) for s in range(100050, 100150)]
    assert {tuple(sorted({s.row for s in case.spawns})) for case in cases} == set(LANE_TRIPLES)
    assert all(len(case.spawns) == 9 and not case.mowers and not case.plants for case in cases)
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
    assert asset_value(env.public) == 100
    assert env.step(env.codec.encode(Place("sunflower", 0, 0)))[1] == 0
    assert asset_value(env.public) == 100
    reward = sum(env.step(0)[1] for _ in range(1000))
    assert env.public.sun == 75 and asset_value(env.public) == 125
    assert reward == pytest.approx(25 / 30000)


@pytest.mark.parametrize("invest", [False, True])
def test_saving_cuda_states_observations_rewards_match_reference(invest):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pvz_rl.envs.cuda_features import CudaFeatures
    from pvz_rl.envs.cuda_lessons import LessonCudaBatch

    cfg = load_config()
    envs = [PvZEnv(cfg, family="saving") for _ in LANE_TRIPLES]
    cases = [saving_case(lanes) for lanes in LANE_TRIPLES]
    for env, case in zip(envs, cases):
        env.reset(seed=4, options={"scenario": case})
    batch = LessonCudaBatch(len(cases), zombie_capacity=9, max_step_ticks=1)
    cp = batch.cp
    flowers = [0] * len(cases)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        batch.reset(
            cases, [4] * len(cases), allowed=[255] * len(cases), natural_sun=[False] * len(cases)
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
    env.reset(seed=4, options={"scenario": saving_case((0, 2, 4))})
    run_control(env)
    path = tmp_path / "saving.pvzdemo"
    env.recorder.save(path)
    assert verify_replay(path).state_hash() == env.game.state_hash()
    playback = open_playback(path)
    assert playback.game.rules.game["sky_sun_amount"] == 0
    playback.seek(1000)
    assert playback.game.observe().sun == 25
    playback.seek(playback.end_tick)
    assert playback.game.state_hash() == env.game.state_hash()


def test_cuda_mixed_income_events_cap_restore_and_atomic_failure():
    from pvz_rl.envs.action_timing import ActionPhaseGame
    from pvz_rl.envs.cuda_lessons import LessonCudaBatch, lesson_kernel_source
    from pvz_rl.envs.lesson_rules import sky_rules

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
    assert [g.observe().sun for g in games] == [9940, 9965]
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
    from pvz_rl.envs.cuda_env import CudaVecEnv

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = load_config()
    cfg["training"].update(n_envs=3, batch_size=128)
    cases = [("easy", family, 4) for family in ("saving", "saving", "preset")]
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
        assert [ref.public.sun for ref in references] == [100, 100, 75]
        for i, ref in enumerate(references):
            assert env.batch.state_hash(i) == ref.game.state_hash()
        untouched = env.batch.state_hash(1)
        with env.device_context():
            env.reset_indices([0, 2], [cases[2], cases[0]])
        references[0].reset(seed=4, options={"family": "preset"})
        references[2].reset(seed=4, options={"family": "saving"})
        assert env.batch.state_hash(1) == untouched
        for _ in range(1000):
            env.step_tensors(torch.zeros(3, dtype=torch.long, device="cuda"), autoreset=False)
            for ref in references:
                ref.step(0)
        for i, ref in enumerate(references):
            assert env.batch.state_hash(i) == ref.game.state_hash()
    finally:
        env.close()
