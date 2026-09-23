"""Economy necessity proof plus independent feasible and failing controls.

These controls only verify tasks; they never supply learner actions or examples.
"""

import argparse
import copy
import random
from itertools import combinations

import numpy as np
import pytest
import torch
from pvz_game import LevelSpec, Place, Rules, Spawn

from pvz_rl.cli import configured
from pvz_rl.config import load_config, validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.provenance import write_json
from pvz_rl.rewards import potential
from pvz_rl.scenarios import namespace_seed, scenario
from pvz_rl.training_requirements import resume_protocol, transfer_protocol

LANE_TRIPLES = list(combinations(range(5), 3))


def saving_case(lanes):
    # Independent literal fixture: assert shipped configuration agrees separately.
    return LevelSpec(
        "saving",
        tuple(Spawn(1000, "basic", lane) for lane in lanes),
        initial_sun=50,
        mowers=False,
    )


def control_action(env, flowers, invest):
    obs = env.public
    if invest and flowers < 2:
        candidate = Place("sunflower", flowers, 1)
    else:
        defended = {p.row for p in obs.plants if p.plant_type == "peashooter"}
        exposed = sorted({z.row for z in obs.zombies} - defended)
        if not exposed:
            return 0, flowers
        candidate = Place("peashooter", exposed[0], 0)
    action = env.codec.encode(candidate)
    if not env.action_masks()[action]:
        return 0, flowers
    return action, flowers + int(candidate.plant_type == "sunflower")


def test_free_sun_cannot_fund_all_threatened_lanes_before_first_breach():
    cfg, rules = load_config(), Rules()
    lesson = cfg["curriculum"]["lessons"]["saving"]
    assert lesson == dict(
        spawn_ticks=[1000],
        initial_sun=50,
        lanes_per_spawn=3,
        allowed_plants=["sunflower", "peashooter"],
    )
    g = rules.game
    # Integer displacement is 10 units/tick. Spawning precedes movement, so
    # the first move is at tick 1000 and move 1000 reaches the house at 1999.
    distance = g["spawn_x"] - g["house_x"]
    speed = rules.zombies["basic"]["speed"]
    assert speed % g["tick_rate"] == 0
    travel_ticks = distance // (speed // g["tick_rate"])
    assert travel_ticks == 1000
    breach_tick = 1000 + travel_ticks - 1
    free_budget = 50 + (breach_tick // g["sky_sun_ticks"]) * g["sky_sun_amount"]
    assert breach_tick == 1999 and free_budget == 275
    assert rules.plants["peashooter"]["cost"] == 100
    assert free_budget < 3 * rules.plants["peashooter"]["cost"]
    # With no sunflowers, peas are the only allowed purchasable plant. Each
    # purchase acts in one lane; digging has no refund or relocation. At most
    # two lanes can ever receive a plant, leaving one entirely unblocked.
    assert (50 + 10 * g["sky_sun_amount"]) == 300  # tick 2000 is already too late.


@pytest.mark.parametrize("lanes", LANE_TRIPLES)
@pytest.mark.parametrize("invest", [False, True])
def test_economy_control_and_no_sunflower_failure(lanes, invest):
    env = PvZEnv(load_config(), family="saving")
    env.reset(seed=4, options={"scenario": saving_case(lanes)})
    flowers, purchases = 0, []
    while env.state == "running":
        action, flowers = control_action(env, flowers, invest)
        if action:
            purchases.append((env.public.tick, env.codec.decode(action)))
        env.step(action)
    assert env.state == ("won" if invest else "lost")
    assert env.public.tick == (1831 if invest else 1999)
    assert env.episode_metrics()["attacker_purchases"] == (3 if invest else 2)
    if invest:
        assert [t for t, _ in purchases] == [0, 200, 1000, 1150, 1560]
    env.close()


def test_seeded_cases_cover_all_lane_triples_and_preserve_archived_single_lane():
    cfg, rules = load_config(), Rules()
    cases = [scenario("easy", "saving", s, rules, cfg) for s in range(100050, 100150)]
    assert {tuple(s.row for s in case.spawns) for case in cases} == set(LANE_TRIPLES)
    assert all(len(case.spawns) == 3 and not case.mowers and not case.plants for case in cases)
    old = copy.deepcopy(cfg)
    old["curriculum"]["lessons"]["saving"].update(spawn_ticks=[1200, 1280, 1360])
    del old["curriculum"]["lessons"]["saving"]["lanes_per_spawn"]
    for seed in (0, 42, 100050):
        case = scenario("easy", "saving", seed, rules, old)
        expected = random.Random(namespace_seed("saving", seed)).randrange(5)
        assert [(s.tick, s.row) for s in case.spawns] == [(t, expected) for t in [1200, 1280, 1360]]


@pytest.mark.parametrize("count", [0, 6, 1.5, True])
def test_invalid_lane_count_rejected(count):
    cfg = load_config()
    cfg["curriculum"]["lessons"]["saving"]["lanes_per_spawn"] = count
    with pytest.raises(ValueError, match="lanes_per_spawn"):
        validate_config(cfg)


def test_archived_resume_preserves_economy_and_lesson_and_explicit_transfer_changes_them(tmp_path):
    old = load_config()
    old["reward"]["economy_weight"] = 0.5
    old["training"].update(n_envs=128, rollout_size=16384)
    old["curriculum"]["lessons"]["saving"].update(spawn_ticks=[1200, 1280, 1360])
    del old["curriculum"]["lessons"]["saving"]["lanes_per_spawn"]
    write_json(tmp_path / "metadata.json", {"config": old})
    resumed = configured(
        argparse.Namespace(
            command="train",
            config=None,
            resume=tmp_path / "final.zip",
        )
    )
    assert resumed == old
    transferred = configured(
        argparse.Namespace(
            command="train",
            config="configs/train.toml",
            init_from=tmp_path / "final.zip",
            stage="saving",
        )
    )
    assert transferred["reward"]["economy_weight"] == 0.1
    assert transferred["curriculum"]["lessons"]["saving"]["lanes_per_spawn"] == 3
    assert transfer_protocol(transferred) == transfer_protocol(old)
    assert resume_protocol(transferred, "masked") != resume_protocol(old, "masked")


def test_economy_shaping_reduced_fivefold_without_purchase_bonus():
    cfg = load_config()
    old = copy.deepcopy(cfg)
    old["reward"]["economy_weight"] = 0.5
    env = PvZEnv(cfg, family="saving")
    env.reset(seed=0)
    initial = potential(env.public, cfg)
    assert cfg["reward"]["economy_weight"] == 0.1
    assert initial == pytest.approx(0.1 * 50 / 300)
    assert potential(env.public, old) == pytest.approx(5 * initial)
    _, reward, _, _, _ = env.step(env.codec.encode(Place("sunflower", 0, 0)))
    assert potential(env.public, cfg) == initial
    assert reward == pytest.approx(-0.001 * initial)
    for _ in range(120):
        env.step(0)
    assert env.public.sun == 25
    assert potential(env.public, cfg) == pytest.approx(0.1 * 75 / 300)
    env.close()


@pytest.mark.parametrize("invest", [False, True])
def test_new_saving_cuda_states_observations_rewards_match_reference(invest):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pvz_game.cuda import CudaBatch

    from pvz_rl.cuda_features import CudaFeatures

    cfg = load_config()
    envs = [PvZEnv(cfg, family="saving") for _ in LANE_TRIPLES]
    cases = [saving_case(lanes) for lanes in LANE_TRIPLES]
    for env, case in zip(envs, cases):
        env.reset(seed=4, options={"scenario": case})
    batch = CudaBatch(len(cases), zombie_capacity=3, max_step_ticks=1)
    batch.reset(cases, [4] * len(cases), allowed=[3] * len(cases))  # sunflower + pea
    cp = batch.cp
    flowers = [0] * len(cases)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
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
            if any(actions) or envs[0].public.tick % 200 == 0 or envs[0].state != "running":
                expected = np.stack([e.encoder.encode(e.public) for e in envs])
                np.testing.assert_allclose(
                    features.observations.get(), expected, atol=1e-7, rtol=1e-6
                )
                np.testing.assert_array_equal(
                    batch.masks.get(), np.stack([e.action_masks() for e in envs])
                )
                for i, env in enumerate(envs):
                    assert batch.state_hash(i) == env.game.state_hash()
    assert all(e.state == ("won" if invest else "lost") for e in envs)
    for env in envs:
        env.close()
