from dataclasses import replace

import numpy as np
import pytest
from pvz_game import Dig, Game, LevelSpec, Place, Spawn, Status, Wait

from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.env import EpisodeState, PvZEnv
from pvz_rl.envs.rewards import asset_value


def test_all_406_actions_have_independent_expected_indices(cfg):
    import torch

    from pvz_rl.envs.actions import ActionSchema

    codec = ActionCodec(cfg)
    assert codec.size == 406 and isinstance(codec.decode(0), Wait)
    for p, kind in enumerate(cfg["environment"]["plants"]):
        for row in range(5):
            for col in range(9):
                index = 1 + 45 * p + 9 * row + col
                assert codec.decode(index) == Place(kind, row, col)
                assert codec.encode(Place(kind, row, col)) == index
    for row in range(5):
        for col in range(9):
            assert codec.decode(361 + row * 9 + col) == Dig(row, col)
    assert [codec.encode(codec.decode(i)) for i in range(406)] == list(range(406))
    for device in ("cpu", "cuda"):
        ids = torch.arange(406, device=device)
        kinds, plants, tiles = ActionSchema.unpack(ids)
        torch.testing.assert_close(ActionSchema.pack(kinds, plants, tiles), ids, rtol=0, atol=0)
        assert kinds[0] == 0 and (kinds[1:361] == 2).all() and (kinds[361:] == 1).all()
        assert (plants[361:] == 0).all() and tiles[0] == 0


@pytest.mark.parametrize("action", [-1, 406, 1.5, True, "0"])
def test_malformed_actions_do_not_mutate(cfg, action):
    env = PvZEnv(cfg)
    env.reset(seed=42)
    before = env.game.state_hash()
    with pytest.raises((ValueError, TypeError)):
        env.step(action)
    assert env.game.state_hash() == before


@pytest.mark.parametrize("sun,can_plant", [(49, False), (50, True)])
def test_legality_exact_cost_cooldown_and_dig(cfg, sun, can_plant):
    env = PvZEnv(cfg)
    env.reset(
        seed=42,
        options={"scenario": LevelSpec("money", (Spawn(1000, "basic", 0),), initial_sun=sun)},
    )
    flower = env.codec.encode(Place("sunflower", 0, 0))
    assert bool(env.action_masks()[flower]) is can_plant
    _, _, _, _, info = env.step(flower)
    assert info["accepted"] is can_plant
    assert env.public.tick == 10
    assert env.public.sun == (0 if can_plant else 49)
    if can_plant:
        assert not env.action_masks()[env.codec.encode(Place("sunflower", 0, 1))]
        assert env.action_masks()[env.codec.encode(Dig(0, 0))]
        env.step(env.codec.encode(Dig(0, 0)))
        assert env.public.sun == 0
        assert not env.action_masks()[env.codec.encode(Dig(0, 0))]


def test_mask_matches_every_engine_action_and_cooldown_boundary(cfg):
    env = PvZEnv(cfg)
    env.reset(
        seed=42,
        options={"scenario": LevelSpec("mask", (Spawn(1000, "basic", 0),), initial_sun=500)},
    )
    env.step(env.codec.encode(Place("sunflower", 0, 0)))
    for _ in range(74):
        expected = [env.game.validate_action(action).accepted for action in env.codec.actions]
        np.testing.assert_array_equal(env.action_masks(), expected)
        assert not env.action_masks()[env.codec.encode(Place("sunflower", 1, 1))]
        env.step(0)
    assert env.public.tick == 750
    assert env.action_masks()[env.codec.encode(Place("sunflower", 1, 1))]


def test_single_action_not_repeated_in_tick_batch(cfg):
    env = PvZEnv(cfg)
    env.reset(seed=57)
    control = Game()
    control.reset("standard", 57)
    action = Place("sunflower", 2, 0)
    env.step(env.codec.encode(action))
    control.step(action)
    for _ in range(9):
        control.step(Wait())
    assert env.game.state_hash() == control.state_hash()


def test_empty_scenario_true_terminal_and_reset(cfg):
    env = PvZEnv(cfg)
    env.reset(seed=1, options={"scenario": LevelSpec("empty")})
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["ticks_advanced"] == 1
    assert env.state == EpisodeState.WON
    assert asset_value(env.public) == env.public.sun
    assert not env.action_masks().any()
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=2)
    assert env.state == EpisodeState.RUNNING and env.action_masks()[0]


def test_house_breach_is_loss_not_truncation(cfg):
    env = PvZEnv(cfg)
    env.reset(
        seed=3,
        options={"scenario": LevelSpec("breach", (Spawn(1, "basic", 0, x=0),), mowers=False)},
    )
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step(0)
        if terminated:
            break
    assert terminated and not truncated and env.state == EpisodeState.LOST
    assert info["reward_parts"]["terminal"] == -2
    assert reward < 0 and info["episode_metrics"]["win"] == 0


def test_time_cutoff_keeps_final_observation_and_assets(cfg):
    cfg["environment"].update(cutoff_seconds=1, decision_ticks=13)
    env = PvZEnv(cfg)
    env.reset(seed=4)
    for _ in range(7):
        env.step(0)
    _, reward, terminated, truncated, info = env.step(0)
    assert not terminated and truncated and info["ticks_advanced"] == 9
    assert env.public.tick == 100 and env.public.status == Status.RUNNING
    assert asset_value(env.public) > 0
    assert reward == 0
    assert info["episode_metrics"]["win"] == 0


def test_vector_truncation_has_terminal_observation_for_bootstrapping(cfg):
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg["environment"]["cutoff_seconds"] = 1
    vec = DummyVecEnv([lambda: PvZEnv(cfg)])
    try:
        vec.reset()
        for _ in range(9):
            vec.step([0])
        reset_obs, _, dones, infos = vec.step([0])
        assert dones[0] and infos[0]["TimeLimit.truncated"]
        assert not np.array_equal(reset_obs[0], infos[0]["terminal_observation"])
    finally:
        vec.close()


def test_future_schedule_ids_and_names_are_not_observation_features(cfg):
    env = PvZEnv(cfg)
    a, _ = env.reset(seed=4, options={"scenario": LevelSpec("hidden-A", (Spawn(800, "basic", 0),))})
    mask_a = env.action_masks()
    b, _ = env.reset(
        seed=876, options={"scenario": LevelSpec("hidden-B", (Spawn(1900, "buckethead", 4),))}
    )
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(mask_a, env.action_masks())
    env.step(env.codec.encode(Place("sunflower", 2, 0)))
    c = env.encoder.encode(env.public)
    changed = replace(
        env.public,
        plants=tuple(replace(p, id=999) for p in env.public.plants),
        level="different-label",
    )
    np.testing.assert_array_equal(c, env.encoder.encode(changed))


def test_hybrid_only_proposes_legal_single_actions(cfg):
    cfg["conditions"]["hybrid"] = dict(masked=True, shaped=True, curriculum=False, hybrid=True)
    env = PvZEnv(cfg, condition="hybrid")
    env.reset(seed=42)
    assert env.action_space.n == 5
    for _ in range(80):
        before = env.public.tick
        candidates = env.candidates()
        for candidate in candidates:
            if candidate is not None:
                assert env.game.validate_action(candidate).accepted
        action = int(np.flatnonzero(env.action_masks())[-1])
        env.step(action)
        assert env.public.tick - before <= 10
    env.reset(seed=43)
    mask = env.action_masks()
    unavailable = int(np.flatnonzero(~mask)[0])
    _, _, _, _, info = env.step(unavailable)
    assert not info["accepted"] and env.public.tick == 10


def test_training_reset_stays_inside_training_split(cfg):
    env = PvZEnv(cfg, training=True)
    for seed in (0, 42, 100001, 200001):
        env.reset(seed=seed)
        assert 1000 <= env.episode_seed <= 99999
    env.set_progress(1234)
    env.step(0)
    assert env.progress == 1234 + cfg["training"]["n_envs"]
