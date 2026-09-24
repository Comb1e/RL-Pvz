import copy

import numpy as np
import pytest
from pvz_game import LevelSpec, Place, Spawn

from pvz_rl.config import research_config, runtime_settings, validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.timing import TrainingTimings


def plain(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["runtime"] = dict.fromkeys(runtime_settings(cfg), False)
    return cfg


def test_legacy_runtime_defaults_and_compatibility(cfg):
    old = copy.deepcopy(cfg)
    old.pop("runtime")
    validate_config(old)
    assert runtime_settings(old) == runtime_settings(cfg)
    assert research_config(old) == research_config(cfg) == research_config(plain(cfg))
    cfg["runtime"]["coalesce_masks"] = "false"
    with pytest.raises(ValueError, match="Runtime"):
        validate_config(cfg)


def test_phase_timing_excludes_validation_and_captures_final_update():
    ticks = iter([1, 3, 6, 100, 104, 105])
    timing = TrainingTimings(clock=lambda: next(ticks))
    timing.end_update()  # No preceding update at startup.
    timing.begin_collection()
    timing.end_collection()
    timing.end_update()
    timing.begin_collection()  # Long validation gap must not enter either phase.
    timing.end_collection()
    timing.end_update()
    timing.end_update()  # Idempotent.
    assert timing.snapshot() == {
        "collection_seconds": 6,
        "optimization_seconds": 4,
        "last_collection_seconds": 4,
        "last_optimization_seconds": 1,
    }


def test_legality_cache_requeries_only_after_relevant_public_change(cfg, monkeypatch):
    env = PvZEnv(cfg)
    env.reset(
        seed=42,
        options={"scenario": LevelSpec("quiet", (Spawn(2000, "basic", 0),), initial_sun=200)},
    )
    calls = []
    original = env.game.legal_actions

    def query():
        calls.append(env.public.tick)
        return original()

    monkeypatch.setattr(env.game, "legal_actions", query)
    expected = [env.game.validate_action(a).accepted for a in env.codec.actions]
    np.testing.assert_array_equal(env.action_masks(), expected)
    for _ in range(5):
        env.step(0)
        np.testing.assert_array_equal(env.action_masks(), expected)
    assert calls == [0]  # Time passes without changing legality.
    # Plant occupancy, sun thresholds, and cooldown change. Then the bomb destroys itself.
    env.step(env.codec.encode(Place("cherry_bomb", 2, 0)))
    for _ in range(14):
        expected = [env.game.validate_action(a).accepted for a in env.codec.actions]
        np.testing.assert_array_equal(env.action_masks(), expected)
        env.step(0)
    assert len(calls) >= 3
    env.reset(seed=43)
    np.testing.assert_array_equal(
        env.action_masks(), [env.game.validate_action(a).accepted for a in env.codec.actions]
    )
    assert calls[-1] == 0


@pytest.mark.parametrize("level", ["easy", "standard", "hard"])
def test_cached_and_reference_game_hashes_rewards_and_masks_match(cfg, level):
    from pvz_rl.frozen_baseline import choose_action

    reference, cached = PvZEnv(plain(cfg), level=level), PvZEnv(cfg, level=level)
    left, _ = reference.reset(seed=42)
    right, _ = cached.reset(seed=42)
    while reference.state == "running":
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(reference.action_masks(), cached.action_masks())
        action = reference.codec.encode(choose_action(reference.public_board()))
        left, r1, term1, trunc1, info1 = reference.step(action)
        right, r2, term2, trunc2, info2 = cached.step(action)
        assert (r1, term1, trunc1, info1) == (r2, term2, trunc2, info2)
        assert reference.game.state_hash() == cached.game.state_hash()
    assert reference.state == ("won" if level == "easy" else "lost")
    # The 100 Hz seeded schedules and 0.1-second control cadence are versioned.
