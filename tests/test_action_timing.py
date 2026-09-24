"""Multiple actions at one tick, native combat, and replay controls."""

import copy

import numpy as np
import pytest
from pvz_game import Dig, Game, LevelSpec, Place, Spawn, Status, Wait
from pvz_game.replay import Playback, read_recording, write_recording

from pvz_rl.action_timing import ActionPhaseGame
from pvz_rl.config import validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.recordings import ActionPhasePlayback, open_playback, verify_replay, watch_recording
from pvz_rl.rewards import asset_value, reward_parts


def ready(cfg, *, record=False):
    env = PvZEnv(cfg, record=record)
    env.reset(
        seed=7,
        options={"scenario": LevelSpec("actions", (Spawn(1000, "basic", 0),), initial_sun=500)},
    )
    return env


def test_multiple_actions_at_tick_zero_update_legality_and_preserve_time(per_tick_cfg):
    env = ready(per_tick_cfg)
    for i, kind in enumerate(("peashooter", "sunflower", "wall_nut")):
        _, _, ended, truncated, info = env.step(env.codec.encode(Place(kind, 0, i)))
        assert not ended and not truncated and info["accepted"]
        assert info["ticks_advanced"] == 0 and env.public.tick == 0
        assert not env.action_masks()[env.codec.encode(Place(kind, 1, 0))]
        assert not env.action_masks()[env.codec.encode(Place("potato_mine", 0, i))]
    assert env.public.sun == 300 and len(env.public.plants) == 3
    assert all(
        c.cooldown_ticks == c.recharge_ticks
        for c in env.public.cards
        if c.plant_type in ("peashooter", "sunflower", "wall_nut")
    )
    env.step(env.codec.encode(Dig(0, 1)))
    assert env.public.tick == 0 and env.public.sun == 300
    assert len(env.public.plants) == 2
    _, _, _, _, info = env.step(0)
    assert info["ticks_advanced"] == 1 and env.public.tick == 1
    assert env.episode_metrics()["max_actions_per_tick"] == 4
    assert env.episode_metrics()["instant_actions"] == 4


@pytest.mark.parametrize("action", [Place("sunflower", 0, 0), Dig(0, 0)])
def test_action_then_wait_matches_unchanged_engine_tick(per_tick_cfg, action):
    env = PvZEnv(per_tick_cfg)
    env.reset(seed=42)
    control = Game()
    control.reset("standard", 42)
    accepted = control.validate_action(action).accepted
    env.step(env.codec.encode(action))
    if accepted:
        env.step(0)
    control.step(action)
    assert env.game.state_hash() == control.state_hash()


def test_invalid_actions_advance_and_never_spend_or_change_board(per_tick_cfg):
    env = ready(per_tick_cfg)
    env.step(env.codec.encode(Place("peashooter", 0, 0)))
    _, _, _, _, info = env.step(env.codec.encode(Place("peashooter", 0, 1)))
    assert not info["accepted"] and info["ticks_advanced"] == 1
    assert env.public.sun == 400 and len(env.public.plants) == 1
    env.step(env.codec.encode(Dig(4, 8)))
    assert env.public.tick == 2 and env.metrics["invalid_actions"] == 2


def test_exact_cost_cooldown_and_many_operations_without_an_artificial_cap(per_tick_cfg):
    env = ready(per_tick_cfg)
    env.step(env.codec.encode(Place("sunflower", 0, 0)))
    for _ in range(749):
        env.step(0)
    assert env.public.tick == 749
    assert not env.action_masks()[env.codec.encode(Place("sunflower", 0, 1))]
    env.step(0)
    assert env.action_masks()[env.codec.encode(Place("sunflower", 0, 1))]
    env.step(env.codec.encode(Place("sunflower", 0, 1)))
    assert env.public.tick == 750
    assert next(c for c in env.public.cards if c.plant_type == "sunflower").cooldown_ticks == 750

    from pvz_game import InitialPlant

    env.reset(
        options={
            "scenario": LevelSpec(
                "dig-all",
                (Spawn(1000, "basic", 0),),
                plants=tuple(InitialPlant("wall_nut", r, c) for r in range(5) for c in range(9)),
            )
        }
    )
    for r in range(5):
        for c in range(9):
            env.step(env.codec.encode(Dig(r, c)))
    assert env.public.tick == 0 and not env.public.plants
    assert env.episode_metrics()["max_actions_per_tick"] == 45
    env.step(0)
    assert env.public.tick == 1


@pytest.mark.parametrize("terminal,expected", [(Status.WON, 1), (Status.LOST, -2)])
def test_configured_terminal_rewards_preserve_physical_assets(per_tick_cfg, terminal, expected):
    from dataclasses import replace

    before = Game().reset("easy", 2)
    after = replace(before, status=terminal)
    parts = reward_parts(before, after, per_tick_cfg)
    assert parts["terminal"] == expected and parts["total"] == expected
    assert asset_value(after) == asset_value(before)


def test_wait_cutoff_terminal_precedence_and_restricted_request(per_tick_cfg):
    per_tick_cfg["environment"]["cutoff_seconds"] = 1
    env = ready(per_tick_cfg)
    for _ in range(99):
        assert env.step(0)[2:4] == (False, False)
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated and truncated and info["ticks_advanced"] == 1
    assert info["reward_parts"]["terminal"] == 0 and asset_value(env.public) > 0
    env.reset(options={"scenario": LevelSpec("empty")})
    assert env.step(0)[2:4] == (True, False)
    env.reset(options={"scenario": LevelSpec("last-tick-win", (Spawn(100, "basic", 0, x=0),))})
    for _ in range(99):
        assert env.step(0)[2:4] == (False, False)
    assert env.step(0)[2:4] == (True, False)
    assert env.state == "won"
    per_tick_cfg["environment"]["cutoff_seconds"] = 10
    env = PvZEnv(per_tick_cfg)
    env.reset(options={"scenario": LevelSpec("loss", (Spawn(1, "basic", 0, x=0),), mowers=False)})
    while env.state == "running":
        _, _, _, _, info = env.step(0)
    assert info["reward_parts"]["terminal"] == -2
    lesson = PvZEnv(per_tick_cfg, family="placement")
    lesson.reset(seed=42)
    info = lesson.step(lesson.codec.encode(Place("sunflower", 0, 0)))[4]
    assert not info["accepted"] and info["ticks_advanced"] == 1


def test_zero_time_purchase_dig_has_no_free_reward(per_tick_cfg):
    env = ready(per_tick_cfg)
    total = sum(
        env.step(env.codec.encode(action))[1]
        for action in (Place("peashooter", 0, 0), Dig(0, 0), Wait())
    )
    assert env.public.tick == 1
    assert total == pytest.approx(-100 / 3000)
    assert env.episode_metrics()["discounted_return"] == pytest.approx(total)


def recorded_actions(cfg, path, *, trailing=False):
    env = ready(cfg, record=True)
    env.step(env.codec.encode(Place("peashooter", 0, 0)))
    env.step(env.codec.encode(Place("sunflower", 1, 0)))
    env.step(0)
    env.step(env.codec.encode(Dig(1, 0)))
    env.step(0)
    if trailing:
        env.step(env.codec.encode(Place("wall_nut", 2, 0)))
    env.recorder.save(path)
    return env


@pytest.mark.parametrize("trailing", [False, True])
def test_replay_roundtrip_seek_hashes_and_zero_time_final_operations(
    per_tick_cfg, tmp_path, trailing
):
    path = tmp_path / "game.pvzdemo"
    env = recorded_actions(per_tick_cfg, path, trailing=trailing)
    playback = open_playback(path)
    assert isinstance(playback, ActionPhasePlayback)
    assert playback.current_tick == 0 and len(playback.game.observe().plants) == 2
    assert playback.verify().state_hash() == env.game.state_hash()
    identity = id(playback.game)
    for tick in (0, 2, 1, 2, 0):
        reference = open_playback(path)
        while reference.current_tick < tick:
            assert reference.step().ticks_advanced == 1
        playback.seek(tick)
        assert id(playback.game) == identity
        assert playback.game.state_hash() == reference.game.state_hash()
        assert playback.last_operation == reference.last_operation
    plain = tmp_path / "game.json"
    write_recording(read_recording(path), plain)
    assert verify_replay(plain).state_hash() == env.game.state_hash()
    assert (
        Playback(path).verify().state_hash() == env.game.state_hash()
    )  # Explicit version prevents silently misplaying it upstream.


def test_zero_tick_only_and_empty_recording_and_corrupt_hashes(per_tick_cfg, tmp_path):
    env = ready(per_tick_cfg, record=True)
    path = tmp_path / "zero.pvzdemo"
    env.recorder.save(path)
    assert open_playback(path).done
    env.step(env.codec.encode(Place("peashooter", 0, 0)))
    env.recorder.save(path)
    playback = open_playback(path)
    assert playback.done and playback.end_tick == playback.start_tick == 0
    assert playback.game.state_hash() == env.game.state_hash()
    data = read_recording(path)
    data["entries"][0]["state_hash"] = "broken"
    with pytest.raises(ValueError, match="state mismatch"):
        open_playback(data)
    data["entries"][0].pop("state_hash")
    data["final_hash"] = "broken"
    with pytest.raises(ValueError, match="final state mismatch"):
        open_playback(data)


def test_native_viewer_receives_project_playback(per_tick_cfg, tmp_path, monkeypatch):
    import pvz_game.ui

    path = tmp_path / "game.pvzdemo"
    env = recorded_actions(per_tick_cfg, path)

    class Viewer:
        def __init__(self, *, speed):
            assert speed == 2

        def run(self):
            assert isinstance(self.playback, ActionPhasePlayback)
            assert self.game is self.playback.game
            self.playback.verify()
            assert self.game.state_hash() == env.game.state_hash()
            self.playback.seek(0)

    monkeypatch.setattr(pvz_game.ui, "App", Viewer)
    watch_recording(path, speed=2)


def test_mask_cache_equivalence_between_instant_actions(per_tick_cfg):
    stock_cfg = copy.deepcopy(per_tick_cfg)
    stock_cfg["runtime"]["cache_legal_actions"] = False
    envs = [ready(cfg) for cfg in (per_tick_cfg, stock_cfg)]
    for action in (Place("peashooter", 0, 0), Place("sunflower", 1, 0), Dig(0, 0), Wait()):
        observations = [e.step(e.codec.encode(action))[0] for e in envs]
        np.testing.assert_array_equal(*observations)
        np.testing.assert_array_equal(*(e.action_masks() for e in envs))


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("environment", "action_timing", "bad"),
        ("environment", "decision_ticks", 10),
        ("reward", "loss_penalty", -2),
        ("reward", "loss_penalty", True),
    ],
)
def test_invalid_timing_and_loss_config(per_tick_cfg, section, key, value):
    per_tick_cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(per_tick_cfg)


def test_zero_tick_api_rejects_wait_and_recovers_after_error():
    game = ActionPhaseGame()
    game.reset("easy", 1)
    before = game.state_hash()
    with pytest.raises(ValueError):
        game.step(Wait(), ticks=0)
    with pytest.raises(TypeError):
        game.step(Place("peashooter", "bad", 0), ticks=0)
    assert game.state_hash() == before
    assert game.step().ticks_advanced == 1


def test_action_phase_replay_midgame_cache_boundary_and_completion_rewind(per_tick_cfg, tmp_path):
    from pvz_rl.recordings import ActionPhaseRecorder

    env = ready(per_tick_cfg)
    for _ in range(7):
        env.step(0)
    recorder = ActionPhaseRecorder(env.game, metadata={"outcome": "truncated"})
    expected = {7: env.game.state_hash()}
    for tick in range(8, 213):
        recorder.step(Wait())
        if tick in (107, 212):  # Exactly at the native relative cache boundary, then final tick.
            recorder.step(
                Place("wall_nut" if tick == 107 else "sunflower", 0, 0 if tick == 107 else 1),
                ticks=0,
            )
        expected[tick] = env.game.state_hash()
    path = tmp_path / "midgame.pvzdemo"
    recorder.save(path)
    playback = open_playback(path)
    playback.verify()
    assert playback.display_outcome == "truncated"
    identity = id(playback.game)
    for tick in (107, 106, 108, 212, 7, 107, 212):
        playback.seek(tick)
        assert id(playback.game) == identity and playback.game.state_hash() == expected[tick]
        assert playback.done == (tick == 212)
        assert playback.display_outcome == ("truncated" if tick == 212 else "running")


def test_instant_actions_do_not_add_video_frames(per_tick_cfg, tmp_path):
    import shutil
    import subprocess

    from pvz_rl.video import export_replay

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    per_tick_cfg["visualization"].update(final_hold_seconds=0, video_size=[1000, 600])
    source, video = tmp_path / "instant.pvzdemo", tmp_path / "instant.mp4"
    env = recorded_actions(per_tick_cfg, source, trailing=True)
    result = export_replay(source, video, per_tick_cfg)
    assert result["frames"] == 2 and result["duration_seconds"] == 2 / 25
    assert result["final_state_hash"] == env.game.state_hash()
    assert (result["width"], result["height"]) == (1000, 600)
    decoded = subprocess.run(
        [result["encoder"]["path"], "-v", "error", "-i", str(video), "-f", "null", "-"],
        capture_output=True,
        timeout=30,
    )
    assert decoded.returncode == 0, decoded.stderr
