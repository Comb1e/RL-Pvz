import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pvz_game import Game, LevelSpec, Place, Spawn
from pvz_game.replay import Playback, Recorder, read_recording, write_recording

from pvz_rl.cli import main
from pvz_rl.config import validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.evaluation import evaluate
from pvz_rl.provenance import file_hash, verify_engine, write_json
from pvz_rl.recordings import open_playback
from pvz_rl.training import load_policy
from pvz_rl.visualization import visualize_run


@pytest.mark.parametrize("changed", ["package", "simulation", "source", "rules"])
def test_strict_installed_game_verification(cfg, monkeypatch, changed):
    import pvz_rl.provenance as provenance

    if changed == "package":
        monkeypatch.setattr(provenance.importlib.metadata, "version", lambda _: "1.0.0")
    elif changed == "simulation":
        monkeypatch.setattr(provenance, "ENGINE_VERSION", "2.0.0")
    elif changed == "source":
        monkeypatch.setattr(provenance, "source_manifest", lambda _: {})
    else:
        original = provenance.engine_pin()
        original["rules_hash"] = "wrong"
        monkeypatch.setattr(provenance, "engine_pin", lambda: original)
    with pytest.raises(RuntimeError, match="[Dd]iffer"):
        verify_engine(cfg)


@pytest.mark.parametrize(
    "old_pin",
    [
        "b3cfbd886ab378313a1fdb57ee43a9a1b36a0793",
        "a47056d8141ec635d3ff3f4d5561d6a75cfca2cc",
    ],
)
def test_old_checkpoint_rejected_before_model_deserialization(cfg, tmp_path, old_pin):
    cfg["engine_commit"] = old_pin
    cfg.pop("engine_package_version")
    write_json(tmp_path / "metadata.json", {"config": cfg, "condition": "masked"})
    with pytest.raises(RuntimeError, match="fresh training"):
        load_policy(tmp_path / "old.zip")


@pytest.mark.parametrize("defeated,total", [(0, 15), (2, 15), (15, 15), (0, 0), (200, 200)])
def test_native_121_progress_counter_and_rendering_purity(cfg, monkeypatch, defeated, total):
    from dataclasses import replace

    from pvz_game.rendering import _BoardCanvas

    from pvz_rl.rendering import render_observation

    env = PvZEnv(cfg)
    env.reset(seed=42)
    before = env.game.state_hash()
    obs = replace(
        env.public, counts=replace(env.public.counts, defeated=defeated, initial_total=total)
    )
    labels = []
    original = _BoardCanvas.text

    def capture(self, value, *args, **kwargs):
        labels.append(str(value))
        return original(self, value, *args, **kwargs)

    monkeypatch.setattr(_BoardCanvas, "text", capture)
    assert render_observation(obs).shape == (600, 1000, 3)
    assert f"{defeated}/{total}" in labels
    assert "ZOMBIES DEFEATED / TOTAL" in labels
    assert env.game.state_hash() == before


def test_compact_recording_metadata_and_input_isolation(cfg, tmp_path):
    cfg["environment"]["cutoff_seconds"] = 1
    rows = evaluate(
        cfg,
        seeds=[100000],
        levels=["easy"],
        output=tmp_path / "eval",
        baseline="wait",
        record=True,
        checkpoint_hash="a" * 64,
    )
    row = rows[0]
    path = Path(row["replay"])
    assert path.suffix == ".pvzdemo" and path.read_bytes().startswith(b"\x1f\x8b")
    assert row["replay_hash"] == file_hash(path)
    native = Playback(path)
    assert native.display_outcome == "running"
    assert native.metadata["checkpoint_sha256"] == "a" * 64
    assert native.metadata["experiment"]["engine"]["package_version"] == "1.4.0"
    native.verify()
    assert native.display_outcome == "truncated"
    assert native.metadata["termination_reason"] == "time_limit"
    plain = tmp_path / "same.json"
    write_recording(read_recording(path), plain)
    assert Playback(plain).verify().state_hash() == native.game.state_hash()
    left, right = (
        PvZEnv(cfg, record=True),
        PvZEnv(
            cfg,
            record=True,
            replay_metadata={
                "policy_id": "different",
                "experiment": {"secret": 55},
                "checkpoint_sha256": "f" * 64,
            },
        ),
    )
    a, _ = left.reset(seed=1)
    b, _ = right.reset(seed=1)
    np.testing.assert_array_equal(a, b)
    assert left.game.state_hash() == right.game.state_hash()
    np.testing.assert_array_equal(left.action_masks(), right.action_masks())


@pytest.mark.parametrize("suffix", [".pvzdemo", ".json.gz", ".json"])
def test_native_metadata_precedence_and_seek(suffix, tmp_path):
    game = Game()
    game.reset(LevelSpec("later", (Spawn(1000, "basic", 2),), initial_sun=200), 2)
    game.step(ticks=17)  # Recording starts mid-game.
    recorder = Recorder(game, metadata={"policy_id": "native", "checkpoint_sha256": "a" * 64})
    recorder.step(Place("sunflower", 2, 0), ticks=13)
    recorder.step(ticks=25)
    recorder.update_metadata({"outcome": "truncated", "termination_reason": "time_limit"})
    path = tmp_path / ("game" + suffix)
    recorder.save(path)
    write_json(
        path.with_suffix(".metadata.json"),
        {"policy_id": "legacy", "outcome": "lost", "checkpoint_hash": "b" * 64},
    )
    playback = open_playback(path, fallback={"outcome": "won", "policy_id": "fallback"})
    identity = id(playback.game)
    assert playback.metadata["policy_id"] == "native"
    assert playback.metadata["checkpoint_sha256"] == "a" * 64
    playback.verify()
    assert playback.display_outcome == "truncated"
    for tick in (20, 55, 17, 30, 54, 55):
        reference = Playback(path)
        while reference.current_tick < tick:
            reference.step()
        playback.seek(tick)
        assert id(playback.game) == identity
        assert playback.game.state_hash() == reference.game.state_hash()
        assert playback.last_operation == reference.last_operation
        assert playback.display_outcome == ("truncated" if tick == 55 else "running")


@pytest.mark.parametrize("natural", [False, True])
def test_legacy_sidecar_and_outcome_precedence(tmp_path, natural):
    game = Game()
    game.reset(LevelSpec("empty") if natural else "easy", 0)
    recorder = Recorder(game)
    recorder.step()
    path = tmp_path / "legacy.json"
    recorder.save(path)
    assert "metadata" not in read_recording(path)
    write_json(
        path.with_suffix(".metadata.json"), {"status": "truncated", "checkpoint_hash": "c" * 64}
    )
    playback = open_playback(path)
    assert playback.display_outcome == "running"
    playback.verify()
    assert playback.display_outcome == ("won" if natural else "truncated")


def test_default_training_records_without_pygame_or_ffmpeg(tmp_path):
    script = r"""
import builtins, sys
from pvz_rl.config import load_config
from pvz_rl.training import train
cfg = load_config()
assert cfg['visualization']['demos'] and not cfg['visualization']['videos']
# Retain the archived periodic-export control independently of mastery scheduling.
cfg['training']['validation_schedule'] = 'periodic'
cfg['environment']['cutoff_seconds'] = 1
cfg['training'].update(budget_unit='decisions', device='cuda', total_steps=64, rollout_steps_per_env=64, rollout_size=64, batch_size=32, n_envs=1, n_epochs=1, hidden_sizes=[32,32], eval_interval=64)
cfg['visualization']['ffmpeg'] = 'missing-ffmpeg'
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name == 'pygame' or name.startswith('pygame.'):
        raise AssertionError('Default demo recording must not import pygame')
    return original(name, *args, **kwargs)
builtins.__import__ = checked
train(cfg, 'masked', 101, sys.argv[1], validation_limit=1)
"""
    run = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "-c", script, str(run)], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((run / "visualizations/demos.json").read_text())
    demos = manifest["demos"]
    assert {d["level"] for d in demos} == {"easy", "standard", "hard"}
    assert {d["checkpoint_hash"] for d in demos} == {file_hash(run / "best.zip")}
    assert all(d["replay"].endswith(".pvzdemo") for d in demos)
    assert not (run / "visualizations/videos").exists()
    assert "--watch --speed 2" in (run / "visualizations/index.html").read_text()


def test_archived_report_and_video_never_load_old_model(cfg, tmp_path, monkeypatch):
    import pvz_rl.training as training

    def forbidden(*args, **kwargs):
        raise AssertionError("Old model must not be loaded")

    monkeypatch.setattr(training, "load_policy", forbidden)
    cfg["engine_commit"] = "b3cfbd886ab378313a1fdb57ee43a9a1b36a0793"
    cfg.pop("engine_package_version")
    cfg["visualization"]["final_hold_seconds"] = 0
    write_json(tmp_path / "metadata.json", {"config": cfg, "family": "preset"})
    write_json(tmp_path / "best.json", {"checkpoint_hash": "a" * 64})
    replay = tmp_path / "visualizations/game.json"
    game = Game()
    game.reset(LevelSpec("empty"), 0)
    recorder = Recorder(game)
    recorder.step()
    recorder.save(replay)
    demo = dict(
        replay="game.json",
        checkpoint_hash="a" * 64,
        level="easy",
        outcome="won",
        simulated_seconds=0.05,
        scenario_seed=100000,
    )
    write_json(tmp_path / "visualizations/demos.json", {"demos": [demo]})
    result = visualize_run(tmp_path, videos=False)
    assert result["state"] == "complete" and "Archived" in result["note"]
    if shutil.which("ffmpeg"):
        assert visualize_run(tmp_path, videos=True)["state"] == "complete"
        assert (tmp_path / "visualizations/videos/easy-100000.mp4").exists()
        manifest = json.loads((tmp_path / "visualizations/demos.json").read_text())
        manifest["demos"][0]["replay_hash"] = file_hash(replay)
        write_json(tmp_path / "visualizations/demos.json", manifest)
        altered = read_recording(replay)
        altered["metadata"] = {"policy_id": "tampered"}
        write_recording(altered, replay)
        result = visualize_run(tmp_path, videos=True)
        assert result["state"] == "failed" and "checksum" in result["error"]


def test_video_options_respect_config_and_mutual_exclusion(cfg, tmp_path, monkeypatch):
    import pvz_rl.training as training

    received = []
    monkeypatch.setattr(
        training, "train", lambda cfg, *a, **kw: received.append(cfg["visualization"]["videos"])
    )
    for flags in ([], ["--videos"], ["--no-videos"]):
        main(["train", "--output", str(tmp_path / "unused"), *flags])
    assert received == [False, True, False]
    with pytest.raises(SystemExit):
        main(["train", "--output", str(tmp_path), "--videos", "--no-videos"])
    config = (
        Path("src/pvz_rl/data/research.toml").read_text().replace("videos = false", "videos = true")
    )
    custom = tmp_path / "custom.toml"
    custom.write_text(config)
    main(["train", "--output", str(tmp_path), "--config", str(custom)])
    assert received[-1] is True


def test_watch_uses_native_payload_speed_and_legacy_labels(tmp_path, monkeypatch, capsys):
    import pvz_game.ui

    game = Game()
    game.reset("easy", 1)
    recorder = Recorder(game)
    recorder.step()
    path = tmp_path / "old.json"
    recorder.save(path)
    write_json(path.with_suffix(".metadata.json"), {"outcome": "truncated"})

    class Viewer:
        def __init__(self, replay_path, speed):
            assert speed == 2
            playback = Playback(replay_path)
            playback.verify()
            assert playback.display_outcome == "truncated"

        def run(self):
            pass

    monkeypatch.setattr(pvz_game.ui, "App", Viewer)
    main(["replay", str(path), "--watch", "--speed", "2"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "running" and result["outcome"] == "truncated"
    for args in (["--speed", "2"], ["--watch", "--speed", "3"]):
        with pytest.raises(SystemExit):
            main(["replay", str(path), *args])


@pytest.mark.parametrize("size", [[0, 820], [1281, 820], [1280], [1280, 819], "1280x820"])
def test_invalid_video_dimensions(cfg, size):
    cfg["visualization"]["video_size"] = size
    with pytest.raises(ValueError, match="even dimensions"):
        validate_config(cfg)
