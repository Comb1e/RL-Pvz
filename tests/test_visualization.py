import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pvz_game import LevelSpec, Spawn

from pvz_rl.config import output_settings, research_config, validate_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.learning.training import ResearchCallback, load_policy, train
from pvz_rl.monitoring.progress import Phase, ProgressReporter
from pvz_rl.presentation.video import export_replay, ffmpeg_info
from pvz_rl.presentation.visualization import (
    build_run_report,
    read_series,
    run_segments,
    visualize_run,
)
from pvz_rl.provenance import file_hash, write_json


def test_progress_throttle_phases_and_persistence(tmp_path, capsys):
    now = [0.0]
    with ProgressReporter(tmp_path / "train.log", 15, clock=lambda: now[0]) as progress:
        progress.emit("Startup", force=True)
        for _ in range(100):
            assert not progress.emit("episode spam")
        now[0] = 15
        assert progress.emit("Aggregate progress")
        progress.phase(Phase.VALIDATING, "Validation started")
        progress.phase(Phase.COMPLETE, "Complete")
    text = capsys.readouterr().out
    assert "episode spam" not in text and text.count("\n") == 4
    assert "[validating]" in text
    assert (tmp_path / "train.log").read_text("utf-8") == text


def test_old_config_and_output_only_compatibility(cfg):
    old = copy.deepcopy(cfg)
    old.pop("logging")
    old.pop("visualization")
    validate_config(old)
    cfg["logging"]["progress_seconds"] = 2
    cfg["visualization"]["videos"] = False
    assert research_config(old) == research_config(cfg)
    assert output_settings(old)["logging"]["progress_seconds"] == 15
    cfg["training"]["learning_rate"] *= 2
    assert research_config(old) != research_config(cfg)


def test_hardware_report_only_never_loads_checkpoint_and_respects_resume_cutoff(
    tmp_path, monkeypatch
):
    from pvz_rl.config import load_config

    cfg = load_config()
    ancestor, run = tmp_path / "previous", tmp_path / "resumed"
    ancestor.mkdir()
    run.mkdir()
    write_json(ancestor / "metadata.json", {"config": cfg})
    write_json(
        run / "metadata.json",
        {"config": cfg, "resume": str(ancestor / "latest.zip"), "resume_steps": 128},
    )
    rows = [
        dict(
            session="one",
            seconds=i,
            training_steps=steps,
            activity=activity,
            phase="warmup",
            gpu_percent=50,
            gpu_memory_percent=None,
            process_cpu_percent=150,
            system_cpu_percent=20,
            gpu_error="missing memory field",
        )
        for i, (steps, activity) in enumerate(
            [(0, "training"), (128, "validation"), (256, "training")]
        )
    ]
    (ancestor / "hardware-metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    (run / "hardware-metrics.jsonl").write_text(
        json.dumps(dict(rows[0], session="two", phase="formal")) + '\n{"partial":'
    )
    (run / "best.zip").write_bytes(b"not a checkpoint")

    def forbidden(*args, **kwargs):
        raise AssertionError("Report-only must not generate demonstrations")

    monkeypatch.setattr("pvz_rl.presentation.visualization.create_demonstrations", forbidden)
    assert len(run_segments(run)[0][1]["hardware"]) == 2
    assert visualize_run(run, report_only=True)["state"] == "complete"
    assert (run / "visualizations/hardware.png").stat().st_size > 1000
    assert "device-wide GPU" in (run / "visualizations/index.html").read_text("utf-8")
    # Only the partial tail is recoverable; corruption in the middle remains visible.
    (run / "hardware-metrics.jsonl").write_text('{"bad":\n' + json.dumps(rows[0]))
    with pytest.raises(json.JSONDecodeError):
        run_segments(run)


@pytest.mark.learning
def test_shared_checkpoint_reports_and_videos(smoke_cfg, tmp_path, monkeypatch):
    import pvz_rl.learning.training as training

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    smoke_cfg["visualization"].update(enabled=True, videos=True, final_hold_seconds=0)
    smoke_cfg["evaluation"]["replays_per_outcome"] = 0  # Demos have their own fixed quota.
    identities, stages = [], set()
    original = ResearchCallback._on_rollout_start

    def inspect(self):
        identities.append((id(self.model), id(self.model.policy.optimizer)))
        stages.add(tuple(self.snapshot()["next_episode_tasks"][0]))
        original(self)

    monkeypatch.setattr(ResearchCallback, "_on_rollout_start", inspect)
    loads = []
    original_load = training.load_policy

    def tracked_load(checkpoint, *args, **kwargs):
        loads.append(Path(checkpoint).name)
        return original_load(checkpoint, *args, **kwargs)

    monkeypatch.setattr(training, "load_policy", tracked_load)
    run = train(smoke_cfg, "masked", 101, tmp_path / "shared", validation_limit=1)
    assert len(set(identities)) == 1 and stages == {("saving",)}
    assert loads == ["best.zip"]
    metrics = read_series(run / "training-metrics.jsonl")
    assert [m["training_steps"] for m in metrics] == [128]
    assert [m["completed_updates"] for m in metrics] == [2]
    assert all(m["optimization"]["value_loss"] is not None for m in metrics)
    assert all(m["optimization"]["critic_optimizer_steps"] > 0 for m in metrics)
    assert all(m["optimization"]["post_update_type_kl"] is not None for m in metrics)
    assert metrics[-1]["rolling_by_task"]["saving"]["completed_games"] > 0
    assert sum(t["transitions"] for t in metrics[-1]["task_counts"].values()) == 128
    assert (run / "tensorboard").exists()
    demo_data = json.loads((run / "visualizations/demos.json").read_text())
    demos = demo_data["demos"]
    assert {r["level"] for r in demos} == {"easy", "standard", "hard"}
    assert {r["checkpoint_hash"] for r in demos} == {file_hash(run / "best.zip")}
    assert {r["scenario_seed"] for r in demos} == {100000}
    assert all(r["outcome"] == "truncated" for r in demos)
    page = (run / "visualizations/index.html").read_text("utf-8")
    assert page.count("<video controls") == 3 and "autoplay" not in page
    assert "Task-specific training diagnostics" in page and "task-curves.png" in page
    assert "Net realized value accounting" in page and "accounting-curves.png" in page
    assert metrics[-1]["rolling_discounted_return"] is not None
    assert metrics[-1]["rolling_development"] is not None
    assert "https://" not in page
    for path in re.findall(r'(?:src|href)="([^"]+)"', page):
        assert (run / "visualizations" / path).exists()
    for demo in demos:
        video = run / "visualizations" / demo["video"]
        info = json.loads(video.with_suffix(".video.json").read_text())
        assert info["verified"] and info["final_state_hash"] == demo["state_hash"]
        assert info["fps"] == smoke_cfg["visualization"]["video_fps"] == 25
        assert info["frames"] == 1 + int(smoke_cfg["environment"]["cutoff_seconds"] * 25)
        assert (info["width"], info["height"]) == (1280, 820)
        result = subprocess.run(
            [ffmpeg_info(smoke_cfg)["path"], "-v", "error", "-i", str(video), "-f", "null", "-"],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
    checkpoint_hash = file_hash(run / "best.zip")
    assert visualize_run(run, videos=False)["state"] == "complete"
    assert file_hash(run / "best.zip") == checkpoint_hash


@pytest.mark.learning
def test_output_does_not_change_learning_and_failed_export_keeps_checkpoint(smoke_cfg, tmp_path):
    plain = train(smoke_cfg, "masked", 103, tmp_path / "plain", validation_limit=1)
    smoke_cfg["visualization"].update(enabled=True, videos=True, ffmpeg="missing-pvz-ffmpeg")
    visual = train(smoke_cfg, "masked", 103, tmp_path / "visual", validation_limit=1)
    assert json.loads((visual / "status.json").read_text())["state"] == "complete"
    assert json.loads((visual / "visualizations/status.json").read_text())["state"] == "failed"
    assert (visual / "visualizations/index.html").exists()
    left, _ = load_policy(plain / "final.zip")
    right, _ = load_policy(visual / "final.zip")
    for key, value in left.policy.state_dict().items():
        assert torch.equal(value, right.policy.state_dict()[key])
    assert (
        json.loads((plain / "best.json").read_text())["steps"]
        == json.loads((visual / "best.json").read_text())["steps"]
    )
    env = PvZEnv(smoke_cfg)
    for level in ("easy", "standard", "hard"):
        obs, _ = env.reset(seed=100000, options={"level": level})
        a, _ = left.predict(obs, deterministic=True, action_masks=env.action_masks())
        b, _ = right.predict(obs, deterministic=True, action_masks=env.action_masks())
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("outcome", ["won", "lost", "truncated"])
def test_video_outcomes_and_bad_replay_preserve_existing_video(cfg, tmp_path, outcome):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    cfg["environment"]["cutoff_seconds"] = 10 if outcome == "lost" else 1
    cfg["visualization"]["final_hold_seconds"] = 0
    scenario = {
        "won": LevelSpec("empty"),
        "lost": LevelSpec("breach", (Spawn(1, "basic", 0, x=0),), mowers=False),
        "truncated": LevelSpec("later", (Spawn(1000, "basic", 0),)),
    }[outcome]
    env = PvZEnv(cfg, record=True)
    env.reset(seed=4, options={"scenario": scenario})
    while env.state == "running":
        env.step(0)
    assert env.state == outcome
    source, dest = tmp_path / "game.json", tmp_path / "game.mp4"
    env.recorder.save(source)
    record = export_replay(source, dest, cfg, context={"outcome": outcome})
    assert record["outcome"] == outcome and record["final_state_hash"] == env.game.state_hash()
    previous = file_hash(dest)
    broken = json.loads(source.read_text())
    broken["final_hash"] = "wrong"
    write_json(source, broken)
    with pytest.raises(ValueError, match="final state mismatch"):
        export_replay(source, dest, cfg)
    assert file_hash(dest) == previous
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_empty_old_report_and_resume_segments(cfg, tmp_path):
    cfg["profile"] = "baseline"  # Labels must not hide current-method diagnostics.
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    write_json(old / "metadata.json", {"config": research_config(cfg)})
    page = build_run_report(old)
    assert "No validated checkpoint yet" in page.read_text("utf-8")
    assert (old / "visualizations/learning-diagnostics.png").is_file()
    for run, steps in ((old, [64, 128]), (new, [128, 192])):
        (run / "training-metrics.jsonl").write_text(
            "\n".join(json.dumps({"training_steps": n}) for n in steps), encoding="utf-8"
        )
    write_json(
        new / "metadata.json",
        {
            "config": cfg,
            "resume": str(old / "latest.zip"),
            "resume_steps": 64,
        },
    )
    segments = run_segments(new)
    assert [
        [r["training_steps"] for r in series["training-metrics"]] for _, series in segments
    ] == [[64], [128, 192]]


def test_missing_encoder_reports_actionable_error(cfg, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="visualization.ffmpeg"):
        ffmpeg_info(cfg)


def test_encoder_failure_cleans_up_and_keeps_previous_output(cfg, tmp_path, monkeypatch):
    import pvz_rl.presentation.video as video

    env = PvZEnv(cfg, record=True)
    env.reset(seed=2)
    env.step(0)
    replay, destination = tmp_path / "record.json", tmp_path / "record.mp4"
    env.recorder.save(replay)
    destination.write_bytes(b"previous successful artifact")
    monkeypatch.setattr(video, "ffmpeg_info", lambda _: {"path": sys.executable})
    with pytest.raises(RuntimeError, match="FFmpeg"):
        export_replay(replay, destination, cfg)
    assert destination.read_bytes() == b"previous successful artifact"
    assert not list(tmp_path.glob("*.tmp.mp4"))
