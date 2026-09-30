import io
import json
from types import SimpleNamespace

import numpy as np
import torch

from pvz_rl.config import load_config
from pvz_rl.learning.cohort import CohortPhase
from pvz_rl.learning.training import ResearchCallback
from pvz_rl.monitoring.progress import ProgressReporter
from pvz_rl.presentation.live_view import Activity


class ActiveAccumulator:
    """Terminal formatting must never read (and so synchronize) fitting accumulators."""

    def __getattr__(self, name):
        raise AssertionError(f"terminal progress read an active accumulator: {name}")


def test_compact_summary_events_and_unchanged_detailed_snapshot(tmp_path):
    cfg = load_config()
    now, stream = [0.0], io.StringIO()
    progress = ProgressReporter.from_settings(
        tmp_path / "train.log", cfg, clock=lambda: now[0], stream=stream
    )
    callback = ResearchCallback(cfg, "masked", 101, tmp_path, progress=progress)
    optimizer = SimpleNamespace(state={0: {"step": torch.tensor(7.0)}})
    callback.model = SimpleNamespace(
        num_timesteps=1000,
        training_games=100,
        phase="collect",
        _n_updates=2,
        cohort_games=4,
        cohort_start_steps=900,
        _phase_times={"collect": 12.0},
        env=SimpleNamespace(enabled_envs=np.array([True, False, False, True]), live_view=None),
        policy=SimpleNamespace(optimizer=optimizer),
        _pass_metrics=ActiveAccumulator(),
        _fit_epoch=1,
    )
    try:
        callback.log_progress(force=True)
        (line,) = stream.getvalue().splitlines()
        assert "cohort 3 2/4 games | games 100/10,000 | transitions 1,000 | " in line
        assert line.endswith("| collect n/a fit n/a | optimizer step 7")
        status = json.loads((tmp_path / "status.json").read_text("utf-8"))
        for key in ("rolling_return", "hardware", "cohort", "task_counts", "compilation_status"):
            assert key in status
        # Snapshots keep their 15-second cadence; the terminal waits 60 seconds.
        now[0] = 30
        callback.model.num_timesteps = 1100
        callback.log_progress()
        assert len(stream.getvalue().splitlines()) == 1
        assert json.loads((tmp_path / "status.json").read_text("utf-8"))["training_steps"] == 1100
        now[0] = 61
        callback.log_progress()
        assert "transitions 1,100" in stream.getvalue().splitlines()[-1]
        now[0] = 130
        callback.log_progress()
        assert len(stream.getvalue().splitlines()) == 2
        callback.cohort_phase(CohortPhase.COLLECT, CohortPhase.FINALIZE_REWARDS)
        assert (
            stream.getvalue()
            .splitlines()[-1]
            .endswith("[updating] Cohort 3 collected: 4 games, 200 transitions in 12.0s (17/s)")
        )
        callback.model.phase, callback.model._n_updates = "idle", 3
        callback.model.cohort_metrics = dict(
            q_optimizer_steps=4,
            fit_seconds=20.0,
            collection_seconds=12.0,
            optimization_seconds=20.0,
            window_seconds=32.0,
            transitions_per_second=6.25,
        )
        callback._on_rollout_end()
        assert (
            stream.getvalue()
            .splitlines()[-1]
            .endswith(
                "Cohort 3 fitted: 4 passes in 20.0s after 12.0s collection; "
                "6 transitions/s; optimizer step 7"
            )
        )
        assert (
            json.loads((tmp_path / "status.json").read_text("utf-8"))["cohort"]["q_optimizer_steps"]
            == 4
        )
        # A capturable optimizer keeps its step on the device; never read it.
        optimizer.state[0]["step"] = SimpleNamespace(device=torch.device("cuda"))
        assert callback.compact_status(callback.snapshot())[0].endswith("optimizer step 0")
    finally:
        callback.close()
        progress.close()
    assert "Run average" not in stream.getvalue() and "Hardware" not in stream.getvalue()
    assert (tmp_path / "train.log").read_text("utf-8") == stream.getvalue()


def test_finalize_rewards_uses_updating_viewer_activity(tmp_path):
    cfg = load_config()
    callback = ResearchCallback(cfg, "masked", 101, tmp_path)
    activities = []

    class Viewer:
        def set_activity(self, activity):
            activities.append(activity)

    callback.model = SimpleNamespace(
        num_timesteps=0,
        training_games=0,
        env=SimpleNamespace(live_view=Viewer()),
        phase="finalize_rewards",
    )
    try:
        callback.hardware_context("finalize_rewards")
    finally:
        callback.close()
    assert activities == [Activity.UPDATING]
