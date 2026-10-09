import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
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
        cohort_number=3,
        _pending_episodes={
            1: dict(
                metrics=dict(win=True, return_value=1.0, development=0.01, time_reward_pending=True)
            ),
            2: dict(
                metrics=dict(
                    win=False, return_value=-2.0, development=-0.02, time_reward_pending=True
                )
            ),
        },
        cohort_start_steps=900,
        _phase_times={"collect": 12.0},
        env=SimpleNamespace(enabled_envs=np.array([True, False, False, True]), live_view=None),
        policy=SimpleNamespace(optimizer=optimizer),
        _pass_metrics=ActiveAccumulator(),
        _fit_epoch=1,
    )
    for item in callback.model._pending_episodes.values():
        item["metrics"]["return"] = item["metrics"].pop("return_value")
    try:
        callback.log_progress(force=True)
        (line,) = stream.getvalue().splitlines()
        assert (
            "cohort 3 done 2/4 wins 1/4 Rraw=-0.5 Rtrain=-0.545~ | "
            "active 2/4 | n/a decisions/game/s | n/a active transitions/s | "
            "2/4 games | games total 100/10,000 | transitions total 1,000 | " in line
        )
        assert line.endswith("| collect n/a fit n/a | optimizer step 7")
        status = json.loads((tmp_path / "status.json").read_text("utf-8"))
        for key in ("rolling_return", "hardware", "cohort", "task_counts", "compilation_status"):
            assert key in status
        assert status["collection_progress"]["active_games"] == 2
        assert "includes fitting" in status["lifetime_throughput"]["denominator"]
        # Snapshots keep their 15-second cadence; the terminal waits 60 seconds.
        now[0] = 30
        callback.model.num_timesteps = 1100
        callback.log_progress()
        assert len(stream.getvalue().splitlines()) == 1
        assert json.loads((tmp_path / "status.json").read_text("utf-8"))["training_steps"] == 1100
        now[0] = 61
        callback.log_progress()
        assert "transitions total 1,100" in stream.getvalue().splitlines()[-1]
        now[0] = 130
        callback.log_progress()
        assert len(stream.getvalue().splitlines()) == 2
        callback.cohort_phase(CohortPhase.COLLECT, CohortPhase.FINALIZE_REWARDS)
        assert (
            stream.getvalue()
            .splitlines()[-1]
            .endswith(
                "[updating] Cohort 3 collected: 4 games, 200 transitions in 12.0s (17/s); done 2/4 wins 1/4 Rraw=-0.5 Rtrain=-0.545~"
            )
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
                "6 transitions/s; optimizer step 7; done 2/4 wins 1/4 Rraw=-0.5 Rtrain=-0.545~"
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


def test_current_cohort_means_finalize_reset_and_ignore_rolling_rows(tmp_path):
    from pvz_rl.learning.objective import finalize_episode_metrics

    callback = ResearchCallback(load_config(), "masked", 101, tmp_path)
    callback.model = SimpleNamespace(cohort_games=2, _pending_episodes={})
    try:
        assert callback.cohort_text() == "done 0/2 wins 0/2 Rraw=n/a Rtrain=n/a"
        callback.recent.append({"win": True, "return": 1000})
        rows = [
            {
                "win": True,
                "return": 1.0,
                "development": 0.01,
                "simulated_seconds": 80,
                "time_reward_pending": True,
            },
            {
                "win": False,
                "return": -2.0,
                "development": -0.02,
                "simulated_seconds": 160,
                "time_reward_pending": True,
            },
        ]
        callback.model._pending_episodes = {
            index: dict(metrics=row) for index, row in enumerate(rows)
        }
        summary = callback.cohort_summary()
        assert summary["completed"] == 2 and summary["wins"] == 1
        assert summary["raw_return"] == -0.5 and summary["provisional"]
        assert summary["training_return"] == pytest.approx(-0.545)
        finalize_episode_metrics(rows, callback.cfg)
        summary = callback.cohort_summary()
        assert summary["raw_return"] == pytest.approx(-0.49)
        assert summary["training_return"] == pytest.approx(-0.535)
        assert not summary["provisional"] and not callback.cohort_text().endswith("~")
        callback.model._pending_episodes = {}
        callback.model.cohort_games = 128
        assert callback.cohort_text() == "done 0/128 wins 0/128 Rraw=n/a Rtrain=n/a"
    finally:
        callback.close()


def test_interval_live_workers_and_cached_execution_never_read_device(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Host telemetry tried to synchronize the device")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    now, stream = [0.0], io.StringIO()
    cfg = load_config()
    progress = ProgressReporter.from_settings(
        tmp_path / "train.log", cfg, clock=lambda: now[0], stream=stream
    )
    callback = ResearchCallback(cfg, "masked", 101, tmp_path, progress=progress)
    enabled = np.ones(4, dtype=bool)
    execution = dict(
        scope="current_cohort", completed_phase_seconds={"storage": 2.0}, pending_device_spans=3
    )
    callback.model = SimpleNamespace(
        num_timesteps=0,
        training_games=0,
        phase="collect",
        cohort_games=4,
        cohort_number=1,
        _pending_episodes={},
        _stats={"collection_execution": execution},
        env=SimpleNamespace(enabled_envs=enabled, live_view=None),
        _pass_metrics=ActiveAccumulator(),
        policy=SimpleNamespace(optimizer=SimpleNamespace(state={0: {"step": ActiveAccumulator()}})),
    )
    callback.last_weights = [1.0, 0.0, 0.0]
    callback.locals = {"infos": []}
    try:
        assert progress.due()
        callback.log_progress(force=True)
        now[0] = 2
        callback.model.num_timesteps = 8
        enabled[2:] = False
        assert callback._on_step()
        now[0] = 5
        callback.model.num_timesteps = 14
        enabled[1:] = False
        assert callback._on_step()
        row = callback.snapshot()
        assert row["collection_progress"]["active_games"] == 1
        assert row["collection_progress"]["interval_active_game_seconds"] == 14
        assert row["collection_progress"]["interval_decisions_per_game_per_second"] == 1
        assert row["collection_progress"]["interval_active_transitions_per_second"] == 2.8
        assert row["collection_execution"] == execution
        execution["completed_phase_seconds"]["storage"] = 99
        assert row["collection_execution"]["completed_phase_seconds"]["storage"] == 2
        callback.model.collection_execution = dict(execution, pending_device_spans=0)
        assert callback.snapshot()["collection_execution"]["pending_device_spans"] == 0
    finally:
        callback.close()
        progress.close()


def test_device_profiler_defers_unfinished_spans_and_snapshot_is_host_only():
    from pvz_rl.monitoring.cuda_diagnostics import DeviceProfiler

    class Event:
        done = False

        def synchronize(self):
            raise AssertionError("Profiler must never synchronize")

    calls = []

    def elapsed(start, end):
        assert end.done
        calls.append((start, end))
        return 25

    profiler = DeviceProfiler(SimpleNamespace(cuda=SimpleNamespace(get_elapsed_time=elapsed)), True)
    start, end = Event(), Event()
    profiler.pending.append(("encoding", start, end))
    profiler.host("storage", 0.1)
    assert profiler.flush() == {"storage": 0.1}
    assert len(profiler.pending) == 1 and not profiler._pool and not calls
    end.done = True
    assert profiler.flush() == {"storage": 0.1, "encoding": 0.025}
    assert not profiler.pending and len(profiler._pool) == 2 and len(calls) == 1
    assert profiler.flush() == {"storage": 0.1, "encoding": 0.025}
    profiler.cp = ActiveAccumulator()
    row = profiler.snapshot()
    row["completed_phase_seconds"]["storage"] = 42
    assert profiler.snapshot()["completed_phase_seconds"]["storage"] == 0.1
    assert row["scope"] == "current_cohort" and row["pending_device_spans"] == 0
    disabled = DeviceProfiler(ActiveAccumulator())
    with disabled.track("nothing"):
        pass
    assert disabled.snapshot()["completed_phase_seconds"] == {}


def test_benchmark_collection_counts_live_work_not_capacity(monkeypatch):
    from pvz_rl.monitoring import collection_benchmark
    from pvz_rl.monitoring.progress import CollectionProgress

    times = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(
        collection_benchmark,
        "CollectionProgress",
        lambda: CollectionProgress(clock=lambda: next(times)),
    )
    enabled = np.zeros(128, dtype=bool)
    enabled[[17, 54, 91]] = True
    model = SimpleNamespace(
        num_timesteps=1000, env=SimpleNamespace(enabled_envs=enabled, num_envs=128)
    )

    def step(callback):
        model.num_timesteps += int(enabled.sum())
        enabled[np.flatnonzero(enabled)[-1]] = False

    model._collect_step = step
    row = collection_benchmark.collect(model, None, 32)
    assert row["lifetime_active_transitions"] == 6
    assert row["active_games"] == 0 and row["interval_active_game_seconds"] == 6
    assert row["interval_decisions_per_game_per_second"] == 1
    assert row["interval_active_transitions_per_second"] == 2


def test_benchmark_matrix_cli_headless_controls(tmp_path, monkeypatch):
    from pvz_rl.monitoring import collection_benchmark

    assert collection_benchmark.ENVIRONMENT_COUNTS == (128, 64, 32, 16, 4, 1)
    assert collection_benchmark.DEFAULT_CASES == ("sparse", "mixed", "crowded")
    assert collection_benchmark.REPETITIONS == 3
    calls = []
    monkeypatch.setattr(
        collection_benchmark, "benchmark", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "collection_benchmark",
            "--output",
            str(tmp_path / "bench.json"),
            "--env-counts",
            "4",
            "1",
            "--cases",
            "mixed",
            "--scratch-lanes",
            "1",
            "--eager",
            "--no-telemetry",
            "--fit-cycle",
            "--cycle-seconds",
            "20",
        ],
    )
    collection_benchmark.main()
    assert calls[0][1] == dict(
        env_counts=[4, 1],
        cases=["mixed"],
        scratch_lanes=[1],
        fit=True,
        fit_live_games=[128],
        fit_cases=["mixed"],
        cycle_seconds=20.0,
        eager=True,
        telemetry=False,
    )


@pytest.mark.parametrize("live_games", [128, 64, 32, 16, 4, 1])
def test_benchmark_reset_keeps_128_original_noncontiguous_slots(live_games, monkeypatch):
    from contextlib import nullcontext

    from pvz_game.cuda.schema import HEADER

    from pvz_rl.monitoring.collection_benchmark import active_slots, reset

    slots = active_slots(live_games)
    assert len(set(slots)) == live_games and slots.min() >= 0 and slots.max() < 128
    if live_games < 128:
        assert not np.array_equal(np.sort(slots), np.arange(live_games))
    order = []
    header = np.ones((128, len(HEADER)), dtype=np.int64)
    enabled = np.ones(128, dtype=bool)

    def restore(rows):
        assert len(rows) == 128
        header[:, HEADER.index("enabled")] = 1

    env = SimpleNamespace(
        num_envs=128,
        enabled_envs=enabled,
        device_context=nullcontext,
        batch=SimpleNamespace(header=header, restore=restore),
        cp=SimpleNamespace(asarray=np.asarray),
        _refresh_growth_bounds=lambda rows: None,
        features=SimpleNamespace(
            initialize_home=lambda rows: None, encode=lambda: None, obs_tensor="host-fixture"
        ),
        profiler=SimpleNamespace(flush=lambda: {}, seconds={}),
    )
    model = SimpleNamespace(
        env=env,
        _buffer=SimpleNamespace(close=lambda: order.append("close")),
        _drain=lambda: order.append("drain"),
        _close_sequences=lambda: order.append("close-sequences"),
    )

    def begin(callback):
        order.append("begin")
        enabled[:] = True
        model.cohort_games = 128

    model._begin = begin
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    reset(model, None, {}, 101, live_count=live_games)
    assert order == ["drain", "close-sequences", "close", "begin"]
    assert model.cohort_games == 128 and env.num_envs == 128
    assert np.array_equal(np.flatnonzero(enabled), np.sort(slots))
    assert np.array_equal(header[:, HEADER.index("enabled")], enabled)


def test_benchmark_probe_lane_control_survives_reallocation(monkeypatch):
    from pvz_rl.envs.probes import CounterfactualCollector
    from pvz_rl.monitoring.collection_benchmark import CollectionProbes

    calls = []
    monkeypatch.setattr(
        CounterfactualCollector, "_allocate", lambda probe: calls.append(probe.lanes)
    )
    probe = CollectionProbes.__new__(CollectionProbes)
    probe.requested_lanes = 1
    for _ in range(2):
        probe.lanes = 2
        probe._allocate()
    assert calls == [1, 1]


def test_benchmark_quiescent_wait_belongs_to_last_live_batch(monkeypatch):
    from pvz_rl.monitoring import collection_benchmark
    from pvz_rl.monitoring.progress import CollectionProgress

    now = [0.0]
    monkeypatch.setattr(
        collection_benchmark, "CollectionProgress", lambda: CollectionProgress(clock=lambda: now[0])
    )
    enabled = np.zeros(128, dtype=bool)
    enabled[[17, 54]] = True
    model = SimpleNamespace(num_timesteps=0, env=SimpleNamespace(enabled_envs=enabled))
    waits = []

    def step(callback):
        model.num_timesteps += int(enabled.sum())
        enabled[:] = False
        now[0] += 1

    def quiesce():
        waits.append(True)
        now[0] += 3

    model._collect_step = step
    row = collection_benchmark.collect(model, None, 16, quiesce=quiesce)
    assert waits == [True] and row["lifetime_active_transitions"] == 2
    assert row["lifetime_collection_seconds"] == 4
    assert row["interval_active_game_seconds"] == 8
    assert row["interval_decisions_per_game_per_second"] == 0.25
    assert row["interval_active_transitions_per_second"] == 0.5


def test_benchmark_rebuilds_actual_and_probe_cutoff_features(monkeypatch):
    from contextlib import nullcontext

    from pvz_rl.monitoring import collection_benchmark

    events = []
    cfg = load_config()
    env = SimpleNamespace(
        cfg=cfg, condition="masked", device_context=nullcontext, batch=object(), profiler=object()
    )

    class Features:
        def __init__(self, batch, cfg, condition):
            self.cutoff = cfg["environment"]["cutoff_seconds"]
            self.profiler = None
            self.module = SimpleNamespace(get_function=lambda name: (name, self.cutoff))

    def release():
        events.append("release")
        probes.features = None

    def allocate():
        events.append("allocate")
        probes.features = Features(env.batch, env.cfg, env.condition)
        probes.features.profiler = env.profiler

    probes = SimpleNamespace(
        features=Features(env.batch, cfg, env.condition),
        release_workspaces=release,
        _ensure_workspaces=allocate,
    )
    model = SimpleNamespace(env=env, _probes=probes, _drain=lambda: events.append("drain"))
    monkeypatch.setattr(collection_benchmark, "CudaFeatures", Features)
    for cutoff in (1, 1200):
        collection_benchmark.rebuild_cutoff_features(model, cutoff)
        assert env.features.cutoff == probes.features.cutoff == cutoff
        assert env.features.profiler is probes.features.profiler is env.profiler
        assert probes.copy_kernel == ("copy_probe_state", cutoff)
    assert events == ["drain", "release", "allocate"] * 2


@pytest.mark.parametrize(
    "terminal,timed_out,valid",
    [(-2.5, True, True), (0, True, False), (-1, True, False), (-2.5, False, False)],
)
def test_benchmark_cycle_guards_timeout_reward_and_restores_cutoff(
    monkeypatch, terminal, timed_out, valid
):
    from pvz_rl.envs.cuda_features import REWARD_FIELDS
    from pvz_rl.monitoring import collection_benchmark

    cfg = load_config()
    cfg["reward"]["loss_penalty"] = 2.5
    original_cutoff = cfg["environment"]["cutoff_seconds"]
    slots = collection_benchmark.active_slots(1)
    rewards = np.zeros((128, len(REWARD_FIELDS)))
    rewards[slots, REWARD_FIELDS.index("terminal")] = terminal
    transitions = np.zeros((128, 4))
    transitions[slots, 1] = timed_out
    env = SimpleNamespace(
        cfg=cfg,
        enabled_envs=np.zeros(128, dtype=bool),
        batch=SimpleNamespace(rules=SimpleNamespace(game={"tick_rate": 100})),
        last_reward_parts_host=rewards,
        last_transition_host=transitions,
    )
    model = SimpleNamespace(
        env=env,
        _fit_epoch=0,
        _phase_times={},
        _buffer=SimpleNamespace(finalize=lambda: {}),
        _finalize_rewards=lambda callback: None,
        _phase=lambda *args: None,
        _synchronize=lambda callback: None,
        cohort_metrics={},
    )
    cutoffs, resets, passes, warmups = [], [], [], []

    def rebuild(model, cutoff):
        cutoffs.append(cutoff)
        model.env.cfg["environment"]["cutoff_seconds"] = cutoff

    def fit(callback):
        model._fit_epoch += 1
        passes.append(model._fit_epoch)

    monkeypatch.setattr(collection_benchmark, "rebuild_cutoff_features", rebuild)
    monkeypatch.setattr(
        collection_benchmark,
        "reset",
        lambda *args, **kwargs: resets.append(cfg["environment"]["cutoff_seconds"]),
    )
    monkeypatch.setattr(
        collection_benchmark, "measure", lambda *args, **kwargs: dict(kwargs, seconds=1.0)
    )
    monkeypatch.setattr(
        collection_benchmark,
        "collect",
        lambda model, callback, decisions, **kwargs: warmups.append(decisions),
    )
    model._drain = lambda: None
    model._fit_step = fit
    if valid:
        row = collection_benchmark.fit_cycle(
            model, None, {"tick": 1}, 16, None, None, {"live_games": 1}, 5
        )
        assert passes == [1, 2, 3, 4] and resets == [1] + [original_cutoff] * 5
        assert warmups == [8]
        assert [row["repeat"] for row in row["after_warmed"]] == [0, 1, 2]
        assert all(trial["activity"] == "after-fit-warm" for trial in row["after_warmed"])
        assert row["before"]["cutoff_terminal_reward"] == -2.5
        assert row["before"]["cutoff_terminal_games"] == 1
        assert "setup_seconds" in row["before"] and "setup_seconds" in row["after"]
    else:
        with pytest.raises(RuntimeError, match="configured terminal loss"):
            collection_benchmark.fit_cycle(
                model, None, {"tick": 1}, 16, None, None, {"live_games": 1}, 5
            )
        assert not passes and resets == [1]
    assert cutoffs == [1, original_cutoff]
    assert cfg["environment"]["cutoff_seconds"] == original_cutoff


@pytest.mark.parametrize(
    "rates,regression,all_slower",
    [
        ([94, 94, 94], True, True),
        ([95, 95, 95], False, False),
        ([94, 94, 101], True, False),
        ([110, 110, 110], False, False),
    ],
)
def test_benchmark_warmed_median_regression_boundary(rates, regression, all_slower):
    from pvz_rl.monitoring.collection_benchmark import warmed_comparison

    primary = [dict(repeat=repeat, transitions_per_second=100) for repeat in range(3)]
    after = [dict(repeat=repeat, transitions_per_second=rate) for repeat, rate in enumerate(rates)]
    row = warmed_comparison(primary, after)
    assert row["primary_warmed_median"] == 100
    assert row["after_warmed_median"] == float(np.median(rates))
    assert row["median_regression_over_5_percent"] is regression
    assert row["all_after_trials_over_5_percent_slower"] is all_slower


def test_benchmark_cycle_budget_exact_boundary(monkeypatch):
    from pvz_rl.monitoring import collection_benchmark

    monkeypatch.setattr(collection_benchmark, "perf_counter", lambda: 120)
    collection_benchmark.check_cycle_budget(121)
    with pytest.raises(TimeoutError, match="wall budget"):
        collection_benchmark.check_cycle_budget(120)


@pytest.mark.parametrize(
    "settings",
    [
        dict(decisions=0),
        dict(decisions=41),
        dict(env_counts=[]),
        dict(env_counts=[1, 1]),
        dict(cases=["unknown"]),
        dict(scratch_lanes=[3]),
        dict(cycle_seconds=0),
    ],
)
def test_benchmark_rejects_invalid_controls_before_cuda(tmp_path, monkeypatch, settings):
    from pvz_rl.monitoring.collection_benchmark import benchmark

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid benchmark controls reached CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden)
    with pytest.raises(ValueError):
        benchmark(tmp_path / "bench.json", **settings)
