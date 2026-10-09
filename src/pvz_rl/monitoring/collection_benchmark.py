"""Headless cold/warm snapshot controls and opt-in bounded four-pass fitting.

Warmup and replicates follow PyTorch 2.8's torch.utils.benchmark.Timer. Collection
uses its existing host handoff, not a new per-step wait. The optional cycle uses
complete cutoff games and the current fitter; never learn, mastery or checkpoints.
"""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Spawn
from pvz_game.cuda.schema import HEADER
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config
from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures
from pvz_rl.envs.probes import CounterfactualCollector
from pvz_rl.learning.cohort import CohortPhase
from pvz_rl.learning.training import build_model, vector_env
from pvz_rl.monitoring.hardware import HardwareMonitor
from pvz_rl.monitoring.progress import CollectionProgress
from pvz_rl.provenance import write_json

CASES = ("sparse", "mixed", "crowded", "rejection-heavy", "accepted-action")
DEFAULT_CASES = CASES[:3]
ENVIRONMENT_COUNTS = (128, 64, 32, 16, 4, 1)
ORIGINAL_ENVIRONMENTS = ENVIRONMENT_COUNTS[0]
SCRATCH_LANES = (1, 2)
WARMUP_DECISIONS = 8
REPETITIONS = 3
FIT_PASSES = 4
PROTOCOL = "collection-active-snapshot-v4"


class CollectionCallback(BaseCallback):
    def _on_step(self):
        return True


class CollectionProbes(CounterfactualCollector):
    """Retain the requested scratch control across fit workspace release."""

    def __init__(self, env, lanes):
        self.requested_lanes = lanes
        super().__init__(env)

    def _allocate(self):
        self.lanes = self.requested_lanes
        super()._allocate()


def active_slots(live_count, environments=ORIGINAL_ENVIRONMENTS):
    """A fixed noncontiguous subset of original slots, including for one game."""
    if live_count not in ENVIRONMENT_COUNTS or environments != ORIGINAL_ENVIRONMENTS:
        raise ValueError("Active controls require supported live counts in 128 original slots")
    return (np.arange(live_count, dtype=np.intp) * 37 + 17) % environments


def snapshot(case):
    if case not in CASES:
        raise ValueError(f"Unknown collection case: {case}")
    count = 60 if case == "crowded" else 20 if case == "mixed" else 0
    plants = 40 if case == "accepted-action" else 12 if case in ("mixed", "crowded") else 0
    game = Game()
    game.reset(
        LevelSpec(
            "collection-control",
            tuple(Spawn(1, "basic", index % 5, x=9000 - index) for index in range(count))
            + (Spawn(100000, "basic", 0),),
            initial_sun=0 if case == "rejection-heavy" else 1000,
            plants=tuple(
                InitialPlant("peashooter", index // 9, index % 9) for index in range(plants)
            ),
        ),
        101,
    )
    game.step(ticks=1)
    raw = game.snapshot()
    if case == "crowded":
        raw["projectiles"] = [
            dict(id=raw["next_id"] + index, row=index % 5, x=500 + index, damage=20, icy=False)
            for index in range(200)
        ]
        raw["next_id"] += 200
        game.restore(raw)
        raw = game.snapshot()
    return raw


def reset(model, callback, raw, seed, live_count=ORIGINAL_ENVIRONMENTS):
    model._drain()
    model._close_sequences()
    if model._buffer is not None:
        model._buffer.close()
    torch.manual_seed(seed)
    model._begin(callback)
    env = model.env
    slots = active_slots(live_count, env.num_envs)
    env.enabled_envs[:] = False
    env.enabled_envs[slots] = True
    with env.device_context():
        env.batch.restore([raw] * env.num_envs)
        env.batch.header[:, HEADER.index("enabled")] = env.cp.asarray(env.enabled_envs)
        env._refresh_growth_bounds(range(env.num_envs))
        env.features.initialize_home(range(env.num_envs))
        env.features.encode()
        model._last_obs = env.features.obs_tensor
    torch.cuda.synchronize()
    env.profiler.flush()
    env.profiler.seconds.clear()


def collect(model, callback, decisions, *, quiesce=None):
    """Count committed live transitions, never padded slots or counterfactuals."""
    progress = CollectionProgress()
    progress.observe(
        transitions=model.num_timesteps,
        active_games=int(model.env.enabled_envs.sum()),
        cohort=0,
        phase="collect",
    )
    for decision in range(decisions):
        if not model.env.enabled_envs.any():
            break
        model._collect_step(callback)
        if quiesce is not None and (decision == decisions - 1 or not model.env.enabled_envs.any()):
            quiesce()
        progress.observe(
            transitions=model.num_timesteps,
            active_games=int(model.env.enabled_envs.sum()),
            cohort=0,
            phase="collect",
        )
    return progress.snapshot()


def capture_setup_seconds(model):
    """Read cumulative capture setup counters without device queries."""
    return {
        name: network.entity.collection_metrics.get("capture_setup_seconds", 0.0)
        for name, network in (("online", model.policy), ("ema", getattr(model, "_teacher", None)))
        if network is not None
    }


def measure(model, callback, hardware, process, decisions, **context):
    env = model.env
    hardware.update(**context)
    torch.cuda.reset_peak_memory_stats()
    cpu_before = process.cpu_times()
    capture_before = capture_setup_seconds(model)
    started = perf_counter()
    work = collect(model, callback, decisions, quiesce=model._drain)
    elapsed = perf_counter() - started
    capture_after = capture_setup_seconds(model)
    cpu_after = process.cpu_times()
    phases = env.profiler.flush()
    with hardware.lock:
        samples = [
            sample
            for sample in hardware.samples
            if all(sample.get(key) == value for key, value in context.items())
        ]
    means = {}
    for key in ("gpu_percent", "system_cpu_percent", "process_cpu_percent"):
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        means[key] = float(np.mean(values)) if values else None
    import cupy

    transitions = work["lifetime_active_transitions"]
    execution = copy.deepcopy(getattr(model, "collection_execution", {}))
    execution["timings"] = env.profiler.snapshot()
    return dict(
        **context,
        seconds=elapsed,
        capture_setup_seconds={
            name: value - capture_before[name] for name, value in capture_after.items()
        },
        transitions=transitions,
        transitions_per_second=transitions / elapsed if elapsed > 0 else None,
        decisions_per_game_per_second=work["interval_decisions_per_game_per_second"],
        active_games=work["active_games"],
        mean_active_games=work["interval_mean_active_games"],
        active_game_seconds=work["lifetime_active_game_seconds"],
        phases_seconds=phases,
        collection_execution=execution,
        active_slot_ids=active_slots(context["live_games"]).tolist(),
        utilization=means,
        process_cpu_percent=100
        * (cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system)
        / max(elapsed, 1e-9),
        host_rss_mib=process.memory_info().rss / 2**20,
        pinned_transfer_mib=env.handoff.nbytes / 2**20,
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        cupy_reserved_mib=cupy.get_default_memory_pool().total_bytes() / 2**20,
        trajectory_bytes=model._buffer.entity_size * 44
        + model._buffer.size * model._buffer.dtype.itemsize,
    )


def rebuild_cutoff_features(model, cutoff):
    """Recompile actual and probe cutoff/reward kernels outside timed collection."""
    model._drain()
    env = model.env
    probes = model._probes
    probes.release_workspaces()
    env.cfg["environment"]["cutoff_seconds"] = cutoff
    with env.device_context():
        env.features = CudaFeatures(env.batch, env.cfg, env.condition)
        env.features.profiler = env.profiler
        probes.copy_kernel = env.features.module.get_function("copy_probe_state")
        probes._ensure_workspaces()


def release_benchmark_model(model):
    """Release captured bound encoders at quiescence, before deleting the model."""
    model._drain()
    model._close_sequences()
    for network in (model.policy, getattr(model, "_teacher", None)):
        if network is not None:
            network.entity.release_collection()
            network.entity.release_fitting()
    probes = getattr(model, "_probes", None)
    if probes is not None:
        probes.release_workspaces()
    if model._buffer is not None:
        model._buffer.close()


def check_cycle_budget(deadline):
    if perf_counter() >= deadline:
        raise TimeoutError("Bounded collection/fitting cycle exceeded its wall budget")


def warmed_comparison(primary, after):
    """Compare only the three normal-snapshot warmed trials, never terminal/cold work."""
    for rows in (primary, after):
        if len(rows) != REPETITIONS or {row["repeat"] for row in rows} != set(range(REPETITIONS)):
            raise ValueError("Warmed comparison requires exactly three distinct trials per control")
        if any(
            not np.isfinite(row["transitions_per_second"]) or row["transitions_per_second"] <= 0
            for row in rows
        ):
            raise ValueError("Warmed comparison requires positive finite transition rates")
    baseline = float(np.median([row["transitions_per_second"] for row in primary]))
    rates = [row["transitions_per_second"] for row in after]
    median = float(np.median(rates))
    threshold = baseline * 0.95
    return dict(
        metric="active_transitions_per_second",
        primary_warmed_median=baseline,
        after_warmed_median=median,
        relative_change=median / baseline - 1,
        median_regression_over_5_percent=median < threshold,
        all_after_trials_over_5_percent_slower=all(rate < threshold for rate in rates),
    )


def fit_cycle(model, callback, raw, decisions, hardware, process, context, seconds):
    """One bounded collect -> four-pass fit -> collect; normal terminal cutoffs."""
    env = model.env
    previous_cutoff = env.cfg["environment"]["cutoff_seconds"]
    terminal_raw = copy.deepcopy(raw)
    terminal_raw["tick"] = env.batch.rules.game["tick_rate"] - decisions
    deadline = perf_counter() + seconds
    started = perf_counter()
    try:
        rebuild_cutoff_features(model, 1)
        reset(model, callback, terminal_raw, 101, live_count=context["live_games"])
        setup_before = perf_counter() - started
        check_cycle_budget(deadline)
        before = measure(
            model, callback, hardware, process, decisions, **context, activity="before-fit"
        )
        check_cycle_budget(deadline)
        if env.enabled_envs.any():
            raise RuntimeError("Bounded cycle did not collect complete cutoff games")
        slots = active_slots(context["live_games"])
        terminal = env.last_reward_parts_host[slots, REWARD_FIELDS.index("terminal")]
        loss = -env.cfg["reward"]["loss_penalty"]
        if not np.all(env.last_transition_host[slots, 1]) or not np.all(terminal == loss):
            raise RuntimeError("Bounded cutoff games must receive the configured terminal loss")
        before.update(
            setup_seconds=setup_before,
            cutoff_terminal_reward=loss,
            cutoff_terminal_games=len(slots),
        )
        model._phase_times["collect"] = before["seconds"]
        started = perf_counter()
        model._finalize_rewards(callback)
        model.prefit_errors = model._buffer.finalize()
        preparation = perf_counter() - started
        model._phase_times["returns"] = preparation
        model._phase(CohortPhase.FIT, callback)
        started = perf_counter()
        while model._fit_epoch < FIT_PASSES:
            check_cycle_budget(deadline)
            model._fit_step(callback)
        model._phase_times["fit"] = perf_counter() - started
        model._phase(CohortPhase.SYNCHRONIZE, callback)
        model._synchronize(callback)
        fitting = perf_counter() - started
        phase_memory = copy.deepcopy(model.cohort_metrics.get("phase_memory", []))
    finally:
        started = perf_counter()
        rebuild_cutoff_features(model, previous_cutoff)
    check_cycle_budget(deadline)
    reset(model, callback, raw, 101, live_count=context["live_games"])
    check_cycle_budget(deadline)
    setup_after = perf_counter() - started
    after = measure(model, callback, hardware, process, decisions, **context, activity="after-fit")
    after["setup_seconds"] = setup_after
    check_cycle_budget(deadline)
    reset(model, callback, raw, 101, live_count=context["live_games"])
    check_cycle_budget(deadline)
    collect(model, callback, WARMUP_DECISIONS, quiesce=model._drain)
    check_cycle_budget(deadline)
    after_warmed = []
    for repeat in range(REPETITIONS):
        check_cycle_budget(deadline)
        reset(model, callback, raw, 101, live_count=context["live_games"])
        check_cycle_budget(deadline)
        after_warmed.append(
            measure(
                model,
                callback,
                hardware,
                process,
                decisions,
                **dict(context, repeat=repeat),
                activity="after-fit-warm",
            )
        )
        check_cycle_budget(deadline)
    return dict(
        **context,
        optimizer_steps=FIT_PASSES,
        passes=FIT_PASSES,
        cutoff_seconds=1,
        setup_scope="actual/probe cutoff feature rebuild and reset; excluded from collection and fit timings",
        preparation_seconds=preparation,
        fit_seconds=fitting,
        phase_memory=phase_memory,
        before=before,
        after=after,
        after_warmed=after_warmed,
        comparison_scope="primary normal-snapshot warmed trials versus after_warmed; excludes before and after cold",
    )


def benchmark(
    output,
    decisions=32,
    *,
    env_counts=ENVIRONMENT_COUNTS,
    cases=DEFAULT_CASES,
    scratch_lanes=SCRATCH_LANES,
    fit=False,
    fit_live_games=(ORIGINAL_ENVIRONMENTS,),
    fit_cases=("mixed",),
    cycle_seconds=120,
    eager=False,
    telemetry=True,
):
    output = Path(output)
    if type(decisions) is not int or not 1 <= decisions <= 40:
        raise ValueError("Use 1..40 decisions to retain the accepted-action snapshot control")
    for values, choices in (
        (env_counts, ENVIRONMENT_COUNTS),
        (cases, CASES),
        (scratch_lanes, SCRATCH_LANES),
        (fit_live_games, ENVIRONMENT_COUNTS),
        (fit_cases, CASES),
    ):
        if (
            not values
            or len(set(values)) != len(values)
            or any(value not in choices for value in values)
        ):
            raise ValueError("Benchmark controls must be nonempty, unique supported choices")
    if not 0 < cycle_seconds <= 300:
        raise ValueError("Use a cycle wall budget in (0, 300] seconds")
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["visualization"].update(enabled=False, live_enabled=False, demos=False, videos=False)
    cfg["training"].update(until_stage_complete=False, total_games=10000, n_epochs=FIT_PASSES)
    cfg["training"]["n_envs"] = ORIGINAL_ENVIRONMENTS
    cfg["training"]["performance"].update(compile_kernels=not eager, telemetry=telemetry)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    result = dict(
        protocol=PROTOCOL,
        measured_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        formal_training=False,
        headless=True,
        optimizer_steps=0,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        environments=ORIGINAL_ENVIRONMENTS,
        live_games=list(env_counts),
        cases=list(cases),
        scratch_lanes=list(scratch_lanes),
        fit_cycle=fit,
        fit_live_games=list(fit_live_games),
        fit_cases=list(fit_cases),
        cycle_seconds=cycle_seconds,
        decisions_per_repeat=decisions,
        warmup_decisions=WARMUP_DECISIONS,
        repetitions=REPETITIONS,
        config=copy.deepcopy(cfg),
        measurements=[],
        comparison="two-lane versus the current one-lane memory fallback, not a historical trainer",
        cold_scope="fresh environment and policy; process and disk kernel caches may already be warm",
        references=[
            "https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/benchmark/utils/timer.py",
            "https://github.com/cupy/cupy/blob/v13.6.0/cupy/cuda/stream.pyx",
        ],
    )
    if output.exists():
        saved = json.loads(output.read_text("utf-8"))
        identity = (
            "protocol",
            "config",
            "decisions_per_repeat",
            "environments",
            "live_games",
            "cases",
            "scratch_lanes",
            "fit_cycle",
            "fit_live_games",
            "fit_cases",
            "cycle_seconds",
        )
        if any(saved.get(key) != result[key] for key in identity):
            raise ValueError("Existing benchmark has different settings; use a new output path")
        result["measurements"] = saved["measurements"]
    process = psutil.Process()
    with HardwareMonitor(output.with_suffix(".hardware.jsonl"), device="cuda") as hardware:
        for count in env_counts:
            for case in cases:
                raw = snapshot(case)
                for lanes in scratch_lanes:
                    cycle = fit and count in fit_live_games and case in fit_cases
                    context = dict(
                        phase=case,
                        case=case,
                        environments=ORIGINAL_ENVIRONMENTS,
                        live_games=count,
                        scratch_lanes=lanes,
                    )
                    completed = {
                        (row["sample_type"], row["repeat"])
                        for row in result["measurements"]
                        if all(row.get(key) == value for key, value in context.items())
                    }
                    expected = {
                        ("cold", None),
                        *(("warm", repeat) for repeat in range(REPETITIONS)),
                    }
                    if cycle:
                        expected.add(("cycle", None))
                    if expected <= completed:
                        continue
                    case_cfg = copy.deepcopy(cfg)
                    env = model = None
                    started = perf_counter()
                    try:
                        env = vector_env(case_cfg, "masked", 101)
                        model = build_model(case_cfg, "masked", env, 101)
                        model.trajectory_root = output.parent
                        model.set_logger(configure(folder=None, format_strings=[]))
                        callback = CollectionCallback()
                        callback.init_callback(model)
                        model._probes = CollectionProbes(env, lanes)
                        branch = (
                            9
                            if case == "accepted-action"
                            else 1
                            if case == "rejection-heavy"
                            else 0
                        )
                        with torch.no_grad():
                            model.policy.branch_head[-1].weight.zero_()
                            model.policy.branch_head[-1].bias.zero_()
                            model.policy.branch_head[-1].bias[branch] = 1
                        reset(model, callback, raw, 101, live_count=count)
                        setup = perf_counter() - started

                        def record(row):
                            result["measurements"].append(row)
                            result["optimizer_steps"] = sum(
                                item.get("optimizer_steps", 0) for item in result["measurements"]
                            )
                            write_json(output, result)
                            print(
                                f"{case} {count}/{ORIGINAL_ENVIRONMENTS} live games {lanes}-lane {row['sample_type']} "
                                f"repeat {row['repeat']}: {row.get('transitions_per_second', 'cycle')} "
                                "active transitions/s",
                                flush=True,
                            )

                        if ("cold", None) not in completed:
                            cold = measure(
                                model,
                                callback,
                                hardware,
                                process,
                                decisions,
                                **context,
                                sample_type="cold",
                                repeat=None,
                                activity="collect",
                            )
                            record(dict(cold, setup_seconds=setup))
                        reset(model, callback, raw, 101, live_count=count)
                        collect(model, callback, WARMUP_DECISIONS)
                        for repeat in range(REPETITIONS):
                            if ("warm", repeat) in completed:
                                continue
                            reset(model, callback, raw, 101, live_count=count)
                            record(
                                measure(
                                    model,
                                    callback,
                                    hardware,
                                    process,
                                    decisions,
                                    **context,
                                    sample_type="warm",
                                    repeat=repeat,
                                    activity="collect",
                                )
                            )
                        if cycle and ("cycle", None) not in completed:
                            cycle_context = dict(context, sample_type="cycle", repeat=None)
                            cycle_row = fit_cycle(
                                model,
                                callback,
                                raw,
                                decisions,
                                hardware,
                                process,
                                cycle_context,
                                cycle_seconds,
                            )
                            primary = [
                                row
                                for row in result["measurements"]
                                if row["sample_type"] == "warm"
                                and all(row.get(key) == value for key, value in context.items())
                            ]
                            cycle_row["warmed_comparison"] = warmed_comparison(
                                primary, cycle_row["after_warmed"]
                            )
                            record(cycle_row)
                    finally:
                        if model is not None:
                            release_benchmark_model(model)
                        if env is not None:
                            env.close()
                        del model, env
                        torch.cuda.empty_cache()
                        torch._dynamo.reset()
    result["optimizer_steps"] = sum(row.get("optimizer_steps", 0) for row in result["measurements"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=32)
    parser.add_argument(
        "--env-counts",
        "--live-games",
        type=int,
        nargs="+",
        choices=ENVIRONMENT_COUNTS,
        default=list(ENVIRONMENT_COUNTS),
        help="Live games in the fixed 128 original simulator slots",
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(DEFAULT_CASES))
    parser.add_argument(
        "--scratch-lanes", type=int, nargs="+", choices=SCRATCH_LANES, default=list(SCRATCH_LANES)
    )
    parser.add_argument("--fit-cycle", action="store_true")
    parser.add_argument(
        "--fit-live-games",
        type=int,
        nargs="+",
        choices=ENVIRONMENT_COUNTS,
        default=[ORIGINAL_ENVIRONMENTS],
    )
    parser.add_argument("--fit-cases", nargs="+", choices=CASES, default=["mixed"])
    parser.add_argument("--cycle-seconds", type=float, default=120)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--no-telemetry", action="store_true")
    args = parser.parse_args()
    try:
        benchmark(
            args.output,
            args.decisions,
            env_counts=args.env_counts,
            cases=args.cases,
            scratch_lanes=args.scratch_lanes,
            fit=args.fit_cycle,
            fit_live_games=args.fit_live_games,
            fit_cases=args.fit_cases,
            cycle_seconds=args.cycle_seconds,
            eager=args.eager,
            telemetry=not args.no_telemetry,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
