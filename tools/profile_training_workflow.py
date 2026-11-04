"""Time the real training CLI, stopping collection after a bounded wall window.

Example: python tools/profile_training_workflow.py --output runs/diagnostics/workflow.json
--collect-seconds 180 -- train --stage peashooter --until-stage-complete ...
Use a separate training --output directory: stopping retains its recovery archive.
Timings are inclusive host wall times; nested operations must not be added together.
"""

import argparse
import json
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-seconds", type=float, default=180)
    parser.add_argument("--stop-after-collect", action="store_true")
    parser.add_argument(
        "--cprofile", action="store_true", help="Record Python calls during collection"
    )
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.training_args
    if command[:1] == ["--"]:
        command = command[1:]
    if command[:1] != ["train"] or args.collect_seconds <= 0:
        parser.error("Supply a positive collection window and -- train ...")
    started = perf_counter()
    spans, events, samples = (
        defaultdict(lambda: dict(calls=0, seconds=0.0, max_seconds=0.0)),
        [],
        [],
    )
    model = None
    collection_started = None
    collection_ended = None
    final_collection = None
    drain_started = None
    scoped = {
        phase: defaultdict(lambda: dict(calls=0, seconds=0.0, max_seconds=0.0))
        for phase in ("startup", "collection", "recovery")
    }
    patches = []
    stop = threading.Event()
    profile = None
    if args.cprofile:
        import cProfile

        profile = cProfile.Profile()

    def snapshot():
        row = dict(elapsed_seconds=perf_counter() - started)
        if model is not None:
            row.update(phase=str(model.phase), steps=model.num_timesteps)
            row["simulated_seconds_per_env"] = (
                getattr(model, "_simulation_ticks", 0)
                / max(1, model.n_envs)
                / model.env.batch.rules.game["tick_rate"]
            )
            env = model.env
            if getattr(env, "enabled_envs", None) is not None:
                row["active_games"] = int(env.enabled_envs.sum())
            for name, attr in (
                ("teacher", "_peashooter_teacher"),
                ("outcomes", "_wait_outcomes"),
                ("pretraining", "_pretrainer"),
                ("factual_targets", "_reach"),
            ):
                owner = getattr(model, attr, None)
                if owner is not None:
                    row[name] = {
                        key: value for key, value in owner.metrics.items() if key != "examples"
                    }
                    if name == "teacher":
                        row["teacher_frontier"] = dict(
                            prepared=getattr(owner, "next_root", None),
                            initialized=getattr(owner, "initialized", None),
                        )
            row["collection_execution"] = getattr(model, "collection_execution", {})
        return row

    def timing_scope():
        return (
            "startup"
            if collection_started is None
            else "collection"
            if collection_ended is None
            else "recovery"
        )

    def record(key, duration, scope):
        for target in (spans, scoped[scope]):
            stat = target[key]
            stat["calls"] += 1
            stat["seconds"] += duration
            stat["max_seconds"] = max(stat["max_seconds"], duration)

    def end_collection():
        nonlocal collection_ended, final_collection
        if collection_ended is None:
            collection_ended = perf_counter()
            final_collection = snapshot()
            stop.set()
            if profile is not None:
                profile.disable()

    def patch(owner, name, label=None, before=None):
        original = getattr(owner, name)
        key = label or f"{owner.__name__}.{name}"

        @wraps(original)
        def measured(*values, **keywords):
            if before:
                before(*values, **keywords)
            scope = timing_scope()
            begin = perf_counter()
            try:
                return original(*values, **keywords)
            finally:
                duration = perf_counter() - begin
                record(key, duration, scope)
                if spans[key]["calls"] == 1 or duration >= 5:
                    events.append(
                        dict(
                            operation=key, start_seconds=begin - started, duration_seconds=duration
                        )
                    )

        setattr(owner, name, measured)
        patches.append((owner, name, original))

    def patch_property(owner, name):
        original = getattr(owner, name)
        key = f"{owner.__name__}.{name}"

        def measured(value):
            begin = perf_counter()
            scope = timing_scope()
            try:
                return original.fget(value)
            finally:
                duration = perf_counter() - begin
                record(key, duration, scope)

        setattr(owner, name, property(measured))
        patches.append((owner, name, original))

    from pvz_rl import cli
    from pvz_rl.envs import cuda_env
    from pvz_rl.envs.fork_execution import ForkExecution
    from pvz_rl.envs.fork_service import ForkService
    from pvz_rl.learning import (
        forecast_pretraining,
        peashooter_teacher,
        sequential_training,
        training,
        training_requirements,
        wait_outcomes,
    )
    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.cuda_q import CudaCohortLifecycle
    from pvz_rl.learning.reach_supervision import ReachSupervision

    events.append(
        dict(operation="imports", start_seconds=0, duration_seconds=perf_counter() - started)
    )
    for name in ("initial_weights", "require_cuda_training", "vector_env", "build_model"):
        patch(training, name)
    patch(training_requirements, "_cuda_probe")
    patch(cuda_env.CudaVecEnv, "start_live_view")
    patch(CudaCohortLifecycle, "_setup_learn")

    def remember(value, *unused, **kwargs):
        nonlocal model
        model = value

    def begin_collection(value, *unused, **kwargs):
        nonlocal collection_started
        remember(value)
        if collection_started is None:
            collection_started = perf_counter()
            events.append(
                dict(operation="collection_enter", start_seconds=collection_started - started)
            )
            if profile is not None:
                profile.enable()

    def check_window(value, *unused, **kwargs):
        nonlocal drain_started
        remember(value)
        if collection_started is None:
            begin_collection(value)
        if drain_started is None and not value.env.enabled_envs.any():
            drain_started = perf_counter()
            events.append(
                dict(operation="all_main_games_finished", start_seconds=drain_started - started)
            )
        if (
            collection_started is not None
            and perf_counter() - collection_started >= args.collect_seconds
        ):
            end_collection()
            raise KeyboardInterrupt("Workflow profiling collection window completed")

    def check_phase(value, phase, *unused, **kwargs):
        remember(value)
        if args.stop_after_collect and phase == CohortPhase.FINALIZE_REWARDS:
            end_collection()
            raise KeyboardInterrupt("Workflow profiling completed one collection")

    patch(sequential_training.CudaSequentialQ, "_begin", before=begin_collection)
    patch(sequential_training.CudaSequentialQ, "_prepare_collection", before=check_window)
    for name in ("_collect_step", "_fit_step"):
        patch(sequential_training.CudaSequentialQ, name, before=remember)
    patch(sequential_training.CudaSequentialQ, "_phase", before=check_phase)
    for name in ("__init__", "advance", "_generate", "_play", "_effects", "_fit"):
        patch(forecast_pretraining.ForecastPretraining, name)
    for name in ("advance", "before", "observe", "_initialize", "_allocate", "_fill", "_run_chunk"):
        patch(peashooter_teacher.PeashooterTeacher, name)
    for name in ("advance", "finalize", "_spill_roots", "_root", "_allocate"):
        patch(wait_outcomes.WaitOutcomes, name)
    for name in ("prepare", "continue_exercises"):
        patch(ReachSupervision, name)
    patch_property(wait_outcomes.WaitOutcomes, "estimated_bytes")
    for owner in (ForkService, ForkExecution):
        patch(owner, "advance")
    for name in ("download_snapshots", "upload_snapshots"):
        patch(wait_outcomes, name)
    patch(peashooter_teacher, "download_snapshots")
    for name in ("put", "_spill", "_encode"):
        patch(peashooter_teacher.SnapshotStore, name)
    for name in ("step_device", "step_host"):
        patch(cuda_env.CudaVecEnv, name)

    def pulse():
        while not stop.wait(15):
            row = snapshot()
            samples.append(row)
            print("WORKFLOW " + json.dumps(row, default=str), flush=True)

    thread = threading.Thread(target=pulse, daemon=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    thread.start()
    try:
        cli.main(command)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        thread.join()
        if profile is not None:
            profile.disable()
            profile.dump_stats(str(args.output.with_suffix(".pstats")))
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
        args.output.write_text(
            json.dumps(
                dict(
                    protocol="actual-training-workflow-v1",
                    python_call_profiling=args.cprofile,
                    command=command,
                    elapsed_seconds=perf_counter() - started,
                    collection_started_seconds=None
                    if collection_started is None
                    else collection_started - started,
                    collection_window_seconds=None
                    if collection_ended is None
                    else collection_ended - collection_started,
                    final_collection=final_collection,
                    pretraining=getattr(model, "pretraining_metrics", None),
                    workload=None
                    if model is None
                    else dict(
                        cutoff_seconds=model.cfg["environment"]["cutoff_seconds"],
                        pretrain_roots=model.cfg["training"]["forecasts"]["pretrain_roots"],
                        pretrain_passes=model.cfg["training"]["forecasts"]["pretrain_passes"],
                        auxiliary_hybrid=model.cfg["training"]["performance"]["auxiliary_hybrid"],
                        forecaster_pretrained=getattr(model, "forecaster_pretrained", False),
                    ),
                    timings=dict(spans),
                    timings_by_phase={phase: dict(values) for phase, values in scoped.items()},
                    label_drain_started_seconds=None
                    if drain_started is None
                    else drain_started - started,
                    events=events,
                    samples=samples,
                    timing_notes=[
                        "Inclusive host wall spans overlap; do not add nested operations.",
                        "Collection keeps the real configured cutoff, weights, pretraining, viewer and callback paths; startup is reported separately.",
                        "Recovery archive writing and cleanup follow the bounded collection window; they are not collection.",
                    ],
                ),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Workflow timing saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
