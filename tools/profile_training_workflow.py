"""Time the real training CLI for a bounded window or one fully fitted cohort.

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
        "--full-cohort", action="store_true", help="Stop after one fully fitted cohort"
    )
    parser.add_argument(
        "--process-trace", action="store_true", help="Capture ordered CPU/CUDA spans"
    )
    parser.add_argument(
        "--cprofile", action="store_true", help="Record Python calls during collection"
    )
    parser.add_argument(
        "--kernel-profile",
        action="store_true",
        help="Capture bounded CUPTI/cProfile windows; use separate unprofiled timing runs",
    )
    parser.add_argument(
        "--stop-file", type=Path, help="Stop at the next bounded operation when this file exists"
    )
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.training_args
    if command[:1] == ["--"]:
        command = command[1:]
    if command[:1] != ["train"] or args.collect_seconds <= 0:
        parser.error("Supply a positive collection window and -- train ...")
    if args.full_cohort and args.stop_after_collect:
        parser.error("Choose --full-cohort or --stop-after-collect")
    if args.process_trace and not args.full_cohort:
        parser.error("--process-trace requires --full-cohort")
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
    trace = None
    process_trace = None
    cohort_finished = None
    fitting_started = None
    phase_events = []
    fitting_passes = []
    label_signature = None
    scoped = {
        phase: defaultdict(lambda: dict(calls=0, seconds=0.0, max_seconds=0.0))
        for phase in ("startup", "collection", "fitting", "recovery")
    }
    patches = []
    stop = threading.Event()
    profile = None
    kernels = None
    stage_trace = None
    if args.cprofile:
        import cProfile

        profile = cProfile.Profile()

    def snapshot():
        row = dict(elapsed_seconds=perf_counter() - started)
        if model is not None:
            row.update(phase=str(model.phase), steps=model.num_timesteps)
            row["completed_games"] = len(getattr(model, "_pending_episodes", {}))
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
            else "fitting"
            if args.full_cohort and cohort_finished is None
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
            if profile is not None:
                profile.disable()

    def patch(owner, name, label=None, before=None):
        original = getattr(owner, name)
        key = label or f"{owner.__name__}.{name}"

        @wraps(original)
        def measured(*values, **keywords):
            nonlocal stage_trace
            if before:
                before(*values, **keywords)
            scope = timing_scope()
            begin = perf_counter()
            ends = []
            if args.process_trace and key == "CudaSequentialQ._begin":
                env = values[0].env
                if stage_trace is None:
                    from profiling_metrics import DeviceStageTrace

                    stage_trace = DeviceStageTrace(env, started)
                stream = env.cp.cuda.ExternalStream(
                    env.stream.cuda_stream, device_id=env.batch.device
                )
                ends.append((stage_trace.begin(key, stream), stream))
            epoch = getattr(values[0], "_fit_epoch", None) if name == "_fit_step" else None
            try:
                return original(*values, **keywords)
            finally:
                for end, stream in ends:
                    end.record(stream)
                duration = perf_counter() - begin
                record(key, duration, scope)
                if epoch is not None and values[0]._fit_epoch != epoch:
                    fitting_passes.append(
                        dict(
                            pass_number=epoch + 1,
                            end_seconds=perf_counter() - started,
                            total_loss=values[0]._stats.get("total_loss"),
                            primary_loss=values[0]._stats.get("primary_loss"),
                        )
                    )
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
        peashooter_teacher,
        sequential_training,
        training,
        training_requirements,
        wait_outcomes,
    )
    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.cuda_q import CudaCohortLifecycle

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

    def restore_observers():
        if trace is not None:
            trace.restore()
        if kernels is not None:
            kernels.close()
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)

    def check_window(value, *unused, **kwargs):
        nonlocal drain_started, trace
        remember(value)
        if args.stop_file and args.stop_file.exists():
            restore_observers()
            end_collection()
            raise KeyboardInterrupt("Workflow stop file requested a checkpoint")
        if collection_started is None:
            begin_collection(value)
        if args.process_trace and trace is None:
            from draw_collection_process import CollectionTrace

            trace = CollectionTrace(value)
            callback = unused[0] if unused else None
            if callback is not None:
                for name in ("update_locals", "on_step", "log_progress"):
                    if hasattr(callback, name):
                        trace.wrap(callback, name, f"callbacks: {name}")
            trace.wrap(
                value,
                "_fit_step",
                "fit pass / forward backward optimizer",
                device_label="fit forward backward optimizer",
            )
            trace.wrap(
                value, "_finalize_rewards", "finalize rewards", device_label="finalize rewards"
            )
            trace.wrap(
                value._buffer,
                "finalize",
                "assemble fitting targets",
                device_label="assemble fitting targets",
            )
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
            restore_observers()
            raise KeyboardInterrupt("Workflow profiling collection window completed")

    def check_phase(value, phase, *unused, **kwargs):
        nonlocal fitting_started, label_signature
        remember(value)
        phase_events.append(dict(phase=phase.value, start_seconds=perf_counter() - started))
        if phase == CohortPhase.FINALIZE_REWARDS and (args.full_cohort or args.stop_after_collect):
            end_collection()
            if args.full_cohort:
                import hashlib

                from pvz_rl.config import digest

                signature = hashlib.sha256()
                for block_index, block in enumerate(value._buffer.blocks):
                    rows = block[
                        : min(
                            value._buffer.block_rows,
                            value._buffer.size - block_index * value._buffer.block_rows,
                        )
                    ]
                    for field in (
                        "action",
                        "accepted",
                        "tick",
                        "duration",
                        "executed_action",
                        "baseline_outcomes",
                        "effect_valid",
                        "effect_labels",
                    ):
                        if field in rows.dtype.names:
                            signature.update(rows[field].tobytes())
                teacher = getattr(value, "_peashooter_teacher", None)
                label_signature = dict(
                    trajectories_and_targets=signature.hexdigest(),
                    teacher=None if teacher is None else digest(sorted(teacher.labels.items())),
                )
        if phase == CohortPhase.FIT:
            fitting_started = perf_counter()
        if args.stop_after_collect and phase == CohortPhase.FINALIZE_REWARDS:
            restore_observers()
            raise KeyboardInterrupt("Workflow profiling completed one collection")

    original_sync = sequential_training.CudaSequentialQ._synchronize

    def synchronize(value, callback):
        nonlocal cohort_finished, process_trace
        capture_ended = perf_counter()
        if trace is not None:
            with value.env.device_context():
                value.env.cp.cuda.get_current_stream().synchronize()
            process_trace = trace.finish(
                args.output.with_suffix(".process.json"),
                settings=dict(value.cfg["training"]["performance"]),
                auxiliary=dict(getattr(getattr(value.env, "auxiliary", None), "metrics", {})),
                games=value.cohort_games,
                ended=collection_ended,
                setup_seconds=trace.origin - collection_started,
                workload=dict(
                    cutoff_seconds=value.cfg["environment"]["cutoff_seconds"],
                    live_view=value.cfg["visualization"]["live_enabled"],
                ),
            )
            process_trace["cohort_finished_seconds"] = capture_ended - trace.origin
            process_trace["fitting_started_seconds"] = fitting_started - trace.origin
            args.output.with_suffix(".process.json").write_text(
                json.dumps(process_trace, indent=2), encoding="utf-8"
            )
        original_sync(value, callback)
        cohort_finished = perf_counter()
        if args.full_cohort:
            restore_observers()
            raise KeyboardInterrupt("Workflow profiling completed one fully fitted cohort")

    sequential_training.CudaSequentialQ._synchronize = synchronize
    patches.append((sequential_training.CudaSequentialQ, "_synchronize", original_sync))

    patch(sequential_training.CudaSequentialQ, "_begin", before=begin_collection)
    patch(sequential_training.CudaSequentialQ, "_prepare_collection", before=check_window)
    patch(sequential_training.CudaSequentialQ, "_collect_step", before=remember)
    patch(sequential_training.CudaSequentialQ, "_fit_step", before=check_window)
    patch(sequential_training.CudaSequentialQ, "_phase", before=check_phase)
    for name in ("advance", "before", "observe", "_initialize", "_allocate", "_fill", "_run_chunk"):
        patch(peashooter_teacher.PeashooterTeacher, name)
    for name in ("advance", "finalize", "_spill_roots", "_root", "_allocate"):
        patch(wait_outcomes.WaitOutcomes, name)
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

    if args.kernel_profile:
        from profiling_metrics import KernelWindows

        args.output.parent.mkdir(parents=True, exist_ok=True)
        kernels = KernelWindows(args.output.with_suffix(".kernels.json"), origin=started)
        kernels.wrap(sequential_training.CudaSequentialQ, "_collect_step", "decisions", skip=30)
        kernels.wrap(peashooter_teacher.PeashooterTeacher, "_run_chunk", "teacher", skip=20)
        kernels.wrap(sequential_training.CudaSequentialQ, "_fit_step", "fitting", skip=2)

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
        restore_observers()
        stop.set()
        thread.join()
        if profile is not None:
            profile.disable()
            profile.dump_stats(str(args.output.with_suffix(".pstats")))
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
                    workload=None
                    if model is None
                    else dict(
                        cutoff_seconds=model.cfg["environment"]["cutoff_seconds"],
                        auxiliary_hybrid=model.cfg["training"]["performance"]["auxiliary_hybrid"],
                    ),
                    timings=dict(spans),
                    timings_by_phase={phase: dict(values) for phase, values in scoped.items()},
                    label_drain_started_seconds=None
                    if drain_started is None
                    else drain_started - started,
                    cohort_finished_seconds=None
                    if cohort_finished is None
                    else cohort_finished - started,
                    cohort_metrics=getattr(model, "cohort_metrics", None),
                    label_signature=label_signature,
                    fitting_passes=fitting_passes,
                    stage_gpu_events=[] if stage_trace is None else stage_trace.finish(),
                    phase_events=phase_events,
                    process_trace=process_trace,
                    events=events,
                    samples=samples,
                    timing_notes=[
                        "Inclusive host wall spans overlap; do not add nested operations.",
                        "Collection keeps the real configured cutoff, weights, viewer and callback paths; startup is reported separately.",
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
