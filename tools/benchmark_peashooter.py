"""Frozen-policy Peashooter cohort timing, including oracle and four fitting passes."""

import argparse
import cProfile
import faulthandler
import gc
import hashlib
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from time import perf_counter


def close_interval_owners(model):
    owner = getattr(model, "_wait_outcomes", None)
    if owner is not None:
        owner.close()
        model._wait_outcomes = None
    service = getattr(model.env, "auxiliary", None)
    if service is not None:
        service.close()


def measure(base, settings, args, workspace):
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from pvz_rl.config import digest
    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.tile_curriculum import initial_tile_state
    from pvz_rl.learning.training import build_model, initial_weights, vector_env

    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_progress = perf_counter()

        def log_progress(self):
            if perf_counter() - self.last_progress < 30:
                return
            self.last_progress = perf_counter()
            teacher = getattr(self.model, "_peashooter_teacher", None)
            print(
                json.dumps(
                    dict(
                        event="progress",
                        phase=self.model.phase.value,
                        main_completed=len(self.model._pending_episodes),
                        roots=len(teacher.roots) if teacher else 0,
                        labeled=len(teacher.labels) if teacher else 0,
                        oracle_completed=teacher.metrics["completed_rollouts"] if teacher else 0,
                        auxiliary=getattr(
                            getattr(self.model.env, "auxiliary", None), "metrics", {}
                        ),
                        chunks={
                            k: b.chunk_ticks
                            for k, b in getattr(
                                getattr(self.model.env, "auxiliary", None), "gpu", {}
                            ).items()
                        },
                    )
                ),
                flush=True,
            )

        def _on_step(self):
            self.log_progress()
            return True

    cfg = deepcopy(base)
    cfg["training"]["random_cohort_seeds"] = False  # Seed-matched performance panels.
    count = settings["n_envs"]
    cfg["training"].update(n_envs=count, total_games=count * (args.repetitions + 2))
    cfg["training"]["peashooter_rules"].update(
        {
            k: v
            for k, v in settings.items()
            if k in ("oracle_lanes", "oracle_vram_mib", "oracle_chunk_ticks", "prepare_workers")
        }
    )
    from pvz_rl.learning.performance import PERFORMANCE_SETTINGS

    cfg["training"]["performance"].update(
        {k: v for k, v in settings.items() if k in PERFORMANCE_SETTINGS}
    )
    mode = settings.get("mode", "hybrid")
    cfg["training"]["performance"]["auxiliary_hybrid"] = mode in ("cuda", "hybrid", "deferred")
    cfg["training"]["performance"]["interval_backend"] = (
        "packed_async" if mode in ("interval", "interval-deferred") else "legacy"
    )
    if mode in ("cuda", "deferred"):
        cfg["training"]["performance"]["auxiliary_cpu_workers"] = 0
    # The production cohort-size resolver is used when present.
    cfg["training"]["peashooter_rules"]["n_envs"] = count
    from pvz_rl.learning.peashooter_rules import validate_rule_settings

    validate_rule_settings(cfg)
    torch.manual_seed(args.seed)
    if args.automatic_layout:
        cfg["curriculum"]["run_stage"] = None
    try:
        env = vector_env(cfg, "masked", args.seed)
    finally:
        cfg["curriculum"]["run_stage"] = "peashooter"
    model = build_model(cfg, "masked", env, args.seed)
    if args.weights:
        weights, _ = initial_weights(args.weights, cfg)
        model.policy.load_state_dict(weights)
    frozen = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    model.tile_skill_state = initial_tile_state(cfg)
    model.trajectory_root = workspace
    model.set_logger(configure(folder=str(workspace), format_strings=[]))
    callback = Callback()
    callback.init_callback(model)
    samples = []
    try:
        for repeat in range(args.repetitions + args.warmup):
            model.tile_skill_state = initial_tile_state(cfg)
            model.policy.load_state_dict(frozen)
            model.policy.optimizer.state.clear()
            if getattr(model, "_teacher", None) is not None:
                model._teacher.load_state_dict(model.policy.runtime_state_dict())
            for slot, rng in enumerate(env.queue.rngs):
                rng.bit_generator.state = np.random.default_rng(
                    args.seed * 1000 + slot % args.case_panel
                ).bit_generator.state
            env._reset_seeds()
            env._episode_serial[:] = [0] * env.num_envs
            for slot in range(env.num_envs):
                env.action_journal.reset(slot, -1)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            model._begin(callback)
            setup_done = perf_counter()
            if getattr(env, "auxiliary", None) is not None:
                env.auxiliary.defer_until_seal = mode in ("deferred", "interval-deferred")
            trace = None
            if args.process_trace and repeat >= args.warmup:
                from draw_collection_process import CollectionTrace

                trace = CollectionTrace(model)
            decision_seconds, spawn_seconds = [], []
            while model.phase == CohortPhase.COLLECT:
                if model._prepare_collection(callback):
                    decision_start = perf_counter()
                    model._collect_step(callback)
                    duration = perf_counter() - decision_start
                    decision_seconds.append(duration)
                    if env._last_spawned.any():
                        spawn_seconds.append(duration)
            torch.cuda.synchronize()
            collected = perf_counter()
            teacher = model._peashooter_teacher
            labels = sorted((k, v) for k, v in teacher.labels.items())
            outcomes = [p["metrics"]["win"] for p in model._pending_episodes.values()]
            if len(outcomes) != count:
                raise RuntimeError(
                    "Benchmark completed a different number of real games than configured"
                )
            episode_metrics = [p["metrics"] for p in model._pending_episodes.values()]
            metrics = deepcopy(teacher.metrics)
            from pvz_rl.envs.cuda_snapshots import download_snapshots

            physical_signature = digest(download_snapshots(env.batch, indices=range(env.num_envs)))
            target_hash = hashlib.sha256()
            for index in range(model._buffer.size):
                row = model._buffer._row_view(index)
                for field in (
                    "action",
                    "accepted",
                    "tick",
                    "duration",
                    "executed_action",
                    "effect_valid",
                ):
                    target_hash.update(row[field].tobytes())
                target_hash.update(row["effect_labels"][row["effect_valid"]].tobytes())
            auxiliary = deepcopy(getattr(getattr(env, "auxiliary", None), "metrics", {}))
            intervals = deepcopy(getattr(getattr(model, "_wait_outcomes", None), "metrics", {}))
            if trace is not None:
                trace.finish(
                    args.output.with_suffix(
                        f".{mode}-{settings.get('auxiliary_cpu_workers', 0)}-{repeat}.process.json"
                    ),
                    settings=settings,
                    auxiliary=auxiliary,
                    games=count,
                    ended=collected,
                    setup_seconds=trace.origin - started,
                    workload=dict(
                        cutoff_seconds=args.seconds,
                        initial_weights=str(args.weights) if args.weights else "random",
                        private_pretraining="bypassed",
                        live_view=False,
                        seed=args.seed,
                    ),
                )
            print(
                json.dumps(
                    dict(
                        event="collected",
                        repeat=repeat,
                        seconds=collected - started,
                        oracle=metrics,
                    )
                ),
                flush=True,
            )
            fitting_started = perf_counter()
            fit_passes = []
            if not args.collect_only:
                model._finalize_rewards(callback)
                model._buffer.finalize(cohort=model.cohort_number, diagnostics=False)
                model._phase(CohortPhase.FIT, callback)
                metrics["scratch_retained_mib"] = teacher.metrics.get("scratch_retained_mib")
                while model.phase == CohortPhase.FIT:
                    epoch = model._fit_epoch
                    model._fit_step(callback)
                    if model._fit_epoch != epoch:
                        fit_passes.append(
                            dict(
                                pass_number=epoch + 1,
                                end_seconds=perf_counter() - fitting_started,
                                total_loss=model._stats.get("total_loss"),
                                primary_loss=model._stats.get("primary_loss"),
                            )
                        )
                    callback.log_progress()
            torch.cuda.synchronize()
            finished = perf_counter()
            fitting_seconds = 0.0 if args.collect_only else finished - fitting_started
            total_seconds = collected - started + fitting_seconds
            samples.append(
                dict(
                    warmup=repeat < args.warmup,
                    games=len(outcomes),
                    wins=sum(outcomes),
                    setup_seconds=setup_done - started,
                    auxiliary=auxiliary,
                    intervals=intervals,
                    decision_latency_seconds=dict(
                        zip(
                            ("p50", "p95", "p99"),
                            np.percentile(decision_seconds, [50, 95, 99]).tolist(),
                            strict=True,
                        )
                    ),
                    spawn_latency_seconds=dict(
                        zip(
                            ("p50", "p95", "p99"),
                            np.percentile(spawn_seconds, [50, 95, 99]).tolist(),
                            strict=True,
                        )
                    )
                    if spawn_seconds
                    else None,
                    timeouts=sum(m["status"] == "truncated" for m in episode_metrics),
                    mean_simulated_seconds=mean(m["simulated_seconds"] for m in episode_metrics),
                    collection_seconds=collected - started,
                    post_collection_audit_seconds=fitting_started - collected,
                    fitting_seconds=fitting_seconds,
                    fitting_passes=fit_passes,
                    total_seconds=total_seconds,
                    seconds_per_env=total_seconds / len(outcomes),
                    collect_seconds_per_env=(collected - started) / len(outcomes),
                    signature=digest(
                        dict(
                            labels=labels,
                            outcomes=outcomes,
                            targets=target_hash.hexdigest(),
                            physical=physical_signature,
                        )
                    ),
                    oracle=metrics,
                    optimizer_commits=model.optimizer_commits,
                    simulator_capacity=env.num_envs,
                    torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                    cupy_reserved_mib=env.cp.get_default_memory_pool().total_bytes() / 2**20,
                    execution=deepcopy(model.execution_state),
                )
            )
            print(json.dumps(dict(settings=settings, **samples[-1])), flush=True)
            # Preserve completed measurements even if a later candidate is
            # interrupted; partial cohorts never enter this journal.
            with args.output.with_suffix(".samples.jsonl").open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(dict(settings=settings, **samples[-1])) + "\n")
            model._close_batches()
            if model._fitting_spool is not None:
                model._fitting_spool.close()
                model._fitting_spool = None
            model._peashooter_objective = None
            close_interval_owners(model)
            teacher.close()
            model._peashooter_teacher = None
            model._buffer.close()
            model._buffer = None
            model.phase = CohortPhase.IDLE
        measured = samples[args.warmup :]
        times = [s["seconds_per_env"] for s in measured]
        if len({s["signature"] for s in samples}) != 1:
            raise RuntimeError("Frozen seeded workload changed between repeated cohorts")
        return dict(
            **settings,
            samples=samples,
            mean_seconds_per_env=mean(times),
            stdev_seconds_per_env=stdev(times) if len(times) > 1 else 0,
            mean_collect_seconds_per_env=mean(s["collect_seconds_per_env"] for s in measured),
        )
    finally:
        model._close_batches()
        # Parameter sweeps create multiple models in one process. Dynamo can
        # retain their code objects after Python owners are dropped; explicitly
        # release graph buffers before admitting the next configuration.
        torch.cuda.synchronize()
        for network in (model.policy, getattr(model, "_teacher", None)):
            if network is not None:
                for encoder in network.encoders:
                    encoder.release_fitting()
                    encoder.release_collection()
        close_interval_owners(model)
        if getattr(model, "_peashooter_teacher", None):
            model._peashooter_teacher.close()
        if model._buffer:
            model._buffer.close()
        env.close()
        del frozen, callback, model, env
        gc.collect()
        torch.cuda.empty_cache()
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, help="JSON list of parameter combinations")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--process-trace",
        action="store_true",
        help="Record CPU/GPU process-state timings for drawing",
    )
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--collect-only", action="store_true", help="Include all labeling/drain; skip fitting"
    )
    parser.add_argument(
        "--automatic-layout",
        action="store_true",
        help="Include inactive capacity retained by the automatic stage pipeline",
    )
    parser.add_argument("--gpu-profile", default="auto")
    parser.add_argument(
        "--case-panel",
        type=int,
        default=128,
        help="Repeat this fixed seed panel across larger cohorts",
    )
    parser.add_argument("--envs", type=int, nargs="+")
    parser.add_argument("--lanes", type=int, nargs="+")
    parser.add_argument("--vram-mib", type=int, nargs="+")
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--chunks", type=int, nargs="+")
    parser.add_argument("--auxiliary-workers", type=int, nargs="+")
    parser.add_argument("--auxiliary-ms", type=float, nargs="+")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("sync", "cuda", "hybrid", "deferred", "interval", "interval-deferred"),
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0 or args.seconds < 1 or args.case_panel < 1:
        parser.error("Use positive repetitions, cutoff seconds and case-panel size")
    if args.profile:
        faulthandler.dump_traceback_later(120, repeat=True)
    import psutil
    import torch

    from pvz_rl.config import digest, load_config
    from pvz_rl.provenance import file_hash

    for p in psutil.process_iter(["cmdline"]):
        command = p.info["cmdline"] or []
        if "pvz_rl" in command and "train" in command:
            raise RuntimeError("A training job is running; benchmark after it finishes")
    cfg = load_config()
    from pvz_rl.gpu import apply_gpu_profile, select_gpu

    device = select_gpu("0")
    cfg, profile_name = apply_gpu_profile(cfg, args.gpu_profile, device=device)
    cfg["curriculum"]["run_stage"] = "peashooter"
    cfg["environment"]["cutoff_seconds"] = args.seconds
    cfg["visualization"].update(enabled=False, live_enabled=False, demos=False, videos=False)
    if args.eager:
        cfg["training"]["performance"].update(compile_collection=False, compile_fitting=False)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    rules = cfg["training"]["peashooter_rules"]
    from pvz_rl.config import cohort_envs

    keys = ("n_envs", "oracle_lanes", "oracle_vram_mib", "oracle_chunk_ticks", "prepare_workers")
    axes = (
        args.envs or [cohort_envs(cfg, "peashooter")],
        args.lanes or [rules["oracle_lanes"]],
        args.vram_mib or [rules["oracle_vram_mib"]],
        args.chunks or [rules["oracle_chunk_ticks"]],
        args.workers or [rules.get("prepare_workers", 0)],
    )
    settings = (
        json.loads(args.settings.read_text())
        if args.settings
        else [dict(zip(keys, row, strict=True)) for row in product(*axes)]
    )
    if not args.settings and (args.auxiliary_workers or args.auxiliary_ms or args.modes):
        settings = [
            dict(
                base,
                mode=mode,
                auxiliary_cpu_workers=workers,
                auxiliary_cuda_target_ms=milliseconds,
            )
            for base in settings
            for mode, workers, milliseconds in product(
                args.modes or ["hybrid"],
                args.auxiliary_workers
                or [cfg["training"]["performance"].get("auxiliary_cpu_workers", 8)],
                args.auxiliary_ms
                or [cfg["training"]["performance"].get("auxiliary_cuda_target_ms", 2.0)],
            )
        ]
    result = dict(
        protocol="peashooter_full_cohort_benchmark_v1",
        optimizer_passes=0 if args.collect_only else cfg["training"]["n_epochs"],
        implementation="hybrid_exact_simulation_v1",
        gpu_profile=profile_name,
        automatic_layout=args.automatic_layout,
        config_digest=digest(cfg),
        weight_sha256=file_hash(args.weights) if args.weights else None,
        performance=deepcopy(cfg["training"]["performance"]),
        case_panel=args.case_panel,
        formal_training=False,
        weights=str(args.weights),
        device=torch.cuda.get_device_name(),
        seconds=args.seconds,
        seed=args.seed,
        measurements=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".samples.jsonl").write_text("", encoding="utf-8")
    profile = cProfile.Profile()
    if args.profile:
        profile.enable()
    with TemporaryDirectory(prefix="peashooter-benchmark-") as directory:
        for setting in settings:
            try:
                row = measure(cfg, setting, args, Path(directory))
            except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                row = dict(**setting, infeasible=str(exc))
            result["measurements"].append(row)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.profile:
        profile.disable()
        profile.dump_stats(str(args.output.with_suffix(".prof")))
    feasible = [r for r in result["measurements"] if "samples" in r]
    for count in {row["n_envs"] for row in feasible}:
        if (
            len(
                {
                    s["signature"]
                    for row in feasible
                    if row["n_envs"] == count
                    for s in row["samples"]
                }
            )
            > 1
        ):
            raise RuntimeError(
                "Seed-matched backends produced different trajectories or exact labels"
            )
    result["fastest"] = min(feasible, key=lambda r: r["mean_seconds_per_env"], default=None)
    reference = {
        row["n_envs"]: mean(
            s["decision_latency_seconds"]["p95"] for s in row["samples"] if not s["warmup"]
        )
        for row in feasible
        if row.get("mode") in ("deferred", "interval-deferred")
    }
    for row in feasible:
        p95 = mean(s["decision_latency_seconds"]["p95"] for s in row["samples"] if not s["warmup"])
        row["main_p95_ratio"] = (
            p95 / reference[row["n_envs"]] if row["n_envs"] in reference else None
        )
    qualified = [
        row
        for row in feasible
        if row.get("mode") in ("cuda", "hybrid", "interval")
        and row["main_p95_ratio"] is not None
        and row["main_p95_ratio"] <= 1.05
    ]
    result["winner"] = min(
        qualified or feasible, key=lambda r: r["mean_seconds_per_env"], default=None
    )
    result["latency_target"] = (
        ("met" if qualified else "unmet") if reference else "unmeasured: include deferred mode"
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
