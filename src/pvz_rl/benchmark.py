"""Measure warmed-up PPO collection/update speed without changing the strategy."""

import copy
import gc
import hashlib
import statistics
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3.common.logger import configure

from .config import output_settings, runtime_settings
from .progress import Phase, ProgressReporter
from .provenance import metadata, write_json
from .timing import TimingCallback
from .training import build_model, vector_env


def policy_digest(model):
    digest = hashlib.sha256()
    for name, value in model.policy.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def measure(cfg, seed, steps, output, *, deadline=None, load_monitor=None):
    """One warmup rollout, then timed learning; setup and warmup are separate."""
    # SB3 callbacks retain cyclic references to models. Reclaim the previous
    # measurement before reporting this run's CUDA memory, then warm up afresh.
    gc.collect()
    if cfg["training"]["device"] == "cuda":
        torch.cuda.empty_cache()
    started = perf_counter()
    env = vector_env(cfg, "masked", seed)
    try:
        if cfg.get("simulation", {}).get("benchmark_shared_mix"):
            from .budget import budget_target
            from .curriculum import teaching_enabled

            env.env_method("set_progress", budget_target(cfg))
            if teaching_enabled(cfg):
                env.env_method("set_curriculum_stage", 4)
        model = build_model(cfg, "masked", env, seed)
        model.set_logger(configure(str(output), []))
        setup_seconds = perf_counter() - started
        started = perf_counter()
        model.learn(cfg["training"]["rollout_size"])
        cuda = model.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(model.device)
        warmup_seconds = perf_counter() - started
        initial_steps = model.num_timesteps
        if hasattr(env, "features"):
            env.features.profiler.flush()
            env.features.profiler.seconds.clear()
            env.phases = dict.fromkeys(env.phases, 0.0)
        if cuda:
            torch.cuda.reset_peak_memory_stats(model.device)
        timing = TimingCallback(deadline=deadline, load_monitor=load_monitor)
        started = perf_counter()
        from .training import TrainingDeadline

        stopped = False
        try:
            model.learn(steps, callback=timing, reset_num_timesteps=False)
        except TrainingDeadline:
            stopped = True
            timing.timings.end_update()
        if cuda:
            torch.cuda.synchronize(model.device)
        seconds = perf_counter() - started
        collected = model.num_timesteps - initial_steps
        return {
            "steps": collected,
            "warmup_steps": initial_steps,
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "seconds": seconds,
            "decisions_per_second": collected / seconds,
            "games_per_minute": timing.games / seconds * 60,
            "simulation_ticks_per_second": timing.ticks / seconds,
            "completed_games": timing.games,
            "device_phase_seconds": env.features.profiler.flush()
            if hasattr(env, "features")
            else None,
            "host_phase_seconds": env.phases if hasattr(env, "phases") else None,
            **timing.timings.snapshot(),
            "cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(model.device) / 2**20 if cuda else None
            ),
            "cuda_peak_reserved_mib": (
                torch.cuda.max_memory_reserved(model.device) / 2**20 if cuda else None
            ),
            "policy_sha256": policy_digest(model),
            "state": "deadline" if stopped else "complete",
        }
    finally:
        env.close()


def benchmark(
    cfg,
    output,
    *,
    steps=16384,
    workers=(1, 4, 8),
    devices=("cpu", "cuda"),
    repeats=3,
    compare_runtime=False,
):
    if steps < 1 or repeats < 1 or not workers or not devices:
        raise ValueError("Benchmark needs positive steps, repeats, and worker/device choices")
    if any(n < 1 or cfg["training"]["rollout_size"] % n for n in workers):
        raise ValueError("Positive worker counts must divide rollout size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "metadata.json",
        metadata(
            cfg,
            kind="hardware-benchmark",
            requested_steps=steps,
            workers=list(workers),
            devices=list(devices),
            repeats=repeats,
            compare_runtime=compare_runtime,
        ),
    )
    records = []
    torch.set_num_threads(cfg["training"]["torch_threads"])
    variants = ["reference", "configured"] if compare_runtime else ["configured"]
    progress = ProgressReporter(
        output / "benchmark.log", output_settings(cfg)["logging"]["progress_seconds"]
    )
    try:
        progress.phase(Phase.COLLECTING, "Benchmark: one warmup rollout before each measurement")
        for device in devices:
            for n_envs in workers:
                if device == "cuda" and not torch.cuda.is_available():
                    records.append({"device": device, "n_envs": n_envs, "state": "unavailable"})
                    write_json(output / "measurements.json", records)
                    continue
                for repeat in range(repeats):
                    # Alternate order to reduce first-run and thermal bias.
                    for variant in variants[:: 1 if repeat % 2 == 0 else -1]:
                        local = copy.deepcopy(cfg)
                        local["training"].update(device=device, n_envs=n_envs)
                        if variant == "reference":
                            local["runtime"] = dict.fromkeys(runtime_settings(cfg), False)
                        progress.emit(
                            f"{device}, {n_envs} workers, {variant}, repeat {repeat + 1}/{repeats}",
                            force=True,
                        )
                        row = {
                            "device": device,
                            "n_envs": n_envs,
                            "variant": variant,
                            "repeat": repeat,
                            "runtime": runtime_settings(local),
                            **measure(local, 800 + repeat, steps, output),
                        }
                        records.append(row)
                        write_json(output / "measurements.json", records)
                        progress.emit(
                            f"{row['decisions_per_second']:.0f} decisions/s; "
                            f"collection {row['collection_seconds']:.2f}s; "
                            f"updates {row['optimization_seconds']:.2f}s",
                            force=True,
                        )
                for variant in variants:
                    group = [
                        r
                        for r in records
                        if r["state"] == "complete"
                        and (r["device"], r["n_envs"], r["variant"]) == (device, n_envs, variant)
                    ]
                    records.append(
                        {
                            "device": device,
                            "n_envs": n_envs,
                            "variant": variant,
                            "state": "aggregate",
                            "median_decisions_per_second": statistics.median(
                                r["decisions_per_second"] for r in group
                            ),
                        }
                    )
                write_json(output / "measurements.json", records)
        winners = [r for r in records if r["state"] == "aggregate" and r["variant"] == "configured"]
        if not winners:
            raise RuntimeError("None of the requested training devices is available")
        winner = max(winners, key=lambda r: r["median_decisions_per_second"])
        recommendation = {
            "device": winner["device"],
            "n_envs": winner["n_envs"],
            "benchmark_steps": steps,
            "repeats": repeats,
            "runtime": runtime_settings(cfg),
            "note": "Warmup/setup/validation/report time excluded. Freeze hardware before comparisons.",
        }
        write_json(output / "recommendation.json", recommendation)
        if compare_runtime:
            comparisons = []
            complete = [r for r in records if r["state"] == "complete"]
            for row in (r for r in complete if r["variant"] == "configured"):
                reference = next(
                    r
                    for r in complete
                    if r["variant"] == "reference"
                    and (r["device"], r["n_envs"], r["repeat"])
                    == (row["device"], row["n_envs"], row["repeat"])
                )
                comparisons.append(
                    {
                        "device": row["device"],
                        "n_envs": row["n_envs"],
                        "repeat": row["repeat"],
                        "speedup": row["decisions_per_second"] / reference["decisions_per_second"],
                        "identical_policy_weights": row["policy_sha256"]
                        == reference["policy_sha256"],
                    }
                )
            write_json(output / "runtime-comparison.json", comparisons)
        progress.phase(Phase.COMPLETE, "Benchmark measurements saved")
        return recommendation
    finally:
        progress.close()
