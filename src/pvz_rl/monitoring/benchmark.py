"""Measure warmed-up Q collection/update speed without changing the strategy."""

import gc
import hashlib
from time import perf_counter

import torch
from stable_baselines3.common.logger import configure

from pvz_rl.learning.training import build_model, vector_env
from pvz_rl.learning.training_requirements import require_cuda_training
from pvz_rl.monitoring.timing import TimingCallback


def policy_digest(model):
    digest = hashlib.sha256()
    for name, value in model.policy.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def measure(cfg, seed, steps, output, *, deadline=None, load_monitor=None, live_view=False):
    """One warmup window, then timed learning; setup and warmup are separate."""
    require_cuda_training(cfg)
    # SB3 callbacks retain cyclic references to models. Reclaim the previous
    # measurement before reporting this run's CUDA memory, then warm up afresh.
    gc.collect()
    if cfg["training"]["device"] == "cuda":
        torch.cuda.empty_cache()
    started = perf_counter()
    env = vector_env(cfg, "masked", seed)
    try:
        if cfg.get("simulation", {}).get("benchmark_shared_mix"):
            from pvz_rl.learning.budget import budget_target
            from pvz_rl.learning.curriculum import STAGES, teaching_enabled

            env.env_method("set_progress", budget_target(cfg) or 0)
            if teaching_enabled(cfg):
                env.env_method("set_curriculum_stage", len(STAGES) - 1)
        model = build_model(cfg, "masked", env, seed)
        if live_view:
            env.start_live_view()
        model.set_logger(configure(str(output), []))
        setup_seconds = perf_counter() - started
        started = perf_counter()
        model.learn(1)
        cuda = model.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(model.device)
        warmup_seconds = perf_counter() - started
        viewer_process, viewer_cpu_start = None, None
        if env.live_view is not None and env.live_view.session.enabled:
            import psutil

            viewer_process = psutil.Process(env.live_view.session.process.pid)
            cpu = viewer_process.cpu_times()
            viewer_cpu_start = cpu.user + cpu.system
        initial_steps = model.num_timesteps
        if hasattr(env, "features"):
            env.features.profiler.flush()
            env.features.profiler.seconds.clear()
            env.phases = dict.fromkeys(env.phases, 0.0)
        if env.live_view is not None:
            env.live_view.stats = dict.fromkeys(env.live_view.stats, 0)
            env.live_view.session.stats = dict.fromkeys(env.live_view.session.stats, 0)
        if cuda:
            torch.cuda.reset_peak_memory_stats(model.device)
        timing = TimingCallback(deadline=deadline, load_monitor=load_monitor)
        started = perf_counter()
        from pvz_rl.learning.training import TrainingDeadline

        stopped = False
        try:
            model.learn(steps, callback=timing, reset_num_timesteps=False)
        except TrainingDeadline:
            stopped = True
        if cuda:
            torch.cuda.synchronize(model.device)
        seconds = perf_counter() - started
        collected = model.num_timesteps - initial_steps
        viewer_resources = {}
        if viewer_process is not None:
            try:
                cpu = viewer_process.cpu_times()
                viewer_resources = {
                    "cpu_percent": (cpu.user + cpu.system - viewer_cpu_start) / seconds * 100,
                    "ram_mib": viewer_process.memory_info().rss / 2**20,
                }
            except psutil.NoSuchProcess:
                viewer_resources = {"error": "viewer exited during measurement"}
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
            "cohort": model.cohort_metrics,
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
            "live_view": {**env.live_view.stats, **env.live_view.session.stats, **viewer_resources}
            if env.live_view is not None
            else None,
            "state": "deadline" if stopped else "complete",
        }
    finally:
        env.close()
