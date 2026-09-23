"""Bounded, repeated CUDA collection+optimization comparisons."""

import copy
import ctypes
import os
import statistics
import subprocess
import threading
from pathlib import Path
from time import perf_counter, process_time

import torch

from .benchmark import measure
from .config import gpu_defaults, resolve_rollout
from .progress import Phase, ProgressReporter
from .provenance import metadata, write_json


def system_cpu_times():
    if os.name != "nt":
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        values = list(map(int, fields))
        return values[3] + values[4], sum(values)
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None

    def number(t):
        return t.dwLowDateTime + (t.dwHighDateTime << 32)

    return number(idle), number(kernel) + number(user)


class LoadMonitor:
    """One low-frequency sampler; records system load without managing other jobs."""

    def __init__(self):
        self.samples, self.phase = [], "starting"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        previous = system_cpu_times()
        while not self.stop.is_set():
            row = {"seconds": perf_counter() - self.started, "phase": self.phase}
            current = system_cpu_times()
            if previous and current and current[1] > previous[1]:
                row["system_cpu_percent"] = 100 * (
                    1 - (current[0] - previous[0]) / (current[1] - previous[1])
                )
            previous = current
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=4,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if output.returncode == 0:
                    values = [float(x.strip()) for x in output.stdout.splitlines()[0].split(",")]
                    row.update(
                        zip(
                            ("gpu_percent", "gpu_memory_percent", "gpu_memory_mib", "gpu_watts"),
                            values,
                        )
                    )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row["error"] = repr(exc)
            self.samples.append(row)
            self.stop.wait(1.0)

    def __enter__(self):
        self.started = perf_counter()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


def recommend(rows, *, parity_passed=False, provisional=True):
    aggregates = {}
    variation = {}
    for name in {r["profile"] for r in rows if r["state"] == "complete"}:
        group = [r for r in rows if r["profile"] == name and r["state"] == "complete"]
        if len(group) == 3:
            rates = [r["decisions_per_second"] for r in group]
            aggregates[name] = statistics.median(rates)
            variation[name] = statistics.stdev(rates) / statistics.mean(rates)
    candidates = {
        int(name.removeprefix("cuda-")): rate
        for name, rate in aggregates.items()
        if name.startswith("cuda-")
        and name.removeprefix("cuda-").isdigit()
        and variation[name] <= 0.10
    }
    selected = min(
        (n for n, rate in candidates.items() if rate >= 0.95 * max(candidates.values())),
        default=None,
    )
    return {
        "n_envs": selected,
        "device": "cuda",
        "simulator": "cuda",
        "rollout_steps_per_env": 128,
        "median_decisions_per_second": aggregates,
        "coefficient_of_variation": variation,
        "stability_limit": 0.10,
        "parity_passed": parity_passed,
        "provisional": provisional,
        "promote_default": bool(parity_passed and not provisional and selected is not None),
        "note": "Complete collection+PPO updates; setup, warmup, validation and export reported separately. No promotion from utilization alone.",
    }


def benchmark_gpu(cfg, output, *, minutes=15, steps=16384, env_counts=None):
    from .training_requirements import require_cuda_training

    require_cuda_training(cfg)
    if not 0 < minutes <= 30 or steps < 1:
        raise ValueError("GPU benchmark must be bounded to 0 < minutes <= 30 and positive steps")
    defaults = gpu_defaults()
    counts = defaults["benchmark_env_counts"] if env_counts is None else env_counts
    if (
        not counts
        or any(type(n) is not int or n < 1 for n in counts)
        or len(set(counts)) != len(counts)
        or any(n * 128 % cfg["training"]["batch_size"] for n in counts)
    ):
        raise ValueError(
            "Benchmark env counts must be unique positive integers compatible with batch_size"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "metadata.json",
        metadata(
            cfg,
            kind="gpu-backend-benchmark",
            minutes=minutes,
            steps=steps,
            repeats=3,
            env_counts=list(counts),
            minimum_rollouts=defaults["benchmark_min_rollouts"],
        ),
    )
    rows = []
    progress = ProgressReporter(output / "benchmark.log", 15)
    deadline = perf_counter() + minutes * 60
    torch.set_num_threads(cfg["training"]["torch_threads"])
    profiles = [
        *((f"cuda-{n}", "cuda", n, 128) for n in counts),
    ]
    try:
        progress.phase(
            Phase.COLLECTING, "GPU benchmark: three repetitions, setup/warmup reported separately"
        )
        with LoadMonitor() as monitor:
            for repeat in range(3):
                for name, backend, n, rollout_steps in profiles[:: 1 if repeat % 2 == 0 else -1]:
                    if perf_counter() >= deadline:
                        break
                    local = copy.deepcopy(cfg)
                    local["simulation"] = {
                        "backend": backend,
                        "profile": True,
                        "benchmark_shared_mix": True,
                    }
                    local["training"].update(n_envs=n, device="cuda")
                    resolve_rollout(local, per_env=rollout_steps)
                    monitor.phase = f"{name}/repeat-{repeat + 1}"
                    progress.emit(monitor.phase, force=True)
                    cpu_start = process_time()
                    try:
                        result = measure(
                            local,
                            800 + repeat,
                            max(steps, n * rollout_steps * defaults["benchmark_min_rollouts"]),
                            output,
                            deadline=deadline,
                            load_monitor=monitor,
                        )
                    except (MemoryError, RuntimeError) as exc:
                        result = dict(
                            state="failed",
                            error=repr(exc),
                            decisions_per_second=0,
                            games_per_minute=0,
                        )
                    row = {
                        "profile": name,
                        "simulator": backend,
                        "n_envs": n,
                        "rollout_steps_per_env": rollout_steps,
                        "repeat": repeat,
                        "main_process_cpu_seconds": process_time() - cpu_start,
                        **result,
                    }
                    rows.append(row)
                    write_json(output / "measurements.json", rows)
                    progress.emit(
                        f"{name}: {row['decisions_per_second']:.0f} decisions/s, {row['games_per_minute']:.2f} games/min; {row['state']}",
                        force=True,
                    )
            write_json(output / "load-samples.json", monitor.samples)
        recommendation = recommend(rows)
        write_json(output / "recommendation.json", recommendation)
        progress.phase(
            Phase.COMPLETE,
            "Benchmark complete; promotion additionally requires documented correctness and uncontested load",
        )
        return recommendation
    finally:
        progress.close()
