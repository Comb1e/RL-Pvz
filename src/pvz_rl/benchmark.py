"""Measure complete PPO collection/update throughput on CPU and CUDA."""

import copy
import statistics
from pathlib import Path
from time import perf_counter

import torch

from .provenance import metadata, write_json
from .training import build_model, vector_env


def benchmark(cfg, output, *, steps=16384, workers=(1, 4, 8), devices=("cpu", "cuda"), repeats=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metadata.json", metadata(cfg, kind="hardware-benchmark"))
    records = []
    torch.set_num_threads(cfg["training"]["torch_threads"])
    for device in devices:
        for n_envs in workers:
            if device == "cuda" and not torch.cuda.is_available():
                records.append({"device": device, "n_envs": n_envs, "state": "unavailable"})
                continue
            if cfg["training"]["rollout_size"] % n_envs:
                raise ValueError("Worker count must divide rollout size")
            speeds = []
            for repeat in range(repeats):
                local = copy.deepcopy(cfg)
                local["training"].update(device=device, n_envs=n_envs)
                env = vector_env(local, "masked", 800 + repeat)
                try:
                    model = build_model(local, "masked", env, 800 + repeat)
                    started = perf_counter()
                    model.learn(steps)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    seconds = perf_counter() - started
                    speeds.append(model.num_timesteps / seconds)
                    records.append(
                        {
                            "device": device,
                            "n_envs": n_envs,
                            "repeat": repeat,
                            "steps": model.num_timesteps,
                            "seconds": seconds,
                            "decisions_per_second": speeds[-1],
                            "state": "complete",
                        }
                    )
                finally:
                    env.close()
            records.append(
                {
                    "device": device,
                    "n_envs": n_envs,
                    "state": "aggregate",
                    "median_decisions_per_second": statistics.median(speeds),
                }
            )
            write_json(output / "measurements.json", records)
    winners = [r for r in records if r["state"] == "aggregate"]
    winner = max(winners, key=lambda r: r["median_decisions_per_second"])
    recommendation = {
        "device": winner["device"],
        "n_envs": winner["n_envs"],
        "benchmark_steps": steps,
        "repeats": repeats,
        "note": "Freeze this choice before formal comparisons.",
    }
    write_json(output / "recommendation.json", recommendation)
    return recommendation
