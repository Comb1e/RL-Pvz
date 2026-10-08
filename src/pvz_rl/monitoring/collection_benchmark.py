"""Warmed, collection-only snapshot controls; never starts fitting or training."""

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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from pvz_rl.config import load_config
from pvz_rl.envs.probes import CounterfactualCollector
from pvz_rl.learning.training import build_model, vector_env
from pvz_rl.monitoring.hardware import HardwareMonitor
from pvz_rl.provenance import write_json

CASES = ("sparse", "mixed", "crowded", "rejection-heavy", "accepted-action")


class CollectionCallback(BaseCallback):
    def _on_step(self):
        return True


def snapshot(case):
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


def reset(model, callback, raw, seed):
    if model._buffer is not None:
        model._buffer.close()
    torch.manual_seed(seed)
    model._begin(callback)
    env = model.env
    with env.device_context():
        env.batch.restore([raw] * env.num_envs)
        env._refresh_growth_bounds(range(env.num_envs))
        env.features.initialize_home(range(env.num_envs))
        env.features.encode()
        model._last_obs = env.features.obs_tensor
    torch.cuda.synchronize()
    env.profiler.flush()
    env.profiler.seconds.clear()


def benchmark(output, decisions=32):
    cfg = load_config()
    cfg["visualization"].update(enabled=False, live_enabled=False)
    cfg["training"]["until_stage_complete"] = False
    cfg["training"]["total_games"] = 10000
    torch.set_num_threads(cfg["training"]["torch_threads"])
    result = dict(
        measured_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        formal_training=False,
        optimizer_steps=0,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        environments=cfg["training"]["n_envs"],
        decisions_per_repeat=decisions,
        warmup_decisions=8,
        repetitions=3,
        config=copy.deepcopy(cfg),
        measurements=[],
        comparison="two-lane versus the current one-lane memory fallback, not a historical trainer",
    )
    process = psutil.Process()
    if output.exists():
        saved = json.loads(output.read_text("utf-8"))
        if saved["config"] != result["config"] or saved["decisions_per_repeat"] != decisions:
            raise ValueError("Existing benchmark has different settings; use a new output path")
        result["measurements"] = saved["measurements"]
    with HardwareMonitor(output.with_suffix(".hardware.jsonl"), device="cuda") as hardware:
        for case in CASES:
            raw = snapshot(case)
            for lanes in (1, 2):
                completed = {
                    row["repeat"]
                    for row in result["measurements"]
                    if row["case"] == case and row["scratch_lanes"] == lanes
                }
                if completed == {0, 1, 2}:
                    continue
                env = vector_env(cfg, "masked", 101)
                model = build_model(cfg, "masked", env, 101)
                model.trajectory_root = output.parent
                model.set_logger(configure(folder=None, format_strings=[]))
                callback = CollectionCallback()
                callback.init_callback(model)
                try:
                    model._probes = CounterfactualCollector(env)
                    if lanes == 1:
                        model._probes.lanes = 1
                        model._probes._allocate()
                        model._probes.features.profiler = env.profiler
                    branch = (
                        9 if case == "accepted-action" else 1 if case == "rejection-heavy" else 0
                    )
                    with torch.no_grad():
                        model.policy.branch_head[-1].weight.zero_()
                        model.policy.branch_head[-1].bias.zero_()
                        model.policy.branch_head[-1].bias[branch] = 1
                    reset(model, callback, raw, 101)
                    for _ in range(8):
                        model._collect_step(callback)
                    for repeat in range(3):
                        if repeat in completed:
                            continue
                        reset(model, callback, raw, 101)
                        hardware.update(phase=case, activity=f"{lanes}-lane", repeat=repeat)
                        torch.cuda.reset_peak_memory_stats()
                        cpu_before = process.cpu_times()
                        started = perf_counter()
                        for _ in range(decisions):
                            model._collect_step(callback)
                        elapsed = perf_counter() - started
                        cpu_after = process.cpu_times()
                        phases = env.profiler.flush()
                        samples = [
                            sample
                            for sample in hardware.samples
                            if sample.get("phase") == case
                            and sample.get("activity") == f"{lanes}-lane"
                            and sample.get("repeat") == repeat
                        ]
                        means = {
                            key: float(
                                np.mean(
                                    [
                                        sample[key]
                                        for sample in samples
                                        if sample.get(key) is not None
                                    ]
                                )
                            )
                            if any(sample.get(key) is not None for sample in samples)
                            else None
                            for key in ("gpu_percent", "system_cpu_percent", "process_cpu_percent")
                        }
                        import cupy

                        transitions = decisions * env.num_envs
                        row = dict(
                            case=case,
                            scratch_lanes=lanes,
                            repeat=repeat,
                            seconds=elapsed,
                            transitions=transitions,
                            transitions_per_second=transitions / elapsed,
                            phases_seconds=phases,
                            utilization=means,
                            process_cpu_percent=100
                            * (
                                cpu_after.user
                                + cpu_after.system
                                - cpu_before.user
                                - cpu_before.system
                            )
                            / elapsed,
                            host_rss_mib=process.memory_info().rss / 2**20,
                            pinned_transfer_mib=env.handoff.nbytes / 2**20,
                            torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                            cupy_reserved_mib=cupy.get_default_memory_pool().total_bytes() / 2**20,
                            trajectory_bytes=model._buffer.entity_size * 44
                            + model._buffer.size * model._buffer.dtype.itemsize,
                        )
                        result["measurements"].append(row)
                        write_json(output, result)
                        print(
                            f"{case} {lanes}-lane repeat {repeat + 1}: {row['transitions_per_second']:.0f} transitions/s",
                            flush=True,
                        )
                finally:
                    if model._buffer is not None:
                        model._buffer.close()
                    env.close()
                    del model, env
                    torch.cuda.empty_cache()
                    torch._dynamo.reset()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=32)
    args = parser.parse_args()
    if args.decisions < 1 or args.decisions > 40:
        parser.error("Use 1..40 decisions to retain the accepted-action snapshot control")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark(args.output, args.decisions)


if __name__ == "__main__":
    main()
