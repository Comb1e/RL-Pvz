"""Bounded factual collection benchmark; no fitting, validation or formal training."""

import argparse
import gc
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter


def measure(base, cap, vram, chunk, repetitions, workspace):
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.tile_curriculum import initial_tile_state
    from pvz_rl.learning.training import build_model, vector_env

    class Callback(BaseCallback):
        def _on_step(self):
            return True

    cfg = deepcopy(base)
    # Keep the historical Peashooter workload for reproducible replay benchmarks.
    cfg["training"].pop("peashooter_rules", None)
    cfg["training"]["counterfactual"]["peashooter"] = deepcopy(cfg["training"]["counterfactual"]["normal"])
    cfg["training"]["counterfactual"].update(
        resident_envs=cap, scratch_vram_mib=vram, chunk_decisions=chunk
    )
    env = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", env, 101)
    model.tile_skill_state = initial_tile_state(cfg)
    if cfg["curriculum"].get("run_stage") is None:
        from pvz_rl.learning.tile_curriculum import TileSkillState

        model.tile_skill_state = TileSkillState(
            phase="peashooter", stage_order=["peashooter"]
        ).to_dict()
    model.trajectory_root = workspace
    model.set_logger(configure(folder=str(workspace), format_strings=[]))
    callback = Callback()
    callback.init_callback(model)
    samples = []
    try:
        for repeat in range(repetitions + 1):
            # Every capacity sees identical main RNG and oracle identities.
            for slot, rng in enumerate(env.queue.rngs):
                rng.bit_generator.state = np.random.default_rng(
                    101000 + repeat * 100003 + slot
                ).bit_generator.state
            torch.manual_seed(101 + repeat)
            torch.cuda.manual_seed_all(101 + repeat)
            env._episode_serial[:] = [repeat] * env.num_envs
            torch.cuda.synchronize()
            started = perf_counter()
            model._begin(callback)
            # Public reachable-style resource control to exercise immediate forks,
            # then ordinary cadence, costs, cooldowns and main inference throughout.
            env.batch.header[:, 2] = 200
            env.batch.cooldowns[:] = 0
            env.features.encode()
            model._last_obs = env.features.obs_tensor
            while model.phase == CohortPhase.COLLECT:
                if model._prepare_collection(callback):
                    model._collect_step(callback)
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            outcomes = [bool(p["metrics"]["win"]) for p in model._pending_episodes.values()]
            from pvz_rl.config import digest

            signature = digest(
                dict(
                    logs=model._counterfactual.logs,
                    alternatives=[
                        (r["identity"], [model._counterfactual.jobs[j]["tile"] for j in r["jobs"]])
                        for r in model._counterfactual.roots
                    ],
                    outcomes=outcomes,
                )
            )
            samples.append(
                dict(
                    warmup=repeat == 0,
                    seconds=elapsed,
                    seconds_per_env=elapsed / len(outcomes),
                    signature=signature,
                    **model._counterfactual.metrics,
                )
            )
            model._counterfactual.close()
            model._counterfactual = None
            model._buffer.close()
            model._buffer = None
            model.phase = CohortPhase.IDLE
        return dict(
            cap=cap,
            physical_main_envs=env.num_envs,
            vram_mib=vram,
            chunk=chunk,
            samples=samples,
            mean_seconds_per_env=mean(r["seconds_per_env"] for r in samples[1:]),
        )
    finally:
        if getattr(model, "_counterfactual", None):
            model._counterfactual.close()
        if model._buffer:
            model._buffer.close()
        env.close()
        del callback, model, env
        gc.collect()
        torch.cuda.empty_cache()
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--caps", type=int, nargs="+", default=[256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    )
    parser.add_argument("--vram", type=int, nargs="+", default=[2048])
    parser.add_argument("--chunks", type=int, nargs="+", default=[16])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--automatic-layout", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.seconds < 1 or any(
        value < 1 for values in (args.caps, args.vram, args.chunks) for value in values
    ):
        parser.error("Use positive capacities, VRAM, chunks, seconds and repetitions")
    import psutil
    import torch

    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if "pvz_rl" in command and "train" in command:
            raise RuntimeError(
                "A training job is running; wait for it to finish before benchmarking"
            )
    from pvz_rl.config import digest, load_config

    cfg = load_config()
    cfg["curriculum"]["run_stage"] = None if args.automatic_layout else "peashooter"
    if args.automatic_layout:
        # Retain the actual 640-lane automatic allocation; its attacker exercises
        # are not collected in this bounded Peashooter performance control.
        cfg["training"]["tile_scenarios"].pop("attacker", None)
    cfg["training"].update(
        n_envs=128, total_games=128 * (args.repetitions + 2), until_stage_complete=False
    )
    cfg["training"]["performance"].update(compile_collection=False, compile_fitting=False)
    cfg["environment"]["cutoff_seconds"] = args.seconds
    cfg["visualization"].update(enabled=False, live_enabled=False, demos=False, videos=False)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        protocol="factual_collection_benchmark_v1",
        main_envs=128,
        seconds=args.seconds,
        automatic_layout=args.automatic_layout,
        formal_training=False,
        optimizer_steps=0,
        config_digest=digest(cfg),
        device=torch.cuda.get_device_name(),
        repetitions=args.repetitions,
        measurements=[],
    )
    with TemporaryDirectory(prefix="factual-benchmark-") as directory:
        for cap in args.caps:
            for vram in args.vram:
                for chunk in args.chunks:
                    try:
                        row = measure(cfg, cap, vram, chunk, args.repetitions, Path(directory))
                    except MemoryError as exc:
                        row = dict(cap=cap, vram_mib=vram, chunk=chunk, infeasible=str(exc))
                    result["measurements"].append(row)
                    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "cap", "vram_mib", "chunk", "mean_seconds_per_env", "infeasible"
                                )
                                if k in row
                            }
                        ),
                        flush=True,
                    )
    feasible = [r for r in result["measurements"] if "samples" in r]
    signatures = [[s["signature"] for s in r["samples"]] for r in feasible]
    if signatures and any(s != signatures[0] for s in signatures):
        raise RuntimeError("Scratch capacity changed main commands or alternative identities")
    result["winner"] = min(
        feasible, key=lambda r: (r["mean_seconds_per_env"], r["vram_mib"], r["cap"]), default=None
    )
    result["capacity_invariance"] = True
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
