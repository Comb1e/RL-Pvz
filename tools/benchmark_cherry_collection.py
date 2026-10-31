"""Compare total collection seconds per env; no fitting or training checkpoints."""

import argparse
import gc
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


def measure(cfg, count, workers, oracle, repetitions, warmups, workspace, *, cuda_warmup=None):
    # Keep Torch out of the module scope: Windows CPU workers import __mp_main__.
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.tile_curriculum import initial_tile_state
    from pvz_rl.learning.training import build_model, vector_env

    class Callback(BaseCallback):
        def _on_step(self):
            return True

    cfg = deepcopy(cfg)
    cfg["training"]["tile_scenarios"].update(enabled=True, cherry_bomb_n_envs=count)
    cfg["training"]["search"]["enabled"] = False
    cfg["training"]["performance"].update(
        cherry_prepare_workers=workers,
        cherry_oracle=oracle,
        cherry_prefetch=False,
    )
    if cuda_warmup is not None:
        cfg["training"]["performance"]["cherry_cuda_warmup"] = cuda_warmup
    warmup_backend = (
        "cuda"
        if oracle == "cuda" and cfg["training"]["performance"].get("cherry_cuda_warmup", False)
        else "cpu"
    )
    cfg["training"].update(
        total_games=count * (repetitions + warmups + 1), until_stage_complete=False
    )
    cfg["curriculum"]["run_stage"] = "cherry_bomb"
    cfg["visualization"].update(enabled=False, live_enabled=False, demos=False, videos=False)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    env = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", env, 101)
    model.trajectory_root = workspace
    model.tile_skill_state = initial_tile_state(cfg)
    model.set_logger(configure(folder=str(workspace), format_strings=[]))
    callback = Callback()
    callback.init_callback(model)
    rows = []
    initial = deepcopy([r.bit_generator.state for r in env.queue.rngs])
    try:
        for rep in range(warmups + repetitions):
            for rng, state in zip(env.queue.rngs, initial, strict=True):
                rng.bit_generator.state = deepcopy(state)
            torch.cuda.synchronize()
            started = perf_counter()
            model._begin(callback)
            torch.cuda.synchronize()
            prepared = perf_counter()
            while model.phase == CohortPhase.COLLECT:
                model._collect_step(callback)
            torch.cuda.synchronize()
            ended = perf_counter()
            completed = len(model._pending_episodes)
            if completed != count:
                raise RuntimeError(f"Collected {completed}/{count} episodes")
            prep = env.queue._preparation
            rows.append(
                dict(
                    envs=count,
                    workers=prep.workers,
                    oracle=oracle,
                    scratch_lanes=prep.cuda.batch.n if prep.cuda else 0,
                    repetition=rep,
                    warmup=rep < warmups,
                    preparation_seconds=prepared - started,
                    play_seconds=ended - prepared,
                    collect_seconds=ended - started,
                    seconds_per_env=(ended - started) / completed,
                    device_used_mib=(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])
                    / 2**20,
                )
            )
            model._buffer.close()
            model._buffer = None
            model.phase = CohortPhase.IDLE
            model._n_updates += 1
        measured = rows[warmups:]
        return dict(
            envs=count,
            workers=workers,
            oracle=oracle,
            warmup_backend=warmup_backend,
            mean_seconds_per_env=sum(r["seconds_per_env"] for r in measured) / repetitions,
            samples=rows,
        )
    finally:
        if model._buffer is not None:
            model._buffer.close()
        env.close()
        model.logger.close()
        cp = env.cp
        del callback, model, env
        gc.collect()
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()


def main():
    from pvz_rl.config import load_config
    from pvz_rl.gpu import apply_gpu_profile, select_gpu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", type=int, default=[256, 512, 640, 1024])
    parser.add_argument("--workers", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--oracles", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--warmup-backends", nargs="+", choices=["cpu", "cuda"], default=["cuda"])
    parser.add_argument("--lanes", type=int)
    parser.add_argument("--vram-mib", type=int)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.envs, args.repetitions, args.warmups) < 1 or min(args.workers) < 0:
        parser.error("envs/repetitions/warmups must be positive; workers must be nonnegative")
    cfg, profile = apply_gpu_profile(load_config(), device=select_gpu())
    for arg, key in (
        (args.lanes, "cherry_oracle_lanes"),
        (args.vram_mib, "cherry_oracle_vram_mib"),
    ):
        if arg is not None:
            if arg < 1:
                parser.error("lanes and vram-mib must be positive")
            cfg["training"]["performance"][key] = arg
    report = dict(profile=profile, settings=cfg["training"]["performance"], combinations=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pvz-cherry-benchmark-") as workspace:
        for count in args.envs:
            for oracle in args.oracles:
                for workers in args.workers:
                    for backend in args.warmup_backends if oracle == "cuda" else ["cpu"]:
                        result = measure(
                            cfg,
                            count,
                            workers,
                            oracle,
                            args.repetitions,
                            args.warmups,
                            workspace,
                            cuda_warmup=backend == "cuda",
                        )
                        report["combinations"].append(result)
                        best = min(report["combinations"], key=lambda r: r["mean_seconds_per_env"])
                        report["fastest"] = {k: v for k, v in best.items() if k != "samples"}
                        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                        print(
                            json.dumps({k: v for k, v in result.items() if k != "samples"}),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
