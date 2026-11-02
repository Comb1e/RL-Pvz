"""Compare total collection seconds per env; no fitting or training checkpoints."""

import argparse
import gc
import json
import math
import random
import statistics
from copy import deepcopy
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


def _worker_ready():
    import os
    from time import sleep

    sleep(0.05)  # Let every spawned process take a task before timing collection.
    return os.getpid()


def measure(
    cfg,
    count,
    workers,
    oracle,
    repetitions,
    warmups,
    workspace,
    *,
    cuda_warmup=None,
    stage="cherry_bomb",
    lanes=None,
    vram_mib=None,
    seed=101,
    warmup_games=0,
):
    # Keep Torch out of the module scope: Windows CPU workers import __mp_main__.
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from pvz_rl.config import digest
    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.tile_curriculum import initial_tile_state
    from pvz_rl.learning.training import build_model, vector_env

    class Callback(BaseCallback):
        def _on_step(self):
            return True

    cfg = deepcopy(cfg)
    warmups = max(warmups, math.ceil(warmup_games / count))
    tile = cfg["training"]["tile_scenarios"]
    tile["enabled"] = True
    attacker = stage == "attacker_pre"
    if attacker:
        tile["attacker"]["n_envs"] = count
    else:
        tile["cherry_bomb_n_envs"] = count
    cfg["training"].pop("peashooter_rules", None)
    perf = cfg["training"]["performance"]
    prefix = "attacker" if attacker else "cherry"
    perf.update({f"{prefix}_prepare_workers": workers, f"{prefix}_prefetch": False})
    if not attacker:
        perf["cherry_oracle"] = oracle
    if cuda_warmup is not None:
        perf[f"{prefix}_cuda_warmup"] = cuda_warmup
    if lanes is not None:
        perf[f"{prefix}_oracle_lanes"] = lanes
    if vram_mib is not None:
        perf[f"{prefix}_oracle_vram_mib"] = vram_mib
    from pvz_rl.envs.tile_scenarios import preparation_settings

    resources = preparation_settings(cfg, stage)
    warmup_backend = (
        "cuda" if oracle in ("cuda", "distance") and resources.get("cuda_warmup", False) else "cpu"
    )
    cfg["training"].update(
        total_games=count * (repetitions + warmups + 1), until_stage_complete=False
    )
    cfg["curriculum"]["run_stage"] = stage
    cfg["visualization"].update(enabled=False, live_enabled=False, demos=False, videos=False)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    env = vector_env(cfg, "masked", seed)
    model = build_model(cfg, "masked", env, seed)
    model.trajectory_root = workspace
    model.tile_skill_state = initial_tile_state(cfg)
    model.set_logger(configure(folder=str(workspace), format_strings=[]))
    callback = Callback()
    callback.init_callback(model)
    rows = []
    try:
        startup = perf_counter()
        env.queue.tile_scenario = stage
        prep = env.queue._bomb_preparation()
        prep.submit([], skill=stage)
        ready = set()
        while prep.pool is not None and len(ready) < prep.workers:
            ready.update(
                f.result()
                for f in [prep.pool.submit(_worker_ready) for _ in range(2 * prep.workers)]
            )
            if perf_counter() - startup > 30:
                raise RuntimeError("Preparation workers did not become ready within 30 seconds")
        worker_startup_seconds = perf_counter() - startup
        if attacker:
            from pvz_rl.envs.attacker_scenarios import motion_bound

            motion_bound.cache_clear()
        for rep in range(warmups + repetitions):
            # Each repetition uses a new common panel, independent of config order.
            # Across env counts, the first N lanes receive exactly the same seeds.
            import numpy as np

            panel = rep if rep < warmups else 100000 + rep - warmups
            for index, rng in enumerate(env.queue.rngs):
                rng.bit_generator.state = np.random.default_rng(
                    (seed + panel * 100003) * 1000 + index
                ).bit_generator.state
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
            labels = [
                (
                    index,
                    getattr(board["snapshot"], "_captured_digest", None),
                    board.get("supplied_plant"),
                    board["feasible_tiles"],
                    board.get("fallback"),
                )
                for index, board in sorted(env.tile_boards.items())
            ]
            rows.append(
                dict(
                    envs=count,
                    workers=prep.workers,
                    oracle=oracle,
                    scratch_lanes=prep.cuda.batch.n if prep.cuda else 0,
                    repetition=rep,
                    panel=panel,
                    warmup=rep < warmups,
                    preparation_seconds=prepared - started,
                    play_seconds=ended - prepared,
                    collect_seconds=ended - started,
                    seconds_per_env=(ended - started) / completed,
                    device_used_mib=(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])
                    / 2**20,
                    generation_failures=env.queue.generation_failures,
                    panel_digest=digest(labels),
                    panel_prefix_digest=digest(labels[:64]),
                )
            )
            model._buffer.close()
            model._buffer = None
            model.phase = CohortPhase.IDLE
            model._n_updates += 1
        measured = rows[warmups:]
        values = [r["seconds_per_env"] for r in measured]
        return dict(
            stage=stage,
            envs=count,
            workers=workers,
            oracle=oracle,
            warmup_backend=warmup_backend,
            oracle_lanes=resources.get("oracle_lanes", 256),
            oracle_vram_mib=resources.get("oracle_vram_mib", 256),
            worker_startup_seconds=worker_startup_seconds,
            warmup_games=warmups * count,
            mean_seconds_per_env=statistics.mean(values),
            median_seconds_per_env=statistics.median(values),
            stdev_seconds_per_env=statistics.stdev(values) if len(values) > 1 else 0,
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
    import torch

    from pvz_rl.config import digest, load_config
    from pvz_rl.gpu import apply_gpu_profile, select_gpu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["cherry_bomb", "cherry_bomb_obstacle", "attacker_pre"],
        default="cherry_bomb",
    )
    parser.add_argument("--envs", nargs="+", type=int, default=[256, 512, 640, 1024])
    parser.add_argument("--workers", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--oracles", nargs="+", choices=["cpu", "cuda", "distance"])
    parser.add_argument("--warmup-backends", nargs="+", choices=["cpu", "cuda"], default=["cuda"])
    parser.add_argument("--lanes", nargs="+", type=int, default=[None])
    parser.add_argument("--vram-mib", nargs="+", type=int, default=[None])
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--warmup-games",
        type=int,
        default=0,
        help="Warm at least this many episodes before measurements",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Skip combinations already recorded")
    args = parser.parse_args()
    if min(*args.envs, args.repetitions, args.warmups) < 1 or min(args.workers) < 0:
        parser.error("envs/repetitions/warmups must be positive; workers must be nonnegative")
    if args.warmup_games < 0:
        parser.error("warmup-games must be nonnegative")
    if any(value is not None and value < 1 for value in (*args.lanes, *args.vram_mib)):
        parser.error("lanes and vram-mib must be positive")
    oracles = args.oracles or (["distance"] if args.stage == "attacker_pre" else ["cpu", "cuda"])
    if (args.stage == "attacker_pre" and oracles != ["distance"]) or (
        args.stage != "attacker_pre" and "distance" in oracles
    ):
        parser.error("attacker_pre uses distance labels; bomb stages use cpu/cuda oracles")
    cfg, profile = apply_gpu_profile(load_config(), device=select_gpu())
    report = dict(
        protocol="placement_collection_v2",
        config_digest=digest(cfg),
        stage=args.stage,
        seed=args.seed,
        profile=profile,
        settings=cfg["training"]["performance"],
        combinations=[],
    )
    if args.resume and args.output.is_file():
        report = json.loads(args.output.read_text("utf-8"))
        if report.get("stage") != args.stage or report.get("seed") != args.seed:
            parser.error("The existing benchmark has a different stage or seed")
        if report.get("protocol") != "placement_collection_v2" or report.get(
            "config_digest"
        ) != digest(cfg):
            parser.error("Benchmark protocol/configuration changed; use a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combinations = [
        (count, workers, oracle, backend, lanes, vram)
        for count, workers, oracle, lanes, vram in product(
            args.envs, args.workers, oracles, args.lanes, args.vram_mib
        )
        for backend in (args.warmup_backends if oracle != "cpu" else ["cpu"])
    ]
    random.Random(args.seed).shuffle(combinations)
    with TemporaryDirectory(prefix="pvz-placement-benchmark-") as workspace:
        for count, workers, oracle, backend, lanes, vram in combinations:
            # Collect failed measurement traceback cycles before another allocation.
            gc.collect()
            from pvz_rl.envs.tile_scenarios import preparation_settings

            defaults = preparation_settings(cfg, args.stage)
            identity = dict(
                stage=args.stage,
                envs=count,
                workers=workers,
                oracle=oracle,
                warmup_backend=backend,
                oracle_lanes=lanes or defaults.get("oracle_lanes", 256),
                oracle_vram_mib=vram or defaults.get("oracle_vram_mib", 256),
            )
            previous = [
                row
                for row in report["combinations"]
                if all(row.get(k) == v for k, v in identity.items())
            ]
            if args.resume and any(
                row.get("status") == "infeasible"
                or (
                    sum(not s["warmup"] for s in row["samples"]) >= args.repetitions
                    and row.get("warmup_games", 0) >= max(args.warmup_games, args.warmups * count)
                )
                for row in previous
            ):
                continue
            try:
                result = measure(
                    cfg,
                    count,
                    workers,
                    oracle,
                    args.repetitions,
                    args.warmups,
                    workspace,
                    cuda_warmup=backend == "cuda",
                    stage=args.stage,
                    lanes=lanes,
                    vram_mib=vram,
                    seed=args.seed,
                    warmup_games=args.warmup_games,
                )
            except (MemoryError, torch.OutOfMemoryError) as error:
                result = dict(identity, status="infeasible", error=str(error))
            report["combinations"] = [row for row in report["combinations"] if row not in previous]
            report["combinations"].append(result)
            measured = [row for row in report["combinations"] if "mean_seconds_per_env" in row]
            if measured:
                best = min(measured, key=lambda r: r["mean_seconds_per_env"])
                report["fastest"] = {k: v for k, v in best.items() if k != "samples"}
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
