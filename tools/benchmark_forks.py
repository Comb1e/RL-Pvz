"""Explicitly bounded, seed-matched fork screening and full-cohort timing."""

import argparse
import copy
import gc
import hashlib
import io
import json
import subprocess
import tempfile
import time
import types
import zipfile
from itertools import product
from pathlib import Path

import numpy as np


def reference_class(revision):
    code = subprocess.check_output(
        ["git", "show", f"{revision}:src/pvz_rl/envs/fork_service.py"], text=True
    )
    module = types.ModuleType("reference_fork_service")
    exec(compile(code, "<reference_fork_service>", "exec"), module.__dict__)
    # The reference has no reusable caches to invalidate.
    module.ForkService.invalidate = lambda self, *args, **kwargs: None
    return module.ForkService


def finish(service, indices, actions=None):
    service.begin(indices, actions)
    while not service.advance():
        pass
    return service.take()


def fixture(count):
    from pvz_game import Game, LevelSpec, Spawn, Wait
    from pvz_game.config import InitialPlant
    from pvz_game.cuda import CudaBatch

    from pvz_rl.envs.cuda_snapshots import upload_snapshots

    source = CudaBatch(count, zombie_capacity=32, max_step_ticks=1024)
    snapshots = []
    for i in range(count):
        g = Game()
        plants = tuple(InitialPlant("peashooter", row, 0) for row in range(i % 6))
        g.reset(
            LevelSpec(
                "fork-benchmark",
                (
                    Spawn(1, "basic", i % 5, x=3000 + i % 5 * 1000),
                    Spawn(1, "conehead", (i + 1) % 5, x=8000),
                    Spawn(10000, "basic", 0),
                ),
                plants=plants,
                mowers=i % 2 == 0,
                initial_sun=1000,
            ),
            101 + i,
        )
        g.step(Wait(), ticks=1)
        g._cooldowns = {kind: 0 for kind in g._cooldowns}
        snapshots.append(g.snapshot())
    upload_snapshots(source, snapshots, indices=list(range(count)))
    legal = source.action_masks_device().get()
    roots, actions = [], []
    for i in range(count):
        rng = np.random.default_rng(101 + i)
        available = np.flatnonzero(legal[i])[1:]
        selected = rng.choice(available, min(5, len(available)), replace=False)
        roots.extend([i] * len(selected))
        actions.extend(selected)
    return source, snapshots, roots, actions


def screen(args):
    import cupy as cp

    from pvz_rl.envs.cuda_snapshots import upload_snapshots
    from pvz_rl.envs.fork_service import ForkService

    source, snapshots, roots, actions = fixture(args.envs)
    old = reference_class(args.reference)
    candidates = [dict(reference=True, lanes=256, chunk_ticks=256)]
    candidates += [
        dict(
            reference=False, lanes=lane, chunk_ticks=chunk, fork_threads=threads, fork_packed=packed
        )
        for packed, threads, lane, chunk in product(
            [True], [32, 64, 128, 256], [256, 512, 1024, 2048], [256, 512, 1024]
        )
    ]
    candidates += [
        dict(reference=False, lanes=256, chunk_ticks=256, fork_threads=32, fork_packed=False)
    ]
    records = []
    signature = None
    for candidate in candidates:
        settings = dict(probe="conehead", vram_mib=3072, **candidate)
        ctor = old if candidate["reference"] else ForkService
        startup = time.perf_counter()
        service = ctor(source, settings)
        startup = time.perf_counter() - startup
        samples = []
        for repeat in range(2):
            upload_snapshots(source, snapshots, indices=list(range(args.envs)))
            if not candidate["reference"]:
                service.invalidate()
            cp.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            baseline = finish(service, list(range(args.envs)))
            effects = finish(service, roots, actions)
            cp.cuda.get_current_stream().synchronize()
            samples.append(time.perf_counter() - started)
            data = np.concatenate((baseline.ravel(), effects.ravel()))
            if signature is None:
                signature = data
            assert np.array_equal(data, signature), candidate
        records.append(
            dict(
                settings=candidate,
                startup_seconds=startup,
                seconds=samples[1],
                seconds_per_env=samples[1] / args.envs,
                metrics=service.metrics,
            )
        )
        print(json.dumps(records[-1]), flush=True)
        args.output.write_text(json.dumps(records, indent=2))
        service.close()
        del service
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


def cohort(args):
    import torch

    import pvz_rl.envs.fork_service as module
    from pvz_rl.config import load_config
    from pvz_rl.envs.cuda_env import CudaVecEnv, ScenarioQueue
    from pvz_rl.learning import training_artifacts
    from pvz_rl.learning.cohort import CohortPhase
    from pvz_rl.learning.sequential_training import CudaSequentialQ
    from pvz_rl.learning.training import train

    base = load_config()
    base["curriculum"]["run_stage"] = args.stage
    t = base["training"]
    t.update(
        n_envs=args.envs, total_games=args.envs, max_minutes=args.minutes, random_cohort_seeds=False
    )
    t["tile_scenarios"]["cherry_bomb_n_envs"] = args.envs
    t["tile_scenarios"]["attacker"]["n_envs"] = args.envs
    t["peashooter_rules"]["n_envs"] = args.envs
    base["environment"]["cutoff_seconds"] = (
        max(120, args.game_seconds)
        if args.stage in ("cherry_bomb", "cherry_bomb_obstacle", "attacker_pre")
        else args.game_seconds
    )
    t["performance"].update(compile_collection=False, compile_fitting=False)
    base["visualization"].update(enabled=False, live_enabled=False, videos=False, demos=False)
    selected = json.loads(args.settings.read_text()) if args.settings else {}
    ctor, finish_run = module.ForkService, training_artifacts.finish_autonomous_run
    phase_method = CudaSequentialQ._phase
    draft_method, reset_method = ScenarioQueue._draft, CudaVecEnv.reset_indices
    if args.active_fixture:
        import random
        from dataclasses import replace

        from pvz_game import Spawn
        from pvz_game.config import InitialPlant, resolve_level

        def active_draft(queue, indices, rngs, seeds=None):
            staged = draft_method(queue, indices, rngs, seeds)
            return [
                (
                    *row[:4],
                    replace(
                        resolve_level(row[4], queue.rules, random.Random(row[3])),
                        initial_sun=500,
                        spawns=(Spawn(1, "basic", 0, x=5500), Spawn(1, "conehead", 1, x=8500)),
                        plants=(InitialPlant("peashooter", 2, 0), InitialPlant("wall_nut", 3, 4)),
                    ),
                )
                for row in staged
            ]

        def active_reset(env, indices, *args, **kwargs):
            result = reset_method(env, indices, *args, **kwargs)
            with env.device_context():
                env.batch.cooldowns[env.cp.asarray(indices)] = 0
                env.features.encode()
            return result

        if args.stage not in ("peashooter", "easy", "standard", "shared"):
            raise ValueError("Active fixture is for full-game stages only")
        ScenarioQueue._draft, CudaVecEnv.reset_indices = active_draft, active_reset
    records = []
    signature = None
    try:
        for variant in args.variants:
            module.ForkService = reference_class(args.reference) if variant == "reference" else ctor
            for repeat in range(args.repetitions + 1):
                cfg = copy.deepcopy(base)
                if variant != "reference":
                    cfg["training"]["forecasts"].update(
                        {
                            k: v
                            for k, v in selected.items()
                            if k in ("lanes", "vram_mib", "chunk_ticks")
                        }
                    )
                    cfg["training"]["performance"].update(
                        {k: v for k, v in selected.items() if k.startswith("fork_")}
                    )
                cfg["training"]["performance"]["fork_suffix_cache"] = variant != "uncached"
                captured = {}

                def capture_phase(model, phase, callback):
                    phase_method(model, phase, callback)
                    if phase == CohortPhase.FIT:
                        hasher = hashlib.sha256()
                        for block_index, block in enumerate(model._buffer.blocks):
                            block = block[
                                : min(
                                    len(block),
                                    model._buffer.size - block_index * model._buffer.block_rows,
                                )
                            ]
                            for name in (
                                "action",
                                "accepted",
                                "target",
                                "components",
                                "baseline_outcomes",
                                "effect_actions",
                                "effect_valid",
                                "effect_labels",
                            ):
                                hasher.update(np.ascontiguousarray(block[name]).tobytes())
                        captured["training_data_sha256"] = hasher.hexdigest()

                CudaSequentialQ._phase = capture_phase

                def capture(path, **kwargs):
                    captured["status"] = json.loads((path / "status.json").read_text())
                    captured["cohorts"] = [
                        json.loads(s)
                        for s in (path / "training-metrics.jsonl").read_text().splitlines()
                    ]
                    captured["episodes"] = [
                        json.loads(s)
                        for s in (path / "training-episodes.jsonl").read_text().splitlines()
                    ]
                    with zipfile.ZipFile(path / "final.zip") as archive:
                        weights = torch.load(
                            io.BytesIO(archive.read("policy.pth")),
                            map_location="cpu",
                            weights_only=True,
                        )
                    hasher = hashlib.sha256()
                    for name, tensor in sorted(weights.items()):
                        hasher.update(name.encode())
                        hasher.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                    captured["weights_sha256"] = hasher.hexdigest()

                training_artifacts.finish_autonomous_run = capture
                with tempfile.TemporaryDirectory(prefix="pvz-fork-benchmark-") as folder:
                    started = time.perf_counter()
                    train(cfg, "masked", args.seed, Path(folder) / "run")
                    elapsed = time.perf_counter() - started
                    record = dict(
                        variant=variant,
                        warmup=repeat == 0,
                        stage=args.stage,
                        active_fixture=args.active_fixture,
                        envs=args.envs,
                        seed=args.seed,
                        reference=args.reference,
                        settings={
                            **{
                                k: cfg["training"]["forecasts"][k]
                                for k in ("probe", "lanes", "vram_mib", "chunk_ticks")
                            },
                            **{
                                k: v
                                for k, v in cfg["training"]["performance"].items()
                                if k.startswith("fork_")
                            },
                        },
                        elapsed_seconds=elapsed,
                        complete="cohorts" in captured,
                        data=captured,
                    )
                    # Never extrapolate a timed-out or otherwise incomplete cohort.
                    if "cohorts" in captured:
                        observed = (captured["training_data_sha256"], captured["weights_sha256"])
                        if signature is None:
                            signature = observed
                        if observed != signature:
                            raise AssertionError("Fork execution changed training data or weights")
                        row = captured["cohorts"][-1]["cohort"]
                        record.update(
                            total_seconds_per_env=row.get("seconds_per_completed_env"),
                            collect_seconds_per_env=row.get("collection_seconds", 0) / args.envs,
                        )
                    records.append(record)
                    args.output.write_text(json.dumps(records, indent=2))
                    print(json.dumps({k: v for k, v in record.items() if k != "data"}), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
    finally:
        module.ForkService, training_artifacts.finish_autonomous_run = ctor, finish_run
        CudaSequentialQ._phase = phase_method
        ScenarioQueue._draft, CudaVecEnv.reset_indices = draft_method, reset_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screen", "cohort"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="4cd11ee")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--stage", default="easy")
    parser.add_argument("--game-seconds", type=int, default=20)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--settings", type=Path)
    parser.add_argument(
        "--active-fixture",
        action="store_true",
        help="Diagnostic ready-card board with visible threats; never a mastery measurement",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("reference", "optimized", "uncached"),
        default=["reference", "optimized"],
    )
    args = parser.parse_args()
    if min(args.envs, args.game_seconds, args.minutes, args.repetitions) <= 0:
        parser.error("Benchmark budgets and repetitions must be positive")
    if args.output.exists():
        parser.error("Use a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (screen if args.mode == "screen" else cohort)(args)


if __name__ == "__main__":
    main()
