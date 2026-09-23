"""Commands are explicit: availability checks never launch formal research runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    load_config,
    research_config,
    resolve_rollout,
    seed_values,
    validate_config,
)
from .provenance import file_hash, metadata, write_json
from .training_requirements import TRAINING_CONDITIONS, require_cuda_training


def common(parser):
    parser.add_argument("--config", type=Path)


def video_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--videos",
        action="store_true",
        default=None,
        help="export MP4 videos in addition to compact demos",
    )
    group.add_argument(
        "--no-videos",
        dest="videos",
        action="store_false",
        help="skip MP4 encoding; retain compact demos and curves",
    )


def training_options(parser):
    common(parser)
    parser.add_argument("--hardware", type=Path, help="benchmark recommendation.json")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--games", type=int, help="total completed training games across all workers"
    )
    budget.add_argument(
        "--steps", type=int, help="legacy decision-budget mode; for archived protocols only"
    )
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--device", choices=("cuda",))
    parser.add_argument("--rollout-size", type=int)
    parser.add_argument("--simulator", choices=("cuda",))
    parser.add_argument(
        "--rollout-steps-per-env", type=int, help="decisions per parallel game; CUDA default 128"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--max-minutes",
        type=float,
        help="cumulative time cap per run, including validation and demos",
    )
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument(
        "--eval-games",
        type=int,
        help="periodic validation interval for diagnostic/fixed or archived runs; teaching uses stage success",
    )
    video_options(parser)


def configured(args):
    cfg = load_config(args.config)
    checkpoint = getattr(args, "resume", None) or getattr(args, "init_from", None)
    if checkpoint and not args.config:
        run = args.output if args.command == "suite" else checkpoint.parent
        saved = json.loads((run / "metadata.json").read_text("utf-8"))
        cfg = saved["config"]
        if (
            getattr(args, "init_from", None)
            and cfg.get("policy", {}).get("kind") == "spatial_grouped_v3"
        ):
            cfg["policy"]["kind"] = "spatial_grouped_v4"
        if args.command == "train":
            for name, field in (
                ("seed", "learner_seed"),
                ("condition", "condition"),
                ("validation_count", "validation_limit"),
            ):
                if getattr(args, name, None) is None:
                    setattr(args, name, saved.get(field))
    if args.command == "train":
        args.seed = 101 if getattr(args, "seed", None) is None else args.seed
        args.condition = getattr(args, "condition", None) or "masked"
        if getattr(args, "stage", None) is not None:
            cfg["curriculum"]["run_stage"] = args.stage
    if getattr(args, "hardware", None):
        hardware = json.loads(args.hardware.read_text("utf-8"))
        if type(hardware.get("n_envs")) is not int or hardware["n_envs"] < 1:
            raise ValueError("Hardware benchmark has no usable parallel-game recommendation")
        cfg["training"].update({k: hardware[k] for k in ("n_envs", "device")})
        if "simulator" in hardware:
            cfg["simulation"] = {"backend": hardware["simulator"]}
        if "rollout_steps_per_env" in hardware:
            cfg["training"]["rollout_steps_per_env"] = hardware["rollout_steps_per_env"]
    for arg, key in (
        ("steps", "total_steps"),
        ("games", "total_games"),
        ("n_envs", "n_envs"),
        ("device", "device"),
        ("rollout_size", "rollout_size"),
        ("batch_size", "batch_size"),
        ("max_minutes", "max_minutes"),
        ("eval_interval", "eval_interval"),
        ("eval_games", "eval_interval_games"),
    ):
        value = getattr(args, arg, None)
        if value is not None:
            cfg["training"][key] = value
    if getattr(args, "games", None) is not None:
        cfg["training"]["budget_unit"] = "games"
    elif getattr(args, "steps", None) is not None and args.command in ("train", "suite"):
        cfg["training"]["budget_unit"] = "decisions"
    selected_simulator = getattr(args, "simulator", None)
    if selected_simulator is not None:
        cfg["simulation"] = {"backend": selected_simulator}
    resolve_rollout(
        cfg,
        per_env=getattr(args, "rollout_steps_per_env", None),
        total=getattr(args, "rollout_size", None),
        new_cuda=selected_simulator == "cuda",
    )
    if (
        getattr(args, "eval_games", None) is not None
        and cfg["training"].get("budget_unit") != "games"
    ):
        raise ValueError("--eval-games requires a game-count training budget")
    if (
        getattr(args, "eval_interval", None) is not None
        and cfg["training"].get("budget_unit") == "games"
    ):
        raise ValueError(
            "Use --eval-games with game-count training; --eval-interval is for legacy --steps"
        )
    if getattr(args, "videos", None) is not None:
        cfg.setdefault("visualization", {})["videos"] = args.videos
    validate_config(cfg)
    if args.command in ("train", "suite", "benchmark-gpu"):
        require_cuda_training(cfg, getattr(args, "condition", "masked"), runtime=False)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pvz-rl", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    doctor = subs.add_parser("doctor", help="check engine, packages, Gym API, and CUDA")
    common(doctor)
    doctor.add_argument("--output", type=Path)
    training = subs.add_parser("train", help="train one shared policy across all difficulties")
    training_options(training)
    training.add_argument("--condition", choices=TRAINING_CONDITIONS, help="default: masked")
    training.add_argument("--seed", type=int, help="learner seed; default: saved seed or 101")
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--validation-count", type=int, help="pilot-only reduced validation set")
    training.add_argument("--family", choices=("preset", "diagnostic"), default="preset")
    from .curriculum import STAGES

    training.add_argument(
        "--stage", choices=STAGES, help="train only this teaching stage; stop at mastery or budget"
    )
    continuation = training.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path, help="continue with saved budget and progress")
    continuation.add_argument(
        "--init-from", type=Path, help="load compatible weights into a fresh experiment or stage"
    )
    evaluation = subs.add_parser("evaluate", help="evaluate a checkpoint or non-learning baseline")
    common(evaluation)
    source = evaluation.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--baseline", choices=("wait", "random_legal", "heuristic", "random_strategy")
    )
    evaluation.add_argument(
        "--split", choices=("development", "validation", "test", "ood"), default="validation"
    )
    evaluation.add_argument("--count", type=int)
    evaluation.add_argument("--levels", nargs="+", choices=("easy", "standard", "hard"))
    evaluation.add_argument(
        "--family",
        default="preset",
        choices=(
            "preset",
            "diagnostic",
            "placement",
            "saving",
            "redistributed",
            "faster",
            "concentrated",
        ),
    )
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--record", action="store_true")
    gpu_bench = subs.add_parser(
        "benchmark-gpu", help="bounded CUDA throughput comparison at configurable parallelism"
    )
    common(gpu_bench)
    gpu_bench.add_argument("--output", type=Path, required=True)
    gpu_bench.add_argument("--minutes", type=float, default=15)
    gpu_bench.add_argument("--steps", type=int, default=16384)
    gpu_bench.add_argument(
        "--env-counts", type=int, nargs="+", help="parallel-game counts (default: 128 256 512 1024)"
    )
    suite = subs.add_parser(
        "suite",
        help="train the shared spatial policy across learner seeds, then evaluate and report",
    )
    training_options(suite)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--resume", action="store_true")
    report = subs.add_parser("report", help="generate statistics and figures from evaluation JSONL")
    common(report)
    report.add_argument("--episodes", type=Path, nargs="+", required=True)
    report.add_argument("--curves", type=Path, nargs="*", default=[])
    report.add_argument("--output", type=Path, required=True)
    replay = subs.add_parser("replay", help="verify or watch an engine replay")
    replay.add_argument("path", type=Path)
    replay.add_argument("--watch", action="store_true")
    replay.add_argument("--speed", type=float, help="native viewer speed; requires --watch")
    replay.add_argument("--video", type=Path, help="export a verified offscreen MP4")
    common(replay)
    visual = subs.add_parser(
        "visualize", help="regenerate a run's offline report and shared-policy demos"
    )
    common(visual)
    visual.add_argument("--run", required=True, type=Path)
    video_options(visual)
    args = parser.parse_args(argv)

    if args.command == "replay":
        from pvz_game.replay import validate_speed

        from .recordings import open_playback

        if args.speed is not None and not args.watch:
            parser.error("--speed requires --watch")
        try:
            speed = validate_speed(1 if args.speed is None else args.speed)
        except ValueError as exc:
            parser.error(str(exc))
        playback = open_playback(args.path)
        game = playback.verify()
        print(
            json.dumps(
                {
                    "verified": True,
                    "status": game.observe().status.value,
                    "outcome": playback.display_outcome,
                    "metadata": playback.metadata,
                    "state_hash": game.state_hash(),
                }
            )
        )
        if args.watch:
            from .recordings import watch_recording

            # Pass the normalized payload so legacy sidecars retain their cutoff labels.
            watch_recording(args.path, speed=speed)
        if args.video:
            from .video import export_replay

            export_replay(args.path, args.video, load_config(args.config))
        return

    cfg = configured(args)
    if args.command == "visualize":
        from .visualization import visualize_run

        original = json.loads((args.run / "metadata.json").read_text("utf-8"))["config"]
        if args.config and research_config(cfg) != research_config(original):
            raise ValueError(
                "Visualization config must preserve the checkpoint's research settings"
            )
        result = visualize_run(args.run, cfg=cfg if args.config else original, videos=args.videos)
        print(f"Report: {(args.run / 'visualizations' / 'index.html').resolve()}")
        if result["state"] != "complete":
            raise SystemExit(1)
    elif args.command == "doctor":
        import torch
        from gymnasium.utils.env_checker import check_env

        from .env import PvZEnv

        details = metadata(cfg, kind="availability-check")
        from .cuda_diagnostics import cuda_doctor

        details["cuda_simulator"] = cuda_doctor()
        for condition in cfg["conditions"]:
            env = PvZEnv(cfg, condition=condition)
            check_env(env, skip_render_check=True)
            env.close()
        details["gymnasium_checks"] = f"passed: {len(cfg['conditions'])} reference conditions"
        from tempfile import TemporaryDirectory

        from pvz_game import Place
        from pvz_game.replay import read_recording

        from .recordings import open_playback

        with TemporaryDirectory(prefix="pvz-doctor-") as temporary:
            replay_path = Path(temporary) / "probe.pvzdemo"
            env = PvZEnv(cfg, record=True)
            env.reset(seed=1)
            info = env.step(env.codec.encode(Place("sunflower", 0, 0)))[4]
            env.step(0)
            env.recorder.update_metadata({"policy_id": "doctor"})
            env.recorder.save(replay_path)
            probe = open_playback(replay_path)
            probe.verify()
            assert probe.game.state_hash() == env.game.state_hash()
            details["compact_replay"] = {
                "available": read_recording(replay_path)["metadata"]["policy_id"] == "doctor",
                "seek_supported": hasattr(probe, "seek"),
                "action_timing": cfg["environment"].get("action_timing", "fixed"),
                "instant_placement": info["ticks_advanced"] == 0,
            }
        try:
            from .rendering import render_observation

            frame = render_observation(env.public)
            details["rendering"] = {"available": True, "shape": list(frame.shape)}
        except Exception as exc:
            details["rendering"] = {"available": False, "error": str(exc)}
        try:
            from .video import ffmpeg_info

            details["video"] = {"available": True, **ffmpeg_info(cfg)}
        except Exception as exc:
            details["video"] = {"available": False, "error": str(exc)}
        details["training_ready"] = details["cuda_simulator"]["available"]
        details["cuda_available"] = torch.cuda.is_available()
        if details["cuda_available"]:
            value = (torch.ones(2, device="cuda") + 1).sum().item()
            details.update(cuda_device=torch.cuda.get_device_name(), cuda_tensor_check=value == 4)
        if args.output:
            write_json(args.output, details)
        print(
            json.dumps(
                {
                    k: details[k]
                    for k in (
                        "engine",
                        "gymnasium_checks",
                        "cuda_available",
                        "training_ready",
                        "cuda_simulator",
                        "rendering",
                        "video",
                        "compact_replay",
                    )
                },
                indent=2,
            )
        )
        if not details["training_ready"]:
            raise SystemExit(1)
    elif args.command == "train":
        from .training import train

        train(
            cfg,
            args.condition,
            args.seed,
            args.output,
            validation_limit=args.validation_count,
            family=args.family,
            resume=args.resume,
            init_from=args.init_from,
        )
        print(f"Training artifacts: {args.output.resolve()}")
    elif args.command == "evaluate":
        from .evaluation import evaluate, summarize
        from .training import load_policy

        policy, condition, learner_seed, checkpoint_hash = None, "masked", None, None
        if args.checkpoint:
            policy, data = load_policy(args.checkpoint)
            if args.config and research_config(cfg) != research_config(data["config"]):
                raise ValueError(
                    "Evaluation config must match the checkpoint; omit --config to reuse it"
                )
            cfg = cfg if args.config else data["config"]
            condition, learner_seed = data["condition"], data["learner_seed"]
            checkpoint_hash = file_hash(args.checkpoint)
            if (
                data["family"] in ("diagnostic", "placement", "saving")
                and args.family != data["family"]
            ):
                raise ValueError(
                    f"Diagnostic checkpoints must be evaluated with --family {data['family']}"
                )
        if args.family in cfg["evaluation"]["ood_families"] and args.split != "ood":
            raise ValueError("Changed scenario families must use --split ood")
        if args.split == "ood" and args.family not in cfg["evaluation"]["ood_families"]:
            raise ValueError("OOD split requires a changed scenario family")
        if args.family in ("diagnostic", "placement", "saving") and args.split not in (
            "development",
            "validation",
        ):
            raise ValueError("Diagnostic tasks are not formal test evidence")
        if args.split == "development":
            count = args.count or 10
            if not 1 <= count <= 10:
                raise ValueError("Development evaluation uses seeds 0-9")
            seeds = list(range(count))
        else:
            seeds = seed_values(cfg, args.split, args.count)
        levels = args.levels or cfg["evaluation"]["ood_levels" if args.split == "ood" else "levels"]
        details = metadata(
            cfg,
            condition=condition,
            learner_seed=learner_seed,
            checkpoint_hash=checkpoint_hash,
            checkpoint=str(args.checkpoint),
            baseline=args.baseline,
            split=args.split,
            family=args.family,
        )
        rows = evaluate(
            cfg,
            policy=policy,
            condition=condition,
            baseline=args.baseline,
            learner_seed=learner_seed,
            seeds=seeds,
            levels=levels,
            family=args.family,
            split=args.split,
            output=args.output,
            record=args.record,
            training_steps=policy.num_timesteps if policy else 0,
            checkpoint_hash=checkpoint_hash,
        )
        write_json(args.output / "metadata.json", details)
        print(json.dumps(summarize(rows), indent=2))
    elif args.command == "benchmark-gpu":
        from .gpu_benchmark import benchmark_gpu

        print(
            json.dumps(
                benchmark_gpu(
                    cfg,
                    args.output,
                    minutes=args.minutes,
                    steps=args.steps,
                    env_counts=args.env_counts,
                ),
                indent=2,
            )
        )
    elif args.command == "suite":
        from .suite import run_suite

        result = run_suite(cfg, args.output, resume=args.resume)
        print(json.dumps({"state": result["state"], "output": str(args.output.resolve())}))
    elif args.command == "report":
        from .reporting import make_report

        make_report(args.episodes, args.output, cfg, args.curves)
        print(f"Report: {(args.output / 'report.md').resolve()}")
