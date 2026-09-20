"""Commands are explicit: availability checks never launch formal research runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, research_config, seed_values, validate_config
from .provenance import file_hash, metadata, write_json


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
    parser.add_argument(
        "--steps", type=int, help="total training decisions, not extra resume steps"
    )
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--rollout-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-interval", type=int)
    video_options(parser)


def configured(args):
    cfg = load_config(args.config)
    if getattr(args, "hardware", None):
        hardware = json.loads(args.hardware.read_text("utf-8"))
        cfg["training"].update({k: hardware[k] for k in ("n_envs", "device")})
    for arg, key in (
        ("steps", "total_steps"),
        ("n_envs", "n_envs"),
        ("device", "device"),
        ("rollout_size", "rollout_size"),
        ("batch_size", "batch_size"),
        ("eval_interval", "eval_interval"),
    ):
        value = getattr(args, arg, None)
        if value is not None:
            cfg["training"][key] = value
    if getattr(args, "videos", None) is not None:
        cfg.setdefault("visualization", {})["videos"] = args.videos
    validate_config(cfg)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pvz-rl", description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    pilot = subs.add_parser(
        "pilot", help="pure-RL diagnostics and matched comparisons, capped at 30 minutes"
    )
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--minutes", type=float, default=30)
    doctor = subs.add_parser("doctor", help="check engine, packages, Gym API, and CUDA")
    common(doctor)
    doctor.add_argument("--output", type=Path)
    training = subs.add_parser("train", help="train one shared policy across all difficulties")
    training_options(training)
    training.add_argument(
        "--condition", default="masked", choices=("masked", "unmasked", "sparse", "mixed", "hybrid")
    )
    training.add_argument("--seed", type=int, default=101, help="learner seed")
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--validation-count", type=int, help="pilot-only reduced validation set")
    training.add_argument("--family", choices=("preset", "diagnostic"), default="preset")
    training.add_argument("--resume", type=Path, help="checkpoint from an interrupted run")
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
    bench = subs.add_parser("benchmark", help="compare complete collection/update throughput")
    common(bench)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--steps", type=int, default=16384)
    bench.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    bench.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument(
        "--compare-runtime",
        action="store_true",
        help="pair current data transport with stock SB3 using identical training settings",
    )
    suite = subs.add_parser(
        "suite", help="run all 25 training jobs, held-out evaluation, and report"
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

    if args.command == "pilot":
        from .pilot import run_pilot

        print(json.dumps(run_pilot(args.output, args.minutes), indent=2))
        return

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
            from pvz_game.ui import App

            # Pass the normalized payload so legacy sidecars retain their cutoff labels.
            App(replay_path=playback.data, speed=speed).run()
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
        for condition in cfg["conditions"]:
            env = PvZEnv(cfg, condition=condition)
            check_env(env, skip_render_check=True)
            env.close()
        details["gymnasium_checks"] = "passed: all five conditions"
        from tempfile import TemporaryDirectory

        from pvz_game.replay import Playback, Recorder, read_recording, write_recording

        with TemporaryDirectory(prefix="pvz-doctor-") as temporary:
            replay_path = Path(temporary) / "probe.pvzdemo"
            recorder = Recorder(env.game, metadata={"policy_id": "doctor"})
            write_recording(recorder.to_dict(), replay_path)
            probe = Playback(replay_path)
            probe.verify()
            details["compact_replay"] = {
                "available": read_recording(replay_path)["metadata"]["policy_id"] == "doctor",
                "seek_supported": hasattr(probe, "seek"),
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
                        "rendering",
                        "video",
                        "compact_replay",
                    )
                },
                indent=2,
            )
        )
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
    elif args.command == "benchmark":
        from .benchmark import benchmark

        print(
            json.dumps(
                benchmark(
                    cfg,
                    args.output,
                    steps=args.steps,
                    workers=args.workers,
                    devices=args.devices,
                    repeats=args.repeats,
                    compare_runtime=args.compare_runtime,
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
