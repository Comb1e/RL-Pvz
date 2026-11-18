"""Recollect one fixed opening panel, fitting one pass between accuracy checks."""

import argparse
from copy import deepcopy
from pathlib import Path
from random import Random

from pvz_rl.config import load_config, validate_config
from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.gpu import apply_gpu_profile, select_gpu
from pvz_rl.learning.fixed_prediction import PROTOCOL, fixed_seeds
from pvz_rl.learning.training import train
from pvz_rl.provenance import write_json


def control_config(cfg, seed):
    """Keep architecture, scene distribution and optimizer tuning; fix only the panel."""
    cfg = deepcopy(cfg)
    cfg["curriculum"]["run_stage"] = "pre_prediction"
    settings = cfg["training"]["tile_scenarios"]["pre_prediction"]
    lower, upper = cfg["splits"]["train"]
    settings["fixed_scene_seeds"] = Random(namespace_seed(PROTOCOL, seed)).sample(
        range(lower, upper + 1), settings["n_envs"]
    )
    cfg["training"].update(
        until_stage_complete=True,
        stage_streak=1,
        random_cohort_seeds=False,
    )
    cfg["training"].pop("cohorts_per_epoch", None)
    # The diagnostic owns its learning settings; live controls cannot alter them.
    cfg["visualization"].update(
        enabled=False,
        live_enabled=False,
        demos=False,
        videos=False,
    )
    validate_config(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, help="Fresh learner and fixed-panel seed; default 101")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-profile", default="rtx4070-laptop")
    parser.add_argument("--resume", type=Path, help="Continue this control's interrupted.zip")
    args = parser.parse_args()
    if args.output.exists() and (
        args.resume is None or args.output.resolve() != args.resume.resolve().parent
    ):
        parser.error("Use a new --output directory, or resume its own interrupted archive")
    if args.resume:
        if args.config:
            parser.error("--resume retains the saved configuration; omit --config")
        from pvz_rl.learning.checkpoints import inspect_checkpoint

        saved = inspect_checkpoint(args.resume)
        cfg, seed = saved["config"], saved["learner_seed"]
        if fixed_seeds(cfg) is None or cfg["curriculum"].get("run_stage") != "pre_prediction":
            parser.error("--resume requires an interrupted fixed pre_prediction control")
        if args.seed is not None and args.seed != seed:
            parser.error("--seed differs from the saved learner seed")
    else:
        seed = 101 if args.seed is None else args.seed
        if not 0 <= seed < 2**32:
            parser.error("--seed must be in [0, 2**32)")
        cfg = control_config(load_config(args.config), seed)
    device = select_gpu(args.gpu)
    cfg, _ = apply_gpu_profile(cfg, args.gpu_profile, device=device, path=args.config)
    try:
        train(cfg, "masked", seed, args.output, resume=args.resume)
    finally:
        if args.output.is_dir():
            # These diagnostic artifacts survive normal completed-run compaction.
            write_json(args.output / "overfit-config.json", cfg)
            write_json(
                args.output / "scene-seeds.json",
                dict(
                    protocol=PROTOCOL,
                    learner_seed=seed,
                    seeds=fixed_seeds(cfg),
                ),
            )


if __name__ == "__main__":
    main()
