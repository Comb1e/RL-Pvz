"""Explicitly budgeted, seed-matched forecast ablations through the real trainer.

This tool does not infer a learning-speed benefit from loss or a short run. It
retains per-cohort curves and real episode outcomes for later sample-efficiency
comparisons; ordinary completed-run artifact retention remains in effect.
"""

import argparse
import copy
import gc
import json
from pathlib import Path

from pvz_rl.config import load_config
from pvz_rl.learning import training_artifacts
from pvz_rl.learning.training import train
from pvz_rl.provenance import write_json

MODES = ("none", "threat", "effect", "both", "both-injection", "both-injection-shaping")


def configure(base, mode, *, games, minutes, half_life=None, pretrain_roots=None):
    cfg = copy.deepcopy(base)
    cfg["curriculum"]["run_stage"] = "easy"
    training = cfg["training"]
    training.update(total_games=games, max_minutes=minutes, until_stage_complete=False)
    training["tile_scenarios"]["enabled"] = False
    settings = training["forecasts"]
    settings.update(
        enabled=True,
        shaping_weight=0.01 if mode.endswith("shaping") else 0.0,
        injection_after_games=0 if "injection" in mode else 2**63 - 1,
    )
    for name, enabled in (
        ("threat", mode not in ("none", "effect")),
        ("effect", mode not in ("none", "threat")),
    ):
        if not enabled:
            training[name + "_prediction_weight"] = 0.0
    if mode in ("none", "threat"):
        settings["pretrain_roots"] = 0
    elif pretrain_roots is not None:
        settings["pretrain_roots"] = pretrain_roots
    if half_life is not None:
        settings["initial_discount_half_life_steps"] = half_life
        settings["discount_half_life_steps"] = half_life
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--half-lives", type=int, nargs="+", help="Agent-step discount sweep")
    parser.add_argument("--pretrain-roots", type=int)
    args = parser.parse_args()
    if (
        args.games <= 0
        or args.minutes <= 0
        or any(value <= 0 for value in args.half_lives or [])
        or (args.pretrain_roots is not None and args.pretrain_roots < 0)
    ):
        parser.error("Game/time budgets must be positive; pretraining roots nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    base = load_config(args.config)
    manifest = dict(protocol="pvz-rl/forecast-ablation-v1", seeds=args.seeds, runs=[])
    for half_life in args.half_lives or [None]:
        for seed in args.seeds:
            for mode in args.modes:
                name = f"{mode}-{seed}-half-{half_life or 'schedule'}"
                cfg = configure(
                    base,
                    mode,
                    games=args.games,
                    minutes=args.minutes,
                    half_life=half_life,
                    pretrain_roots=args.pretrain_roots,
                )
                finish = training_artifacts.finish_autonomous_run
                captured = {}

                def capture(path, *, retain=True):
                    for filename in (
                        "training-metrics.jsonl",
                        "training-episodes.jsonl",
                        "status.json",
                    ):
                        source = path / filename
                        if source.exists():
                            text = source.read_text(encoding="utf-8")
                            captured[filename] = (
                                json.loads(text)
                                if filename.endswith(".json")
                                else [json.loads(line) for line in text.splitlines() if line]
                            )
                    write_json(args.output / (name + ".json"), captured)
                    if retain:
                        finish(path)

                training_artifacts.finish_autonomous_run = capture
                try:
                    train(cfg, "masked", seed, args.output / name)
                finally:
                    training_artifacts.finish_autonomous_run = finish
                if not captured:
                    capture(args.output / name, retain=False)
                manifest["runs"].append(
                    dict(
                        name=name, seed=seed, mode=mode, half_life=half_life, report=name + ".json"
                    )
                )
                write_json(args.output / "comparison.json", manifest)
                gc.collect()


if __name__ == "__main__":
    main()
