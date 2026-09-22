"""Bounded development comparison; never run the two-hour research matrix.

Run from the research checkout. Four five-minute allowances include each run's
validation and cleanup. Both profiles use the same five normal validation seeds.
"""

import argparse
import json
from pathlib import Path

from pvz_rl.config import load_config
from pvz_rl.training import train
from pvz_rl.training_requirements import require_cuda_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sc2-learning-check"))
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["sc2-inspired", "sc2-plant-rewards"],
        choices=["sc2-inspired", "sc2-efficient", "sc2-plant-rewards"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102])
    parser.add_argument("--minutes", type=float, default=5)
    args = parser.parse_args()
    if not 0 < args.minutes * len(args.configs) * len(args.seeds) <= 30:
        parser.error(
            "Learning checks must have a positive combined allowance of at most 30 minutes"
        )
    require_cuda_training(load_config(Path("configs") / f"{args.configs[0]}.toml"))
    output = args.output
    for index, seed in enumerate(args.seeds):
        names = list(args.configs)
        if index % 2:
            names.reverse()
        for name in names:
            cfg = load_config(Path("configs") / f"{name}.toml")
            cfg["training"]["max_minutes"] = args.minutes
            cfg["visualization"].update(enabled=False, demos=False, videos=False)
            run = output / f"{name}-{seed}"
            if run.exists():
                raise FileExistsError(run)
            train(cfg, "masked", seed, run, validation_limit=5)
            print(
                "PILOT_RESULT",
                json.dumps({"run": str(run), **json.loads((run / "status.json").read_text())}),
                flush=True,
            )


if __name__ == "__main__":
    main()
