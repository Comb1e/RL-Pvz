"""Bounded execution calibration and learning-study scaffold.

Every execution trial is an isolated process.  A result is eligible only when
the subprocess completed all repetitions and reported the required memory
headroom.  The script never launches formal training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic

from pvz_rl.config import _read_config

REQUIRED_CANDIDATE_KEYS = {"species", "tile", "tokens", "groups", "threads"}


def tuning_candidates(tuning):
    """Return validated, de-duplicated execution candidates."""
    candidates = [dict(candidate) for candidate in tuning.get("candidates", [])]
    if not candidates:
        raise ValueError("tuning.candidates must contain at least one execution candidate")
    if any(set(candidate) != REQUIRED_CANDIDATE_KEYS for candidate in candidates):
        raise ValueError(
            "each tuning candidate must define species, tile, tokens, groups and threads"
        )
    unique = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate[name] for name in sorted(REQUIRED_CANDIDATE_KEYS))
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _median(result):
    measurements = result["measurements"]
    repeats = min((len(m["repetitions"]) for m in measurements), default=0)
    totals = [sum(m["repetitions"][i]["seconds"] for m in measurements) for i in range(repeats)]
    return sorted(totals)[len(totals) // 2] if totals else float("inf")


def run_trial(args, candidate, output, timeout):
    output.unlink(missing_ok=True)
    workload = getattr(args, "workload", "synthetic")
    module = "pipeline_benchmark" if workload == "full-cycle" else "fit_benchmark"
    command = [sys.executable, "-m", "pvz_rl.monitoring." + module, "--output", str(output),
               "--gpu", args.gpu, "--species-microbatch", str(candidate["species"]),
               "--tile-microbatch", str(candidate["tile"]), "--token-budget", str(candidate["tokens"]),
               "--groups", str(candidate["groups"]), "--threads", str(candidate["threads"])]
    if workload == "full-cycle":
        command.extend(("--seconds", str(timeout), "--stage", args.stage))
    else:
        command.extend(("--rows-per-width", str(args.rows_per_width), "--repeats", str(args.repeats)))
    if workload == "trajectory":
        command.extend(("--trajectory", str(args.trajectory)))
    if getattr(args, "initial_weights", None):
        command.extend(("--init-from", str(args.initial_weights)))
    if args.no_compile:
        command.append("--no-compile")
    if args.config is not None:
        command.extend(("--config", str(args.config)))
    started = monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except subprocess.TimeoutExpired:
        return {"candidate": candidate, "complete": False, "reason": "timeout"}
    if completed.returncode or not output.exists():
        return {"candidate": candidate, "complete": False, "reason": "subprocess",
                "stderr": completed.stderr[-2000:]}
    result = json.loads(output.read_text("utf-8"))
    if workload == "full-cycle":
        result["measurements"] = [dict(workload="full-cycle", repetitions=[dict(
            seconds=result.get("cycle_seconds", float("inf")),
            peak_allocated_mib=result["torch_peak_mib"], free_mib=result["free_mib"],
            total_mib=result["total_mib"])])]
    if result.get("candidate") not in (None, candidate):
        return {"candidate": candidate, "complete": False, "reason": "candidate mismatch"}
    repetitions = [r for m in result["measurements"] for r in m["repetitions"]]
    headroom = all(r["free_mib"] >= max(r["total_mib"] * args.headroom_fraction,
                                         args.headroom_mib) for r in repetitions)
    return {"candidate": candidate, "complete": bool(result.get("complete") and headroom),
            "headroom": headroom, "median_seconds": _median(result),
            "elapsed_seconds": monotonic() - started, "raw": result}


def pipeline_acceptance(trials, settings):
    if len(trials) != 4 or not all(t.get("complete") for t in trials):
        return dict(eligible=False, reason="two completed paired comparisons required")
    baseline = [trials[0]["median_seconds"], trials[3]["median_seconds"]]
    candidate = [trials[1]["median_seconds"], trials[2]["median_seconds"]]
    changes = [c / a - 1 for a, c in zip(baseline, candidate, strict=True)]
    median_change = sum(changes) / 2
    eligible = (median_change <= -settings["pipeline_minimum_improvement"] and
                max(changes) <= settings["pipeline_maximum_regression"])
    return dict(eligible=eligible, paired_relative_changes=changes, median_relative_change=median_change)


def choose(trials, tolerance):
    valid = [trial for trial in trials if trial.get("complete")]
    if not valid:
        return None
    fastest = min(trial["median_seconds"] for trial in valid)
    eligible = [trial for trial in valid if trial["median_seconds"] <= fastest * (1 + tolerance)]
    def memory_then_time(trial):
        peak = max(r["peak_allocated_mib"]
                   for m in trial["raw"]["measurements"] for r in m["repetitions"])
        return peak, trial["median_seconds"]

    return min(eligible, key=memory_then_time)


def run_learning_study(args, root, timeout):
    """Run paired, equal-game pilots only when explicitly requested."""
    if not args.initial_weights:
        return {"complete": False, "reason": "--initial-weights is required"}
    root.mkdir(parents=True, exist_ok=True)
    studies = []
    deadline = monotonic() + timeout
    multipliers = tuple(float(value) for value in args.tuning.get(
        "learning_rate_multipliers", (1.0, 0.5, 2.0)))
    seeds = tuple(int(value) for value in args.tuning.get("learner_seeds", (101, 102)))
    profiles = {0.5: "lr-half", 1.0: "current", 2.0: "lr-double"}
    for multiplier in multipliers:
        profile = profiles.get(multiplier)
        if profile is None:
            raise ValueError(f"No learning profile is defined for multiplier {multiplier:g}")
        for seed in seeds:
            cohort = "first" if seed == seeds[0] else "second"
            remaining = deadline - monotonic()
            if remaining < 30:
                studies.append({"multiplier": multiplier, "seed": seed, "cohort": cohort,
                                "complete": False, "reason": "time budget exhausted"})
                continue
            output = root / f"learn-{multiplier:g}-{seed}"
            command = [sys.executable, "-m", "pvz_rl", "train", "--output", str(output),
                       "--init-from", str(args.initial_weights), "--gpu", args.gpu,
                       "--gpu-profile", "auto", "--learning-profile", profile,
                       "--seed", str(seed), "--stage", args.stage, "--games", str(args.study_games),
                       "--viewer", "off"]
            if args.config is not None:
                command.extend(("--config", str(args.config)))
            started = monotonic()
            try:
                run = subprocess.run(command, capture_output=True, text=True, timeout=remaining)
                complete = run.returncode == 0 and output.exists()
                studies.append({"multiplier": multiplier, "seed": seed, "cohort": cohort,
                                "complete": complete, "seconds": monotonic() - started,
                                "stderr": run.stderr[-2000:] if not complete else ""})
            except subprocess.TimeoutExpired:
                studies.append({"multiplier": multiplier, "seed": seed, "cohort": cohort,
                                "complete": False, "reason": "timeout"})
    expected = len(multipliers) * len(seeds)
    return {"complete": len(studies) == expected and all(item["complete"] for item in studies),
            "paired_trials": studies,
            "curriculum_split_required": True,
            "promotion": "requires win-rate evidence and confirmation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="visible index or complete CUDA UUID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--stage", default="easy")
    parser.add_argument("--workload", choices=("synthetic", "trajectory", "full-cycle"), default="synthetic")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--repeats", type=int,
                        help="measured repetitions (defaults to tuning.repetitions)")
    parser.add_argument("--rows-per-width", type=int,
                        help="rows per entity width (defaults to tuning.rows_per_width)")
    parser.add_argument("--headroom-fraction", type=float,
                        help="minimum free VRAM fraction (defaults to tuning.minimum_headroom_fraction)")
    parser.add_argument("--headroom-mib", type=float,
                        help="minimum free VRAM MiB (defaults to tuning.minimum_headroom_mib)")
    parser.add_argument("--learning-study", action="store_true",
                        help="run paired two-seed, two-cohort learning pilots after execution trials")
    parser.add_argument("--no-compile", action="store_true",
                        help="rehearse eager fitting on low-memory GPUs")
    parser.add_argument("--study-games", type=int, default=256,
                        help="completed games per paired pilot (two default 128-game cohorts)")
    args = parser.parse_args()
    if args.workload == "trajectory" and args.trajectory is None:
        parser.error("trajectory workload requires --trajectory")
    if args.workload == "full-cycle" and args.initial_weights is None:
        parser.error("full-cycle workload requires --initial-weights")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = _read_config(args.config)
    tuning = raw.get("tuning", {})
    args.repeats = args.repeats or int(tuning.get("repetitions", 3))
    args.rows_per_width = args.rows_per_width or int(tuning.get("rows_per_width", 8))
    args.headroom_fraction = (args.headroom_fraction if args.headroom_fraction is not None
                              else float(tuning.get("minimum_headroom_fraction", 0.20)))
    args.headroom_mib = (args.headroom_mib if args.headroom_mib is not None
                         else float(tuning.get("minimum_headroom_mib", 1024)))
    args.tuning = tuning
    candidates = tuning_candidates(tuning)
    if args.workload == "full-cycle":
        pair = tuning_candidates({"candidates": tuning["pipeline_candidates"]})
        candidates = [pair[0], pair[1], pair[1], pair[0]]
    started = monotonic()
    trials = []
    for index, candidate in enumerate(candidates):
        remaining = args.minutes * 60 - (monotonic() - started)
        if remaining < 30:
            break
        path = args.output / f"trial-{index:02d}.json"
        trials.append(run_trial(args, candidate, path, remaining))
    result = {"complete": bool(trials) and all(trial["complete"] for trial in trials),
              "gpu": args.gpu, "stage": args.stage,
              "initial_weights": str(args.initial_weights) if args.initial_weights else None,
              "trials": trials,
              "recommendation": choose(trials, float(tuning.get("within_fastest_fraction", 0.05)))}
    result["workload"] = args.workload
    result["eligible_for_profile_promotion"] = False
    if args.workload == "full-cycle":
        gate = pipeline_acceptance(trials, tuning)
        result["profile_acceptance"] = gate
        result["eligible_for_profile_promotion"] = gate["eligible"]
        if not gate["eligible"]:
            result["recommendation"] = None
    if args.learning_study:
        result["learning_study"] = run_learning_study(
            args, args.output / "learning", max(30, args.minutes * 60 - (monotonic() - started))
        )
    (args.output / "recommendation.json").write_text(json.dumps(result, indent=2), "utf-8")
    print(json.dumps({"recommendation": result["recommendation"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
