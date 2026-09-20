"""Time-bounded development comparisons; never touches final-test cases."""

import math
from pathlib import Path
from time import perf_counter

from .config import learning_profile, seed_values
from .provenance import metadata, write_json
from .training import train
from .visualization import read_json, read_series

PROFILES = ("baseline", "tactical", "grouped", "pure-rl")
SEEDS = (101, 102)


def comparison_summary(output, diagnostics_passed):
    """Use a shared completed validation budget, never each method's best score."""
    output = Path(output)
    measurements, protocols, missing = {}, set(), []
    for seed in SEEDS:
        for profile in PROFILES:
            run = output / f"{profile}-{seed}"
            series = read_series(run / "learning-curve.jsonl")
            measurements[profile, seed] = {r["training_steps"]: r for r in series}
            if not series:
                missing.append(f"{profile}-{seed}")
            for file in run.glob("validation/*/episodes.jsonl"):
                protocols.update(r.get("game_protocol_hash") for r in read_series(file))
    if len(protocols) > 1 or None in protocols:
        raise ValueError("Pilot profiles must share the same verified game/evaluation protocol")
    common = set.intersection(*(set(v) for v in measurements.values()))
    if common:
        reference = learning_profile("baseline")
        expected_cases = {
            ("validation", "preset", level, seed)
            for level in reference["evaluation"]["levels"]
            for seed in seed_values(reference, "validation", 5)
        }
        for budget in common:
            for seed in SEEDS:
                for profile in PROFILES:
                    episodes = read_series(
                        output / f"{profile}-{seed}" / "validation" / str(budget) / "episodes.jsonl"
                    )
                    cases = {
                        (r.get("split"), r.get("family"), r.get("level"), r.get("scenario_seed"))
                        for r in episodes
                    }
                    if (
                        cases != expected_cases
                        or len(episodes) != len(expected_cases)
                        or any(
                            r.get("learner_seed") != seed
                            or r.get("profile") != profile
                            or r.get("training_steps") != budget
                            for r in episodes
                        )
                    ):
                        raise ValueError(
                            "Pilot comparison requires identical complete validation cases and profile identities"
                        )
                    score = sum(r["win"] for r in episodes) / len(episodes)
                    if not math.isclose(
                        score, measurements[profile, seed][budget]["macro_win_rate"], abs_tol=1e-12
                    ):
                        raise ValueError("Pilot curve score differs from episode records")
    budget = max(common) if common else None
    rows = (
        [
            {
                "profile": profile,
                "learner_seed": seed,
                "training_steps": budget,
                "macro_win_rate": measurements[profile, seed][budget]["macro_win_rate"],
            }
            for seed in SEEDS
            for profile in PROFILES
        ]
        if budget is not None
        else []
    )
    differences = (
        {
            str(seed): measurements["pure-rl", seed][budget]["macro_win_rate"]
            - measurements["baseline", seed][budget]["macro_win_rate"]
            for seed in SEEDS
        }
        if budget is not None
        else {}
    )
    recommended = bool(
        diagnostics_passed and len(differences) == 2 and all(d > 0 for d in differences.values())
    )
    result = {
        "state": "recommended_for_larger_development_study"
        if recommended
        else "experimental_inconclusive",
        "diagnostics_passed": diagnostics_passed,
        "matched_steps": budget,
        "missing_runs_or_validation": missing,
        "rows": rows,
        "candidate_difference_vs_baseline": differences,
        "shared_game_protocol_hash": next(iter(protocols), None),
        "common_validation_budgets": sorted(common),
        "limitations": "Two learner seeds and five validation cases per difficulty; no final-test evidence.",
    }
    write_json(output / "comparison.json", result)
    lines = [
        "# Pure-RL development pilot",
        "",
        f"Conclusion: **{result['state']}**.",
        "",
        f"Learning diagnostics passed: {diagnostics_passed}. Matched decisions: {budget}.",
        "",
        "Normal-game macro win rate at the same completed budget (not best-checkpoint scores).",
        "",
        "| Profile | Learner seed | Decisions | Win rate |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['profile']} | {r['learner_seed']} | {r['training_steps']} | {r['macro_win_rate']:.1%} |"
        for r in rows
    )
    if missing:
        lines.extend(["", "Missing comparisons: " + ", ".join(missing)])
    if budget is None:
        lines.extend(
            [
                "",
                "No budget is complete for all profiles and both seeds; no paired improvement claim is possible.",
            ]
        )
    lines.extend(
        [
            "",
            result["limitations"],
            "",
            "Each run has its own logs, settings, curves and checkpoints. Profiles and engine pins are never pooled.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rows:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, seed in zip(axes, SEEDS):
            for profile in PROFILES:
                series = measurements[profile, seed]
                steps = sorted(common)
                ax.plot(
                    steps, [series[n]["macro_win_rate"] for n in steps], marker="o", label=profile
                )
            ax.set(
                title=f"Learner seed {seed}",
                xlabel="Matched training decisions",
                ylabel="Normal-game macro win rate",
                ylim=(-0.02, 1.02),
            )
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "comparison.png", dpi=150)
        plt.close(fig)
        with (output / "report.md").open("a", encoding="utf-8") as stream:
            stream.write("\n![Matched validation curves](comparison.png)\n")
    return result


def run_pilot(output, minutes=30):
    if not math.isfinite(minutes) or not 0 < minutes <= 30:
        raise ValueError("Pilot minutes must be positive and at most 30")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    deadline = started + minutes * 60
    base = learning_profile("baseline")
    write_json(
        output / "metadata.json",
        metadata(base, pilot_minutes=minutes, profiles=PROFILES, learner_seeds=SEEDS),
    )
    diagnostics = {}
    jobs = []
    failure = None
    try:
        diagnostic_deadline = min(deadline, started + 300)
        for i, lesson in enumerate(("placement", "saving")):
            if perf_counter() >= diagnostic_deadline:
                diagnostics[lesson] = {"state": "not_run", "passed": False}
                continue
            cfg = learning_profile("pure-rl")
            cfg["training"].update(total_steps=131072, eval_interval=16384)
            cfg["visualization"]["demos"] = False
            run = output / f"diagnostic-{lesson}"
            slot = (diagnostic_deadline - perf_counter()) / (2 - i)
            train(
                cfg,
                "masked",
                101,
                run,
                family=lesson,
                validation_limit=20,
                deadline=min(diagnostic_deadline, perf_counter() + slot),
            )
            best = read_json(run / "best.json", {})
            diagnostics[lesson] = {
                "state": "complete",
                "passed": best.get("macro_win_rate", 0) >= 0.9,
                "best": best,
                "steps": read_json(run / "status.json", {}).get("steps"),
            }
            write_json(output / "diagnostics.json", diagnostics)
        order = [
            (profile, seed)
            for seed in SEEDS
            for profile in (PROFILES if seed == SEEDS[0] else reversed(PROFILES))
        ]
        for i, (profile, seed) in enumerate(order):
            if perf_counter() >= deadline:
                jobs.append({"profile": profile, "learner_seed": seed, "state": "not_run"})
                continue
            cfg = learning_profile(profile)
            cfg["training"].update(total_steps=262144, eval_interval=32768)
            cfg["visualization"]["demos"] = False
            slot = (deadline - perf_counter()) / (len(order) - i)
            run = output / f"{profile}-{seed}"
            train(
                cfg,
                "masked",
                seed,
                run,
                validation_limit=5,
                deadline=min(deadline, perf_counter() + slot),
            )
            jobs.append(
                {"profile": profile, "learner_seed": seed, **read_json(run / "status.json", {})}
            )
            write_json(output / "jobs.json", jobs)
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        write_json(output / "diagnostics.json", diagnostics)
        write_json(output / "jobs.json", jobs)
        result = comparison_summary(
            output, len(diagnostics) == 2 and all(d["passed"] for d in diagnostics.values())
        )
        write_json(
            output / "status.json",
            {
                "state": "failed" if failure else "complete",
                "error": failure,
                "wall_seconds": perf_counter() - started,
                "minutes": minutes,
                "comparison": result["state"],
            },
        )
    return result
