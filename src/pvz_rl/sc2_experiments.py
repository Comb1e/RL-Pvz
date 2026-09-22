"""Explicit development comparisons. Never evaluates final-test seeds."""

import copy
import json
import math
import tomllib
from importlib.resources import files
from pathlib import Path

import numpy as np

from .config import digest, learning_profile, load_config, validate_config
from .provenance import write_json
from .visualization import read_json, read_series


def settings():
    return tomllib.loads(files("pvz_rl").joinpath("data/sc2.toml").read_text("utf-8"))


def sc2_profile(letter):
    spec = settings()
    if letter not in spec["profiles"]:
        raise ValueError("SC2 profile must be A, B, C, D, E, or F")
    cfg = learning_profile("baseline" if letter == "A" else "grouped", load_config())
    cfg["profile"] = {"E": "sc2-inspired", "F": "sc2-long-horizon"}.get(letter, f"sc2-{letter}")
    cfg["training"].update(spec["training"])
    cfg["training"]["learner_seeds"] = spec["learner_seeds"]
    cfg["simulation"] = {"backend": "cuda"}
    if letter in "CDEF":
        cfg["training"]["exploration"] = copy.deepcopy(spec["exploration"])
    if letter in "DEF":
        cfg["curriculum"] = learning_profile("pure-rl")["curriculum"]
        cfg["curriculum"]["residency"] = "episode_start_stage"
    if letter in "EF":
        cfg["policy"] = spec["policy"]
    if letter == "F":
        cfg["reward"]["gamma"] = 0.9999
    cfg["diagnostics"] = {"early_dig_seconds": spec["early_dig_seconds"]}
    validate_config(cfg)
    return cfg


def comparison_protocol(cfg):
    """Verify shared controls while naming the deliberate gamma ablation."""
    return digest(
        {
            "engine": [
                cfg[k] for k in ("engine_commit", "engine_version", "engine_package_version")
            ],
            "environment": cfg["environment"],
            "reward_coefficients": {k: v for k, v in cfg["reward"].items() if k != "gamma"},
            "validation": cfg["splits"]["validation"],
            "levels": cfg["evaluation"]["levels"],
            "training": {k: v for k, v in cfg["training"].items() if k != "exploration"},
            "simulation": cfg["simulation"],
        }
    )


def comparison_report(output):
    root, spec = Path(output), settings()
    rows, observations, missing, protocols, identities = [], {}, [], set(), {}
    for seed in spec["learner_seeds"]:
        for profile in spec["profiles"]:
            run = root / f"{profile}-{seed}"
            meta = read_json(run / "metadata.json", {})
            if meta:
                protocols.add(comparison_protocol(meta["config"]))
                cfg = meta["config"]
                identity = digest(
                    {
                        k: cfg.get(k)
                        for k in ("profile", "encoding", "policy", "curriculum", "reward")
                    }
                    | {"exploration": cfg["training"].get("exploration")}
                )
                identities.setdefault(profile, set()).add(identity)
                if (
                    meta["learner_seed"] != seed
                    or cfg["profile"] != sc2_profile(profile)["profile"]
                ):
                    raise ValueError("Comparison profile or learner identity differs")
            series = read_series(run / "learning-curve.jsonl")
            final = next((r for r in reversed(series) if r.get("final")), None)
            status = read_json(run / "status.json", {})
            if final is None and series and status.get("state") == "complete":
                final = series[-1]
            if not final or status.get("validation_pending"):
                missing.append(f"{profile}-{seed}")
                continue
            # A completed validation at the final weights is required. Do not
            # substitute the best score encountered at any earlier checkpoint.
            if final["training_steps"] != status.get("steps"):
                missing.append(f"{profile}-{seed}")
                continue
            candidates = list(
                (run / "validation").glob(f"{final['training_steps']}*/episodes.jsonl")
            )
            expected = {
                (level, seed)
                for level in meta["config"]["evaluation"]["levels"]
                for seed in range(
                    meta["config"]["splits"]["validation"][0],
                    meta["config"]["splits"]["validation"][1] + 1,
                )
            }
            episodes = next(
                (read_series(p) for p in candidates if len(read_series(p)) == len(expected)), []
            )
            cases = {(r["level"], r["scenario_seed"]) for r in episodes}
            if cases != expected or any(r["split"] != "validation" for r in episodes):
                raise ValueError("Comparison requires every fixed normal-game validation case")
            if len({r["game_protocol_hash"] for r in episodes}) != 1:
                raise ValueError("Mixed evaluation protocols within one run")
            protocol = digest(
                {
                    "engine": meta["engine"],
                    "environment": cfg["environment"],
                    "reward": cfg["reward"],
                }
            )
            if any(
                r["game_protocol_hash"] != protocol
                or r["learner_seed"] != seed
                or r["profile"] != cfg["profile"]
                or r["family"] != "preset"
                for r in episodes
            ):
                raise ValueError("Episode provenance differs from the comparison run")
            if not math.isclose(
                sum(r["win"] for r in episodes) / len(episodes), final["macro_win_rate"]
            ):
                raise ValueError("Validation score differs from episode evidence")
            observations[profile, seed] = {
                (r["level"], r["scenario_seed"]): r["win"] for r in episodes
            }
            rows.append(
                {
                    "profile": profile,
                    "learner_seed": seed,
                    "games": final["training_games"],
                    "decisions": final["training_steps"],
                    "wall_seconds": status["wall_seconds"],
                    "nominal_games": final.get("nominal_games"),
                    "macro_win_rate": final["macro_win_rate"],
                    "levels": final["levels"],
                    "stop_reason": status.get("stop_reason"),
                }
            )
    if len(protocols) > 1:
        raise ValueError("SC2 comparisons require the same engine and evaluation/resource controls")
    if any(len(values) > 1 for values in identities.values()):
        raise ValueError("Cannot pool different agent profiles under one method name")
    diagnostic = read_json(root / "diagnostics.json", {})
    diagnostics_passed = all(
        diagnostic.get(str(seed), {}).get("passed", False) for seed in spec["learner_seeds"]
    )
    comparisons = {}
    for profile in spec["profiles"][1:]:
        paired, per_seed = [], {}
        for seed in spec["learner_seeds"]:
            if (profile, seed) not in observations or ("A", seed) not in observations:
                continue
            candidate, baseline = observations[profile, seed], observations["A", seed]
            ordered = sorted(candidate)
            diff = np.array([candidate[k] - baseline[k] for k in ordered], dtype=float).reshape(
                3, -1
            )
            paired.append(diff)
            per_seed[str(seed)] = float(diff.mean())
        interval = None
        if len(paired) == len(spec["learner_seeds"]):
            values = np.stack(paired)
            rng = np.random.default_rng(1739)
            boots = []
            for _ in range(2000):
                # Preserve both method pairing and the shared scenario seed
                # across difficulties, resampling learner runs separately.
                learners = rng.integers(0, len(values), len(values))
                seeds = rng.integers(0, values.shape[2], values.shape[2])
                boots.append(values[learners][:, :, seeds].mean())
            interval = np.quantile(boots, [0.025, 0.975]).tolist()
        comparisons[profile] = {
            "difference_by_learner": per_seed,
            "paired_95_percent_interval": interval,
            "recommended_for_further_study": diagnostics_passed
            and len(per_seed) == 2
            and all(value > 0 for value in per_seed.values()),
        }
    result = {
        "state": "experimental",
        "rows": rows,
        "missing": missing,
        "diagnostics_passed": diagnostics_passed,
        "comparisons": comparisons,
        "limitations": "Development validation with two learner seeds; no held-out evidence. Actual game counts include rollout overshoot. Gamma F is a deliberate ablation.",
    }
    write_json(root / "comparison.json", result)
    lines = [
        "# SC2-inspired development comparison",
        "",
        result["limitations"],
        "",
        "| Profile | Learner | Actual games | Decisions | Minutes | Final macro win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['profile']} | {r['learner_seed']} | {r['games']} | {r['decisions']} | {r['wall_seconds'] / 60:.1f} | {r['macro_win_rate']:.1%} |"
        for r in rows
    ]
    lines += [
        "",
        f"Learning diagnostics passed: {diagnostics_passed}.",
        "",
        "Missing comparisons: " + (", ".join(missing) or "none"),
        "",
        "```json",
        json.dumps(comparisons, indent=2),
        "```",
    ]
    if rows:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis_row, seed in zip(axes, spec["learner_seeds"]):
            for profile in spec["profiles"]:
                points = read_series(root / f"{profile}-{seed}" / "learning-curve.jsonl")
                for ax, key, divisor in zip(axis_row, ("training_games", "wall_seconds"), (1, 60)):
                    if points:
                        ax.plot(
                            [r[key] / divisor for r in points],
                            [r["macro_win_rate"] for r in points],
                            marker=".",
                            label=profile,
                        )
            for ax, label in zip(
                axis_row, ("Actual completed games (includes overshoot)", "Elapsed minutes")
            ):
                ax.set(
                    xlabel=label,
                    ylabel="Normal-game validation win rate",
                    ylim=(-0.02, 1.02),
                    title=f"Learner {seed}",
                )
                ax.legend()
        fig.tight_layout()
        fig.savefig(root / "comparison-curves.png", dpi=150)
        plt.close(fig)
        lines += ["", "![Validation curves](comparison-curves.png)"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run_comparison(output, *, minutes=120, games=10000, diagnostics_only=False):
    from .training import train
    from .training_requirements import require_cuda_training

    require_cuda_training(sc2_profile("E"))

    if not math.isfinite(minutes) or not 0 < minutes <= 120 or games < 1:
        raise ValueError("Use positive games and at most 120 minutes per run")
    root, spec = Path(output), settings()
    root.mkdir(parents=True, exist_ok=False)
    diagnostic = {}
    try:
        for seed in spec["learner_seeds"]:
            cfg = sc2_profile("E")
            cfg["training"].update(
                total_games=spec["diagnostic_games"],
                max_minutes=min(minutes, spec["diagnostic_minutes"]),
            )
            cfg["visualization"]["demos"] = False
            run = root / f"diagnostic-{seed}"
            train(cfg, "masked", seed, run)
            probes = read_series(run / "curriculum-probes.jsonl")
            passed = {p["stage"] for p in probes if p["advanced"]}
            diagnostic[str(seed)] = {
                "passed": {"placement", "saving"} <= passed,
                "advanced_stages": sorted(passed),
            }
            write_json(root / "diagnostics.json", diagnostic)
        if not diagnostics_only:
            for i, seed in enumerate(spec["learner_seeds"]):
                order = spec["profiles"] if i == 0 else list(reversed(spec["profiles"]))
                for profile in order:
                    cfg = sc2_profile(profile)
                    cfg["training"].update(max_minutes=minutes, total_games=games)
                    train(cfg, "masked", seed, root / f"{profile}-{seed}")
    finally:
        comparison_report(root)
    return read_json(root / "comparison.json")
