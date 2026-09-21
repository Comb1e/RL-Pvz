"""Generate a standalone research report and figures from completed evaluations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import read_rows
from .provenance import write_json
from .statistics import bootstrap_interval, paired_difference, result_matrix


def make_report(paths, output, cfg, learning_paths=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = read_rows(paths)
    if not rows:
        raise ValueError("No results supplied")
    protocols = {r.get("protocol_hash") for r in rows}
    if len(protocols) != 1 or None in protocols:
        game_protocols = {r.get("game_protocol_hash") for r in rows}
        if len(game_protocols) != 1 or None in game_protocols:
            raise ValueError(
                "Evaluations must have the same verified observation, reward and game protocol, or an explicit shared game protocol for profile comparisons"
            )
    settings = {
        "replicates": cfg["evaluation"]["bootstrap_replicates"],
        "seed": cfg["evaluation"]["bootstrap_seed"],
    }
    groups = {}
    for row in rows:
        key = (row["split"], row["family"], row["level"], row["policy"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (split, family, level, policy), group in sorted(groups.items()):
        if len({r.get("training_config_hash") for r in group}) != 1:
            raise ValueError("Do not pool different training configurations under one policy name")
        matrix, _, _ = result_matrix(group)
        entry = {
            "split": split,
            "family": family,
            "level": level,
            "policy": policy,
            **bootstrap_interval(matrix, **settings),
        }
        reference = groups.get((split, family, level, "heuristic"))
        if reference and policy != "heuristic":
            entry["difference_vs_heuristic"] = paired_difference(group, reference, **settings)
        summaries.append(entry)
    write_json(output / "statistics.json", summaries)
    lines = [
        "# PVZ reinforcement-learning evaluation",
        "",
        "These results describe the supplied checkpoints and episodes only. "
        "Development and diagnostic results are not final-test evidence.",
        "",
        "| Split / family | Level | Policy | Runs × scenarios | Win rate (95% interval) | Difference vs heuristic |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in summaries:
        difference = r.get("difference_vs_heuristic")
        text = (
            f"{difference['mean']:+.1%} [{difference['low']:+.1%}, {difference['high']:+.1%}]"
            if difference
            else "—"
        )
        lines.append(
            f"| {r['split']} / {r['family']} | {r['level']} | {r['policy']} | "
            f"{r['runs']} × {r['scenarios']} | {r['mean']:.1%} "
            f"[{r['low']:.1%}, {r['high']:.1%}] | {text} |"
        )
    lines.extend(
        [
            "",
            "Intervals resample learner runs and scenario seeds, preserving scenario pairing. "
            "A positive lower bound on the paired difference supports improvement over the heuristic. "
            "Few runs and all-win/all-loss samples can produce misleadingly narrow bootstrap intervals; "
            "inspect sample counts and per-run results before drawing conclusions.",
            "",
            "![Win rates](win-rates.png)",
            "",
        ]
    )
    panels = sorted({(r["split"], r["family"], r["level"]) for r in summaries})
    cols = min(3, len(panels))
    nrows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 5, nrows * 4.2), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        group = [r for r in summaries if (r["split"], r["family"], r["level"]) == panel]
        means = np.array([r["mean"] for r in group])
        low = np.array([r["low"] for r in group])
        high = np.array([r["high"] for r in group])
        ax.bar(range(len(group)), means, color="#46836b")
        ax.errorbar(
            range(len(group)),
            means,
            yerr=[np.maximum(0, means - low), np.maximum(0, high - means)],
            fmt="none",
            color="#253a37",
            capsize=3,
        )
        ax.set_xticks(range(len(group)), [r["policy"] for r in group], rotation=45, ha="right")
        ax.set(ylabel="Win rate", ylim=(0, 1.05), title=" / ".join(panel))
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "win-rates.png", dpi=160)
    plt.close(fig)
    curves = []
    for path in learning_paths:
        curve = read_rows([path])
        if curve:
            curves.append(
                (str(Path(path).parent.parent.name + "/" + Path(path).parent.name), curve)
            )
    if curves:
        units = {r.get("budget_unit", "decisions") for _, curve in curves for r in curve}
        if len(units) != 1:
            raise ValueError("Cannot combine learning curves with game and decision budgets")
        unit = units.pop()
        key = "training_games" if unit == "games" else "training_steps"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for label, curve in curves:
            y = [r["macro_win_rate"] for r in curve]
            axes[0].plot([r[key] for r in curve], y, label=label)
            axes[1].plot([r["wall_seconds"] / 3600 for r in curve], y, label=label)
        for ax, xlabel in zip(axes, (f"Training {unit}", "Wall time (hours)")):
            ax.set(xlabel=xlabel, ylabel="Validation macro win rate", ylim=(-0.02, 1.02))
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "learning-curves.png", dpi=160)
        plt.close(fig)
        lines.extend(["![Validation learning curves](learning-curves.png)", ""])
    lines.extend(
        [
            "## Failure analysis",
            "",
            "House breaches and time cutoffs are recorded separately. Review the predetermined "
            "first winning and failing replays per level to investigate economy, lane coverage, "
            "armor, emergency placement, and mower dependence. These causes require replay analysis; "
            "they are not inferred from survival time alone.",
            "",
            "## Inputs",
            "",
            *[f"- `{Path(p).resolve()}`" for p in paths],
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summaries
