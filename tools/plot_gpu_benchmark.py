"""Draw measured medians and individual repetitions; never invent error bars."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source, output):
    rows = json.loads(Path(source).read_text("utf-8"))
    profiles = list(dict.fromkeys(r["profile"] for r in rows))
    historical_labels = {
        "cpu-current": "CPU · 8 × 512",
        "cpu-equivalent": "CPU · 8 × 128",
        "cuda-equivalent": "CUDA · 8 × 128",
    }
    labels = []
    for name in profiles:
        row = next(r for r in rows if r["profile"] == name)
        labels.append(
            f"{row['simulator'].upper()} · {row['n_envs']} × {row['rollout_steps_per_env']}"
            if all(k in row for k in ("simulator", "n_envs", "rollout_steps_per_env"))
            else historical_labels.get(name, name)
        )
    groups = [
        [r["decisions_per_second"] for r in rows if r["profile"] == p and r["state"] == "complete"]
        for p in profiles
    ]
    if any(len(g) != 3 for g in groups):
        raise ValueError("This comparison needs three complete repetitions per profile")
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    medians = [statistics.median(g) for g in groups]
    ax.barh(labels, medians, color="#4c8ec1", height=0.64)
    for i, values in enumerate(groups):
        ax.scatter(
            values,
            [i - 0.15, i, i + 0.15],
            s=25,
            facecolors="white",
            edgecolors="#273444",
            zorder=3,
        )
        ax.text(max(values) + 190, i, f"{medians[i]:,.0f}", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(medians) * 1.2)
    ax.set_xlabel("Training decisions per second · collection + optimization")
    ax.set_title(
        "PVZ training throughput on RTX 4070 Laptop GPU", loc="left", weight="bold", pad=15
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    fig.suptitle(
        "Bars: medians · dots: three repetitions · labels: games × decisions per rollout\nSetup, warmup, validation and export excluded; identical PPO settings",
        y=0.01,
        fontsize=9,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.source, args.output)
