"""Prediction and factual penetration maps, never planting-action values."""

import numpy as np


def factual_map(labels):
    labels = np.asarray(labels)
    cells = np.arange(9)[None] >= labels[:, None]
    return np.where(labels[:, None] >= 0, cells.astype(float), np.nan)


def report_reach_maps(metrics, output, save, *, stage="normal"):
    import matplotlib.pyplot as plt

    examples = next(
        (
            row.get("reach_tracking", {}).get("examples")
            for row in reversed(metrics)
            if row.get("reach_tracking", {}).get("examples")
        ),
        [],
    )
    if not examples:
        return None
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#78817a")
    fig, axes = plt.subplots(len(examples), 2, figsize=(11, 2.6 * len(examples)), squeeze=False)
    for pair, sample in zip(axes, examples, strict=True):
        for ax, values, title in zip(
            pair,
            (sample["prediction"], factual_map(sample["labels"])),
            ("Predicted penetration", "Reach supervision target"),
            strict=True,
        ):
            ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax.set(
                title=f"{title} — env {sample['env']}, tick {sample['tick']}",
                xticks=range(9),
                xticklabels=range(1, 10),
                yticks=range(5),
                yticklabels=list("ABCDE"),
                xlabel="Column (home is to the left)",
            )
            if ax is pair[1]:
                for lane, conservative in enumerate(sample.get("conservative_lanes", [])):
                    if conservative:
                        ax.text(0, lane, "full-lane failure", color="white", fontsize=8)
    name = f"{stage}-reach-maps"
    save(fig, output, name)
    return (
        f"{stage.replace('_', ' ').title()} reach: prediction and target; full-lane failures are marked; grey lanes have no targets",
        name + ".png",
    )


def draw_reach(surface, font, decision, x, y, width):
    """Compact focused-view pair; returns the occupied vertical height."""
    import pygame

    if not decision or not decision.get("reach_valid"):
        return 0
    cell_w = max(8, min(36, (width - 40) // 18))
    cell_h = 14
    factual = factual_map(decision.get("reach_labels", [-1] * 5))
    for panel, (title, values) in enumerate(
        (
            ("Predicted penetration", decision["reach_prediction"]),
            ("Reach target (may use full-lane failure)", factual),
        )
    ):
        left = x + panel * (9 * cell_w + 22)
        surface.blit(font.render(title, True, (224, 231, 216)), (left, y))
        for row in range(5):
            for col in range(9):
                value = float(values[row][col])
                color = (
                    (70, 80, 75)
                    if not np.isfinite(value)
                    else (int(70 + 180 * value), int(185 - 120 * value), 65)
                )
                pygame.draw.rect(
                    surface,
                    color,
                    (left + col * cell_w, y + 21 + row * cell_h, cell_w - 1, cell_h - 1),
                )
            if panel and decision.get("reach_conservative", 0) & (1 << row):
                pygame.draw.rect(
                    surface, (245, 245, 235), (left, y + 21 + row * cell_h, 9 * cell_w, cell_h), 1
                )
    return 21 + 5 * cell_h + 8
