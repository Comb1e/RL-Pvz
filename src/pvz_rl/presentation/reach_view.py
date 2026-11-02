"""Prediction and factual penetration maps, never planting-action values."""

import numpy as np


def factual_map(labels):
    labels = np.asarray(labels)
    cells = np.arange(9)[None] >= labels[:, None]
    return np.where(labels[:, None] >= 0, cells.astype(float), np.nan)


def report_reach_maps(metrics, output, save):
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
            ("Predicted penetration", "Factual visible-target penetration"),
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
    save(fig, output, "normal-reach-maps")
    return (
        "Visible-zombie reach: probability versus observed outcome (grey lanes have no targets)",
        "normal-reach-maps.png",
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
            ("Factual (pending lanes grey)", factual),
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
    return 21 + 5 * cell_h + 8
