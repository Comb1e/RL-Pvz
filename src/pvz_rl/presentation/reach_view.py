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
                for lane, cleared in enumerate(sample.get("cleared_lanes", [])):
                    if cleared:
                        ax.text(0, lane, "cleared: no remaining spawns", color="black", fontsize=8)
    name = f"{stage}-reach-maps"
    fig.tight_layout()
    save(fig, output, name)
    legacy = any(any(sample.get("conservative_lanes", [])) for sample in examples)
    return (
        f"{stage.replace('_', ' ').title()} reach: prediction and supervision target; "
        + "cleared lanes are marked; "
        + ("historical conservative full-lane targets are marked; " if legacy else "")
        + "grey lanes have no targets",
        name + ".png",
    )


def _draw_grid(surface, values, rect, font=None):
    import pygame

    for row in range(5):
        for col in range(9):
            value = float(values[row][col])
            value = min(1.0, max(0.0, value)) if np.isfinite(value) else np.nan
            color = (
                (70, 80, 75)
                if not np.isfinite(value)
                else (int(70 + 180 * value), int(185 - 120 * value), 65)
            )
            cell = pygame.Rect(
                rect.x + col * rect.w // 9,
                rect.y + row * rect.h // 5,
                rect.w // 9 - 1,
                rect.h // 5 - 1,
            )
            pygame.draw.rect(surface, color, cell)
            if font is not None and cell.w >= 28 and cell.h >= 16 and np.isfinite(value):
                label = font.render(f"{value:.0%}", True, (16, 26, 20))
                surface.blit(label, label.get_rect(center=cell.center))


def draw_live_reach(surface, frame, rect):
    """Latest collection-time prediction beside the board, independent of history browsing."""
    import pygame

    old_clip = surface.get_clip()
    surface.set_clip(old_clip.clip(rect))
    pygame.draw.rect(surface, (24, 39, 32), rect, border_radius=4)
    font = pygame.font.SysFont("Segoe UI", 13 if rect.w >= 230 else 11)
    decision = frame.get("reach_decision") or frame.get("decision") or {}
    surface.blit(
        font.render("Predicted penetration", True, (237, 231, 188)), (rect.x + 8, rect.y + 5)
    )
    valid = decision.get("reach_valid")
    rate = max(1, frame["observation"].tick_rate)
    stamp = (
        f"Decision #{decision['sequence']} @ {decision['tick'] / rate:.2f}s"
        if valid
        else "Awaiting tile decision"
    )
    surface.blit(font.render(stamp, True, (184, 205, 186)), (rect.x + 8, rect.y + 23))
    grid = pygame.Rect(rect.x + 20, rect.y + 56, max(9, rect.w - 28), max(10, rect.h - 80))
    axis_font = pygame.font.SysFont("Segoe UI", max(7, min(11, grid.h // 5 - 2)))
    for col in range(9):
        label = font.render(str(col + 1), True, (184, 205, 186))
        surface.blit(
            label, label.get_rect(midbottom=(grid.x + (col + 0.5) * grid.w / 9, grid.y - 1))
        )
    for row in range(5):
        if axis_font.get_height() <= grid.h // 5:
            label = axis_font.render(chr(65 + row), True, (184, 205, 186))
            surface.blit(
                label, label.get_rect(midright=(grid.x - 3, grid.y + (row + 0.5) * grid.h / 5))
            )
    values = decision["reach_prediction"] if valid else np.full((5, 9), np.nan)
    _draw_grid(surface, values, grid, font)
    legend = "Home <  |  green 0% - red 100%" if rect.w >= 230 else "Home <  |  0% - 100%"
    surface.blit(font.render(legend, True, (184, 205, 186)), (rect.x + 8, rect.bottom - 20))
    surface.set_clip(old_clip)


def draw_reach(surface, font, decision, x, y, width):
    """Focused-view factual target for the selected decision; prediction stays beside the game."""
    import pygame

    if not decision or not decision.get("reach_valid"):
        return 0
    title = (
        "Reach target (historical full-lane label)"
        if decision.get("reach_conservative", 0)
        else "Reach target (grey = pending / no targets)"
    )
    surface.blit(font.render(title, True, (224, 231, 216)), (x, y))
    grid = pygame.Rect(x, y + 21, min(324, width), 70)
    _draw_grid(surface, factual_map(decision.get("reach_labels", [-1] * 5)), grid)
    for row in range(5):
        if decision.get("reach_conservative", 0) & (1 << row):
            pygame.draw.rect(surface, (245, 245, 235), (x, grid.y + row * 14, grid.w, 14), 1)
    cleared = ", ".join(
        chr(65 + r) for r in range(5) if decision.get("reach_cleared", 0) & (1 << r)
    )
    if cleared:
        label = font.render(f"Lanes {cleared} cleared: no remaining spawns", True, (224, 231, 216))
        surface.blit(label, (x, grid.bottom + 5))
        return 120
    return 99
