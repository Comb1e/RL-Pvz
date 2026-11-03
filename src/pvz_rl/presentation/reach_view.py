"""Raw future-cell forecasts and masked factual/snapshot targets."""

import numpy as np


def factual_map(labels, homes=None):
    values = np.asarray(labels, dtype=float)
    if values.shape == (5,):  # Historical report data only.
        values = np.where(values[:, None] >= 0, np.arange(9)[None] >= values[:, None], np.nan)
    else:
        values = np.where(values >= 0, values, np.nan)
    if homes is None:
        return values
    homes = np.asarray(homes, dtype=float)
    return np.column_stack((np.where(homes >= 0, homes, np.nan), values))


def predicted_map(decision):
    return np.column_stack(
        (
            decision.get("home_prediction", [np.nan] * 5),
            decision.get("reach_prediction", np.full((5, 9), np.nan)),
        )
    )


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
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 2.8 * len(examples)), squeeze=False)
    for pair, sample in zip(axes, examples, strict=True):
        values = (
            np.column_stack((sample.get("home_prediction", [np.nan] * 5), sample["prediction"])),
            factual_map(sample["labels"], sample.get("home_labels", [-1] * 5)),
        )
        for ax, grid, title in zip(
            pair,
            values,
            ("Predicted future reach", sample.get("provenance", "Reach target")),
            strict=True,
        ):
            ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax.set(
                title=f"{title} — env {sample['env']}, tick {sample['tick']}",
                xticks=range(10),
                xticklabels=["Home", *range(1, 10)],
                yticks=range(5),
                yticklabels=list("ABCDE"),
                xlabel="Home entry | board cells; grey = unknown",
            )
    name = f"{stage}-reach-maps"
    fig.tight_layout()
    save(fig, output, name)
    return (
        f"{stage.replace('_', ' ').title()} future reach and home danger: raw predictions / masked targets",
        name + ".png",
    )


def _draw_grid(surface, values, rect, font=None):
    import pygame

    width = len(values[0])
    for row in range(5):
        for col in range(width):
            value = float(values[row][col])
            color = (
                (70, 80, 75)
                if not np.isfinite(value)
                else (int(70 + 180 * value), int(185 - 120 * value), 65)
            )
            cell = pygame.Rect(
                rect.x + col * rect.w // width,
                rect.y + row * rect.h // 5,
                rect.w // width - 1,
                rect.h // 5 - 1,
            )
            pygame.draw.rect(surface, color, cell)
            if font is not None and cell.w >= 28 and cell.h >= 16 and np.isfinite(value):
                label = font.render(f"{value:.0%}", True, (16, 26, 20))
                surface.blit(label, label.get_rect(center=cell.center))
    pygame.draw.line(
        surface,
        (210, 220, 200),
        (rect.x + rect.w // width, rect.y),
        (rect.x + rect.w // width, rect.bottom),
        1,
    )


def draw_live_reach(surface, frame, rect):
    """Latest prediction beside the board, with no new policy inference."""
    import pygame

    old_clip = surface.get_clip()
    surface.set_clip(old_clip.clip(rect))
    pygame.draw.rect(surface, (24, 39, 32), rect, border_radius=4)
    font = pygame.font.SysFont("Segoe UI", 13 if rect.w >= 230 else 11)
    decision = frame.get("reach_decision") or frame.get("decision") or {}
    valid = decision.get("reach_valid")
    surface.blit(
        font.render("Predicted future reach", True, (237, 231, 188)), (rect.x + 8, rect.y + 5)
    )
    rate = max(1, frame["observation"].tick_rate)
    stamp = (
        f"Decision #{decision['sequence']} @ {decision['tick'] / rate:.2f}s"
        if valid
        else "Awaiting tile decision"
    )
    surface.blit(font.render(stamp, True, (184, 205, 186)), (rect.x + 8, rect.y + 23))
    grid = pygame.Rect(rect.x + 20, rect.y + 56, max(10, rect.w - 28), max(10, rect.h - 80))
    axis_font = pygame.font.SysFont("Segoe UI", max(7, min(11, grid.h // 5 - 2)))
    for col, text in enumerate(("H", *map(str, range(1, 10)))):
        label = axis_font.render(text, True, (184, 205, 186))
        surface.blit(
            label, label.get_rect(midbottom=(grid.x + (col + 0.5) * grid.w / 10, grid.y - 1))
        )
    for row in range(5):
        if axis_font.get_height() <= grid.h // 5:
            label = axis_font.render(chr(65 + row), True, (184, 205, 186))
            surface.blit(
                label, label.get_rect(midright=(grid.x - 3, grid.y + (row + 0.5) * grid.h / 5))
            )
    _draw_grid(surface, predicted_map(decision) if valid else np.full((5, 10), np.nan), grid, font)
    legend = "H = home danger | green 0% - red 100%" if rect.w >= 250 else "H = home | 0% - 100%"
    surface.blit(font.render(legend, True, (184, 205, 186)), (rect.x + 8, rect.bottom - 20))
    surface.set_clip(old_clip)


def draw_reach(surface, font, decision, x, y, width):
    import pygame

    if not decision or not decision.get("reach_valid"):
        return 0
    kind = (
        "snapshot constraints"
        if decision.get("reach_provenance") == 1
        else "factual future / safe constraints"
    )
    surface.blit(font.render(f"Reach target: {kind}", True, (224, 231, 216)), (x, y))
    surface.blit(
        font.render("Home | 1–9; grey = unknown / pending", True, (184, 205, 186)), (x, y + 20)
    )
    grid = pygame.Rect(x, y + 42, min(360, width), 70)
    _draw_grid(
        surface,
        factual_map(
            decision.get("reach_labels", np.full((5, 9), -1)), decision.get("home_labels", [-1] * 5)
        ),
        grid,
    )
    return 120
