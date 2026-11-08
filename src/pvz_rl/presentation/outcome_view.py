"""Raw post-action field/probe predictions from the decision journal."""

import numpy as np


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
                label = font.render("1" if value >= 0.5 else "0", True, (16, 26, 20))
                surface.blit(label, label.get_rect(center=cell.center))
    pygame.draw.line(
        surface,
        (210, 220, 200),
        (rect.x + rect.w // width, rect.y),
        (rect.x + rect.w // width, rect.bottom),
        1,
    )


def draw_live_outcomes(surface, frame, rect):
    """Latest prediction beside the board, with no new policy inference."""
    import pygame

    old_clip = surface.get_clip()
    surface.set_clip(old_clip.clip(rect))
    pygame.draw.rect(surface, (24, 39, 32), rect, border_radius=4)
    font = pygame.font.SysFont("Segoe UI", 13 if rect.w >= 230 else 11)
    decision = frame.get("outcome_decision") or frame.get("decision") or {}
    has_effects = bool(decision.get("effect_prediction_valid", False))
    surface.blit(
        font.render(
            "Post-action outcomes",
            True,
            (237, 231, 188),
        ),
        (rect.x + 8, rect.y + 5),
    )
    rate = max(1, frame["observation"].tick_rate)
    stamp = (
        f"Decision #{decision['sequence']} @ {decision['tick'] / rate:.2f}s"
        if has_effects
        else "Awaiting decision"
    )
    surface.blit(font.render(stamp, True, (184, 205, 186)), (rect.x + 8, rect.y + 23))
    panels = []
    if has_effects:
        suffix = "Wait" if decision.get("executed_action") == 0 else "proposal"
        panels.extend(
            zip(
                (f"Static field after {suffix}", f"Probe after {suffix}"),
                decision["effect_prediction"],
                strict=True,
            )
        )
    top_margin = 74
    height = max(10, (rect.h - top_margin - 5 - 23 * len(panels)) // max(1, len(panels)))
    grid = pygame.Rect(rect.x + 20, rect.y + top_margin, max(10, rect.w - 28), height)
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
    for index, (title, values) in enumerate(panels):
        top = grid.y + index * (height + 23)
        if title:
            title_top = top - (34 if index == 0 else 18)
            surface.blit(font.render(title, True, (237, 231, 188)), (rect.x + 8, title_top))
        _draw_grid(surface, values, pygame.Rect(grid.x, top, grid.w, height), font)
    if not has_effects:
        message = "Field/probe predictions not recorded"
        surface.blit(
            font.render(message, True, (184, 205, 186)),
            (rect.x + 8, rect.bottom - 39),
        )
    legend = (
        "H = home reached | 1 = depth reaches that cell"
        if rect.w >= 250
        else "H = home | 1 = reached"
    )
    surface.blit(font.render(legend, True, (184, 205, 186)), (rect.x + 8, rect.bottom - 20))
    surface.set_clip(old_clip)
