"""Pre-action field, breach, and probe drawn as three bands. No captions."""

import numpy as np


def _depth_color(value, hue):
    if not np.isfinite(value):
        return (55, 64, 60)
    dark, light = hue
    return tuple(int(dark[channel] + (light[channel] - dark[channel]) * value) for channel in range(3))


def _draw_depth(surface, values, rect, hue):
    import pygame

    width = 10
    for row in range(5):
        for col in range(width):
            raw = values[row][col] if values is not None else np.nan
            try:
                value = float(raw)
            except (TypeError, ValueError, IndexError):
                value = np.nan
            cell = pygame.Rect(
                rect.x + col * rect.w // width,
                rect.y + row * rect.h // 5,
                rect.w // width - 1,
                rect.h // 5 - 1,
            )
            pygame.draw.rect(surface, _depth_color(value, hue), cell)
    if values is None:
        return
    pygame.draw.line(
        surface,
        (232, 236, 214),
        (rect.x + rect.w // width, rect.y),
        (rect.x + rect.w // width, rect.bottom),
        2,
    )


def _draw_breach(surface, classes, rect):
    import pygame

    # Five growing bins, then a gap, then the detached "none" pip.
    gap = max(4, rect.w // 18)
    span = rect.w - gap
    known = classes is not None
    for row in range(5):
        chosen = -1
        if known:
            try:
                chosen = int(classes[row])
            except (TypeError, ValueError, IndexError):
                chosen = -1
        for col in range(6):
            if col < 5:
                cell = pygame.Rect(
                    rect.x + col * (span * 5 // 6) // 5,
                    rect.y + row * rect.h // 5,
                    (span * 5 // 6) // 5 - 1,
                    rect.h // 5 - 1,
                )
            else:
                cell = pygame.Rect(
                    rect.x + span * 5 // 6 + gap,
                    rect.y + row * rect.h // 5,
                    span // 6 - 1,
                    rect.h // 5 - 1,
                )
            on = known and 0 <= chosen < 5 and col <= chosen
            none = known and chosen == 5 and col == 5
            color = (214, 164, 72) if on or none else (42, 52, 48)
            pygame.draw.rect(surface, color, cell)


def _columns(surface, rect, columns, font):
    if font is None:
        return
    for col, text in enumerate(columns):
        label = font.render(str(text), True, (168, 190, 172))
        surface.blit(
            label,
            label.get_rect(midbottom=(rect.x + (col + 0.5) * rect.w / len(columns), rect.y - 3)),
        )


def _rows(surface, rect, font):
    if font is None or font.get_height() > rect.h // 5:
        return
    for row in range(5):
        label = font.render(chr(65 + row), True, (168, 190, 172))
        surface.blit(
            label, label.get_rect(midright=(rect.x - 3, rect.y + (row + 0.5) * rect.h / 5))
        )


def draw_live_outcomes(surface, frame, rect):
    """Field, breach track, and probe. Color and position identify each band."""
    import pygame

    old_clip = surface.get_clip()
    surface.set_clip(old_clip.clip(rect))
    pygame.draw.rect(surface, (24, 39, 32), rect, border_radius=4)
    decision = frame.get("outcome_decision") or frame.get("decision") or {}
    has_effects = bool(decision.get("effect_prediction_valid", False))
    grids = decision.get("effect_prediction") if has_effects else None
    breach = decision.get("breach_prediction") if decision.get("breach_prediction_valid") else None
    axis = pygame.font.SysFont("Segoe UI", max(8, min(12, rect.w // 28)))
    gutter, head = 16, axis.get_height() + 6
    inner = rect.inflate(-8, -8)
    body = pygame.Rect(inner.x + gutter, inner.y + head, max(10, inner.w - gutter), max(10, inner.h - head))
    gap = 6
    field_h = max(10, int(body.h * 0.46))
    breach_h = max(8, int(body.h * 0.16))
    probe_h = max(10, body.h - field_h - breach_h - 2 * gap)
    field_rect = pygame.Rect(body.x, body.y, body.w, field_h)
    breach_rect = pygame.Rect(body.x, field_rect.bottom + gap, body.w, breach_h)
    probe_rect = pygame.Rect(body.x, breach_rect.bottom + gap, body.w, probe_h)
    _columns(surface, field_rect, ("H", *range(1, 10)), axis)
    _rows(surface, field_rect, axis)
    _draw_depth(surface, None if grids is None else grids[0], field_rect, ((22, 120, 70), (190, 220, 140)))
    _rows(surface, breach_rect, axis)
    _draw_breach(surface, breach, breach_rect)
    _rows(surface, probe_rect, axis)
    _draw_depth(surface, None if grids is None else grids[1], probe_rect, ((28, 70, 100), (150, 210, 230)))
    surface.set_clip(old_clip)
