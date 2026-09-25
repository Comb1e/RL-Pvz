"""Thin adapter for the game's public offscreen renderer; no local board drawing."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def board_renderer(size):
    from pvz_game.rendering import BoardRenderer

    return BoardRenderer(size=tuple(size))


def render_context(values=None):
    from pvz_game.rendering import RenderContext

    values = values or {}
    return RenderContext(
        policy_id=values.get("policy_id"),
        outcome=values.get("outcome"),
        termination_reason=values.get("termination_reason"),
        message=values.get("message", values.get("action", "")),
    )


def render_observation(obs, *, context=None):
    """Preserve Gym's writable (600, 1000, 3) uint8 RGB-array contract."""
    frame = board_renderer((1000, 600)).rgb_frame(obs, context=render_context(context))
    return np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3).copy()
