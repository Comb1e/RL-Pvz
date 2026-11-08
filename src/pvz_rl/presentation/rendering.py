"""Thin adapter for the game's public offscreen renderer; no local board drawing."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def board_renderer(size):
    from pvz_game.rendering import BoardRenderer

    return BoardRenderer(size=tuple(size))


def render_context(values=None):
    from pvz_game.rendering import RenderContext

    from pvz_rl.envs.tile_scenarios import is_placement_skill

    values = values or {}
    outcome = values.get("outcome")
    message = values.get("message", values.get("action", ""))
    task = values.get("task")
    reward = values.get("reward")
    if (
        isinstance(reward, (int, float))
        and outcome != "scored"
        and task != "prediction"
        and not is_placement_skill(task)
    ):
        message = " | ".join(
            filter(None, (reward_label(reward, values.get("reward_pending")), message))
        )
    return RenderContext(
        policy_id=values.get("policy_id"),
        # A scored prediction scene ends the exercise while its game is still running.
        outcome=None if outcome == "scored" else outcome,
        termination_reason=values.get("termination_reason"),
        message=message,
    )


def reward_label(value, pending=False):
    """Bare signed return. Prediction and placement boards omit it."""
    if value is None:
        return ""
    return f"{value:+.1f}{'~' if pending else ''}"


def render_observation(obs, *, context=None):
    """Preserve Gym's writable (600, 1000, 3) uint8 RGB-array contract."""
    frame = board_renderer((1000, 600)).rgb_frame(obs, context=render_context(context))
    return np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3).copy()
