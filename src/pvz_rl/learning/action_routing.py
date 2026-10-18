"""Fitting attribution from pre-action public legality and the actual outcome.

These routes never change controller selection, simulator rewards or returns.
CPU plans are shared by verified demonstrations and autonomous trajectories.
"""

from enum import IntEnum

import numpy as np
from pvz_game.cuda.schema import REASONS

from pvz_rl.envs.actions import ActionSchema as A


class FitRoute(IntEnum):
    WAIT_RETURN = 0
    ACCEPTED = 1
    SPECIES_INVALID = 2
    TILE_INVALID = 3


def route_outcomes(actions, accepted, branch_usable):
    """Branch impossibility takes precedence over a bad placement."""
    actions, accepted, usable = np.broadcast_arrays(
        np.asarray(actions), np.asarray(accepted, bool), np.asarray(branch_usable, bool)
    )
    if np.any((actions < 0) | (actions >= A.size)):
        raise ValueError("Routing requires valid recorded proposals")
    return np.where(
        actions == 0,
        FitRoute.WAIT_RETURN,
        np.where(
            accepted,
            FitRoute.ACCEPTED,
            np.where(usable, FitRoute.TILE_INVALID, FitRoute.SPECIES_INVALID),
        ),
    ).astype(np.uint8)


def classify_actions(actions, accepted, legal_masks, reasons=None):
    """Return routes and availability evidence from full pre-action legal masks.

    Unlike policy proposal geometry, these masks include public card availability,
    sun, cooldown and digging occupancy. The engine outcome remains authoritative
    for task restrictions and terminal/outside-space rejection.
    """
    actions = np.asarray(actions, np.int64).reshape(-1)
    accepted = np.asarray(accepted, bool).reshape(-1)
    masks = np.asarray(legal_masks, bool)
    if masks.shape != (len(actions), A.size) or accepted.shape != actions.shape:
        raise ValueError("Routing outcome and public legality shapes disagree")
    if np.any((actions < 0) | (actions >= A.size)):
        raise ValueError("Routing requires valid recorded proposals")
    branches = np.where(actions == 0, 0, 1 + (actions - 1) // A.tiles)
    if np.any(accepted & (actions != 0) & ~masks[np.arange(len(actions)), actions]):
        raise ValueError("Accepted proposal disagrees with pre-action public legality")
    available = np.c_[masks[:, 0], masks[:, 1:].reshape(-1, 9, A.tiles).any(-1)]
    usable = available[np.arange(len(actions)), branches]
    if reasons is not None:
        reasons = np.asarray(reasons)
        if reasons.shape != actions.shape:
            raise ValueError("Routing rejection reasons have the wrong shape")
        if reasons.dtype.kind in "OUS":
            reasons = np.asarray([REASONS.index(r) for r in reasons])
        impossible = np.isin(
            reasons,
            [
                REASONS.index(r)
                for r in (
                    "task_restriction",
                    "game_finished",
                    "outside_space",
                    "card_recharging",
                    "insufficient_sun",
                )
            ],
        )
        usable &= ~impossible | accepted
    return route_outcomes(actions, accepted, usable), usable


def validate_routes(rows):
    """Validate persisted outcome attribution without allocating row selections."""
    expected = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
    if not np.array_equal(rows["route"], expected):
        raise ValueError("Fitting route disagrees with recorded outcome/availability")
    if "rejection_reason" in rows.dtype.names:
        reasons = rows["rejection_reason"]
        if np.any(reasons >= len(REASONS)):
            raise ValueError("Unknown trajectory rejection reason")
        impossible = np.isin(
            reasons,
            [
                REASONS.index(r)
                for r in (
                    "task_restriction",
                    "game_finished",
                    "outside_space",
                    "card_recharging",
                    "insufficient_sun",
                )
            ],
        )
        if np.any((~rows["accepted"]) & impossible & rows["branch_usable"]):
            raise ValueError("Routing branch availability contradicts rejection reason")
    return rows["route"]


def route_indices(rows):
    """Validate persisted attribution and prepare compact CPU row selections."""
    route = validate_routes(rows)
    return {
        "species": np.flatnonzero(route != FitRoute.TILE_INVALID),
        "tiles": np.flatnonzero((route == FitRoute.ACCEPTED) | (route == FitRoute.TILE_INVALID)),
        "ema": np.flatnonzero(route == FitRoute.ACCEPTED),
    }


def workload_counts(rows):
    plans = route_indices(rows)
    return {
        "routes": {
            route.name.lower(): int(np.count_nonzero(rows["route"] == route)) for route in FitRoute
        },
        "online_species_rows": len(plans["species"]),
        "online_tile_rows": len(plans["tiles"]),
        "ema_tile_rows": len(plans["ema"]),
        "skipped_ema_rows": int(np.count_nonzero(rows["action"] != 0)) - len(plans["ema"]),
    }
