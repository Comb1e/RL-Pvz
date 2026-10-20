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


def branch_availability(sun, cooldowns, costs, *, active=None, card_present=None):
    """Build branch availability without using occupancy or tile geometry."""
    sun = np.asarray(sun, dtype=np.float64).reshape(-1)
    cooldowns = np.asarray(cooldowns, dtype=np.int64)
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    if cooldowns.shape != (len(sun), A.plant_types) or len(costs) != A.plant_types:
        raise ValueError("Branch availability fields have incompatible shapes")
    available = np.zeros((len(sun), A.plant_types + 2), dtype=bool)
    available[:, 0] = True
    available[:, -1] = True
    available[:, 1:-1] = (cooldowns == 0) & (sun[:, None] >= costs[None, :])
    if card_present is not None:
        present = np.asarray(card_present, dtype=bool)
        if present.shape != (len(sun), A.plant_types):
            raise ValueError("card_present must have shape [rows, plant_types]")
        available[:, 1:-1] &= present
    if active is not None:
        active = np.asarray(active, dtype=bool).reshape(-1)
        if active.shape != (len(sun),):
            raise ValueError("active must match the branch availability rows")
        available &= active[:, None]
    return available


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


def classify_actions(actions, accepted, legal_masks, reasons=None, *, branch_available=None):
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
    if branch_available is None:
        available = np.c_[masks[:, 0], masks[:, 1:].reshape(-1, 9, A.tiles).any(-1)]
    else:
        available = np.asarray(branch_available, dtype=bool)
        if available.shape != (len(actions), A.plant_types + 2):
            raise ValueError("branch_available must have shape [rows, 10]")
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


def _outcome_counts(rows):
    """Count the rows that will actually contribute to a fitting pass."""
    counts = np.zeros((3, 2), dtype=np.int64)
    if len(rows):
        groups = np.where(
            rows["action"] == 0,
            0,
            np.where(rows["action"] < A.dig_start, 1, 2),
        )
        np.add.at(counts, (groups, rows["accepted"].astype(np.int64)), 1)
    return counts


def compact_invalid_rows(rows, *, group_size=100):
    """Select fitting representatives for repeated invalid proposals.

    Complete-return accounting keeps every source row.  This helper only plans
    the rows sent through the fitting encoders.  Invalid proposals are grouped
    per environment and episode in their original chronological order.  A
    contiguous run is keyed by action, route and rejection reason and is split
    into windows of ``group_size``; the first row of every window is retained.
    All waits and accepted actions remain selected.
    """
    if type(group_size) is not int or isinstance(group_size, bool) or group_size < 1:
        raise ValueError("invalid fitting group size must be a positive integer")
    rows = np.asarray(rows)
    if rows.ndim != 1:
        raise ValueError("fitting rows must be one-dimensional")
    required = {"action", "accepted", "route", "rejection_reason"}
    missing = required - set(rows.dtype.names or ())
    if missing:
        raise ValueError(f"fitting rows are missing routing fields: {sorted(missing)}")
    active = (
        np.asarray(rows["active"], bool)
        if "active" in rows.dtype.names
        else np.ones(len(rows), dtype=bool)
    )
    route = np.asarray(rows["route"], np.int64)
    invalid = active & np.isin(route, (FitRoute.SPECIES_INVALID, FitRoute.TILE_INVALID))
    selected = np.flatnonzero(active).tolist()
    selected_invalid = set()
    invalid_positions = set(np.flatnonzero(invalid).tolist())
    run_count = 0

    # Collection stores vector-environment rows interleaved by slot.  Grouping
    # each slot independently restores its chronological episode order without
    # changing the source row order used for all other fitting work.
    env_values = (
        np.asarray(rows["env"], np.int64)
        if "env" in rows.dtype.names
        else np.zeros(len(rows), dtype=np.int64)
    )
    reset = (
        np.asarray(rows["reset"], bool)
        if "reset" in rows.dtype.names
        else np.zeros(len(rows), dtype=bool)
    )
    done = (
        np.asarray(rows["done"], bool)
        if "done" in rows.dtype.names
        else np.zeros(len(rows), dtype=bool)
    )
    for env in np.unique(env_values[active]):
        positions = np.flatnonzero(env_values == env)
        run_key = None
        run_length = 0
        previous_done = True
        for position in positions:
            if not active[position]:
                run_key = None
                run_length = 0
                previous_done = True
                continue
            if reset[position] or previous_done:
                run_key = None
                run_length = 0
            previous_done = bool(done[position])
            if not invalid[position]:
                run_key = None
                run_length = 0
                continue
            key = (
                int(rows["action"][position]),
                int(route[position]),
                int(rows["rejection_reason"][position]),
            )
            if key != run_key:
                run_key = key
                run_length = 0
                run_count += 1
            if run_length % group_size == 0:
                selected_invalid.add(int(position))
            run_length += 1

    selected = np.asarray(
        [
            position
            for position in selected
            if position not in invalid_positions or position in selected_invalid
        ],
        dtype=np.int64,
    )
    raw_invalid = int(invalid.sum())
    retained_invalid = len(selected_invalid)
    stats = {
        "raw_rows": int(active.sum()),
        "selected_rows": int(len(selected)),
        "raw_invalid_rows": raw_invalid,
        "retained_invalid_rows": retained_invalid,
        "invalid_runs": int(run_count),
        "invalid_windows": retained_invalid,
        "invalid_reduction": (
            float((raw_invalid - retained_invalid) / raw_invalid) if raw_invalid else 0.0
        ),
    }
    return selected, _outcome_counts(rows[selected]), stats


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
