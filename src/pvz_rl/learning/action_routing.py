"""Fitting attribution from pre-action public legality and the actual outcome.

These routes never change controller selection, simulator rewards or returns.
CPU plans are shared by verified demonstrations and autonomous trajectories.
"""

from enum import IntEnum

import numpy as np
from pvz_game.cuda.schema import REASONS

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.actions import dig_branch_available


class FitRoute(IntEnum):
    WAIT_RETURN = 0
    ACCEPTED = 1
    SPECIES_INVALID = 2
    TILE_INVALID = 3


def branch_availability(
    sun, cooldowns, costs, *, plant_count, active=None, card_present=None, digging=None
):
    """Plant branches ignore geometry; Dig requires at least one plant."""
    sun = np.asarray(sun, dtype=np.float64).reshape(-1)
    cooldowns = np.asarray(cooldowns, dtype=np.int64)
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    if cooldowns.shape != (len(sun), A.plant_types) or len(costs) != A.plant_types:
        raise ValueError("Branch availability fields have incompatible shapes")
    plant_count = np.asarray(plant_count)
    if (
        plant_count.shape != (len(sun),)
        or plant_count.dtype.kind not in "iu"
        or np.any(plant_count < 0)
    ):
        raise ValueError("plant_count must contain nonnegative integer counts for each row")
    available = np.zeros((len(sun), A.plant_types + 2), dtype=bool)
    available[:, 0] = True
    available[:, -1] = dig_branch_available(plant_count)
    if digging is not None:
        digging = np.asarray(digging, dtype=bool).reshape(-1)
        if digging.shape != (len(sun),):
            raise ValueError("digging must match the branch availability rows")
        available[:, -1] = dig_branch_available(plant_count, digging)
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


def _validate_outcomes(actions, accepted, usable, reasons=None):
    """Check evidence without repairing simulator/availability disagreements."""
    if np.any(accepted & ~usable):
        raise ValueError("Accepted proposal disagrees with pre-action branch availability")
    if reasons is None:
        return
    reasons = np.asarray(reasons)
    if reasons.shape != actions.shape:
        raise ValueError("Routing rejection reasons have the wrong shape")
    if reasons.dtype.kind in "OUS":
        try:
            reasons = np.asarray([REASONS.index(r) for r in reasons], dtype=np.int64)
        except ValueError as exc:
            raise ValueError("Unknown trajectory rejection reason") from exc
    if reasons.dtype.kind not in "iu" or np.any((reasons < 0) | (reasons >= len(REASONS))):
        raise ValueError("Unknown trajectory rejection reason")
    if np.any(accepted != (reasons == 0)):
        raise ValueError("Action acceptance contradicts rejection reason")
    branch_failure = np.isin(
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
    # The engine reports empty_tile for both an impossible empty-board Dig
    # and a bad tile on a populated board. Explicit pre-action availability
    # distinguishes species-invalid from tile-invalid; the reason is retained.
    tile_failure = reasons == REASONS.index("occupied_tile")
    if np.any((branch_failure & usable) | (tile_failure & ~usable)):
        raise ValueError("Routing branch availability contradicts rejection reason")
    if np.any(
        ((reasons == REASONS.index("occupied_tile")) & ((actions == 0) | (actions >= A.dig_start)))
        | ((reasons == REASONS.index("empty_tile")) & (actions < A.dig_start))
        | (
            np.isin(reasons, [REASONS.index("card_recharging"), REASONS.index("insufficient_sun")])
            & ((actions == 0) | (actions >= A.dig_start))
        )
        | ((reasons == REASONS.index("task_restriction")) & (actions == 0))
    ):
        raise ValueError("Rejection reason contradicts the proposed action branch")


def classify_actions(actions, accepted, legal_masks, reasons=None, *, branch_available):
    """Return routes from explicit pre-action branch availability and outcomes.

    Legal masks check accepted placements only. Plant branch availability is
    independent of geometry; Dig's explicit availability also requires a plant.
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
    available = np.asarray(branch_available, dtype=bool)
    if available.shape != (len(actions), A.plant_types + 2):
        raise ValueError("branch_available must have shape [rows, 10]")
    usable = available[np.arange(len(actions)), branches]
    _validate_outcomes(actions, accepted, usable, reasons)
    return route_outcomes(actions, accepted, usable), usable


def validate_routes(rows):
    """Validate persisted attribution against executed-row evidence."""
    expected = route_outcomes(rows["action"], rows["accepted"], rows["branch_usable"])
    if not np.array_equal(rows["route"], expected):
        raise ValueError("Fitting route disagrees with recorded outcome/availability")
    active = slice(None)
    if "active" in rows.dtype.names and not rows["active"].all():
        active = rows["active"]
    _validate_outcomes(
        rows["action"][active],
        rows["accepted"][active],
        rows["branch_usable"][active],
        rows["rejection_reason"][active] if "rejection_reason" in rows.dtype.names else None,
    )
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


class RepeatedRowCompactor:
    """First-of-window wait/invalid selection, with bounded per-environment carry."""

    def __init__(self, group_size=10):
        if type(group_size) is not int or group_size < 1:
            raise ValueError("repeated fitting group size must be a positive integer")
        self.group_size, self.state = group_size, {}
        self.counts = np.zeros((3, 2), np.int64)
        self.raw = self.raw_invalid = self.retained_invalid = self.selected = 0
        self.raw_wait = self.retained_wait = self.invalid_runs = self.wait_runs = 0

    def select(self, rows):
        rows = np.asarray(rows)
        required = {"action", "accepted", "route", "rejection_reason"}
        if rows.ndim != 1 or required - set(rows.dtype.names or ()):
            raise ValueError("fitting rows require one-dimensional routing fields")
        names = rows.dtype.names
        active = rows["active"] if "active" in names else np.ones(len(rows), bool)
        environments = rows["env"] if "env" in names else np.zeros(len(rows), np.int64)
        invalid = active & np.isin(rows["route"], (FitRoute.SPECIES_INVALID, FitRoute.TILE_INVALID))
        waits = active & (rows["route"] == FitRoute.WAIT_RETURN) & (rows["action"] == 0)
        compactable = invalid | waits
        keep = active & ~compactable
        for env in np.unique(environments):
            ix = np.flatnonzero(environments == env)
            current = rows[ix]
            live, repeated = active[ix], compactable[ix]
            n = len(ix)
            boundary = np.ones(n, bool)
            keys = np.stack(
                [current[k].astype(np.int64) for k in ("action", "route", "rejection_reason")],
                axis=1,
            )
            if n > 1:
                boundary[1:] = np.any(keys[1:] != keys[:-1], axis=1) | ~repeated[:-1] | ~live[:-1]
            for field in ("reset", "run_break"):
                if field in names:
                    boundary |= current[field]
            if "done" in names:
                boundary[1:] |= current["done"][:-1]
            if "episode" in names:
                boundary[1:] |= current["episode"][1:] != current["episode"][:-1]
            if "decision" in names:
                boundary[1:] |= current["decision"][1:] != current["decision"][:-1] + 1
            prior = self.state.get(int(env))
            length = 0
            if prior is not None:
                key, previous_repeated, done, episode, decision, length = prior
                continuation = previous_repeated and not done and tuple(keys[0]) == key
                continuation &= not any(
                    bool(current[k][0]) for k in ("reset", "run_break") if k in names
                )
                if "episode" in names:
                    continuation &= int(current["episode"][0]) == episode
                if "decision" in names:
                    continuation &= int(current["decision"][0]) == decision + 1
                boundary[0] = not continuation
            starts = np.maximum.accumulate(np.where(boundary, np.arange(n), -length))
            ordinals = np.arange(n) - starts
            keep[ix] |= repeated & (ordinals % self.group_size == 0)
            self.invalid_runs += int(np.count_nonzero(invalid[ix] & boundary))
            self.wait_runs += int(np.count_nonzero(waits[ix] & boundary))
            self.state[int(env)] = (
                tuple(keys[-1]),
                bool(repeated[-1]),
                bool(current["done"][-1]) if "done" in names else False,
                int(current["episode"][-1]) if "episode" in names else 0,
                int(current["decision"][-1]) if "decision" in names else 0,
                int(ordinals[-1] + 1) if repeated[-1] else 0,
            )
        selected = np.flatnonzero(keep)
        self.counts += _outcome_counts(rows[selected])
        self.raw += int(active.sum())
        self.raw_invalid += int(invalid.sum())
        self.retained_invalid += int(np.count_nonzero(keep & invalid))
        self.raw_wait += int(waits.sum())
        self.retained_wait += int(np.count_nonzero(keep & waits))
        self.selected += len(selected)
        return selected

    @property
    def diagnostics(self):
        return dict(
            raw_rows=self.raw,
            selected_rows=self.selected,
            raw_invalid_rows=self.raw_invalid,
            retained_invalid_rows=self.retained_invalid,
            invalid_runs=self.invalid_runs,
            invalid_windows=self.retained_invalid,
            invalid_reduction=(1 - self.retained_invalid / self.raw_invalid)
            if self.raw_invalid
            else 0.0,
            raw_wait_rows=self.raw_wait,
            retained_wait_rows=self.retained_wait,
            wait_runs=self.wait_runs,
            wait_windows=self.retained_wait,
            wait_reduction=(1 - self.retained_wait / self.raw_wait) if self.raw_wait else 0.0,
            fitting_reduction=(1 - self.selected / self.raw) if self.raw else 0.0,
        )


def compact_repeated_rows(rows, *, group_size=10):
    """Shared first-of-window compaction; accepted non-wait rows stay intact."""
    compactor = RepeatedRowCompactor(group_size)
    selected = compactor.select(rows)
    return selected, compactor.counts, compactor.diagnostics


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
