"""Simulation-only outcome arithmetic; safe in CPU worker processes."""

import numpy as np

ABSOLUTE_SIZES = (11, 6, 11)  # Field depth, breach time, probe depth.
OUTCOME_SHAPE = (5, len(ABSOLUTE_SIZES))
OUTCOME_SIZE = OUTCOME_SHAPE[0] * OUTCOME_SHAPE[1]
SHOOTER_KINDS = ("peashooter", "snow_pea", "repeater")


def depth_code(x, units=1000, house=-500):
    if x <= house:
        return 10
    return max(0, min(9, 9 - max(0, int(x) // units)))


def time_bin(ticks, tau):
    """The right-open thresholds make an exact tau breach part of bin 1."""
    ticks = np.asarray(ticks)
    return np.where(
        ticks < 0, 5, np.searchsorted(np.array([1, 2, 4, 8]) * tau, ticks, side="right")
    ).astype(np.int8)
