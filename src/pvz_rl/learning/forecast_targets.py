"""Public snapshot constraints and factual, partially observed lane outcomes."""

import numpy as np

from pvz_rl.learning.forecast_labels import time_bin


def snapshot_targets(row):
    """Never turn the zombie lane into a safe-depth or no-breach example."""
    row["threat_depth_allowed"], row["threat_time_allowed"] = 0, 0
    row["threat_depth"], row["threat_time"], row["threat_loss"] = -1, -1, -1
    row["threat_censored"] = False
    if row["forecast_pending"] == 0:
        empty = row["lane_zombies"] == 0
        row["threat_depth_allowed"][empty] = 1
        row["threat_time_allowed"][empty] = 1 << 5
        row["threat_depth"][empty], row["threat_time"][empty] = 0, 5
        row["threat_loss"][empty | row["forecast_mowers"]] = 0
    row["reach_provenance"] = 4
    target_maps(row)


def factual_targets(row, trace, homes, finished, won, tau):
    tick, end = int(row["tick"]), len(trace) - 1
    initial_safe = (row["forecast_pending"] == 0) & (row["lane_zombies"] == 0)
    finished = np.asarray(finished, bool) | initial_safe
    row["threat_depth_allowed"], row["threat_time_allowed"] = 0, 0
    row["threat_depth"], row["threat_time"], row["threat_loss"] = -1, -1, -1
    for horizon, stop in enumerate((tick + 2 * tau, tick + 9 * tau, end)):
        observed = trace[tick : min(stop, end) + 1].max(0)
        for lane, depth in enumerate(observed):
            depth = int(depth)
            exact = won or (horizon < 2 and stop <= end) or finished[lane] or depth == 10
            row["threat_depth"][lane, horizon] = depth
            row["threat_censored"][lane, horizon] = not exact
            row["threat_depth_allowed"][lane, horizon] = (
                1 << depth if exact else 2047 ^ ((1 << depth) - 1)
            )
    elapsed = end - tick
    for lane in range(5):
        if homes[lane] >= tick:
            target = int(time_bin(homes[lane] - tick, tau))
            row["threat_time_allowed"][lane] = 1 << target
            row["threat_time"][lane], row["threat_loss"][lane] = target, 1
        elif won or finished[lane]:
            row["threat_time_allowed"][lane] = 1 << 5
            row["threat_time"][lane], row["threat_loss"][lane] = 5, 0
        else:
            # No event through elapsed: only bins containing a later integer tick
            # and the no-breach class remain possible. Never fabricate survival.
            allowed = (1 << 4) | (1 << 5)
            for code, upper in enumerate((tau, 2 * tau, 4 * tau, 8 * tau)):
                if upper > elapsed + 1:
                    allowed |= 1 << code
            row["threat_time_allowed"][lane] = allowed
    row["reach_provenance"] = 3
    target_maps(row)


def target_maps(row):
    """Display only conclusions supported by the full allowed-class set."""
    row["reach_labels"] = -1
    row["home_labels"] = row["threat_loss"]
    for lane, bits in enumerate(row["threat_depth_allowed"][:, 2]):
        if not bits or bits == 2047:
            continue
        support = [k for k in range(11) if int(bits) & (1 << k)]
        for column in range(9):
            threshold = 9 - column
            if min(support) >= threshold:
                row["reach_labels"][lane, column] = 1
            elif max(support) < threshold:
                row["reach_labels"][lane, column] = 0
