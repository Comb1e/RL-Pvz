"""Public snapshot constraints and factual, partially observed lane outcomes."""

import numpy as np

from pvz_rl.learning.forecast_labels import time_bin

FACTUAL_FIELDS = (
    "threat_depth_allowed",
    "threat_time_allowed",
    "threat_depth",
    "threat_time",
    "threat_loss",
    "threat_censored",
    "reach_labels",
    "home_labels",
    "reach_provenance",
)


def factual_maxima(trace, tau):
    """Exact inclusive 2τ/9τ/end maxima for every tick, using a sparse table."""
    count = len(trace)
    table, stride = [trace], 1
    while stride * 2 <= count:
        previous = table[-1]
        table.append(np.maximum(previous[:-stride], previous[stride:]))
        stride *= 2
    starts = np.arange(count)
    result = np.empty((count, 5, 3), dtype=trace.dtype)
    for horizon, distance in enumerate((2 * tau, 9 * tau, count)):
        ends = np.minimum(starts + distance + 1, count)
        lengths = ends - starts
        levels = np.floor(np.log2(lengths)).astype(np.int64)
        for level in np.unique(levels):
            at = np.flatnonzero(levels == level)
            result[at, :, horizon] = np.maximum(
                table[level][starts[at]], table[level][ends[at] - (1 << int(level))]
            )
    return result


def factual_targets_batch(rows, maxima, homes, finished, won, tau):
    """Array equivalent of factual_targets; preserve all censoring and bitsets."""
    ticks, end = rows["tick"].astype(np.int64), len(maxima) - 1
    depths = maxima[ticks].astype(np.int16)
    safe = (rows["forecast_pending"][:, None] == 0) & (rows["lane_zombies"] == 0)
    done = np.asarray(finished, bool)[None] | safe
    stops = ticks[:, None] + np.array([2 * tau, 9 * tau, end + 1])[None]
    exact = won | ((np.arange(3) < 2)[None, None] & (stops[:, None] <= end))
    exact = exact | done[..., None] | (depths == 10)
    rows["threat_depth"] = depths
    rows["threat_censored"] = ~exact
    rows["threat_depth_allowed"] = np.where(exact, 1 << depths, 2047 ^ ((1 << depths) - 1))
    elapsed = end - ticks
    allowed = np.full((len(rows), 5), (1 << 4) | (1 << 5), np.uint16)
    for code, upper in enumerate((tau, 2 * tau, 4 * tau, 8 * tau)):
        allowed |= np.where(upper > elapsed[:, None] + 1, 1 << code, 0).astype(np.uint16)
    observed = homes[None] >= ticks[:, None]
    survived = ~observed & (won | done)
    bins = time_bin(homes[None] - ticks[:, None], tau)
    rows["threat_time_allowed"] = np.where(observed, 1 << bins, np.where(survived, 1 << 5, allowed))
    rows["threat_time"] = np.where(observed, bins, np.where(survived, 5, -1))
    rows["threat_loss"] = np.where(observed, 1, np.where(survived, 0, -1))
    rows["home_labels"] = rows["threat_loss"]
    rows["reach_provenance"] = 3
    # Final-depth allowed sets are either one exact class or an inclusive tail.
    final = depths[..., 2]
    thresholds = 9 - np.arange(9)
    rows["reach_labels"] = np.where(
        final[..., None] >= thresholds,
        1,
        np.where(exact[..., 2, None], 0, -1),
    )


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
