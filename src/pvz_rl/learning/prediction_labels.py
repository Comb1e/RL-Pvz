"""One static result and one probe label for a prediction episode.

Breach delays stay in simulator ticks. The shared outcome coder bins them.
"""

import numpy as np


def prediction_label_action(done, timed_out, lost, breached=False):
    """Keep forecasting, drop the episode, or probe the quiet board.

    A house entry stays on the static target and still receives the probe.
    The cutoff and a real loss publish nothing.
    """
    del breached
    if not done:
        return "keep"
    if timed_out or lost:
        return "drop"
    return "probe"


def accumulate_static(depth, home, trace, breaches):
    """Fold one guarded step into the episode's deepest code and first home tick."""
    depth[:] = np.maximum(depth, np.asarray(trace).reshape(-1, depth.shape[0]).max(axis=0))
    breaches = np.asarray(breaches)
    replace = (breaches >= 0) & ((home < 0) | (breaches < home))
    home[replace] = breaches[replace]


def freeze_static(depth, home, origin):
    """Field class 10 marks a home entry. The delay is ticks since the first forecast."""
    field = np.array(depth, dtype=np.int64, copy=True)
    entered = np.asarray(home) >= 0
    field[entered] = 10
    delay = np.where(entered, np.asarray(home) - int(origin), -1).astype(np.int64)
    return field, delay


def write_episode_target(buffer, indices, field, delay, probe):
    """Stamp one target onto existing forecast rows. This stores no new row."""
    target = np.zeros((len(field), 3), np.int64)
    target[:, 0] = field
    target[:, 1] = delay
    target[:, 2] = probe
    for index in indices:
        block, offset = divmod(int(index), buffer.block_rows)
        row = buffer.blocks[block][offset]
        row["baseline_outcomes"] = target
        row["null_target_valid"] = True
        row["outcome_ready"] = True
    return target
