"""Semantic probe ownership, independent of disposable field-suffix storage."""

from enum import IntEnum

import numpy as np


def interval_probes_enabled(cfg):
    # Missing settings are intentional: old recoveries keep their original targets.
    return cfg["training"]["forecasts"].get("collection_protocol") == "wait_interval_v2"


class ProbeState(IntEnum):
    UNINITIALIZED = 0
    COMPUTING = 1
    READY = 2


class ProbeIntervals:
    def __init__(self, count):
        self.identity = np.zeros(count, np.int64)
        self.state = np.zeros(count, np.int8)
        self.anchor = np.full(count, -1, np.int64)
        self.probe = np.zeros((count, 5), np.int64)

    def end(self, indices):
        self.identity[indices] += 1
        self.state[indices] = ProbeState.UNINITIALIZED
        self.anchor[indices] = -1
        self.probe[indices] = 0

    def begin(self, indices):
        if np.any(self.state[indices] != ProbeState.UNINITIALIZED):
            raise RuntimeError("Only uninitialized intervals can launch a probe")
        self.state[indices] = ProbeState.COMPUTING

    def complete(self, indices, probe, anchor):
        if np.any(self.state[indices] != ProbeState.COMPUTING):
            raise RuntimeError("Probe publication needs a computing interval")
        self.probe[indices], self.anchor[indices] = probe, anchor
        self.state[indices] = ProbeState.READY

    def snapshot(self):
        return dict(
            version=1,
            **{k: getattr(self, k).copy() for k in ("identity", "state", "anchor", "probe")},
        )

    def restore(self, saved):
        if saved is None or saved.get("version") != 1:
            raise ValueError("Missing or incompatible interval probe recovery")
        for name in ("identity", "state", "anchor", "probe"):
            value, target = np.asarray(saved[name]), getattr(self, name)
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError("Invalid interval probe recovery storage")
            target[:] = value
        if (
            np.any(self.identity < 0)
            or np.any((self.state < 0) | (self.state > 2))
            or np.any((self.probe < 0) | (self.probe > 10))
            or np.any((self.state == ProbeState.READY) & (self.anchor < 0))
        ):
            raise ValueError("Invalid interval probe lifecycle or result")
