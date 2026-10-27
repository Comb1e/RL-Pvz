"""Bounded FP64 population moments merged from centered chunks."""

import numpy as np


class RunningMoments:
    def __init__(self):
        self.count, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not values.size:
            return
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite return diagnostic")
        count = values.size
        mean = float(values.mean())
        centered = values - mean
        m2 = float(np.square(centered).sum())
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    @property
    def variance(self):
        return max(0.0, self.m2 / self.count) if self.count else 0.0
