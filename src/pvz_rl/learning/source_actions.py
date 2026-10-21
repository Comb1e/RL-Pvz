"""Stable host-only action identity; physical storage locations are private."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceActionRef:
    env: int
    episode: int
    decision: int

    def __post_init__(self):
        if any(type(value) is not int or value < 0
               for value in (self.env, self.episode, self.decision)):
            raise ValueError("Action identity fields must be nonnegative integers")


@dataclass(frozen=True)
class AppendReceipt:
    slots: np.ndarray
    rows: np.ndarray
    episodes: np.ndarray
    decisions: np.ndarray
    environments: np.ndarray

    def reference(self, index):
        return SourceActionRef(int(self.environments[index]), int(self.episodes[index]),
                               int(self.decisions[index]))
