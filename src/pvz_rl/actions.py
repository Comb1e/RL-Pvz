"""Stable action indices and a mask derived from the engine's legal-action query."""

import numpy as np
from pvz_game import Dig, Game, Place, Wait

from .config import load_config


class ActionCodec:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        env = cfg["environment"]
        self.rows, self.cols = env["rows"], env["cols"]
        self.plants = tuple(env["plants"])
        self.actions = (
            (Wait(),)
            + tuple(
                Place(kind, row, col)
                for kind in self.plants
                for row in range(self.rows)
                for col in range(self.cols)
            )
            + tuple(Dig(row, col) for row in range(self.rows) for col in range(self.cols))
        )
        self.indices = {action: i for i, action in enumerate(self.actions)}
        self.size = len(self.actions)

    def decode(self, index: int):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("Action index must be an integer")
        if not 0 <= index < self.size:
            raise ValueError("Action index out of range")
        return self.actions[int(index)]

    def encode(self, action) -> int:
        try:
            return self.indices[action]
        except (KeyError, TypeError) as exc:
            raise ValueError("Action does not belong to this schema") from exc

    def mask(self, game: Game) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.bool_)
        for action in game.legal_actions():
            result[self.indices[action]] = True
        return result
