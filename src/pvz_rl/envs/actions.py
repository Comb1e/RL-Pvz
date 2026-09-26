"""Stable action indices and a mask derived from the engine's legal-action query."""

import numpy as np
from pvz_game import Dig, Game, Place, Wait

from pvz_rl.config import load_config


class ActionSchema:
    """Pinned engine transport and conditional policy dimensions in one place."""

    version = "kind_plant_tile_v1"
    wait, dig, plant = 0, 1, 2
    kinds, plant_types, rows, cols = 3, 8, 5, 9
    tiles = rows * cols
    tile_groups = plant_types + 1
    dig_start = 1 + plant_types * tiles
    size = dig_start + tiles

    @classmethod
    def unpack(cls, actions):
        """Decode device tensors; ignored plant/tile arguments use zero."""
        import torch

        actions = actions.long()
        kinds = torch.where(
            actions == 0, cls.wait, torch.where(actions < cls.dig_start, cls.plant, cls.dig)
        )
        plants = ((actions - 1) // cls.tiles).clamp(0, cls.plant_types - 1)
        plants = torch.where(kinds == cls.plant, plants, 0)
        tiles = torch.where(kinds == cls.wait, 0, (actions - 1) % cls.tiles)
        return kinds, plants, tiles

    @classmethod
    def pack(cls, kinds, plants, tiles):
        import torch

        return torch.where(
            kinds == cls.wait,
            0,
            torch.where(kinds == cls.dig, cls.dig_start + tiles, 1 + plants * cls.tiles + tiles),
        ).long()

    @classmethod
    def tile_masks(cls, masks):
        return masks[..., 1:].reshape(*masks.shape[:-1], cls.tile_groups, cls.tiles)


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
