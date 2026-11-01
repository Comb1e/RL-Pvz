"""Owned forks of validated pinned-engine games; combat remains engine code."""

import random
from copy import copy
from functools import lru_cache

from pvz_game import Game, Rules
from pvz_game.config import canonical_hash

# Fail closed if the pinned engine introduces state we have not classified.
_SHARED = {"rules", "_g", "_level"}
_SCALARS = {
    "_ready",
    "_seed",
    "_tick",
    "_status",
    "_sun",
    "_gameplay_rng",
    "_sky_drops",
    "_sky_due",
    "_next_id",
    "_spawn_index",
    "_defeated",
    "_wave",
    "_total_waves",
}
_OWNED = {
    "_rng",
    "_cooldowns",
    "_plants",
    "_zombies",
    "_projectiles",
    "_tiles",
    "_mowers",
    "_events",
}
_OPTIONAL = {"_previous_x"}


@lru_cache(maxsize=1)
def scratch_rules():
    return Rules()


class EngineSnapshot(dict):
    """Process-local provenance for snapshots captured from engine-owned games.

    Ordinary dictionaries, including restored user/checkpoint inputs, still go
    through Game.restore. This marker is only minted by the generation path.
    """

    @classmethod
    def capture(cls, game):
        if type(game) is not Game:
            raise TypeError("Expected an engine-owned Game")
        return cls._seal(game.snapshot())

    @classmethod
    def _seal(cls, snapshot):
        # Internal only: CPU capture or owned CUDA snapshot download.
        result = cls(snapshot)
        result._captured_digest = canonical_hash(result)
        return result

    def unchanged(self):
        return getattr(self, "_captured_digest", None) == canonical_hash(self)


def scratch_game(rules=None):
    """Share already validated immutable Rules, without parsing them again."""
    rules = scratch_rules() if rules is None else rules
    if not isinstance(rules, Rules):
        raise TypeError("Scratch games require validated engine Rules")
    game = Game.__new__(Game)
    game.rules, game._g, game._ready = rules, rules.game, False
    return game


def restored_game(snapshot, rules=None):
    game = scratch_game(rules)
    # The public engine validator runs once at the external snapshot boundary.
    game.restore(snapshot)
    return game


def set_initial_mowers(game, ready_rows):
    """Select only ready/spent carts on a freshly reset owned scratch game."""
    if type(game) is not Game or not game._ready or game._tick != 0:
        raise ValueError("Initial mower states require a freshly reset engine game")
    rows = set(ready_rows)
    if any(type(row) is not int or not 0 <= row < game.rules.game["rows"] for row in rows):
        raise ValueError("Initial mower rows must be within the board")
    for mower in game._mowers:
        mower.state = "ready" if mower.row in rows else "spent"


def fork_game(game):
    """Copy every mutable field; never share RNGs, entities, or event buffers."""
    game._require_ready()
    if type(game) is not Game or set(vars(game)) - _OPTIONAL != _SHARED | _SCALARS | _OWNED:
        raise RuntimeError("Pinned engine scratch state changed; classify fields before forking")
    branch = scratch_game(game.rules)
    for name in _SHARED | _SCALARS:
        setattr(branch, name, getattr(game, name))
    branch._rng = random.Random(0)
    branch._rng.setstate(game._rng.getstate())
    branch._cooldowns = game._cooldowns.copy()
    branch._tiles = game._tiles.copy()
    for name in ("_plants", "_zombies", "_projectiles"):
        setattr(branch, name, {key: copy(value) for key, value in getattr(game, name).items()})
    branch._mowers = [copy(value) for value in game._mowers]
    branch._events = list(game._events)
    if hasattr(game, "_previous_x"):
        branch._previous_x = game._previous_x.copy()
    return branch
