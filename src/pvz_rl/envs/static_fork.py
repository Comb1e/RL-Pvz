"""CPU verification fork. Own all mutable state, silence events, suppress spawns.

Production collection uses the matching device adapter. This intentionally
changes terminal handling only inside the owned fork: a lost lane is removed,
and the other lanes continue. Installed engine code is never modified.
"""

from dataclasses import replace
from functools import lru_cache

import numpy as np
from pvz_game import Dig, Game, Place, Status
from pvz_game.engine import _Zombie

from pvz_rl.envs.outcome_math import depth_code
from pvz_rl.envs.scratch_game import fork_game, restored_game


def reference_tau(rules):
    """Reference tile-crossing time, including both Basic gait animations."""
    basic, game = rules.zombies["basic"], rules.game
    return _reference_tau(game["units_per_tile"], basic["speed"], basic["max_speed"])


@lru_cache(maxsize=16)
def _reference_tau(units, minimum, maximum):
    from pvz_game.mechanics import gait_step

    durations = []
    for gait in (0, 1):
        zombie = _Zombie(1, "basic", 0, 9 * units, 270, 0, "walking")
        zombie.speed, zombie.gait = (minimum + maximum) // 2, gait
        distance = ticks = 0
        while distance < 9 * units:
            distance += gait_step(zombie, units, False)
            ticks += 1
        durations.append(ticks)
    return max(1, int(np.ceil(sum(durations) / 18)))


def apply_action(game, action):
    """The engine placement/Dig operation, without inventing a time tick."""
    accepted = True
    if action:
        tile, species = (action - 1) % 45, (action - 1) // 45
        from pvz_game.config import PLANT_TYPES

        command = (
            Dig(tile // 9, tile % 9)
            if species == 8
            else Place(PLANT_TYPES[species], tile // 9, tile % 9)
        )
        accepted = game.validate_action(command).accepted
        if accepted:
            if species == 8:
                game._remove_plant(game._tiles[tile // 9, tile % 9], "dug")
            else:
                kind = PLANT_TYPES[species]
                game._sun -= game.rules.plants[kind]["cost"]
                game._cooldowns[kind] = game.rules.plants[kind]["recharge_ticks"] + 1
                game._add_plant(kind, tile // 9, tile % 9)
    return accepted


class StaticFork:
    def __init__(self, game, *, action=0, probe="conehead", sky_sun_rate=1.0, field_only=False):
        # Public snapshot boundary validates subclasses as well as external input.
        source = restored_game(game.snapshot(), game.rules) if type(game) is not Game else game
        self.game = fork_game(source)
        self.phase, self.start, self.probe = "field", source._tick, probe
        self.field_ticks = self.probe_ticks = 0
        self.field_only, self.anchor_tick = field_only, -1
        self.sky_sun_rate = sky_sun_rate
        self.labels = np.zeros((5, 4), np.int64)
        self.labels[:, 1] = -1
        self.game._events = []
        self.game._emit = lambda *args, **kwargs: None
        self.game._level = replace(
            self.game._level, spawns=self.game._level.spawns[: self.game._spawn_index]
        )
        self.game._status = Status.RUNNING
        original_damage = self.game._damage
        original_move = self.game._move_zombie

        def damage(zombie, *args, **kwargs):
            self._record(zombie)
            return original_damage(zombie, *args, **kwargs)

        def move(zombie, end):
            self._record(zombie)
            result = original_move(zombie, end)
            self._record(zombie)
            return result

        self.game._damage, self.game._move_zombie = damage, move
        self.accepted = apply_action(self.game, action)
        self._scan()
        self._resolve_breaches()

    def _record(self, zombie):
        if not zombie.headless and zombie.health > 0:
            column = 0 if self.phase == "field" else 3
            # Home is certified only after mower resolution, in advance().
            code = min(
                9, depth_code(zombie.x, self.game._g["units_per_tile"], self.game._g["house_x"])
            )
            self.labels[zombie.row, column] = max(self.labels[zombie.row, column], code)

    def _scan(self):
        for zombie in self.game._zombies.values():
            self._record(zombie)

    def _resolve_breaches(self):
        breached = {
            z.row
            for z in self.game._zombies.values()
            if not z.headless and z.health > 0 and z.x <= self.game._g["house_x"]
        }
        for row in breached:
            if self.phase == "field":
                self.labels[row, 0] = 10
                if self.labels[row, 1] < 0:
                    self.labels[row, 1] = self.game._tick - self.start
                self.labels[row, 2] = 1
            else:
                self.labels[row, 3] = 10
        if breached:
            self.game._zombies = {
                k: z for k, z in self.game._zombies.items() if z.row not in breached
            }

    def _probe(self):
        self.anchor_tick = self.game._tick
        self.phase = "probe"
        self.game._status = Status.RUNNING
        spec, g = self.game.rules.zombies[self.probe], self.game._g
        for row in range(5):
            z = _Zombie(
                self.game._id(),
                self.probe,
                row,
                9 * g["units_per_tile"],
                spec["health"],
                spec["armor"],
                "walking",
            )
            self.game._zombies[z.id] = z
            self.game._start_walk(z)
        self._scan()

    def advance(self, ticks=256):
        """Bounded resumable unit; no simulated-time failure is fabricated."""
        for _ in range(ticks):
            if not any(not z.headless and z.health > 0 for z in self.game._zombies.values()):
                if self.phase == "probe" or self.field_only:
                    self.phase = "complete"
                    return True
                if all(mower.state != "moving" for mower in self.game._mowers):
                    self._probe()
            self.game._status = Status.RUNNING
            drops = self.game._sky_drops
            if self.phase == "field":
                self.field_ticks += 1
            else:
                self.probe_ticks += 1
            self.game._advance()
            if self.game._sky_drops != drops and self.sky_sun_rate != 1:
                self.game._sky_due = self.game._tick + max(
                    1, int((self.game._sky_due - self.game._tick) / self.sky_sun_rate)
                )
            self._scan()
            self._resolve_breaches()
            self.game._status = Status.RUNNING
        return False

    def run(self):
        while not self.advance():
            pass
        return self.labels.copy()
