"""Pinned-kernel adapter for per-episode sky income in mixed CUDA batches.

Only the sky payment becomes a per-game input. Snapshot rules carry the effective
amount so CPU verification and recordings are self-contained. No upstream file
is edited, and arbitrary combat-rule changes remain unsupported.
"""

import copy

from pvz_game import Game, Rules
from pvz_game.cuda import CudaBatch

from .lesson_rules import sky_rules


def lesson_kernel_source(source):
    replacements = {
        "double *f;": "double *f;\n  I sky_sun_amount;",
        "income(G_sky_sun_amount, 0);": "income(sky_sun_amount, 0);",
        "I ticks, I per_tick) {": "I ticks, I per_tick, const I *sky_amounts) {",
        "facts + i * 9};": "facts + i * 9, sky_amounts[i]};",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError(
                "Pinned CUDA sky-income hook changed; verify the engine installation"
            )
        source = source.replace(before, after)
    return source


class LessonCudaBatch(CudaBatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_rules = [self.rules] * self.n
        self._sunless_rules = sky_rules(self.rules, False)
        self._sky_amounts = self.cp.full(self.n, self.rules.game["sky_sun_amount"], self.cp.int64)
        self.source = lesson_kernel_source(self.source)
        self.module = self.cp.RawModule(code=self.source, options=("--std=c++11", "--fmad=false"))
        self._step_kernel = self.module.get_function("step_games")
        self._mask = self.module.get_function("legal_masks")
        self._step = self._step_with_income

    def _step_with_income(self, grid, block, args):
        self._step_kernel(grid, block, (*args, self._sky_amounts))

    def reset(self, levels, seeds, *, indices=None, allowed=None, digging=None, natural_sun=None):
        indices = self._indices(indices)
        enabled = [True] * len(indices) if natural_sun is None else list(natural_sun)
        if len(enabled) != len(indices) or any(type(x) is not bool for x in enabled):
            raise ValueError("One natural_sun boolean is required per reset game")
        if len(levels) != len(indices) or len(seeds) != len(indices):
            raise ValueError("One level and seed required for every reset index")
        snapshots = []
        for level, seed, sky in zip(levels, seeds, enabled):
            game = Game(self.rules if sky else self._sunless_rules)
            game.reset(level, seed)
            snapshots.append(game.snapshot())
        self.restore(snapshots, indices=indices, allowed=allowed, digging=digging)

    def restore(self, snapshots, *, indices=None, allowed=None, digging=None):
        indices = self._indices(indices)
        normalized, rules = [], []
        for snapshot in snapshots:
            rule = Rules(snapshot["rules"])
            if rule.digest not in (self.rules.digest, self._sunless_rules.digest):
                raise ValueError(
                    "CUDA lessons may only disable sky sun; combat rules must stay pinned"
                )
            Game(rule).restore(snapshot)
            raw = copy.deepcopy(snapshot)
            raw.update(rules=self.rules.to_dict(), rules_hash=self.rules.digest)
            normalized.append(raw)
            rules.append(rule)
        # Parent validates the entire batch/capacities before touching device state.
        super().restore(normalized, indices=indices, allowed=allowed, digging=digging)
        self._sky_amounts[self.cp.asarray(indices, dtype=self.cp.int64)] = self.cp.asarray(
            [r.game["sky_sun_amount"] for r in rules], dtype=self.cp.int64
        )
        for index, rule in zip(indices, rules):
            self._episode_rules[index] = rule

    def snapshot(self, index):
        raw = super().snapshot(index)
        rule = self._episode_rules[index]
        raw.update(rules=rule.to_dict(), rules_hash=rule.digest)
        return raw

    def observe(self, index):
        return Game(self._episode_rules[index]).restore(self.snapshot(index))
