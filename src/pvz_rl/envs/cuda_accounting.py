"""Read-only reward accounting hooks for the pinned CUDA simulator."""

from copy import copy
from functools import partial
from weakref import proxy

from pvz_game.cuda import CudaBatch
from pvz_game.cuda.schema import EVENTS

ACCOUNTING_WIDTH = 11
EVENT_CODES = {name: index for index, (name, _) in enumerate(EVENTS)}


def accounting_kernel_source(source):
    replacements = {
        "double *f;": "double *f;\n  I *accounting;\n  I sun_numerator, sun_denominator;",
        "I ticks, I per_tick) {": "I ticks, I per_tick, I *accounting, I sun_numerator, I sun_denominator) {",
        "facts + i * 9};": f"facts + i * 9, accounting + i * {ACCOUNTING_WIDTH}, sun_numerator, sun_denominator}};",
        "    emit(5, a.id, source, dh, da);": "    if (source > 0) accounting[0] += dh + da;\n    emit(5, a.id, source, dh, da);",
        "    emit(4, id, source, h->sun - before, amount);": "    accounting[source == 0 ? 1 : 2] += h->sun - before;\n"
        "    if (source == 1 && h->total_spawns > 0 && sun_denominator * h->spawn_index < sun_numerator * h->total_spawns) accounting[10] += h->sun - before;\n"
        "    emit(4, id, source, h->sun - before, amount);",
        "    *ne = 0;": f"    *ne = 0;\n    for (I j = 0; j < {ACCOUNTING_WIDTH}; j++) accounting[j] = 0;",
        "    if (DIAGNOSTIC) {": (
            f"    if (kind == {EVENT_CODES['SunProduced']}) accounting[3] += b;\n"
            f"    if (kind == {EVENT_CODES['PlantPlaced']}) {{ accounting[4] += PC[a]; accounting[8]++; }}\n"
            f"    if (kind == {EVENT_CODES['ZombieSpawned']}) accounting[5]++;\n"
            f"    if (kind == {EVENT_CODES['ZombieDefeated']}) accounting[6]++;\n"
            f"    if (kind == {EVENT_CODES['ZombieRemoved']}) accounting[7]++;\n"
            f"    if (kind == {EVENT_CODES['PlantRemoved']}) accounting[9]++;\n"
            "    if (DIAGNOSTIC) {"
        ),
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError("Pinned CUDA accounting hook changed; verify the engine")
        source = source.replace(before, after)
    return source


class AccountingCudaBatch(CudaBatch):
    def __init__(self, *args, sun_spawn_fraction=None, **kwargs):
        from pvz_rl.config import load_config

        super().__init__(*args, **kwargs)
        self.sun_spawn_fraction = tuple(
            sun_spawn_fraction or load_config()["reward"]["early_sun_spawn_fraction"]
        )
        self.accounting = self.cp.zeros((self.n, ACCOUNTING_WIDTH), self.cp.int64)
        self.source = accounting_kernel_source(self.source)
        self.module = self.cp.RawModule(code=self.source, options=("--std=c++11", "--fmad=false"))
        self._step_kernel = self.module.get_function("step_games")
        self._mask = self.module.get_function("legal_masks")
        self._step = partial(type(self)._step_with_accounting, proxy(self))

    def compact_view(self, count):
        """Alias an occupied scratch prefix without changing the allocation owner."""
        if type(count) is not int or not 0 < count <= self.n:
            raise ValueError("Compact batch size must be a positive integer within the allocation")
        view = copy(self)
        view.n = count
        for name in (
            "header",
            "plants",
            "zombies",
            "projectiles",
            "mowers",
            "cooldowns",
            "_schedules",
            "_events",
            "_event_counts",
            "facts",
            "masks",
            "accounting",
        ):
            setattr(view, name, getattr(self, name)[:count])
        view._templates = self._templates[:count]
        view._step = partial(type(view)._step_with_accounting, proxy(view))
        return view

    def _step_with_accounting(self, grid, block, args):
        self._step_kernel(
            grid, block, (*args, self.accounting, *map(self.cp.int64, self.sun_spawn_fraction))
        )
