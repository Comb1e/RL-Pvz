"""Reward accounting and scenario-local sky timing for the pinned simulator."""

from copy import copy
from functools import partial
from weakref import proxy

from pvz_game.cuda import CudaBatch

from pvz_rl.envs.sky_sun import validate_sky_sun_rate

# damage, sky income, plant-produced income, early income, then bite damage
# split by the eight plant kinds.  The extra facts stay in the private
# accounting slab and never enter model observations.
ACCOUNTING_WIDTH = 12


def accounting_kernel_source(source):
    replacements = {
        "double *f;": "double *f;\n  I *accounting;\n  I sun_numerator, sun_denominator;\n  double sky_sun_rate;",
        "I ticks, I per_tick) {": "I ticks, I per_tick, I *accounting, I sun_numerator, I sun_denominator, double sky_sun_rate) {",
        "facts + i * 9};": f"facts + i * 9, accounting + i * {ACCOUNTING_WIDTH}, sun_numerator, sun_denominator, sky_sun_rate}};",
        "          + random(0, G_sky_interval_jitter_ticks);": "          + random(0, G_sky_interval_jitter_ticks);\n"
        "      h->sky_due = h->tick + hi(1, (I)floor((double)(h->sky_due - h->tick) / sky_sun_rate));",
        "    emit(5, a.id, source, dh, da);": "    if (source > 0) accounting[0] += dh + da;\n    emit(5, a.id, source, dh, da);",
        "        emit(14, b.id, a.id, d);": "        accounting[4 + b.kind] += d;\n        emit(14, b.id, a.id, d);",
        "    emit(4, id, source, h->sun - before, amount);": "    accounting[source == 0 ? 1 : 2] += h->sun - before;\n"
        "    if (source == 1 && h->total_spawns > 0 && sun_denominator * h->spawn_index < sun_numerator * h->total_spawns) accounting[3] += h->sun - before;\n"
        "    emit(4, id, source, h->sun - before, amount);",
        "    *ne = 0;": f"    *ne = 0;\n    for (I j = 0; j < {ACCOUNTING_WIDTH}; j++) accounting[j] = 0;",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError("Pinned CUDA accounting hook changed; verify the engine")
        source = source.replace(before, after)
    return source


class AccountingCudaBatch(CudaBatch):
    def __init__(self, *args, sun_spawn_fraction=None, **kwargs):
        import torch

        from pvz_rl.config import load_config

        kwargs.setdefault("device", torch.cuda.current_device())
        super().__init__(*args, **kwargs)
        self.sun_spawn_fraction = tuple(
            sun_spawn_fraction or load_config()["reward"]["early_sun_spawn_fraction"]
        )
        self.sky_sun_rate = 1.0
        self.accounting = self.cp.zeros((self.n, ACCOUNTING_WIDTH), self.cp.int64)
        self.source = accounting_kernel_source(self.source)
        from pvz_rl.envs.reach_tracking import forecast_kernel_source

        self.source = forecast_kernel_source(self.source, self.max_step_ticks)
        self.reach_enabled = False
        self.reach_trace = self.cp.zeros(1, self.cp.int64)
        self.reach_error = self.cp.zeros(self.n, self.cp.int64)
        self.module = self.cp.RawModule(code=self.source, options=("--std=c++11", "--fmad=false"))
        self._step_kernel = self.module.get_function("step_games")
        self._mask = self.module.get_function("legal_masks")
        self._step = partial(type(self)._step_with_accounting, proxy(self))

    def set_sky_sun_rate(self, rate):
        self.sky_sun_rate = validate_sky_sun_rate(rate, self.rules)

    def enable_reach(self, saved=None):
        from pvz_rl.envs.reach_tracking import FORECAST_HEADER

        shape = (self.n, FORECAST_HEADER + (self.max_step_ticks + 1) * 5)
        if saved is None:
            self.reach_trace = self.cp.zeros(shape, self.cp.int64)
        else:
            if tuple(saved.shape) != shape:
                raise ValueError("Reach recovery capacity differs from simulator")
            self.reach_trace = self.cp.asarray(saved)
        self.reach_error.fill(0)
        self.reach_enabled = True

    def disable_reach(self):
        self.reach_enabled = False
        self.reach_trace = self.cp.zeros(1, self.cp.int64)
        self.reach_error.fill(0)

    def reset(self, levels, seeds, *, indices=None, **kwargs):
        indices = self._indices(indices)
        super().reset(levels, seeds, indices=indices, **kwargs)
        if self.reach_enabled:
            self.reach_trace[self.cp.asarray(indices)] = 0
            self.reach_error[self.cp.asarray(indices)] = 0
        if self.sky_sun_rate != 1.0:
            from pvz_game.cuda.schema import HEADER

            ix = self.cp.asarray(indices)
            due = HEADER.index("sky_due")
            self.header[ix, due] = self.cp.maximum(
                1, self.cp.floor(self.header[ix, due] / self.sky_sun_rate)
            ).astype(self.cp.int64)

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
        view.reach_error = self.reach_error[:count]
        if self.reach_enabled:
            view.reach_trace = self.reach_trace[:count]
        view._step = partial(type(view)._step_with_accounting, proxy(view))
        return view

    def _step_with_accounting(self, grid, block, args):
        self._step_kernel(
            grid,
            block,
            (
                *args,
                self.accounting,
                *map(self.cp.int64, self.sun_spawn_fraction),
                self.cp.float64(self.sky_sun_rate),
                self.reach_trace,
                self.reach_error,
                self.cp.int64(self.reach_enabled),
            ),
        )
