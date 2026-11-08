"""Persistent device-only static forks with uncensored per-lane terminal handling."""

from functools import partial
from weakref import proxy

import numpy as np
from pvz_game.config import PLANT_TYPES
from pvz_game.cuda import CudaBatch
from pvz_game.cuda.schema import HEADER, MOWER_STATES, PLANT_STATES

from pvz_rl.envs.outcome_math import OUTCOME_SHAPE, OUTCOME_SIZE, SHOOTER_KINDS

_SETTLING_CHECK = r"""
__device__ bool forecast_settled(const Header *h, const Plant *p, const Mower *m) {
  if (h->nq) return false;
  for (I row = 0; row < 5; row++)
    if (m[row].state == FORECAST_MOWER_MOVING) return false;
  for (I i = 0; i < h->np; i++) {
    const Plant &a = p[i];
    if (a.kind == FORECAST_CHOMPER && a.state != FORECAST_PLANT_READY) return false;
    if (a.burst_due && (FORECAST_SHOOTERS & (1 << a.kind))) return false;
  }
  return true;
}
"""

_METHODS = r"""
  __device__ bool forecast_settling() {
    return forecast_phase[0] < 2 && h->status && !forecast_settled(h, p, m);
  }
  __device__ void forecast_position(Zombie &a) {
    if (a.headless || a.health <= 0) return;
    I code = hi(0, 9 - hi(0, floor_div(a.x, G_units_per_tile)));
    I offset = forecast_phase[0] == 0 ? 0 : 2;
    I index = a.row * FORECAST_TERMS + offset;
    forecast[index] = hi(forecast[index], code);
  }
  __device__ void forecast_finish() {
    for (I j = 0; j < h->nz; j++) forecast_position(z[j]);
    I lost[5] = {0,0,0,0,0};
    for (I j = 0; j < h->nz; j++)
      if (!z[j].headless && z[j].health > 0 && z[j].x <= G_house_x) lost[z[j].row] = 1;
    for (I r = 0; r < 5; r++) if (lost[r]) {
      if (forecast_phase[0] == 0) {
        I index = r * FORECAST_TERMS;
        forecast[index] = 10;
        if (forecast[index + 1] < 0) forecast[index + 1] = h->tick - forecast_start[0];
      } else forecast[r * FORECAST_TERMS + 2] = 10;
    }
    I active = 0;
    for (I j = 0; j < h->nz;) {
      if (lost[z[j].row]) {
        for (I k = j; k + 1 < h->nz; k++) z[k] = z[k + 1];
        h->nz--;
      } else {
        if (!z[j].headless && z[j].health > 0) active++;
        j++;
      }
    }
    h->status = active ? 0 : 1;
  }
"""


def static_fork_source(source):
    """Fail closed when any of the pinned source contracts changes."""
    terminal = """    for (I i = 0; i < h->nz; i++)
      if (!z[i].headless && z[i].x <= G_house_x)
        h->status = 2;
    if (!h->status && h->defeated == h->total_spawns)
      h->status = 1;
    if (h->status)
      emit(17, 0, h->status);"""
    hooks = {
        "struct Sim {": _SETTLING_CHECK + "\nstruct Sim {",
        "double *f;": "double *f; I *forecast, *forecast_phase, *forecast_start; double sky_sun_rate;",
        "I ticks, I per_tick) {": "I ticks, I per_tick, I *forecast, I *forecast_phase, I *forecast_start, double sky_sun_rate) {",
        "facts + i * 9};": "facts + i * 9, forecast + i * FORECAST_SIZE, forecast_phase + i, forecast_start + i, sky_sun_rate};",
        "    if (DIAGNOSTIC) {": "    if (false) {",  # no log/event publication, even in diagnostics
        "  __device__ I random(I minimum, I maximum) {": _METHODS
        + "\n  __device__ I random(I minimum, I maximum) {",
        "      a.headless = 1;": "      forecast_position(a);\n      a.headless = 1;",
        "    I hp = a.health, ar = a.armor;": "    forecast_position(a);\n    I hp = a.health, ar = a.armor;",
        "    a.x = end;\n  }\n  __device__ void zombies()": "    a.x = end;\n    forecast_position(a);\n  }\n  __device__ void zombies()",
        "        if (h->tick < a.vault_until) continue;": "        forecast_position(a);\n        if (h->tick < a.vault_until) continue;",
        "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;": "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;\n        forecast_position(a);",
        "    plants();\n    projectiles();": "    for (I j = 0; j < h->nz; j++) forecast_position(z[j]);\n    plants();\n    projectiles();",
        "    if (!h->enabled || h->status)\n      return;": "    if (!h->enabled || (h->status && !(ticks && !action && forecast_settling())))\n      return;",
        "    for (I i = 0; i < ticks && !h->status; i++)": "    for (I i = 0; i < ticks && (!h->status || forecast_settling()); i++)",
        "    if (per_tick && action && h->accepted)\n      return;": "    forecast_finish();\n    if (per_tick && action && h->accepted)\n      return;",
        "          + random(0, G_sky_interval_jitter_ticks);": "          + random(0, G_sky_interval_jitter_ticks);\n      h->sky_due = h->tick + hi(1, (I)floor((double)(h->sky_due - h->tick) / sky_sun_rate));",
        terminal: "    forecast_finish();",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned static-fork hook changed: {before[:70]}")
        source = source.replace(before, after)
    # Probe initialization uses the engine's own spawn construction and gait RNG.
    source += r"""
extern "C" __global__ void start_probes(
    I *headers, const I *plants, I *zombies, const I *mowers, I *phases, I kind) {
  I i = blockIdx.x;
  if (threadIdx.x) return;
  Header *h = reinterpret_cast<Header *>(headers + i * GAME_HEADER_WIDTH);
  const Plant *p = reinterpret_cast<const Plant *>(plants + i * 45 * GAME_PLANT_WIDTH);
  const Mower *m = reinterpret_cast<const Mower *>(mowers + i * 5 * GAME_MOWER_WIDTH);
  if (!h->enabled || !h->status || phases[i] >= 2 || !forecast_settled(h, p, m)) return;
  Zombie *z = reinterpret_cast<Zombie *>(zombies + i * ZCAP * GAME_ZOMBIE_WIDTH);
  for (I row = 0; row < 5; row++) {
    Zombie a = {};
    a.id = h->next_id++; a.kind = kind; a.row = row;
    a.x = G_cols * G_units_per_tile; a.health = ZH[kind]; a.armor = ZA[kind];
    unsigned int x = (unsigned int)h->gameplay_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    a.speed = ZS[kind] + x % (ZSMAX[kind] - ZS[kind] + 1);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    a.gait = x % 2 == 0 ? 1 : 0;
    h->gameplay_rng = x;
    z[h->nz++] = a;
  }
  h->status = 0;
  phases[i] = 2;
}
"""
    shooters = sum(1 << PLANT_TYPES.index(kind) for kind in SHOOTER_KINDS)
    constants = {
        "MOWER_MOVING": MOWER_STATES.index("moving"),
        "CHOMPER": PLANT_TYPES.index("chomper"),
        "PLANT_READY": PLANT_STATES.index("ready"),
        "SHOOTERS": shooters,
        "TERMS": OUTCOME_SHAPE[1],
        "SIZE": OUTCOME_SIZE,
    }
    return "".join(f"#define FORECAST_{key} {value}\n" for key, value in constants.items()) + source


class StaticForkBatch(CudaBatch):
    """Inputs clone D2D; advance() is bounded and snapshot() preserves both phases.

    phase: 0 field (including settling), 2 probe, 3 completed.
    """

    def __init__(self, *args, sky_sun_rate=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.forecast = self.cp.zeros((self.n, *OUTCOME_SHAPE), self.cp.int64)
        self.forecast_phase = self.cp.zeros(self.n, self.cp.int64)
        self.forecast_start = self.cp.zeros(self.n, self.cp.int64)
        self.sky_sun_rate = sky_sun_rate
        self.source = static_fork_source(self.source)
        self.module = self.cp.RawModule(code=self.source, options=("--std=c++11", "--fmad=false"))
        self._step_kernel = self.module.get_function("step_games")
        self._probe_kernel = self.module.get_function("start_probes")
        self._step = partial(type(self)._forecast_step, proxy(self))

    def _forecast_step(self, grid, block, args):
        self._step_kernel(
            grid,
            block,
            (
                *args,
                self.forecast,
                self.forecast_phase,
                self.forecast_start,
                self.cp.float64(self.sky_sun_rate),
            ),
        )

    def start(self, source, indices, actions=None):
        """Copy physical slabs only, never borrow mutable storage or an RNG object."""
        cp = self.cp
        ix = cp.asarray(indices, cp.int64)
        count = len(ix)
        if count > self.n or source.zcap > self.zcap or source.qcap > self.qcap:
            raise ValueError("Static fork storage cannot hold the requested roots")
        self.header[:, HEADER.index("enabled")] = 0
        for name in ("header", "plants", "zombies", "projectiles", "mowers", "cooldowns"):
            dest, original = getattr(self, name), getattr(source, name)
            dest[:count] = 0
            if original.ndim == 3:
                dest[:count, : original.shape[1]] = original[ix]
            else:
                dest[:count] = original[ix]
        self.initialize(count, actions)

    def initialize(self, count, actions=None):
        cp = self.cp
        self.count = count
        self.forecast.fill(0)
        self.forecast[:, :, 1] = -1
        self.forecast_phase.fill(0)
        self.forecast_start[:] = self.header[:, 0]
        self.header[:count, HEADER.index("status")] = 0
        self.header[:count, HEADER.index("enabled")] = 1
        self.header[:count, HEADER.index("total_spawns")] = self.header[
            :count, HEADER.index("spawn_index")
        ]
        self._event_counts.fill(0)
        commands = cp.zeros(self.n, cp.int64)
        if actions is not None:
            commands[:count] = cp.asarray(actions, cp.int64)
        # The trusted low-level kernel supports zero ticks: apply exactly once,
        # without making Wait consume fictitious time or cooldown.
        self._step(
            (self.n,),
            (32,),
            (
                self.header,
                self.plants,
                self.zombies,
                self.projectiles,
                self.mowers,
                self.cooldowns,
                self._schedules,
                self._events,
                self._event_counts,
                self.facts,
                commands,
                self.n,
                0,
                1,
            ),
        )
        self.accepted = self.header[:count, HEADER.index("accepted")].copy()

    def advance(self, ticks=None, probe_kind=2):
        cp = self.cp
        self._probe_kernel(
            (self.n,),
            (32,),
            (
                self.header,
                self.plants,
                self.zombies,
                self.mowers,
                self.forecast_phase,
                cp.int64(probe_kind),
            ),
        )
        self.step_device(
            cp.zeros(self.n, cp.int64),
            ticks=self.max_step_ticks if ticks is None else ticks,
            per_tick=False,
        )
        done = (self.forecast_phase == 2) & (self.header[:, 1] == 1)
        self.forecast_phase[done] = 3
        self.header[done, HEADER.index("enabled")] = 0
        return self.forecast_phase[: self.count] == 3

    def snapshot(self):
        names = (
            "header",
            "plants",
            "zombies",
            "projectiles",
            "mowers",
            "cooldowns",
            "forecast",
            "forecast_phase",
            "forecast_start",
        )
        return dict(
            count=self.count,
            sky_sun_rate=self.sky_sun_rate,
            arrays={name: getattr(self, name)[: self.count].get() for name in names},
        )

    def restore(self, state):
        if state["sky_sun_rate"] != self.sky_sun_rate:
            raise ValueError("Static fork sky-sun settings differ from recovery")
        if not 0 <= state["count"] <= self.n:
            raise MemoryError(
                "Pending static-fork panel needs more resident lanes; free GPU memory and resume"
            )
        expected = {
            "header",
            "plants",
            "zombies",
            "projectiles",
            "mowers",
            "cooldowns",
            "forecast",
            "forecast_phase",
            "forecast_start",
        }
        if set(state["arrays"]) != expected:
            raise ValueError("Incomplete static-fork physical recovery")
        self.header[:, HEADER.index("enabled")] = 0
        for name, value in state["arrays"].items():
            dest = getattr(self, name)
            if dest[: state["count"]].shape != value.shape or value.dtype != np.int64:
                raise ValueError("Static fork recovery has incompatible storage")
            dest[: state["count"]] = self.cp.asarray(value)
        self.count = state["count"]
