"""Private tick-boundary snapshots; observational hooks never change the engine."""

from copy import copy

import numpy as np

from pvz_rl.envs.cuda_snapshots import download_snapshots


def wait_capture_source(source):
    methods = r"""
  __device__ bool wait_active() {
    for (I j = 0; j < h->nz; j++)
      if (!z[j].headless && z[j].health > 0) return true;
    return false;
  }
  __device__ void wait_copy(I slot) {
    I *out = wait_states + slot * wait_stride;
    const I *parts[] = {reinterpret_cast<I *>(h), reinterpret_cast<I *>(p),
      reinterpret_cast<I *>(z), reinterpret_cast<I *>(q), reinterpret_cast<I *>(m), cd};
    I capacity[] = {GAME_HEADER_WIDTH, 45 * GAME_PLANT_WIDTH,
      ZCAP * GAME_ZOMBIE_WIDTH, QCAP * GAME_PROJECTILE_WIDTH, 5 * GAME_MOWER_WIDTH, 8};
    I live[] = {GAME_HEADER_WIDTH, h->np * GAME_PLANT_WIDTH,
      h->nz * GAME_ZOMBIE_WIDTH, h->nq * GAME_PROJECTILE_WIDTH, 5 * GAME_MOWER_WIDTH, 8};
    for (I part = 0; part < 6; part++) {
      for (I j = 0; j < live[part]; j++) out[j] = parts[part][j];
      out += capacity[part];
    }
  }
  __device__ void wait_clear() {
    if (wait_enabled && wait_watch[0] && wait_meta[0] < 0 &&
        wait_meta[1] < 0 && !wait_active()) {
      wait_copy(1); wait_meta[1] = h->tick;
    }
  }
"""
    hooks = {
        "I *reach, *reach_error; I reach_enabled;": "I *reach, *reach_error; I reach_enabled; I *wait_states, *wait_meta, *wait_watch; I wait_stride, wait_enabled;",
        "I reach_enabled) {": "I reach_enabled, I *wait_states, I *wait_meta, I *wait_watch, I wait_stride, I wait_enabled) {",
        "reach_error + i, reach_enabled};": "reach_error + i, reach_enabled, wait_states + i * 2 * wait_stride, wait_meta + i * 3, wait_watch + i, wait_stride, wait_enabled};",
        "  __device__ void advance() {": methods
        + r"""
  __device__ void advance() {
    if (wait_enabled && wait_meta[0] < 0 && h->spawn_index < h->total_spawns &&
        sp[h->spawn_index * 5] <= h->tick + 1) {
      wait_copy(0); wait_meta[0] = h->tick; wait_meta[2] = h->spawn_index;
    }
""",
        "    if (h->status)\n      emit(17, 0, h->status);": "    wait_clear();\n    if (h->status)\n      emit(17, 0, h->status);",
        "    if (per_tick && action && h->accepted)\n      return;": "    wait_clear();\n    if (per_tick && action && h->accepted)\n      return;",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned Wait capture hook changed: {before[:70]}")
        source = source.replace(before, after)
    return source


class WaitCapture:
    def __init__(self, batch):
        self.batch = batch
        self.shapes = [
            getattr(batch, name).shape[1:]
            for name in ("header", "plants", "zombies", "projectiles", "mowers", "cooldowns")
        ]
        self.stride = sum(int(np.prod(shape)) for shape in self.shapes)
        cp = batch.cp
        self.states = cp.empty((batch.n, 2, self.stride), cp.int64)
        self.meta = cp.full((batch.n, 3), -1, cp.int64)
        self.watch = cp.ones(batch.n, cp.int64)

    @property
    def nbytes(self):
        return self.states.nbytes + self.meta.nbytes + self.watch.nbytes

    def arm(self, watch):
        self.meta.fill(-1)
        self.watch[:] = self.batch.cp.asarray(watch, self.batch.cp.int64)

    def read(self, active):
        meta = self.meta.get()
        captures = [{}, {}]
        for slot in range(2):
            selected = np.flatnonzero(active & (meta[:, slot] >= 0)).tolist()
            if not selected:
                continue
            view = copy(self.batch)
            offset = 0
            for name, shape in zip(
                ("header", "plants", "zombies", "projectiles", "mowers", "cooldowns"),
                self.shapes,
                strict=True,
            ):
                size = int(np.prod(shape))
                setattr(
                    view,
                    name,
                    self.states[:, slot, offset : offset + size].reshape(self.batch.n, *shape),
                )
                offset += size
            captures[slot] = dict(
                zip(selected, download_snapshots(view, indices=selected), strict=True)
            )
        return meta, captures


def capture_arguments(batch):
    cp = batch.cp
    capture = getattr(batch, "wait_capture", None)
    if capture is None:
        return (batch.reach_error, batch.reach_error, batch.reach_error, cp.int64(0), cp.int64(0))
    return capture.states, capture.meta, capture.watch, cp.int64(capture.stride), cp.int64(1)
