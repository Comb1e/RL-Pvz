"""Private future-visit timestamps and complete-public-board snapshot constraints."""

import numpy as np

REACH_WIDTH = 10  # nine latest cell-visit ticks+1, then actual home-entry tick+1
LABEL_POLICY = "future-cell-home-v2"


def public_reach_facts(observation, units=1000):
    """CPU reference: occupied bits, provably unreachable bits, cleared flag per lane."""
    facts = np.zeros((5, 3), np.int16)
    rightmost = np.full(5, -(2**31))
    present = np.zeros(5, bool)
    for zombie in observation.zombies:
        if zombie.headless:
            continue
        row, col = zombie.row, int(zombie.x) // units
        present[row] = True
        rightmost[row] = max(rightmost[row], col)
        if 0 <= col < 9:
            facts[row, 0] |= 1 << col
    if observation.counts.not_yet_spawned == 0:
        for row, col in enumerate(rightmost):
            facts[row, 1] = sum(1 << c for c in range(9) if c > col)
            facts[row, 2] = not present[row]
    return facts


def future_labels(facts, trace, tick, *, snapshot_only=False, terminal=False, final_facts=None):
    """Binary targets; -1 is unknown. Later clearance never erases observed visits."""
    facts, trace = np.asarray(facts), np.asarray(trace)
    columns = 1 << np.arange(9)
    occupied = (facts[:, :1] & columns) != 0
    safe = (facts[:, 1:2] & columns) != 0
    cells = np.full((5, 9), -1, np.int8)
    home = np.full(5, -1, np.int8)
    cells[safe] = 0
    home[facts[:, 2].astype(bool)] = 0
    if not snapshot_only:
        if terminal:
            cells.fill(0)
            home.fill(0)
        elif final_facts is not None:
            final_safe = (final_facts[:, 1:2] & columns) != 0
            cells[final_safe] = 0
            home[final_facts[:, 2].astype(bool)] = 0
        cells[trace[:, :9] > int(tick) + 1] = 1
        home[trace[:, 9] > int(tick) + 1] = 1
    cells[occupied] = 1
    return cells, home


REACH_METHOD = r"""
  __device__ void track_reach(Zombie &a, I start) {
    if (!reach_enabled || a.headless) return;
    if (a.row < 0 || a.row >= 5) { *reach_error = 1; return; }
    I left = lo(start, a.x), right = hi(start, a.x);
    for (I c = 0; c < 9; c++)
      if (right >= c * G_units_per_tile && left < (c + 1) * G_units_per_tile)
        reach[a.row * 10 + c] = h->tick + 1;
  }
"""


def reach_kernel_source(source):
    hooks = {
        "double sky_sun_rate;": "double sky_sun_rate;\n  I *reach, *reach_error; I reach_enabled;",
        "double sky_sun_rate) {": "double sky_sun_rate, I *reach, I *reach_error, I reach_enabled) {",
        "sun_denominator, sky_sun_rate};": "sun_denominator, sky_sun_rate, reach + i * 50, reach_error + i, reach_enabled};",
        "      a.headless = 1;": "      track_reach(a, a.x);\n      a.headless = 1;",
        "    a.x = end;\n  }\n  __device__ void zombies()": "    I reach_start = a.x;\n    a.x = end;\n    track_reach(a, reach_start);\n  }\n  __device__ void zombies()",
        "        if (h->tick < a.vault_until) continue;": "        track_reach(a, a.previous_x);\n        if (h->tick < a.vault_until) continue;",
        "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;": "        I reach_start = a.x;\n        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;\n        track_reach(a, reach_start);",
        "  __device__ void advance() {": REACH_METHOD + "\n  __device__ void advance() {",
        "    plants();\n    projectiles();": "    if (reach_enabled) for (I j = 0; j < h->nz; j++) track_reach(z[j], z[j].x);\n    plants();\n    projectiles();",
        "    *ne = 0;": "    *ne = 0;\n    if (reach_enabled) for (I j = 0; j < h->nz; j++) track_reach(z[j], z[j].x);",
        "    if (h->status)\n      emit(17, 0, h->status);": "    if (reach_enabled && h->status == 2)\n      for (I j = 0; j < h->nz; j++)\n        if (!z[j].headless && z[j].x <= G_house_x) reach[z[j].row * 10 + 9] = h->tick + 1;\n    if (h->status)\n      emit(17, 0, h->status);",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned CUDA reach hook changed: {before[:64]}")
        source = source.replace(before, after)
    return source
