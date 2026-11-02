"""Owned, guarded hooks for private identity-aware visible-zombie reach traces.

Each identity keeps its latest entry tick into each column, including outside.
This is a compressed suffix index: combined with a decision's initial positions,
it exactly answers furthest reach after that decision, including re-entry. Slot
compaction cannot change identity. Nothing in this slab enters observations.
"""

REACH_WIDTH = 14  # id, row, last category, defeat tick+1, ten entry ticks+1

REACH_METHOD = r"""
  __device__ void track_reach(Zombie &a, bool defeated = false) {
    if (!reach_enabled || a.headless) return;
    I slot = 0;
    while (slot < ZCAP && reach[slot * 14] && reach[slot * 14] != a.id) slot++;
    if (slot == ZCAP) { *reach_error = 1; return; }
    I *v = reach + slot * 14;
    I col = a.x >= 9 * G_units_per_tile ? 9 : hi(0, a.x / G_units_per_tile);
    if (!v[0]) { v[0] = a.id; v[1] = a.row; v[2] = -1; }
    if (v[1] != a.row || v[3]) { *reach_error = 2; return; }
    if (v[2] != col) { v[4 + col] = h->tick + 1; v[2] = col; }
    if (defeated) v[3] = h->tick + 1;
  }
"""


def reach_kernel_source(source):
    hooks = {
        "double sky_sun_rate;": "double sky_sun_rate;\n  I *reach, *reach_error; I reach_enabled;",
        "double sky_sun_rate) {": "double sky_sun_rate, I *reach, I *reach_error, I reach_enabled) {",
        "sun_denominator, sky_sun_rate};": "sun_denominator, sky_sun_rate, reach + i * ZCAP * 14, reach_error + i, reach_enabled};",
        "      a.headless = 1;": "      track_reach(a, true);\n      a.headless = 1;",
        "    a.x = end;\n  }\n  __device__ void zombies()": "    a.x = end;\n    track_reach(a);\n  }\n  __device__ void zombies()",
        "        if (h->tick < a.vault_until) continue;": "        track_reach(a);\n        if (h->tick < a.vault_until) continue;",
        "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;": "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;\n        track_reach(a);",
        "  __device__ void advance() {": REACH_METHOD + "\n  __device__ void advance() {",
        "    plants();\n    projectiles();": "    if (reach_enabled) for (I j = 0; j < h->nz; j++) track_reach(z[j]);\n    plants();\n    projectiles();",
        "    *ne = 0;": "    *ne = 0;\n    if (reach_enabled) for (I j = 0; j < h->nz; j++) track_reach(z[j]);",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned CUDA reach hook changed: {before[:64]}")
        source = source.replace(before, after)
    return source


def position_category(x, units=1000):
    return min(9, max(0, int(x) // units))


def suffix_labels(targets, trace, tick):
    """Independent host reduction; columns 0..8, outside 9, absent lane -1."""
    import numpy as np

    by_id = {int(row[0]): row for row in trace if row[0]}
    result = np.full(5, -1, dtype=np.int8)
    for identity, lane, initial in targets:
        event = by_id.get(int(identity))
        if event is None or event[1] != lane:
            raise RuntimeError(
                "Reach identity is missing or changed lane; refusing a fabricated label"
            )
        future = np.flatnonzero(event[4:14] > int(tick) + 1)
        column = min(int(initial), int(future.min()) if len(future) else 9)
        lane = int(lane)
        result[lane] = column if result[lane] < 0 else min(result[lane], column)
    return result
