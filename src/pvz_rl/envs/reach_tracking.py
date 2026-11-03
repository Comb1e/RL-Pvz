"""Guarded per-tick depth and actual home-entry hooks for factual hindsight."""

FORECAST_HEADER = 11  # start, five home entries, five first bite-contact ticks
FORECAST_METHOD = r"""
  __device__ void track_forecast(Zombie &a) {
    if (!reach_enabled || a.headless || a.health <= 0) return;
    I offset = h->tick - reach[0];
    if (a.row < 0 || a.row >= 5 || offset < 0 || offset > forecast_ticks) {
      *reach_error = 1; return;
    }
    I code = hi(0, 9 - hi(0, floor_div(a.x, G_units_per_tile)));
    I index = 11 + offset * 5 + a.row;
    reach[index] = hi(reach[index], code);
  }
"""


def forecast_kernel_source(source, ticks):
    hooks = {
        "double sky_sun_rate;": "double sky_sun_rate;\n  I *reach, *reach_error; I reach_enabled;",
        "double sky_sun_rate) {": "double sky_sun_rate, I *reach, I *reach_error, I reach_enabled) {",
        "sun_denominator, sky_sun_rate};": f"sun_denominator, sky_sun_rate, reach + i * {FORECAST_HEADER + (ticks + 1) * 5}, reach_error + i, reach_enabled}};",
        "  __device__ void advance() {": f"  static const I forecast_ticks = {ticks};\n"
        + FORECAST_METHOD
        + "\n  __device__ void advance() {",
        "    I hp = a.health, ar = a.armor;": "    track_forecast(a);\n    I hp = a.health, ar = a.armor;",
        "    a.x = end;\n  }\n  __device__ void zombies()": "    a.x = end;\n    track_forecast(a);\n  }\n  __device__ void zombies()",
        "        if (h->tick < a.vault_until) continue;": "        track_forecast(a);\n        if (h->tick < a.vault_until) continue;",
        "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;": "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;\n        track_forecast(a);",
        "    plants();\n    projectiles();": "    for (I j = 0; j < h->nz; j++) track_forecast(z[j]);\n    plants();\n    projectiles();",
        "      Plant &b = p[best];": "      Plant &b = p[best];\n      if (reach_enabled && reach[6 + a.row] < 0) reach[6 + a.row] = h->tick;",
        "    *ne = 0;": f"    *ne = 0;\n    if (reach_enabled) {{\n      for (I j = 0; j < {FORECAST_HEADER + (ticks + 1) * 5}; j++) reach[j] = 0;\n      reach[0] = h->tick;\n      for (I j = 1; j <= 10; j++) reach[j] = -1;\n      for (I j = 0; j < h->nz; j++) track_forecast(z[j]);\n    }}",
        "    if (h->status)\n      emit(17, 0, h->status);": """    if (reach_enabled && h->status == 2)
      for (I j = 0; j < h->nz; j++)
        if (!z[j].headless && z[j].x <= G_house_x) {
          reach[1 + z[j].row] = h->tick;
          reach[11 + (h->tick - reach[0]) * 5 + z[j].row] = 10;
        }
    if (h->status)
      emit(17, 0, h->status);""",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned forecast trace hook changed: {before[:70]}")
        source = source.replace(before, after)
    return source
