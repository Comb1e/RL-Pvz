"""Guarded per-tick field depth and mower-resolved home-entry observations."""

FIELD_HEADER = 6  # start and five actual home-entry ticks
FIELD_METHOD = r"""
  __device__ void track_field(Zombie &a) {
    if (!field_enabled || a.headless || a.health <= 0) return;
    I offset = h->tick - field[0];
    if (a.row < 0 || a.row >= 5 || offset < 0 || offset > field_ticks) {
      *field_error = 1; return;
    }
    I code = hi(0, 9 - hi(0, floor_div(a.x, G_units_per_tile)));
    I index = 6 + offset * 5 + a.row;
    field[index] = hi(field[index], code);
  }
"""


def field_kernel_source(source, ticks):
    hooks = {
        "double sky_sun_rate;": "double sky_sun_rate;\n  I *field, *field_error; I field_enabled;",
        "double sky_sun_rate) {": "double sky_sun_rate, I *field, I *field_error, I field_enabled) {",
        "sun_denominator, sky_sun_rate};": f"sun_denominator, sky_sun_rate, field + i * {FIELD_HEADER + (ticks + 1) * 5}, field_error + i, field_enabled}};",
        "  __device__ void advance() {": f"  static const I field_ticks = {ticks};\n"
        + FIELD_METHOD
        + "\n  __device__ void advance() {",
        "    I hp = a.health, ar = a.armor;": "    track_field(a);\n    I hp = a.health, ar = a.armor;",
        "    a.x = end;\n  }\n  __device__ void zombies()": "    a.x = end;\n    track_field(a);\n  }\n  __device__ void zombies()",
        "        if (h->tick < a.vault_until) continue;": "        track_field(a);\n        if (h->tick < a.vault_until) continue;",
        "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;": "        a.x -= M_jump_shift_pixels * G_units_per_tile / 80;\n        track_field(a);",
        "    plants();\n    projectiles();": "    for (I j = 0; j < h->nz; j++) track_field(z[j]);\n    plants();\n    projectiles();",
        "    *ne = 0;": f"    *ne = 0;\n    if (field_enabled) {{\n      for (I j = 0; j < {FIELD_HEADER + (ticks + 1) * 5}; j++) field[j] = 0;\n      field[0] = h->tick;\n      for (I j = 1; j <= 5; j++) field[j] = -1;\n      for (I j = 0; j < h->nz; j++) track_field(z[j]);\n    }}",
        "    if (h->status)\n      emit(17, 0, h->status);": """    if (field_enabled && h->status == 2)
      for (I j = 0; j < h->nz; j++)
        if (!z[j].headless && z[j].x <= G_house_x) {
          field[1 + z[j].row] = h->tick;
          field[6 + (h->tick - field[0]) * 5 + z[j].row] = 10;
        }
    if (h->status)
      emit(17, 0, h->status);""",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned forecast trace hook changed: {before[:70]}")
        source = source.replace(before, after)
    return source
