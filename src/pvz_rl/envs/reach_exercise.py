"""Guarded device-side endpoints for private scenario reach continuations."""

# Modes 1/2 retain saved bomb/attacker endpoints; mode 3 observes natural termination.
EXERCISE_WIDTH = 9  # mode, source, reason, tick, cutoff, two task ids, started, continuing
LEGACY_ENDPOINT_POLICY = "placement_endpoints_v1"
NATURAL_ENDPOINT_POLICY = "natural_game_end_v1"
STOP_REASONS = {
    1: "exploded",
    2: "targets_defeated",
    3: "plant_removed",
    4: "engine_ended",
    5: "cutoff",
    6: "rejected",
    7: "behind_target",
    8: "natural_win",
    9: "home_breach",
}

METHODS = r"""
  __device__ void reach_stop(I reason) {
    if (exercise[0] && !exercise[2]) {
      exercise[2] = reason;
      exercise[3] = h->tick;
    }
  }
  __device__ void reach_event(I kind, I id) {
    if (!exercise[0] || exercise[0] == 3 || !exercise[7] || exercise[2]) return;
    if (kind == 6 && exercise[0] == 2) {
      if (exercise[5] == id) exercise[5] = 0;
      if (exercise[6] == id) exercise[6] = 0;
      if (!exercise[5] && !exercise[6]) reach_stop(2);
    }
    if (id == exercise[1]) {
      if (kind == 8 && exercise[0] == 1) reach_stop(1);
      else if (kind == 1) reach_stop(3);
    }
  }
  __device__ void reach_start(I action) {
    if (!exercise[0] || exercise[7]) return;
    exercise[7] = 1;
    if (exercise[0] == 3) return;
    if (!h->accepted || !action) { reach_stop(6); return; }
    if (exercise[0] == 2) {
      I tile = (action - 1) % 45;
      for (I j = 0; j < h->nz; j++)
        if (!z[j].headless && z[j].health > 0 && z[j].row == tile / 9 &&
            tile % 9 * G_units_per_tile > z[j].x) {
          reach_stop(7); break;
        }
    }
  }
"""


def exercise_kernel_source(source):
    hooks = {
        "I *reach, *reach_error; I reach_enabled;": "I *reach, *reach_error; I reach_enabled; I *exercise;",
        "I reach_enabled) {": "I reach_enabled, I *exercise) {",
        "reach_error + i, reach_enabled};": f"reach_error + i, reach_enabled, exercise + i * {EXERCISE_WIDTH}}};",
        "  __device__ void emit(I kind,": METHODS + "\n  __device__ void emit(I kind,",
        "    if (DIAGNOSTIC) {": "    reach_event(kind, id);\n    if (DIAGNOSTIC) {",
        "    if (!reach_enabled || a.headless) return;": "    if (!reach_enabled || a.headless || (exercise[0] && exercise[2])) return;",
        "    if (per_tick && action && h->accepted)": "    reach_start(action);\n    if (per_tick && action && h->accepted)",
        "    for (I i = 0; i < ticks && !h->status; i++)": "    for (I i = 0; i < ticks && !h->status && (!exercise[8] || !exercise[2]); i++)",
        "    if (h->status)\n      emit(17, 0, h->status);": "    if (h->status)\n      emit(17, 0, h->status);\n"
        "    if (exercise[0]) {\n"
        "      if (exercise[0] == 3) {\n"
        "        if (h->status) reach_stop(h->status == 1 ? 8 : 9);\n"
        "      } else if (h->status) reach_stop(4);\n"
        "      else if (h->tick >= exercise[4]) reach_stop(5);\n    }",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned CUDA exercise-reach hook changed: {before[:64]}")
        source = source.replace(before, after)
    return source
