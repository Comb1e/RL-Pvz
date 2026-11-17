"""First entry or quiet-board boundary for Prediction forecasting.

The pinned simulator marks a win when every zombie has been defeated and then
stops stepping. Prediction holds that win until the real lawn is quiet. A house
entry removes that zombie and closes forecasting. An isolated suffix finishes
the fight, settles the defenses and probes them without another model call.

A plant that is sitting still does not keep the lawn open. Armed mines and
digesting or recovering Chompers are quiet. A plant that is still rising,
fusing, biting, or exploding is not.
"""

import torch
from pvz_game.cuda.schema import HEADER, MOWER, MOWER_STATES, PLANT, PLANT_STATES, STATUSES

MOWER_MOVING = MOWER_STATES.index("moving")
LOST = STATUSES.index("lost")
WON = STATUSES.index("won")
# These states still change the lawn. ready, armed, digesting, and recovering do not.
RESOLVING_PLANT_STATES = (
    "arming",
    "rising",
    "fusing",
    "biting",
    "biting_got_one",
    "exploding",
    "detonating",
)
RESOLVING_PLANT_IDS = tuple(PLANT_STATES.index(name) for name in RESOLVING_PLANT_STATES)
PREDICTION_QUIET = "idle_plants"

_WIN = """    if (!h->status && h->defeated == h->total_spawns)
      h->status = 1;"""
_HELD_WIN = """    if (!h->status && h->defeated == h->total_spawns &&
        !(prediction_hold_win && !prediction_board_quiet()))
      h->status = 1;"""
_HOUSE = """    for (I i = 0; i < h->nz; i++)
      if (!z[i].headless && z[i].x <= G_house_x)
        h->status = 2;"""
_HELD_HOUSE = """    for (I i = 0; i < h->nz;) {
      if (!z[i].headless && z[i].health > 0 && z[i].x <= G_house_x) {
        if (prediction_hold_win) {
          if (field_enabled) {
            if (field[1 + z[i].row] < 0) field[1 + z[i].row] = h->tick;
            field[6 + (h->tick - field[0]) * 5 + z[i].row] = 10;
          }
          for (I k = i; k + 1 < h->nz; k++) z[k] = z[k + 1];
          h->nz--;
          continue;
        }
        h->status = 2;
      }
      i++;
    }"""
_RESOLVING_TEST = " || ".join(
    f"state == QUIET_PLANT_{name.upper()}" for name in RESOLVING_PLANT_STATES
)
_QUIET_METHOD = f"""  __device__ bool prediction_board_quiet() {{
    if (h->nz || h->nq || h->spawn_index != h->total_spawns) return false;
    for (I row = 0; row < 5; row++)
      if (m[row].state == QUIET_MOWER_MOVING) return false;
    for (I i = 0; i < h->np; i++) {{
      I state = p[i].state;
      if ({_RESOLVING_TEST}) return false;
    }}
    return true;
  }}
  __device__ bool prediction_entered_house() {{
    if (!prediction_hold_win || !field_enabled) return false;
    for (I row = 0; row < 5; row++) if (field[1 + row] >= 0) return true;
    return false;
  }}
"""


def prediction_last_quiet(header, plants, mowers, active):
    """True where the real board has nothing left to resolve and no spawn due.

    Armed, digesting, and recovering plants are sitting still. A resolving
    state still has to finish.
    """
    count = header[:, HEADER.index("np")]
    state = plants[:, :, PLANT.index("state")]
    slots = torch.arange(state.shape[1], device=state.device)
    resolving = torch.isin(
        state, torch.tensor(RESOLVING_PLANT_IDS, device=state.device, dtype=state.dtype)
    )
    busy = (resolving & (slots < count[:, None])).any(dim=1)
    moving = (mowers[:, :, MOWER.index("state")] == MOWER_MOVING).any(dim=1)
    return (
        active.bool()
        & (header[:, HEADER.index("nz")] == 0)
        & (header[:, HEADER.index("nq")] == 0)
        & ~moving
        & ~busy
        & (header[:, HEADER.index("spawn_index")] == header[:, HEADER.index("total_spawns")])
    )


def prediction_step_done(header, quiet, active, cutoff_tick, breached=None):
    """Cutoff, loss, first entry, or the step that left the quiet board.

    A mower-resolved entry ends forecasting without making the target a loss.
    """
    lost = header[:, HEADER.index("status")] == LOST
    cutoff = header[:, HEADER.index("tick")] >= cutoff_tick
    stop = lost | cutoff | quiet.bool()
    if breached is not None:
        stop |= breached.bool()
    return active.bool() & stop


def prediction_resume_won(header, quiet, active):
    """An already-won board that is still resolving steps again.

    A won board that is already static stays won. That decision is the one
    forecast, and the probe follows.
    """
    won = header[:, HEADER.index("status")] == WON
    return active.bool() & won & ~quiet.bool()


def prediction_quiet_source(source):
    """Hold the win, and a house entry, until the lawn matches the host stop."""
    hooks = {
        "I *motion; I motion_slots;": "I *motion; I motion_slots; I prediction_hold_win;",
        "I motion_slots, I motion_width) {": "I motion_slots, I motion_width, I prediction_hold_win) {",
        "motion + i * motion_width, motion_slots};": "motion + i * motion_width, motion_slots, prediction_hold_win};",
        "  __device__ void advance() {": _QUIET_METHOD + "  __device__ void advance() {",
        "    for (I i = 0; i < ticks && !h->status; i++)": "    for (I i = 0; i < ticks && !h->status && !prediction_entered_house(); i++)",
        _WIN: _HELD_WIN,
        _HOUSE: _HELD_HOUSE,
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned quiet-stop hook changed: {before[:70]}")
        source = source.replace(before, after)
    defines = "".join(
        f"#define QUIET_PLANT_{name.upper()} {index}\n"
        for name, index in zip(RESOLVING_PLANT_STATES, RESOLVING_PLANT_IDS, strict=True)
    )
    return defines + f"#define QUIET_MOWER_MOVING {MOWER_MOVING}\n" + source
