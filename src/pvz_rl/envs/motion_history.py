"""Deterministic per-tick motion on the entities the board already has.

The tracker is keyed by private engine id and never writes that id, a speed,
a gait, or a random draw into the observation. Each token receives the same
twelve integers: eight x displacements aligned to the newest end, how many of
those samples are real, ticks since the last pea, ticks in the current public
state, and the zombie bite phase. A missing sample stays out of the filled
suffix, because a real gait frame can be zero.
"""

from pvz_game import Status, StepResult, Wait
from pvz_game.cuda.schema import MOWER_STATES, PLANT_STATES, ZOMBIE_STATES

HISTORY_TICKS = 8
SHOT_CLAMP = 30
STATE_CLAMP = 50
UNKNOWN_PHASE = 8
MOTION_HEADER = 2
MOTION_STRIDE = 20
PLANT, ZOMBIE, SHOT, MOWER = 1, 2, 3, 4
# Newest displacement is dx_7. The unfilled prefix is padding, not a zero step.
HISTORY_FIELDS = (
    *(f"dx_{index}" for index in range(HISTORY_TICKS)),
    "history_filled",
    "since_shot",
    "state_ticks",
    "seen_phase",
)
_PLANT_STATE = {name: index for index, name in enumerate(PLANT_STATES)}
_ZOMBIE_STATE = {name: index for index, name in enumerate(ZOMBIE_STATES)}
_MOWER_STATE = {name: index for index, name in enumerate(MOWER_STATES)}


def unknown_history(shot_clamp=SHOT_CLAMP):
    """Twelve channels for an entity the tracker has not followed."""
    return (0,) * HISTORY_TICKS + (0, int(shot_clamp), 0, UNKNOWN_PHASE)


def motion_table_slots(zombie_capacity, projectile_capacity):
    """Power-of-two open-address table for one game, capped at 1024 slots."""
    needed = 45 + int(zombie_capacity) + int(projectile_capacity) + 5
    slots = 1
    while slots < needed * 2 and slots < 1024:
        slots *= 2
    return slots


def motion_row_width(slots):
    return MOTION_HEADER + int(slots) * MOTION_STRIDE


def _blank(shot_clamp):
    return {
        "last_x": 0,
        "dx": [],
        "filled": 0,
        "since": int(shot_clamp),
        "state": 0,
        "state_ticks": 0,
        "seen": -1,
        "ready": False,
        "shot": False,
        "seeded": False,
    }


def _push(track, x):
    track["dx"].append(int(x) - int(track["last_x"]))
    if len(track["dx"]) > HISTORY_TICKS:
        del track["dx"][: len(track["dx"]) - HISTORY_TICKS]
    track["filled"] = len(track["dx"])
    track["last_x"] = int(x)


def _finish_tick(track, x, state, shot_clamp, state_clamp, *, zombie=False):
    """Apply one observed position. A seeded birth records the move and stays at 0."""
    if track["seeded"]:
        _push(track, x)
        track["seeded"] = False
        track["state"] = int(state)
        track["shot"] = False
        return
    if not track["ready"]:
        track["last_x"] = int(x)
        track["state"] = int(state)
        track["state_ticks"] = 0
        track["since"] = 0 if track["shot"] else int(shot_clamp)
        track["shot"] = False
        track["seen"] = 1 if zombie else -1
        track["filled"] = 0
        track["dx"] = []
        track["ready"] = True
        return
    _push(track, x)
    if track["state"] != int(state):
        track["state"] = int(state)
        track["state_ticks"] = 0
    elif track["state_ticks"] < state_clamp:
        track["state_ticks"] += 1
    if track["seen"] >= 0:
        track["seen"] += 1
    if track["shot"]:
        track["since"] = 0
        track["shot"] = False
    elif track["since"] < shot_clamp:
        track["since"] += 1


def _channels(track, shot_clamp):
    if track is None or not track["ready"]:
        return unknown_history(shot_clamp)
    dx = [0] * HISTORY_TICKS
    filled = track["filled"]
    if filled:
        dx[-filled:] = track["dx"]
    phase = UNKNOWN_PHASE if track["seen"] < 0 else track["seen"] % HISTORY_TICKS
    return (*dx, filled, track["since"], track["state_ticks"], phase)


class MotionHistory:
    """Online rings for one CPU game. Call sync once, then advance every tick."""

    def __init__(self, cfg, rules):
        history = cfg["encoding"]["history"]
        self.ticks = int(history["ticks"])
        self.shot_clamp = int(history["shot_clamp"])
        self.state_clamp = int(history["state_clamp"])
        self.units = int(rules.game["units_per_tile"])
        self.projectile_offset = int(rules.game["projectile_offset"])
        self.tracks = {}
        self.shots = 0

    def reset(self):
        self.tracks.clear()
        self.shots = 0

    def plant_x(self, plant):
        return int(plant.col) * self.units + self.units // 2

    def sync(self, observation):
        """Remember positions before the first simulated tick. No displacement is stored."""
        self.reset()
        for plant in observation.plants:
            track = _blank(self.shot_clamp)
            track.update(last_x=self.plant_x(plant), state=_PLANT_STATE[plant.state], ready=True)
            self.tracks[(PLANT, plant.id)] = track
        for zombie in observation.zombies:
            track = _blank(self.shot_clamp)
            track.update(
                last_x=int(zombie.x),
                state=_ZOMBIE_STATE[zombie.state],
                seen=0,
                ready=True,
            )
            self.tracks[(ZOMBIE, zombie.id)] = track
        for shot in observation.projectiles:
            track = _blank(self.shot_clamp)
            track.update(last_x=int(shot.x), ready=True)
            self.tracks[(SHOT, shot.id)] = track
        for mower in observation.mowers:
            track = _blank(self.shot_clamp)
            track.update(
                last_x=int(mower.x),
                state=_MOWER_STATE[mower.state],
                ready=True,
            )
            self.tracks[(MOWER, int(mower.row) + 1)] = track

    def advance(self, observation, events):
        """Fold one completed simulator tick into the rings."""
        fired = set()
        spawned = set()
        for event in events:
            if event.kind == "ProjectileFired":
                self.shots += 1
                source = event.get("source")
                if source is not None:
                    fired.add(int(source))
                self._seed_shot(event.entity_id, source, observation)
            elif event.kind == "ZombieSpawned" and event.entity_id is not None:
                spawned.add(int(event.entity_id))
        live = set()
        for plant in observation.plants:
            live.add((PLANT, plant.id))
            track = self.tracks.setdefault((PLANT, plant.id), _blank(self.shot_clamp))
            if plant.id in fired:
                track["shot"] = True
            _finish_tick(
                track,
                self.plant_x(plant),
                _PLANT_STATE[plant.state],
                self.shot_clamp,
                self.state_clamp,
            )
        for zombie in observation.zombies:
            live.add((ZOMBIE, zombie.id))
            track = self.tracks.setdefault((ZOMBIE, zombie.id), _blank(self.shot_clamp))
            _finish_tick(
                track,
                int(zombie.x),
                _ZOMBIE_STATE[zombie.state],
                self.shot_clamp,
                self.state_clamp,
                zombie=zombie.id in spawned and not track["ready"],
            )
        for shot in observation.projectiles:
            live.add((SHOT, shot.id))
            track = self.tracks.setdefault((SHOT, shot.id), _blank(self.shot_clamp))
            _finish_tick(track, int(shot.x), 0, self.shot_clamp, self.state_clamp)
        for mower in observation.mowers:
            ident = int(mower.row) + 1
            live.add((MOWER, ident))
            track = self.tracks.setdefault((MOWER, ident), _blank(self.shot_clamp))
            _finish_tick(
                track,
                int(mower.x),
                _MOWER_STATE[mower.state],
                self.shot_clamp,
                self.state_clamp,
            )
        for key in list(self.tracks):
            if key not in live:
                del self.tracks[key]

    def _seed_shot(self, shot_id, source, observation):
        if shot_id is None or source is None:
            return
        plant = next((item for item in observation.plants if item.id == source), None)
        if plant is None:
            return
        track = self.tracks.setdefault((SHOT, int(shot_id)), _blank(self.shot_clamp))
        if track["ready"]:
            return
        track.update(
            last_x=self.plant_x(plant) + self.projectile_offset,
            ready=True,
            seeded=True,
            since=self.shot_clamp,
            seen=-1,
        )

    def channels(self, category, ident):
        return _channels(self.tracks.get((int(category), int(ident))), self.shot_clamp)

    def export(self):
        tracks = []
        for (category, ident), track in self.tracks.items():
            if not track["ready"]:
                continue
            dx = [0] * HISTORY_TICKS
            if track["filled"]:
                dx[-track["filled"] :] = track["dx"]
            tracks.append(
                {
                    "category": int(category),
                    "id": int(ident),
                    "last_x": int(track["last_x"]),
                    "state": int(track["state"]),
                    "state_ticks": int(track["state_ticks"]),
                    "since": int(track["since"]),
                    "seen": int(track["seen"]),
                    "filled": int(track["filled"]),
                    "dx": dx,
                }
            )
        return {"shots": int(self.shots), "tracks": tracks}


def write_motion_row(row, export, slots):
    """Pack one game's rings into the same open-address layout the CUDA step uses."""
    row[:] = 0
    if not export:
        return
    row[0] = int(export["shots"])
    mask = int(slots) - 1
    for track in export["tracks"]:
        start = (int(track["category"]) * 0x9E3779B1 ^ int(track["id"])) & mask
        placed = False
        for probe in range(int(slots)):
            slot = (start + probe) & mask
            base = MOTION_HEADER + slot * MOTION_STRIDE
            if row[base] != 0:
                continue
            row[base] = int(track["id"])
            row[base + 1] = int(track["category"])
            row[base + 2] = int(track["last_x"])
            row[base + 3] = int(track["state"])
            row[base + 4] = int(track["state_ticks"])
            row[base + 5] = int(track["since"])
            row[base + 6] = int(track["seen"])
            row[base + 7] = int(track["filled"])
            row[base + 8 : base + 16] = track["dx"]
            row[base + 17] = 1
            placed = True
            break
        if not placed:
            raise RuntimeError("Motion table cannot hold the warmed entities")


def install_motion(batch, exports, indices):
    """Copy CPU rings into the device table before the first encode of those games."""
    import numpy as np

    rows = np.zeros((len(indices), batch.motion.shape[1]), np.int64)
    for row, export in zip(rows, exports, strict=True):
        write_motion_row(row, export, batch.motion_slots)
    batch.motion[batch.cp.asarray(list(indices))] = batch.cp.asarray(rows)


def matched_step(game, action, ticks, motion):
    """One action followed by single-tick waits. The final board matches one multi-tick step."""
    if ticks <= 0:
        return game.step(action, ticks=ticks)
    first = game.step(action, ticks=1)
    events = list(first.events)
    if first.ticks_advanced:
        motion.advance(first.observation, first.events)
    status = first.status
    observation = first.observation
    advanced = first.ticks_advanced
    while advanced < ticks and status == Status.RUNNING:
        nxt = game.step(Wait(), ticks=1)
        events.extend(nxt.events)
        if nxt.ticks_advanced:
            motion.advance(nxt.observation, nxt.events)
        status = nxt.status
        observation = nxt.observation
        advanced += nxt.ticks_advanced
        if not nxt.ticks_advanced:
            break
    return StepResult(observation, first.action_result, tuple(events), status, advanced)


def motion_kernel_source(source, *, shot_clamp, state_clamp):
    """Record per-tick motion inside the existing multi-tick launch."""
    methods = f"""
  __device__ I *motion_slot(I slot) {{
    return motion + {MOTION_HEADER} + slot * {MOTION_STRIDE};
  }}
  __device__ I *find_track(I category, I id, bool create) {{
    I mask = motion_slots - 1;
    I start = (category * 0x9E3779B1LL ^ id) & mask;
    I tomb = -1;
    for (I n = 0; n < motion_slots; n++) {{
      I s = (start + n) & mask;
      I *t = motion_slot(s);
      if (t[0] == 0) {{
        if (!create) return nullptr;
        I dest_i = tomb >= 0 ? tomb : s;
        I *dest = motion_slot(dest_i);
        for (I k = 0; k < {MOTION_STRIDE}; k++) dest[k] = 0;
        dest[0] = id;
        dest[1] = category;
        dest[5] = {int(shot_clamp)};
        dest[6] = -1;
        return dest;
      }}
      if (t[0] == -1) {{
        if (tomb < 0) tomb = s;
        continue;
      }}
      if (t[0] == id && t[1] == category) return t;
    }}
    if (!create || tomb < 0) return nullptr;
    I *dest = motion_slot(tomb);
    for (I k = 0; k < {MOTION_STRIDE}; k++) dest[k] = 0;
    dest[0] = id;
    dest[1] = category;
    dest[5] = {int(shot_clamp)};
    dest[6] = -1;
    return dest;
  }}
  __device__ void push_dx(I *t, I x) {{
    for (I k = 0; k < 7; k++) t[8 + k] = t[9 + k];
    t[15] = x - t[2];
    if (t[7] < {HISTORY_TICKS}) t[7]++;
    t[2] = x;
  }}
  __device__ void touch(I category, I id, I x, I state, bool birth) {{
    I *t = find_track(category, id, true);
    if (!t) return;
    if (t[16]) {{
      push_dx(t, x);
      t[16] = 0;
      t[3] = state;
      t[19] = motion[1];
      return;
    }}
    if (!t[17]) {{
      t[2] = x;
      t[3] = state;
      t[4] = 0;
      t[5] = t[18] ? 0 : {int(shot_clamp)};
      t[18] = 0;
      t[6] = birth ? 1 : -1;
      t[7] = 0;
      t[17] = 1;
    }} else {{
      push_dx(t, x);
      if (t[3] != state) {{
        t[3] = state;
        t[4] = 0;
      }} else if (t[4] < {int(state_clamp)}) t[4]++;
      if (t[6] >= 0) t[6]++;
      if (t[18]) {{
        t[5] = 0;
        t[18] = 0;
      }} else if (t[5] < {int(shot_clamp)}) t[5]++;
    }}
    t[19] = motion[1];
  }}
  __device__ void note_shot(Plant &a) {{
    if (!motion) return;
    motion[0]++;
    I *plant = find_track(1, a.id, true);
    if (plant) plant[18] = 1;
    Shot &s = q[h->nq - 1];
    I *shot = find_track(3, s.id, true);
    if (!shot || shot[17]) return;
    shot[2] = s.x;
    shot[3] = 0;
    shot[5] = {int(shot_clamp)};
    shot[6] = -1;
    shot[16] = 1;
    shot[17] = 1;
  }}
  __device__ void note_motion() {{
    if (!motion) return;
    motion[1]++;
    for (I j = 0; j < h->np; j++)
      touch(1, p[j].id, p[j].col * G_units_per_tile + G_units_per_tile / 2, p[j].state, false);
    for (I j = 0; j < h->nz; j++)
      touch(2, z[j].id, z[j].x, z[j].state, true);
    for (I j = 0; j < h->nq; j++)
      touch(3, q[j].id, q[j].x, 0, false);
    for (I j = 0; j < 5; j++)
      touch(4, j + 1, m[j].x, m[j].state, false);
    I epoch = motion[1];
    for (I s = 0; s < motion_slots; s++) {{
      I *t = motion_slot(s);
      if (t[0] > 0 && t[19] != epoch) {{
        t[0] = -1;
        t[17] = 0;
      }}
    }}
  }}
"""
    hooks = {
        "I *wait_states, *wait_meta, *wait_watch; I wait_stride, wait_enabled;": "I *wait_states, *wait_meta, *wait_watch; I wait_stride, wait_enabled; I *motion; I motion_slots;",
        "I wait_enabled) {": "I wait_enabled, I *motion, I motion_slots, I motion_width) {",
        "wait_watch + i, wait_stride, wait_enabled};": "wait_watch + i, wait_stride, wait_enabled, motion + i * motion_width, motion_slots};",
        "  __device__ void advance() {": methods + "\n  __device__ void advance() {",
        "    q[h->nq++] = shot;\n    emit(7, shot.id, a.id);": "    q[h->nq++] = shot;\n    note_shot(a);\n    emit(7, shot.id, a.id);",
        "    wait_clear();\n    if (h->status)\n      emit(17, 0, h->status);": "    note_motion();\n    wait_clear();\n    if (h->status)\n      emit(17, 0, h->status);",
    }
    for before, after in hooks.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Pinned motion hook changed: {before[:70]}")
        source = source.replace(before, after)
    return source
