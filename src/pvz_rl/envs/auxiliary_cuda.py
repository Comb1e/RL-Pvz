"""Event-polled CUDA workers shared by interval and teacher queues."""

import numpy as np
from pvz_game.config import ZOMBIE_TYPES
from pvz_game.cuda.schema import HEADER

from pvz_rl.envs.cuda_lane_oracle import LaneTransfers
from pvz_rl.envs.cuda_snapshots import download_snapshots
from pvz_rl.envs.cuda_static_fork import StaticForkBatch
from pvz_rl.envs.scratch_game import EngineSnapshot
from pvz_rl.envs.snapshot_packet import SnapshotRoot

FORK_STEP = r"""
extern "C" __global__ void advance_auxiliary(
 I *headers, I *plants, I *zombies, I *shots, I *mowers, I *cooldowns,
 I *schedules, I *events, I *event_counts, double *facts,
 I *forecast, I *phases, I *starts, I *anchors, I *counts, I *timers, I ticks, I probe,
 double sky_rate) {
 I i=blockIdx.x; if(threadIdx.x) return;
 unsigned long long begin,end; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin));
 timers[2*i]=timers[2*i+1]=0;
 Sim s = {(Header *)(headers+i*GAME_HEADER_WIDTH), (Plant *)(plants+i*45*GAME_PLANT_WIDTH),
  (Zombie *)(zombies+i*ZCAP*GAME_ZOMBIE_WIDTH), (Shot *)(shots+i*QCAP*GAME_PROJECTILE_WIDTH),
  (Mower *)(mowers+i*5*GAME_MOWER_WIDTH), cooldowns+i*8,
  schedules+i*ZCAP*5, events+i*ECAP*8, event_counts+i, facts+i*9,
  forecast+i*20, phases+i, starts+i, sky_rate};
 if(!s.h->enabled || phases[i]==3) return;
 for(I used=0; ;used++) {
  if(s.h->status) {
   if(phases[i]==0) {
    anchors[i]=s.h->tick;
    for(I row=0;row<5;row++) {
     Zombie a={}; a.id=s.h->next_id++; a.kind=probe; a.row=row;
     a.x=G_cols*G_units_per_tile; a.health=ZH[probe]; a.armor=ZA[probe];
     a.speed=s.random(ZS[probe],ZSMAX[probe]); a.gait=s.random(0,1)==0?1:0;
     s.z[s.h->nz++]=a;
    }
    s.h->status=0; phases[i]=2;
   } else { phases[i]=3; s.h->enabled=0; break; }
  }
  if(used>=ticks) break;
  counts[i*2+(phases[i]==0?0:1)]++;
  s.advance();
 }
 asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end)); timers[2*i]=(I)begin; timers[2*i+1]=(I)end;
}
"""


def device_bytes(batch):
    return sum(v.nbytes for v in vars(batch).values() if isinstance(v, batch.cp.ndarray))


class CudaWorkers:
    def __init__(self, source, family, count, settings, teacher_settings, stream):
        cp = source.cp
        self.cp, self.family, self.stream = cp, family, stream
        self.settings, self.teacher_settings = settings, teacher_settings
        self.pool = cp.cuda.MemoryPool()
        with stream, cp.cuda.using_allocator(self.pool.malloc):
            if family == "outcomes":
                self.batch = StaticForkBatch(
                    count,
                    rules=source.rules,
                    device=source.device,
                    zombie_capacity=source.zcap + 5,
                    projectile_capacity=source.qcap,
                    max_step_ticks=settings["chunk_ticks"],
                    sky_sun_rate=source.sky_sun_rate,
                )
                module = cp.RawModule(
                    code=self.batch.source + FORK_STEP, options=("--std=c++11", "--fmad=false")
                )
                self.kernel = module.get_function("advance_auxiliary")
                self.anchors = cp.full(count, -1, cp.int64)
                self.counts = cp.zeros((count, 2), cp.int64)
                self.timers = cp.zeros((count, 2), cp.int64)
            else:
                from pvz_game.cuda.backend import projectile_bound

                from pvz_rl.learning.peashooter_teacher import _LaneBatch

                self.batch = _LaneBatch(
                    count,
                    rules=source.rules,
                    device=source.device,
                    zombie_capacity=source.zcap,
                    projectile_capacity=source.qcap + projectile_bound(source.rules),
                    max_step_ticks=teacher_settings["oracle_chunk_ticks"],
                    cutoff=teacher_settings["cutoff"],
                    timing=True,
                )
                self.batch.set_sky_sun_rate(source.sky_sun_rate)
                # Compile before collection, including the exact cutoff wrapper.
                self.batch._timed_kernel.compile()
            self.batch.header[:, HEADER.index("enabled")] = 0
            self.transfers = LaneTransfers(cp, min(16 * 2**20, settings["vram_mib"] * 2**20 // 32))
            self.transfers.copy.compile()
            self.slots = [None] * count
            self.actions = cp.zeros(count, cp.int64)
            self.packet = cp.empty((count, 27 if family == "outcomes" else 6), cp.int64)
        self.host_memory = cp.cuda.alloc_pinned_memory(self.packet.nbytes)
        self.host = np.frombuffer(self.host_memory, np.int64, count=self.packet.size).reshape(
            self.packet.shape
        )
        self.event = cp.cuda.Event(disable_timing=True)
        self.pending = False
        self.chunk_ticks = 64
        self.progress_bytes = 0
        self.nbytes = device_bytes(self.batch) + self.packet.nbytes + self.actions.nbytes
        if family == "outcomes":
            self.nbytes += self.anchors.nbytes + self.counts.nbytes + self.timers.nbytes

    def fill(self, entries):
        """Entries are (slot, job). Caller owns roots until this chunk completes."""
        cp, b = self.cp, self.batch
        transfers = []
        prepared = {}
        for slot, job in entries:
            state = job.get("continuation")
            snapshot = state["snapshot"] if state else job["snapshot"]
            key = ("continuation", job["id"], snapshot["tick"]) if state else ("root", job["root"])
            if key not in self.transfers.cache and key not in prepared:
                prepared[key] = (
                    snapshot
                    if isinstance(snapshot, (EngineSnapshot, SnapshotRoot))
                    else EngineSnapshot._seal(snapshot)
                )
            transfers.append((slot, key, prepared.get(key, snapshot)))
            self.slots[slot] = job
            if state:
                job["ticks"] = dict(state.get("ticks", {}))
        self.transfers.fill(b, transfers)
        indices = cp.asarray([slot for slot, _ in entries], cp.int64)
        self.actions[indices] = cp.asarray(
            [0 if j.get("continuation") else j["action"] for _, j in entries]
        )
        b.header[indices, HEADER.index("enabled")] = 1
        if self.family == "outcomes":
            b.header[indices, HEADER.index("status")] = 0
            b.header[indices, HEADER.index("total_spawns")] = b.header[
                indices, HEADER.index("spawn_index")
            ]
            b.forecast[indices] = 0
            b.forecast[indices, :, 1] = -1
            b.forecast_phase[indices] = 0
            b.forecast_start[indices] = b.header[indices, 0]
            self.anchors[indices] = -1
            self.counts[indices] = 0
            # Engine applies actions and scans the initial position at zero ticks.
            b._step(
                (b.n,),
                (32,),
                (
                    b.header,
                    b.plants,
                    b.zombies,
                    b.projectiles,
                    b.mowers,
                    b.cooldowns,
                    b._schedules,
                    b._events,
                    b._event_counts,
                    b.facts,
                    self.actions,
                    b.n,
                    0,
                    1,
                ),
            )
            for slot, job in entries:
                if state := job.get("continuation"):
                    b.header[slot, HEADER.index("status")] = {"running": 0, "won": 1, "lost": 2}[
                        state["snapshot"]["status"]
                    ]
                    b.forecast[slot] = cp.asarray(state["labels"])
                    b.forecast_phase[slot] = {"field": 0, "probe": 2, "complete": 3}[state["phase"]]
                    b.forecast_start[slot] = state["start"]
                    self.anchors[slot] = state["anchor"]
                    self.counts[slot] = cp.asarray(
                        [state["ticks"].get(k, 0) for k in ("field_ticks", "probe_ticks")]
                    )
            self.actions.fill(0)

    def launch(self, ticks):
        cp, b = self.cp, self.batch
        with self.stream, cp.cuda.using_allocator(self.pool.malloc):
            if self.family == "outcomes":
                self.kernel(
                    (b.n,),
                    (32,),
                    (
                        b.header,
                        b.plants,
                        b.zombies,
                        b.projectiles,
                        b.mowers,
                        b.cooldowns,
                        b._schedules,
                        b._events,
                        b._event_counts,
                        b.facts,
                        b.forecast,
                        b.forecast_phase,
                        b.forecast_start,
                        self.anchors,
                        self.counts,
                        self.timers,
                        ticks,
                        ZOMBIE_TYPES.index(self.settings["probe"]),
                        cp.float64(b.sky_sun_rate),
                    ),
                )
                self.packet[:, 25:27] = self.timers
                self.packet[:, 0] = b.header[:, 0]
                self.packet[:, 1] = b.forecast_phase
                self.packet[:, 2] = self.anchors
                self.packet[:, 3:5] = self.counts
                self.packet[:, 5:25] = b.forecast.reshape(b.n, 20)
            else:
                b.step_device(self.actions, ticks=ticks, per_tick=self.teacher_settings["per_tick"])
                self.packet[:, :3] = b.header[:, [0, 1, 12]]
                self.packet[:, 3] = b.header[:, 14]
                self.packet[:, 4:6] = b.kernel_time
                self.actions.fill(0)
            self.packet.get(out=self.host, stream=self.stream, blocking=False)
            self.event.record()
        self.pending = True
        self.progress_bytes += self.packet.nbytes

    def poll(self):
        if not self.pending or not self.event.done:
            return None
        self.pending = False
        self.transfers.inflight = []
        results = []
        for slot, job in enumerate(self.slots):
            if job is None:
                continue
            row = self.host[slot]
            if self.family == "outcomes":
                job["ticks"] = dict(
                    field_ticks=int(row[3]), probe_ticks=int(row[4]), teacher_ticks=0
                )
                if row[1] != 3:
                    continue
                result = dict(
                    outcome=row[5:25].reshape(5, 4).copy(), anchor=int(row[2]), ticks=job["ticks"]
                )
            else:
                if job["action"] and not job.get("placed") and not row[2]:
                    raise RuntimeError("CUDA teacher rejected a saved legal candidate")
                job["placed"] = True
                job.setdefault("ticks", dict(field_ticks=0, probe_ticks=0, teacher_ticks=0))
                previous = job.get(
                    "last_tick",
                    job.get("continuation", {}).get("snapshot", job["snapshot"])["tick"],
                )
                job["ticks"]["teacher_ticks"] += int(row[0]) - previous
                job["last_tick"] = int(row[0])
                if not row[1] and row[0] < job["cutoff"]:
                    continue
                result = dict(
                    success=bool(row[1] == 1),
                    timeout=bool(row[1] == 0),
                    tick=int(row[0]),
                    ticks=dict(job["ticks"]),
                )
            results.append((job["id"], result))
            self.slots[slot] = None
        timer = 25 if self.family == "outcomes" else 4
        times = self.host[:, timer : timer + 2]
        times = times[times[:, 0] > 0]
        milliseconds = float(times[:, 1].max() - times[:, 0].min()) / 1e6 if len(times) else 0.0
        return results, milliseconds

    def continuations(self):
        if self.pending:
            self.event.synchronize()
            raise RuntimeError("Harvest the completed CUDA chunk before checkpointing")
        slots = [i for i, j in enumerate(self.slots) if j is not None]
        snapshots = download_snapshots(self.batch, indices=slots, share_metadata=True)
        states = []
        for slot, snapshot in zip(slots, snapshots, strict=True):
            job = self.slots[slot]
            state = dict(snapshot=snapshot, ticks=dict(job.get("ticks", {})), action_applied=True)
            if self.family == "outcomes":
                row = self.host[slot]
                state.update(
                    phase="field" if row[1] == 0 else "probe",
                    start=int(self.batch.forecast_start[slot].get()),
                    anchor=int(row[2]),
                    labels=row[5:25].reshape(5, 4).copy(),
                )
            states.append((job["id"], state))
        return states

    def close(self):
        if self.pending:
            self.event.synchronize()
        self.transfers.close()
        self.slots.clear()
        self.batch = self.packet = self.actions = None
        if self.family == "outcomes":
            self.anchors = self.counts = self.timers = None
        self.pool.free_all_blocks()
