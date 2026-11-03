// Owned execution adapter. Each worker owns an entire independent engine state.
struct ForkCache {
  I *states, *ticks, *depths, *values, *origins;
  I steps, stride, source_zcap, source_qcap, wait_ticks, zcap, qcap;
};

__device__ I fork_add(I *p, I n) {
  return (I)atomicAdd(reinterpret_cast<unsigned long long *>(p), (unsigned long long)n);
}

__device__ I fork_worker() {
#if FORK_PACKED
  return blockIdx.x * blockDim.x + threadIdx.x;
#else
  return threadIdx.x ? -1 : blockIdx.x;
#endif
}

__device__ void fork_probe(Sim &s, I kind) {
  for (I row = 0; row < 5; row++) {
    Zombie a = {};
    a.id = s.h->next_id++; a.kind = kind; a.row = row;
    a.x = G_cols * G_units_per_tile; a.health = ZH[kind]; a.armor = ZA[kind];
    a.speed = s.random(ZS[kind], ZSMAX[kind]);
    a.gait = s.random(0, 1) == 0 ? 1 : 0;
    s.z[s.h->nz++] = a;
  }
  s.h->status = 0;
  s.forecast_phase[0] = 2;
}

__device__ void fork_pack(Sim &s, I *out, I zcap, I qcap) {
  const I *header = reinterpret_cast<I *>(s.h);
  for (I k = 0; k < GAME_HEADER_WIDTH; k++) out[k] = header[k];
  out += GAME_HEADER_WIDTH;
  const I *parts[] = {reinterpret_cast<I *>(s.p), reinterpret_cast<I *>(s.z),
                     reinterpret_cast<I *>(s.q), reinterpret_cast<I *>(s.m), s.cd};
  I live[] = {s.h->np * GAME_PLANT_WIDTH, s.h->nz * GAME_ZOMBIE_WIDTH,
              s.h->nq * GAME_PROJECTILE_WIDTH, 5 * GAME_MOWER_WIDTH, 8};
  I widths[] = {45 * GAME_PLANT_WIDTH, zcap * GAME_ZOMBIE_WIDTH,
                qcap * GAME_PROJECTILE_WIDTH, 5 * GAME_MOWER_WIDTH, 8};
  for (I part = 0; part < 5; part++) {
    for (I k = 0; k < live[part]; k++) out[k] = parts[part][k];
    out += widths[part];
  }
}

__device__ bool fork_equal(const Header *h, const I *p, const I *z, const I *q,
                           const I *m, const I *cd, const I *saved, I zcap, I qcap) {
  const I *header = reinterpret_cast<const I *>(h);
  for (I k = 0; k < GAME_HEADER_WIDTH; k++) {
    // total_spawns is suppressed; acceptance/advance/enabled are dispatch facts.
    if (k == 8 || k == 12 || k == 13 || k == 14 || k == 17) continue;
    if (header[k] != saved[k]) return false;
  }
  saved += GAME_HEADER_WIDTH;
  const I *parts[] = {p, z, q, m, cd};
  I live[] = {h->np * GAME_PLANT_WIDTH, h->nz * GAME_ZOMBIE_WIDTH,
              h->nq * GAME_PROJECTILE_WIDTH, 5 * GAME_MOWER_WIDTH, 8};
  I widths[] = {45 * GAME_PLANT_WIDTH, zcap * GAME_ZOMBIE_WIDTH,
                qcap * GAME_PROJECTILE_WIDTH, 5 * GAME_MOWER_WIDTH, 8};
  for (I part = 0; part < 5; part++) {
    for (I k = 0; k < live[part]; k++) if (parts[part][k] != saved[k]) return false;
    saved += widths[part];
  }
  return true;
}

extern "C" __global__ void invalidate_fork_cache(
    const I *headers, const I *actions, const bool *active, I *ticks, I steps, I n) {
  I i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Header *h = reinterpret_cast<const Header *>(headers + i * GAME_HEADER_WIDTH);
  if (active[i] && actions[i] && h->accepted)
    for (I e = 0; e < steps; e++) ticks[i * steps + e] = -1;
}

extern "C" __global__ void run_fork_jobs(
    I *headers, I *plants, I *zombies, I *shots, I *mowers, I *cooldowns,
    I *schedules, I *events, I *event_counts, double *facts,
    I *forecast, I *phases, I *starts, I *job_ids, I workers,
    const I *source_h, const I *source_p, const I *source_z, const I *source_q,
    const I *source_m, const I *source_cd,
    const I *roots, const I *actions, I jobs, I *next_job, I *progress, I *results,
    I ticks, I probe, double sky_rate, bool baseline,
    I *cache_states, I *cache_ticks, I *cache_depths, I *cache_values, I *cache_origins,
    I cache_steps, I cache_stride, I source_zcap, I source_qcap, I wait_ticks,
    I cache_zcap, I cache_qcap) {
  ForkCache cache = {cache_states, cache_ticks, cache_depths, cache_values, cache_origins,
                     cache_steps, cache_stride, source_zcap, source_qcap, wait_ticks,
                     cache_zcap, cache_qcap};
  I i = fork_worker();
  if (i < 0 || i >= workers) return;
  I used = 0;
  Sim s = {(Header *)(headers + i * GAME_HEADER_WIDTH),
           (Plant *)(plants + i * 45 * GAME_PLANT_WIDTH),
           (Zombie *)(zombies + i * ZCAP * GAME_ZOMBIE_WIDTH),
           (Shot *)(shots + i * QCAP * GAME_PROJECTILE_WIDTH),
           (Mower *)(mowers + i * 5 * GAME_MOWER_WIDTH), cooldowns + i * 8,
           schedules + i * ZCAP * 5, events + i * ECAP * 8, event_counts + i,
           facts + i * 9, forecast + i * 20, phases + i, starts + i, sky_rate};
  while (used < ticks) {
    I job = job_ids[i];
    if (job < 0) {
      if (*next_job >= jobs) break;
      job = fork_add(next_job, 1);
      if (job >= jobs) break;
      I root = roots[job];
      const Header *original = (const Header *)(source_h + root * GAME_HEADER_WIDTH);
      const I *op = source_p + root * 45 * GAME_PLANT_WIDTH;
      const I *oz = source_z + root * cache.source_zcap * GAME_ZOMBIE_WIDTH;
      const I *oq = source_q + root * cache.source_qcap * GAME_PROJECTILE_WIDTH;
      const I *om = source_m + root * 5 * GAME_MOWER_WIDTH;
      const I *oc = source_cd + root * 8;
      bool hit = false;
      if (baseline && cache.steps) {
        for (I e = 0; e < cache.steps; e++) if (cache.ticks[root * cache.steps + e] == original->tick) {
          if (fork_equal(original, op, oz, oq, om, oc,
                         cache.states + (root * cache.steps + e) * cache.stride,
                         cache.zcap, cache.qcap)) {
            for (I k = 0; k < 20; k++)
              results[job * 20 + k] = cache.values[(root * cache.steps + e) * 20 + k];
            fork_add(progress + 2, 1); fork_add(progress, 1); hit = true;
          } else {
            const I *saved = cache.states + (root * cache.steps + e) * cache.stride;
            fork_add(progress + (saved[4] != original->spawn_index ? 5 : 4), 1);
          }
          break;
        }
      }
      if (hit) continue;
      if (baseline) {
        fork_add(progress + 3, 1);
        bool active = false;
        const Zombie *present = reinterpret_cast<const Zombie *>(oz);
        for (I k = 0; k < original->nz; k++)
          active |= !present[k].headless && present[k].health > 0;
        if (!active) fork_add(progress + 7, 1);
        if (cache.steps && cache.ticks[root * cache.steps + cache.steps - 1] >= 0 &&
            original->tick > cache.ticks[root * cache.steps + cache.steps - 1])
          fork_add(progress + 6, 1);
      }
      // Fused cloning: only occupied prefixes are read. Engine capacities bound writes.
      I *dh = reinterpret_cast<I *>(s.h);
      for (I k = 0; k < GAME_HEADER_WIDTH; k++) dh[k] = source_h[root * GAME_HEADER_WIDTH + k];
      const I *parts[] = {op, oz, oq, om, oc};
      I *dest[] = {reinterpret_cast<I *>(s.p), reinterpret_cast<I *>(s.z),
                  reinterpret_cast<I *>(s.q), reinterpret_cast<I *>(s.m), s.cd};
      I live[] = {s.h->np * GAME_PLANT_WIDTH, s.h->nz * GAME_ZOMBIE_WIDTH,
                  s.h->nq * GAME_PROJECTILE_WIDTH, 5 * GAME_MOWER_WIDTH, 8};
      for (I part = 0; part < 5; part++)
        for (I k = 0; k < live[part]; k++) dest[part][k] = parts[part][k];
      *s.forecast_phase = 0; *s.forecast_start = s.h->tick;
      for (I k = 0; k < 20; k++) s.forecast[k] = k % 4 == 1 ? -1 : 0;
      s.h->enabled = 1; s.h->status = 0; s.h->total_spawns = s.h->spawn_index;
      s.suffix_steps = 0;
      if (baseline && cache.steps) {
        cache.origins[root] = s.h->tick;
        for (I e = 0; e < cache.steps; e++) {
          cache.ticks[root * cache.steps + e] = -1;
          for (I row = 0; row < 5; row++) cache.depths[(root * cache.steps + e) * 5 + row] = 0;
        }
      }
      s.step(actions[job], 0, true);
      job_ids[i] = job;
    }
    I root = roots[job];
    // Ready caches are disposable. A restored in-progress job has no cache origin.
    bool caching = baseline && cache.steps && cache.origins[root] == *s.forecast_start;
    s.suffix_steps = caching ? cache.steps : 0;
    s.suffix_ticks = caching ? cache.ticks + root * cache.steps : nullptr;
    s.suffix_depth = caching ? cache.depths + root * cache.steps * 5 : nullptr;
    if (*s.forecast_phase == 0 && !s.h->status && caching) {
      I elapsed = s.h->tick - *s.forecast_start;
      I e = elapsed / cache.wait_ticks - 1;
      bool lost = false;
      for (I row = 0; row < 5; row++) lost |= s.forecast[row * 4 + 2] != 0;
      if (!lost && elapsed > 0 && elapsed % cache.wait_ticks == 0 && e < cache.steps &&
          s.h->nz <= cache.zcap && s.h->nq <= cache.qcap) {
        cache.ticks[root * cache.steps + e] = s.h->tick;
        fork_pack(s, cache.states + (root * cache.steps + e) * cache.stride,
                  cache.zcap, cache.qcap);
        for (I k = 0; k < s.h->nz; k++) s.forecast_position(s.z[k]);
      }
    }
    if ((*s.forecast_phase == 0 || *s.forecast_phase == 1) && s.h->status)
      fork_probe(s, probe);
    if ((*s.forecast_phase == 2 || *s.forecast_phase == 3) && s.h->status) {
      *s.forecast_phase = 3;
      for (I k = 0; k < 20; k++) results[job * 20 + k] = s.forecast[k];
      if (caching) for (I e = 0; e < cache.steps; e++) {
        I at = cache.ticks[root * cache.steps + e];
        if (at < 0) continue;
        for (I row = 0; row < 5; row++) {
          I offset = (root * cache.steps + e) * 20 + row * 4;
          cache.values[offset] = s.forecast[row * 4 + 2] ? 10 :
              cache.depths[(root * cache.steps + e) * 5 + row];
          I time = s.forecast[row * 4 + 1];
          cache.values[offset + 1] = time < 0 ? -1 : time + *s.forecast_start - at;
          cache.values[offset + 2] = s.forecast[row * 4 + 2];
          cache.values[offset + 3] = s.forecast[row * 4 + 3];
        }
      }
      s.h->enabled = 0; job_ids[i] = -1; fork_add(progress, 1);
      continue;
    }
    s.advance(); used++;
  }
  fork_add(progress + 1, used);
}
