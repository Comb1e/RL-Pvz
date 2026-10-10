// Uses the game's integer structs/rule constants. Only documented public fields
// contribute to the policy vector. Schedules, IDs and task names are never
// read.
__device__ double asset_value(Header &h, Plant *p) {
  double value = h.sun;
  for (I j = 0; j < h.np; j++) value += (double)PC[p[j].kind] * p[j].health / PH[p[j].kind];
  return value;
}

__device__ void clear_record(int *o) {
  for (I f = 0; f < ENTITY_WIDTH; f++) o[f] = 0;
}
__device__ void plant_record(const Header &h, const Plant &a, int *o) {
  clear_record(o);
  o[E_type]=a.kind+1; o[E_state]=a.state+1; o[E_row]=a.row;
  o[E_x]=a.col*G_units_per_tile+G_units_per_tile/2; o[E_health]=a.health;
  o[E_phase_ticks]=hi(0, a.due-h.tick);
}
__device__ void zombie_record(const Header &h, const Zombie &a, int *o) {
  clear_record(o);
  o[E_type]=a.kind+ZOMBIE_TYPE_START; o[E_state]=a.state+ZOMBIE_STATE_START;
  o[E_row]=a.row; o[E_x]=a.x; o[E_health]=a.health; o[E_armor]=a.armor;
  o[E_phase_ticks]=a.state==2 ? hi(0,a.vault_until-h.tick) : a.state==3
    ? (hi(0,G_bite_ticks*2-a.bite_progress)+(h.tick<a.slow_until ? 0:1))
      / (h.tick<a.slow_until ? 1:2) : 0;
  o[E_slow_ticks]=hi(0,a.slow_until-h.tick); o[E_has_pole]=a.has_pole; o[E_headless]=a.headless;
}
__device__ void shot_record(const Shot &a, int *o) {
  clear_record(o);
  o[E_type]=PROJECTILE_TYPE_START+a.icy; o[E_row]=a.row; o[E_x]=a.x; o[E_damage]=a.damage;
}
__device__ void mower_record(const Mower &a, int *o) {
  clear_record(o);
  o[E_type]=MOWER_TYPE; o[E_state]=MOWER_STATE_START+a.state; o[E_row]=a.row; o[E_x]=a.x;
}

// Canonical order within one group: lane (mowers and plants), x, then every
// public field. Equal records are interchangeable; index breaks the tie.
__device__ int record_order(const int *a, const int *b, bool lanes) {
  if (lanes && a[E_row] != b[E_row]) return a[E_row] < b[E_row] ? -1 : 1;
  if (a[E_x] != b[E_x]) return a[E_x] < b[E_x] ? -1 : 1;
  for (I f = 0; f < ENTITY_WIDTH; f++)
    if (a[f] != b[f]) return a[f] < b[f] ? -1 : 1;
  return 0;
}
// Projectile records differ only in x, type, row and damage, compared after the
// same int32 conversion as the stored record.
__device__ int shot_order(const Shot &a, const Shot &b) {
  int values[8] = {(int)a.x, (int)b.x, (int)(PROJECTILE_TYPE_START + a.icy),
                   (int)(PROJECTILE_TYPE_START + b.icy), (int)a.row, (int)b.row,
                   (int)a.damage, (int)b.damage};
  for (int k = 0; k < 8; k += 2)
    if (values[k] != values[k + 1]) return values[k] < values[k + 1] ? -1 : 1;
  return 0;
}

// One block per game. The stable lexicographic order (group, lane, x, fields)
// is reproduced by ranking each entity within its group, so no host read or
// sort is needed. Mowers, plants and zombies are staged once; projectiles are
// ranked from simulator slots, supporting the full allocated capacity. Records
// past ENTITY_LIMIT are omitted and counted; later store positions are cleared.
extern "C" __global__ void encode_canonical(const I *headers, const I *plants,
    const I *zombies, const I *projectiles, const I *mowers, const I *cooldowns,
    int *records, int *entities, bool *mask, float *globals, double *assets,
    I *summary, I n) {
  I i = blockIdx.x;
  if (i >= n) return;
  Header h = ((Header *)headers)[i];
  if (!h.enabled) {
    for (I index = threadIdx.x; index < ENTITY_LIMIT; index += blockDim.x) {
      clear_record(entities + (i * ENTITY_LIMIT + index) * ENTITY_WIDTH);
      mask[i * ENTITY_LIMIT + index] = false;
    }
    for (I index = threadIdx.x; index < GLOBAL_WIDTH; index += blockDim.x)
      globals[i * GLOBAL_WIDTH + index] = 0;
    if (!threadIdx.x) {
      assets[i] = 0;
      for (I index = 0; index < 4; index++) summary[i * 4 + index] = 0;
    }
    return;
  }
  Plant *p = (Plant *)(plants + i * 45 * GAME_PLANT_WIDTH);
  Zombie *z = (Zombie *)(zombies + i * ZCAP * GAME_ZOMBIE_WIDTH);
  Shot *q = (Shot *)(projectiles + i * QCAP * GAME_PROJECTILE_WIDTH);
  Mower *m = (Mower *)(mowers + i * 5 * GAME_MOWER_WIDTH);
  int *staged = records + i * RECORD_CAPACITY * ENTITY_WIDTH;
  int *out = entities + i * ENTITY_LIMIT * ENTITY_WIDTH;
  I staged_count = 5 + h.np + h.nz;
  I kept = lo(staged_count + h.nq, ENTITY_LIMIT);
  if (!threadIdx.x) {
    float *g = globals + i * GLOBAL_WIDTH;
    g[O_sun] = (double)h.sun / COST_SCALE;
    g[O_elapsed] = (double)h.tick / G_tick_rate / CUTOFF_SECONDS;
    g[O_wave] = (double)h.wave / WAVE_SCALE;
    g[O_total_waves] = (double)h.total_waves / WAVE_SCALE;
    g[O_initial] = (double)h.total_spawns / COUNT_SCALE;
    g[O_defeated] = (double)h.defeated / COUNT_SCALE;
    const I recharge[8] = {CD_0, CD_1, CD_2, CD_3, CD_4, CD_5, CD_6, CD_7};
    for (I j=0; j<8; j++) g[O_cooldown_0+j] = (double)cooldowns[i*8+j] / recharge[j];
    g[O_plant_count] = (double)h.np / COUNT_SCALE;
    g[O_zombie_count] = (double)h.nz / COUNT_SCALE;
    g[O_projectile_count] = (double)h.nq / COUNT_SCALE;
    g[O_mower_count] = 5.0 / COUNT_SCALE;
    assets[i] = asset_value(h, p);
    I *s = summary + i * 4;
    s[0] = kept;
    s[1] = hi(0, h.np - (ENTITY_LIMIT - 5));
    s[2] = hi(0, h.nz - hi(0, ENTITY_LIMIT - 5 - h.np));
    s[3] = hi(0, h.nq - hi(0, ENTITY_LIMIT - 5 - h.np - h.nz));
  }
  for (I j = threadIdx.x; j < staged_count; j += blockDim.x) {
    int *o = staged + j * ENTITY_WIDTH;
    if (j < 5) mower_record(m[j], o);
    else if (j < 5 + h.np) plant_record(h, p[j - 5], o);
    else zombie_record(h, z[j - 5 - h.np], o);
  }
  __syncthreads();
  for (I j = threadIdx.x; j < staged_count; j += blockDim.x) {
    I start = j < 5 ? 0 : j < 5 + h.np ? 5 : 5 + h.np;
    I end = j < 5 ? 5 : j < 5 + h.np ? 5 + h.np : staged_count;
    const int *own = staged + j * ENTITY_WIDTH;
    I rank = start;
    for (I k = start; k < end; k++) {
      if (k == j) continue;
      int order = record_order(staged + k * ENTITY_WIDTH, own, j < 5 + h.np);
      rank += order < 0 || (order == 0 && k < j);
    }
    if (rank < ENTITY_LIMIT)
      for (I f = 0; f < ENTITY_WIDTH; f++) out[rank * ENTITY_WIDTH + f] = own[f];
  }
  I available = ENTITY_LIMIT - staged_count;
  for (I j = threadIdx.x; j < h.nq && available > 0; j += blockDim.x) {
    Shot a = q[j];
    I rank = 0;
    // Stop once this shot cannot be kept; ranks of kept shots stay exact.
    for (I k = 0; k < h.nq && rank < available; k++) {
      if (k == j) continue;
      int order = shot_order(q[k], a);
      rank += order < 0 || (order == 0 && k < j);
    }
    if (rank < available) shot_record(a, out + (staged_count + rank) * ENTITY_WIDTH);
  }
  for (I index = threadIdx.x; index < ENTITY_LIMIT; index += blockDim.x) {
    mask[i * ENTITY_LIMIT + index] = index < kept;
    if (index >= kept) clear_record(out + index * ENTITY_WIDTH);
  }
}

// Policy masks expose only board geometry.  The sequential Q controller keeps
// all ten branch values visible; the simulator remains authoritative for
// affordability, cooldown, and rejection reasons.
extern "C" __global__ void policy_masks(const I *headers, const I *plants,
                                         bool *masks, I n) {
  I index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n * ACTION_COUNT)
    return;
  I game = index / ACTION_COUNT;
  I action = index % ACTION_COUNT;
  Header h = ((Header *)headers)[game];
  if (!h.enabled || h.status != 0 || h.tick >= CUTOFF_SECONDS * G_tick_rate) {
    masks[index] = false;
    return;
  }
  bool available = action == 0 || action >= ACTION_DIG_START;
  if (action > 0 && action < ACTION_DIG_START) {
    I tile = (action - 1) % ACTION_TILES;
    I row = tile / 9, col = tile % 9;
    available = true;
    Plant *p = (Plant *)(plants + game * 45 * GAME_PLANT_WIDTH);
    for (I j = 0; j < h.np; j++)
      if (p[j].row == row && p[j].col == col) available = false;
  }
  masks[index] = available;
}

// Copy only occupied simulator slots, never the reserved projectile capacity.
extern "C" __global__ void copy_probe_state(
    const Header *sh, const Plant *sp, const Zombie *sz, const Shot *sq,
    const Mower *sm, const I *sc, const I *ss, const I *sl, const double *sa,
    Header *dh, Plant *dp, Zombie *dz, Shot *dq, Mower *dm, I *dc, I *ds, I *dl,
    double *da, const I *source_ids, const I *target_ids, const bool *active) {
  I row = blockIdx.x, t = threadIdx.x;
  I source = source_ids[row], i = target_ids[row];
  if (!active[row] || !sh[source].enabled) return;
  Header h = sh[source];
  if (t == 0) { dh[i] = h; da[i] = sa[source]; }
  for (I j=t;j<h.np;j+=blockDim.x) dp[i*45+j]=sp[source*45+j];
  for (I j=t;j<h.nz;j+=blockDim.x) dz[i*ZCAP+j]=sz[source*ZCAP+j];
  for (I j=t;j<h.nq;j+=blockDim.x) dq[i*QCAP+j]=sq[source*QCAP+j];
  for (I j=t;j<5;j+=blockDim.x) dm[i*5+j]=sm[source*5+j];
  for (I j=t;j<8;j+=blockDim.x) dc[i*8+j]=sc[source*8+j];
  for (I j=t;j<ZCAP*5;j+=blockDim.x) ds[i*ZCAP*5+j]=ss[source*ZCAP*5+j];
  for (I j=t;j<ZCAP*2;j+=blockDim.x) dl[i*ZCAP*2+j]=sl[source*ZCAP*2+j];
}

extern "C" __global__ void dedup_probes(const int *entities, const I *summary,
    const float *globals, const I *history,
    const bool *valid, I *source, I n, I slots,
    const float *hidden, const float *cell, I memory_width) {
  I game = blockIdx.x;
  if (game >= n) return;
  __shared__ int equal;
  for (I slot = 0; slot < slots; slot++) {
    I row = slot * n + game;
    I chosen = valid[row] ? slot : -1;
    for (I prior = 0; prior < slot && chosen == slot; prior++) {
      I other = prior * n + game;
      if (!valid[other] || summary[row * 4] != summary[other * 4])
        continue;
      if (!threadIdx.x) equal = 1;
      __syncthreads();
      const int *a = entities + row * ENTITY_LIMIT * ENTITY_WIDTH;
      const int *b = entities + other * ENTITY_LIMIT * ENTITY_WIDTH;
      for (I k = threadIdx.x; k < summary[row * 4] * ENTITY_WIDTH; k += blockDim.x)
        if (a[k] != b[k]) atomicExch(&equal, 0);
      for (I k = threadIdx.x; k < HISTORY_WIDTH; k += blockDim.x)
        if (history[row * HISTORY_WIDTH + k] != history[other * HISTORY_WIDTH + k]) atomicExch(&equal, 0);
      for (I k = threadIdx.x; k < GLOBAL_WIDTH; k += blockDim.x)
        if (globals[row * GLOBAL_WIDTH + k] != globals[other * GLOBAL_WIDTH + k]) atomicExch(&equal, 0);
      for (I k = threadIdx.x; k < memory_width; k += blockDim.x)
        if (hidden[row * memory_width + k] != hidden[other * memory_width + k]
            || cell[row * memory_width + k] != cell[other * memory_width + k]) atomicExch(&equal, 0);
      __syncthreads();
      if (equal) chosen = source[other];
      __syncthreads();
    }
    if (!threadIdx.x) source[row] = chosen;
    __syncthreads();
  }
}

// Copy each selected row's kept canonical records to its exclusive-scan offset.
extern "C" __global__ void pack_entities(const int *entities, const I *summary,
    const I *offsets, const bool *selected, int *packed, I rows) {
  I row = blockIdx.x;
  if (row >= rows || !selected[row]) return;
  const int *source = entities + row * ENTITY_LIMIT * ENTITY_WIDTH;
  int *target = packed + offsets[row] * ENTITY_WIDTH;
  for (I k = threadIdx.x; k < summary[row * 4] * ENTITY_WIDTH; k += blockDim.x)
    target[k] = source[k];
}

// Each episode's total roster is bounded by ZCAP. Keep IDs only in accounting.
extern "C" __global__ void home_entries(const I *headers, const I *zombies,
    I *ledger, I *entries, const bool *selected, I n) {
  I i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  entries[i * 2] = entries[i * 2 + 1] = 0;
  if (!selected[i]) return;
  Header h = ((const Header *)headers)[i];
  const Zombie *z = ((const Zombie *)zombies) + i * ZCAP;
  I *seen = ledger + i * ZCAP * 2;
  for (I j = 0; j < h.nz; j++) {
    if (z[j].headless || z[j].health <= 0) continue;
    I stage = (z[j].x < 2 * G_units_per_tile) + (z[j].x < G_units_per_tile);
    I slot = 0;
    while (slot < ZCAP && seen[slot * 2] && seen[slot * 2] != z[j].id) slot++;
    if (slot == ZCAP) { entries[i * 2] = -1; return; }
    I old = seen[slot * 2 + 1];
    for (I s = old; s < stage; s++) entries[i * 2 + s]++;
    seen[slot * 2] = z[j].id;
    seen[slot * 2 + 1] = hi(old, stage);
  }
}

// Reward order mirrors reward_parts, using double intermediates before casting
// the scalar reward to the same float32 rollout storage used by SB3.
extern "C" __global__ void
reward_metrics(const I *headers, const I *old_headers, const I *old_cd,
               const I *actions, const double *facts, const I *accounting, const double *before_assets,
               const double *after_assets, float *rewards, double *parts,
               double *totals, const I *home_entries, I n) {
  I i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Header h = ((Header *)headers)[i], b = ((Header *)old_headers)[i];
  const double *f = facts + i * 9;
  double *v = parts + i * REWARD_SIZE;
  double *t = totals + i * METRIC_SIZE;
  I action = actions[i];
  for (I j = 0; j < REWARD_SIZE; j++)
    v[j] = 0;
  if (!b.enabled) {
    rewards[i] = 0;
    return;
  }
  const I *a = accounting + i * ACCOUNTING_WIDTH;
  v[F_terminal] = h.status == 1 ? R_win_reward : h.status == 2 ? -R_loss_penalty : 0.;
  if (h.status == 0 && h.tick >= CUTOFF_SECONDS * G_tick_rate)
    v[F_terminal] = -R_loss_penalty;
  v[F_plant_kills] = f[0]; v[F_mower_kills] = f[1];
  v[F_nonlethal_health_damage] = f[2]; v[F_empty_mower_activations] = f[4];
  v[F_mower_activations] = f[5]; v[F_mower_activation_sun] = f[6];
  v[F_wall_nut_damage] = f[7]; v[F_empty_explosions] = f[8];
  v[F_effective_damage] = a[0]; v[F_sky_income] = a[1]; v[F_produced_sun] = a[2];
  double resources = after_assets[i] - before_assets[i] - a[1];
  v[F_plant_value_loss] = a[2] - resources;
  v[F_combat_value] = R_basic_zombie_value * a[0] / BASIC_HP;
  v[F_mower_expenditure] = R_mower_value * f[5];
  v[F_net_value] = resources + v[F_combat_value] - v[F_mower_expenditure];
  double scale = R_progress_weight / R_value_scale;
  v[F_early_sun] = a[10];
  v[F_early_sun_bonus] = scale * R_early_sun_extra_multiplier * a[10];
  v[F_development] = scale * v[F_net_value] + v[F_early_sun_bonus];
  v[F_mower_activation_penalty] = -scale * v[F_mower_expenditure];
  v[F_invalid_plant_penalty] =
      action > 0 && action < ACTION_DIG_START && !h.accepted
      ? -R_invalid_plant_penalty : 0.;
  v[F_empty_dig_penalty] =
      action >= ACTION_DIG_START && action < ACTION_COUNT && !h.accepted
      && h.reason == EMPTY_TILE_REASON ? -R_empty_dig_penalty : 0.;
  v[F_home_outer_entries] = home_entries[i * 2];
  v[F_home_inner_entries] = home_entries[i * 2 + 1];
  v[F_home_proximity] = -HOME_PENALTY_0 * home_entries[i * 2]
                       -HOME_PENALTY_1 * home_entries[i * 2 + 1];
  // A corrupted/exhausted ledger must fail at the existing host boundary,
  // never turn the negative overflow sentinel into a positive reward.
  if (home_entries[i * 2] < 0) v[F_home_proximity] = 0.0 / 0.0;
  double total = v[F_terminal] + v[F_development] + v[F_home_proximity]
      + v[F_invalid_plant_penalty] + v[F_empty_dig_penalty];
  v[F_total] = total;
  rewards[i] = (float)total;
  t[0] += total;
  t[T_discounted_return] += pow(GAMMA, t[2]) * total;
  t[T_discounted_outcome_return] += pow(GAMMA, t[2]) * v[F_terminal];
  t[T_discounted_development_return] += pow(GAMMA, t[2]) * v[F_development];
  t[1]++;
  t[2] += h.advanced;
  t[3] += h.advanced == 0;
  if (action && h.accepted) {
    t[4]++;
    t[5] = fmax(t[5], t[4]);
  }
  if (h.advanced)
    t[4] = 0;
  t[6] += !h.accepted;
  // Rejected proposals advance one tick as automatic waits, so metrics count
  // their resolved execution action rather than only explicit Wait commands.
  t[7] += action == 0 || !h.accepted;
  t[8] = fmax(t[8], (double)hi(b.sun, h.sun));
  bool opportunity = false;
  I kinds[3] = {1, 5, 7};
  for (I j = 0; j < 3; j++) {
    I kind = kinds[j];
    if (b.sun >= PC[kind] && old_cd[i * 8 + kind] == 0 && b.np < 45)
      opportunity = true;
  }
  t[9] += opportunity;
  if (action >= 1 && action < ACTION_DIG_START && h.accepted) {
    I kind = (action - 1) / ACTION_TILES;
    t[12 + kind]++;
    t[36 + (action - 1) % ACTION_TILES] = b.tick + 1;
    if (kind == 1 || kind == 5 || kind == 7) {
      t[10]++;
      if (t[11] < 0)
        t[11] = b.tick;
    }
  }
  if (action >= ACTION_DIG_START && action < ACTION_COUNT && h.accepted) {
    I tile = action - ACTION_DIG_START;
    double planted = t[36 + tile];
    t[35] += planted > 0 && b.tick - (planted - 1) <= EARLY_DIG_TICKS;
    t[36 + tile] = 0;
  }
  // Named indices come from the shared Python reporting schema.
  METRIC_ACCUMULATION
  t[T_cumulative_net_value] = t[T_net_value];
  t[T_maximum_net_value] = fmax(t[T_maximum_net_value], t[T_cumulative_net_value]);
  t[T_value_drawdown] = t[T_maximum_net_value] - t[T_cumulative_net_value];
}
