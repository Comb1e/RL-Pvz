// Uses the game's integer structs/rule constants. Only documented public fields
// contribute to the policy vector. Schedules, IDs and task names are never
// read.
__device__ double phi(Header &h, Plant *p) {
  if (h.status)
    return 0;
  double progress = R_defeated_weight * h.defeated / hi(1, h.total_spawns);
  I value = 0;
  for (I j = 0; j < h.np; j++) value += PC[p[j].kind];
  return progress + R_economy_weight * (h.sun + value) / R_economy_scale;
}
__device__ I bin_x(I x) {
  return hi(0, lo(BINS - 1, (x - G_house_x) * BINS / (G_spawn_x - G_house_x)));
}
extern "C" __global__ void encode_state(const I *headers, const I *plants,
                                        const I *zombies, const I *shots,
                                        const I *mowers, const I *cooldowns,
                                        float *output, double *potential, I n) {
  I i = blockIdx.x;
  if (threadIdx.x || i >= n)
    return;
  Header h = ((Header *)headers)[i];
  Plant *p = (Plant *)(plants + i * 45 * 8);
  Zombie *z = (Zombie *)(zombies + i * ZCAP * 15);
  Shot *q = (Shot *)(shots + i * QCAP * 6);
  Mower *m = (Mower *)(mowers + i * 5 * 4);
  const I *cd = cooldowns + i * 8;
  float *o = output + i * OBS_SIZE;
  for (I j = 0; j < OBS_SIZE; j++)
    o[j] = 0;
  for (I j = 0; j < h.np; j++) {
    Plant a = p[j];
    I k = (a.row * 9 + a.col) * 4;
    o[k] = a.kind + 1;
    o[k + 1] = (double)a.health / PH[a.kind];
    o[k + 2] = (double)hi(0, a.due - h.tick) / TIMER_SCALE;
    o[k + 3] = a.state + 1;
  }
  I zs[5 * BINS * 16] = {0};
  I nearest[5 * BINS];
  for (I j = 0; j < 5 * BINS; j++) nearest[j] = 9223372036854775807LL;
  for (I j = 0; j < h.nz; j++) {
    Zombie a = z[j];
    I *cell = zs + (a.row * BINS + bin_x(a.x)) * 16;
    cell[a.kind]++;
    cell[5] += a.health;
    cell[6] += a.armor;
    I region = a.row * BINS + bin_x(a.x);
    nearest[region] = lo(nearest[region], a.x);
    cell[8] += hi(0, a.slow_until - h.tick);
    cell[9] += a.has_pole;
    I timer = 0;
    if (a.state == 2)
      timer = hi(0, a.vault_until - h.tick);
    else if (a.state == 3) {
      bool slow = h.tick < a.slow_until;
      timer = (hi(0, G_bite_ticks * 2 - a.bite_progress) + (slow ? 0 : 1)) /
              (slow ? 1 : 2);
    }
    cell[10] += timer;
    cell[11 + a.state]++;
  }
  for (I j = 0; j < 5 * BINS * 16; j++) {
    I f = j % 16;
    double scale = LOCAL_COUNT;
    if (f == 5)
      scale *= HP_SCALE;
    else if (f == 6)
      scale *= ARMOR_SCALE;
    else if (f == 7)
      scale *= POSITION_SCALE;
    else if (f == 8 || f == 10)
      scale *= TIMER_SCALE;
    o[180 + j] = f == 7
       ? (nearest[j / 16] == 9223372036854775807LL ? 1. : (double)(nearest[j / 16] - G_house_x) / POSITION_SCALE)
       : (double)zs[j] / scale;
  }
  I qs[5 * BINS * 3] = {0};
  for (I j = 0; j < h.nq; j++) {
    Shot a = q[j];
    I *cell = qs + (a.row * BINS + bin_x(a.x)) * 3;
    cell[0]++;
    cell[1] += a.damage;
    cell[2] += a.icy;
  }
  for (I j = 0; j < 5 * BINS * 3; j++)
    o[PROJECTILE_OFFSET + j] =
        (double)qs[j] / (LOCAL_COUNT * (j % 3 == 1 ? DAMAGE_SCALE : 1.));
  I k = GLOBAL_OFFSET;
  o[k++] = (double)h.sun / COST_SCALE;
  o[k++] = (double)h.tick / G_tick_rate / CUTOFF_SECONDS;
  o[k++] = (double)h.wave / WAVE_SCALE;
  o[k++] = (double)h.total_waves / WAVE_SCALE;
  o[k++] = (double)h.total_spawns / COUNT_SCALE;
  o[k++] = (double)h.spawn_index / COUNT_SCALE;
  o[k++] = (double)h.defeated / COUNT_SCALE;
  for (I j = 0; j < 8; j++) o[k++] = (double)cd[j] / hi(1, PR[j]);
  for (I r = 0; r < 5; r++) {
    o[k++] = (double)m[r].x / POSITION_SCALE;
    for (I j = 0; j < 3; j++)
      o[k++] = m[r].state == j;
  }
  potential[i] = phi(h, p);
}
// Reward order mirrors reward_parts, using double intermediates before casting
// the scalar reward to the same float32 rollout storage used by SB3.
extern "C" __global__ void
reward_metrics(const I *headers, const I *old_headers, const I *old_cd,
               const I *actions, const double *facts, const double *before_phi,
               const double *after_phi, float *rewards, double *parts,
               double *totals, I n) {
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
  v[0] = h.status == 1 ? R_win_reward : h.status == 2 ? -R_loss_penalty : 0.;
  v[1] = SHAPED ? R_gamma * after_phi[i] - before_phi[i] : 0.;
  v[2] = f[0]; v[3] = f[1]; v[4] = f[2]; v[5] = f[4];
  v[6] = f[5]; v[7] = f[6]; v[8] = f[7]; v[9] = f[8];
  v[10] = -R_mower_activation_cost * f[5];
  double total = v[0] + v[1] + v[10];
  v[REWARD_SIZE - 1] = total;
  rewards[i] = (float)total;
  t[0] += total;
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
  t[7] += action == 0 || h.reason == 5;
  t[8] = fmax(t[8], (double)hi(b.sun, h.sun));
  bool opportunity = false;
  I kinds[3] = {1, 5, 7};
  for (I j = 0; j < 3; j++) {
    I kind = kinds[j];
    if (b.sun >= PC[kind] && old_cd[i * 8 + kind] == 0 && b.np < 45)
      opportunity = true;
  }
  t[9] += opportunity;
  if (action >= 1 && action <= 360 && h.accepted) {
    I kind = (action - 1) / 45;
    t[12 + kind]++;
    t[36 + (action - 1) % 45] = b.tick + 1;
    if (kind == 1 || kind == 5 || kind == 7) {
      t[10]++;
      if (t[11] < 0)
        t[11] = b.tick;
    }
  }
  if (action >= 361 && action <= 405 && h.accepted) {
    I tile = action - 361;
    double planted = t[36 + tile];
    t[35] += planted > 0 && b.tick - (planted - 1) <= EARLY_DIG_TICKS;
    t[36 + tile] = 0;
  }
  for (I j = 0; j < 9; j++) t[20 + j] += v[2 + j];
}
