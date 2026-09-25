// Uses the game's integer structs/rule constants. Only documented public fields
// contribute to the policy vector. Schedules, IDs and task names are never
// read.
__device__ double asset_value(Header &h, Plant *p) {
  double value = h.sun;
  for (I j = 0; j < h.np; j++) value += (double)PC[p[j].kind] * p[j].health / PH[p[j].kind];
  return value;
}
__device__ I bin_x(I x) {
  return hi(0, lo(BINS - 1, (x - G_house_x) * BINS / (G_spawn_x - G_house_x)));
}
extern "C" __global__ void encode_state(const I *headers, const I *plants,
                                        const I *zombies,
                                        const I *mowers,
                                        float *output, double *assets, I n) {
  I i = blockIdx.x;
  if (threadIdx.x || i >= n)
    return;
  Header h = ((Header *)headers)[i];
  Plant *p = (Plant *)(plants + i * 45 * GAME_PLANT_WIDTH);
  Zombie *z = (Zombie *)(zombies + i * ZCAP * GAME_ZOMBIE_WIDTH);
  Mower *m = (Mower *)(mowers + i * 5 * GAME_MOWER_WIDTH);
  float *o = output + i * OBS_SIZE;
  for (I j = 0; j < OBS_SIZE; j++)
    o[j] = 0;
  for (I j = 0; j < h.np; j++) {
    Plant a = p[j];
    I k = (a.row * 9 + a.col) * 3;
    o[k] = a.kind + 1;
    o[k + 1] = (double)a.health / PH[a.kind];
    o[k + 2] = BEHAVIOR[a.state];
  }
  I zs[5 * BINS * ZOMBIE_WIDTH] = {0};
  I nearest[5 * BINS];
  I nearest_pole[5 * BINS];
  for (I j = 0; j < 5 * BINS; j++) nearest[j] = 9223372036854775807LL;
  for (I j = 0; j < 5 * BINS; j++) nearest_pole[j] = 9223372036854775807LL;
  for (I j = 0; j < h.nz; j++) {
    Zombie a = z[j];
    I region = a.row * BINS + bin_x(a.x);
    I *cell = zs + region * ZOMBIE_WIDTH;
    cell[a.kind]++;
    cell[Z_health] += a.health;
    cell[Z_armor] += a.armor;
    nearest[region] = lo(nearest[region], a.x);
    if (a.has_pole && !a.headless)
      nearest_pole[region] = lo(nearest_pole[region], a.x);
  }
  for (I j = 0; j < 5 * BINS * ZOMBIE_WIDTH; j++) {
    I f = j % ZOMBIE_WIDTH;
    double scale = LOCAL_COUNT;
    if (f == Z_health)
      scale *= HP_SCALE;
    else if (f == Z_armor)
      scale *= ARMOR_SCALE;
    if (f == Z_nearest)
      o[ZOMBIE_OFFSET + j] = nearest[j / ZOMBIE_WIDTH] == 9223372036854775807LL
          ? EMPTY_DISTANCE
          : (double)(nearest[j / ZOMBIE_WIDTH] - G_house_x) / POSITION_SCALE;
    else if (f == Z_nearest_pole)
      o[ZOMBIE_OFFSET + j] = nearest_pole[j / ZOMBIE_WIDTH] == 9223372036854775807LL
          ? EMPTY_DISTANCE
          : (double)(nearest_pole[j / ZOMBIE_WIDTH] - G_house_x) / POSITION_SCALE;
    else
      o[ZOMBIE_OFFSET + j] = (double)zs[j] / scale;
  }
  float *g = o + GLOBAL_OFFSET;
  g[O_sun] = (double)h.sun / COST_SCALE;
  g[O_elapsed] = (double)h.tick / G_tick_rate / CUTOFF_SECONDS;
  g[O_wave] = (double)h.wave / WAVE_SCALE;
  g[O_total_waves] = (double)h.total_waves / WAVE_SCALE;
  g[O_initial] = (double)h.total_spawns / COUNT_SCALE;
  g[O_defeated] = (double)h.defeated / COUNT_SCALE;
  for (I r = 0; r < 5; r++)
    g[O_mower_spent_0 + r] = m[r].state == 2;
  for (I r = 0; r < 5; r++) {
    I nearest = 9223372036854775807LL;
    bool headless = false;
    for (I j = 0; j < h.nz; j++) if (z[j].row == r) {
      if (z[j].x < nearest) { nearest = z[j].x; headless = z[j].headless; }
      else if (z[j].x == nearest && !z[j].headless) headless = false;
    }
    o[HEADLESS_OFFSET + r] = headless;
  }
  assets[i] = asset_value(h, p);
}
// Reward order mirrors reward_parts, using double intermediates before casting
// the scalar reward to the same float32 rollout storage used by SB3.
extern "C" __global__ void
reward_metrics(const I *headers, const I *old_headers, const I *old_cd,
               const I *actions, const double *facts, const I *accounting, const double *before_assets,
               const double *after_assets, float *rewards, double *parts,
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
  const I *a = accounting + i * 3;
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
  v[F_development] = scale * v[F_net_value];
  v[F_mower_activation_penalty] = -scale * v[F_mower_expenditure];
  double total = v[F_terminal] + v[F_development];
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
