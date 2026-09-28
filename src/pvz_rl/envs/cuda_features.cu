// Uses the game's integer structs/rule constants. Only documented public fields
// contribute to the policy vector. Schedules, IDs and task names are never
// read.
__device__ double asset_value(Header &h, Plant *p) {
  double value = h.sun;
  for (I j = 0; j < h.np; j++) value += (double)PC[p[j].kind] * p[j].health / PH[p[j].kind];
  return value;
}
// Each thread emits one real public entity; reserved simulator slots are skipped.
extern "C" __global__ void encode_state(const I *headers, const I *plants,
    const I *zombies, const I *projectiles, const I *mowers, const I *cooldowns,
    int *entities, float *globals, double *assets, I width, I n) {
  I i = blockIdx.x;
  if (i >= n) return;
  Header h = ((Header *)headers)[i];
  Plant *p = (Plant *)(plants + i * 45 * GAME_PLANT_WIDTH);
  Zombie *z = (Zombie *)(zombies + i * ZCAP * GAME_ZOMBIE_WIDTH);
  Shot *q = (Shot *)(projectiles + i * QCAP * GAME_PROJECTILE_WIDTH);
  Mower *m = (Mower *)(mowers + i * 5 * GAME_MOWER_WIDTH);
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
  }
  for (I index=threadIdx.x; index<width; index+=blockDim.x) {
    int *o = entities + (i * width + index) * ENTITY_WIDTH;
    for (I f=0; f<ENTITY_WIDTH; f++) o[f]=0;
    I j=index;
    if (j < h.np) {
      Plant a=p[j];
      o[E_type]=a.kind+1; o[E_state]=a.state+1; o[E_row]=a.row;
      o[E_x]=a.col*G_units_per_tile+G_units_per_tile/2; o[E_health]=a.health;
      o[E_phase_ticks]=hi(0, a.due-h.tick);
    } else if ((j-=h.np) < h.nz) {
      Zombie a=z[j];
      o[E_type]=a.kind+ZOMBIE_TYPE_START; o[E_state]=a.state+ZOMBIE_STATE_START;
      o[E_row]=a.row; o[E_x]=a.x; o[E_health]=a.health; o[E_armor]=a.armor;
      o[E_phase_ticks]=a.state==2 ? hi(0,a.vault_until-h.tick) : a.state==3
        ? (hi(0,G_bite_ticks*2-a.bite_progress)+(h.tick<a.slow_until ? 0:1))
          / (h.tick<a.slow_until ? 1:2) : 0;
      o[E_slow_ticks]=hi(0,a.slow_until-h.tick); o[E_has_pole]=a.has_pole; o[E_headless]=a.headless;
    } else if ((j-=h.nz) < h.nq) {
      Shot a=q[j];
      o[E_type]=PROJECTILE_TYPE_START+a.icy; o[E_row]=a.row; o[E_x]=a.x; o[E_damage]=a.damage;
    } else if ((j-=h.nq) < 5) {
      Mower a=m[j];
      o[E_type]=MOWER_TYPE; o[E_state]=MOWER_STATE_START+a.state; o[E_row]=a.row; o[E_x]=a.x;
    }
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
  v[F_invalid_plant_penalty] =
      action > 0 && action < ACTION_DIG_START && !h.accepted
      ? -R_invalid_plant_penalty : 0.;
  v[F_empty_dig_penalty] =
      action >= ACTION_DIG_START && action < ACTION_COUNT && !h.accepted
      && h.reason == EMPTY_TILE_REASON ? -R_empty_dig_penalty : 0.;
  double total = v[F_terminal] + v[F_development]
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
