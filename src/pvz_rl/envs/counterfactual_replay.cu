// Private replay metadata: slot, cursor, initial action, state, root tick.
// States: queued=0, running=1, resources=2, commands=3, complete=4, discarded=5.
// Stats: discounted economics; current wait length/discount; min/max closed
// run lengths; sums of discount * length^0,1,2; last decision tick; ticks;
// resource waits; maximum queue lag. The prefix is cloned with the root.
extern "C" __global__ void clone_lanes(const unsigned long long *source,
    unsigned long long *target, const I *from, const I *to, I width, I n) {
  I k = (I)blockIdx.x * blockDim.x + threadIdx.x;
  for (; k < width * n; k += (I)gridDim.x * blockDim.x) {
    I lane = k / width, field = k % width;
    target[to[lane] * width + field] = source[from[lane] * width + field];
  }
}
__device__ void close_wait(double *s) {
  double length = s[1];
  if (!length) return;
  s[3] = s[3] == 0 ? length : fmin(s[3], length);
  s[4] = fmax(s[4], length);
  s[5] += s[2]; s[6] += s[2] * length; s[7] += s[2] * length * length;
  s[1] = s[2] = 0;
}
extern "C" __global__ void replay_select(I *headers, const I *plants, const I *cd,
    const I *commands, const I *lengths, const I *published, const bool *ended,
    I *meta, I *actions, bool *eligible, double *stats, I capacity, I cutoff, I n) {
  I i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Header &h = ((Header *)headers)[i];
  I *m = meta + i * 5;
  h.enabled = 0; actions[i] = 0; eligible[i] = false;
  if (m[3] >= 4 || m[0] < 0) return;
  if (h.status || h.tick >= cutoff) { m[3] = 4; return; }
  I slot = m[0], action = 0;
  bool ordinary = false;
  bool initial = m[2] != 0;
  if (initial) { action = m[2]; m[2] = 0; }
  else {
    I cursor = m[1];
    const I *c = commands + (slot * capacity + cursor) * 4;
    // Compress only expired Wait ranges, never skip a non-Wait command.
    while (cursor < lengths[slot] && c[2] == 0 && h.tick >= c[1]
           && (cursor + 1 < lengths[slot] || ended[slot])) {
      cursor++; c += 4;
    }
    m[1] = cursor;
    if (!ended[slot] && h.tick >= published[slot]) { m[3] = 3; return; }
    if (cursor < lengths[slot]) {
      if (h.tick >= c[0]) {
        action = c[2]; ordinary = !action && c[3];
        if (action) {
          stats[i * 12 + 11] = fmax(stats[i * 12 + 11], (double)(h.tick - c[0]));
        }
      }
    } // Mandatory post-main draining Wait is not voluntary passivity.
  }
  m[3] = 1;
  if (action > 0 && action < ACTION_DIG_START) {
    I tile = (action - 1) % 45, kind = (action - 1) / 45;
    const Plant *p = ((const Plant *)plants) + i * 45;
    for (I j = 0; j < h.np; j++) {
      if (p[j].row * 9 + p[j].col == tile) { m[3] = 5; return; }
    }
    if (!(h.allowed & (1LL << kind))) { m[3] = 6; return; }
    if (h.sun < PC[kind] || cd[i * 8 + kind] > 0) {
      // The initial alternative was checked before cloning and must be legal.
      if (initial) { m[3] = 6; return; }
      action = 0; ordinary = false; m[3] = 2; stats[i * 12 + 10]++;
    } else if (!initial) m[1]++;
  } else if (action >= ACTION_DIG_START) {
    I tile = action - ACTION_DIG_START;
    const Plant *p = ((const Plant *)plants) + i * 45;
    bool occupied = false;
    for (I j = 0; j < h.np; j++) occupied |= p[j].row * 9 + p[j].col == tile;
    if (!occupied) action = 0; // Real interval, empty Dig is a no-op.
    m[1]++;
  }
  actions[i] = action; eligible[i] = ordinary; h.enabled = 1;
}
extern "C" __global__ void replay_accumulate(const I *headers, const I *before,
    const double *parts, const bool *eligible, double *stats, I *meta,
    const I *component_indices, const double *component_weights, I component_count,
    I cutoff, I n) {
  I i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Header h = ((const Header *)headers)[i], b = ((const Header *)before)[i];
  if (!b.enabled) return;
  double *s = stats + i * 12;
  const double *v = parts + i * REWARD_SIZE;
  double value = 0;
  for (I k = 0; k < component_count; k++)
    value += component_weights[k] * v[component_indices[k]];
  double factor = pow(GAMMA, (double)b.tick);
  s[0] += factor * value;
  if (eligible[i]) { s[1]++; s[2] += factor; }
  else close_wait(s);
  s[8] = b.tick; s[9] += h.advanced;
  if (meta && (h.status || h.tick >= cutoff)) meta[i * 5 + 3] = 4;
}
