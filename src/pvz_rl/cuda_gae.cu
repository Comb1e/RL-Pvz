extern "C" __global__ void gae(const float *reward, const float *value,
                               const float *starts, const float *last_value,
                               const float *done, float *advantage,
                               float *returns, int steps, int games,
                               float gamma, float lambda) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= games)
    return;
  float last = 0;
  for (int t = steps - 1; t >= 0; --t) {
    int k = t * games + i;
    float mask = 1 - (t == steps - 1 ? done[i] : starts[k + games]);
    float next = t == steps - 1 ? last_value[i] : value[k + games];
    float delta = reward[k] + gamma * next * mask - value[k];
    last = delta + gamma * lambda * mask * last;
    advantage[k] = last;
    returns[k] = last + value[k];
  }
}
