# Sources actually used

## Persistent exploration

- Dabney et al., [Temporally-Extended epsilon-Greedy Exploration](https://arxiv.org/abs/2006.01782):
  inspected temporal persistence as a way to reduce dithering. This project commits
  to an alternative species until it can be planted; it does not sample random
  repetition durations or claim the paper's performance results.
- Bacon et al., [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
  and its [project](https://github.com/jeanharb/option_critic): inspected temporal
  abstraction and explicit option termination. The local controller has deterministic
  readiness/episode termination, not learned options, option critics or policy gradients.

## Selective event memory

- Campos et al., [Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/abs/1708.06834)
  and the [official project](https://imatge-upc.github.io/skiprnn-2017-telecombcn/):
  inspected for selective recurrent-state updates on low-information sequences.
  This project uses deterministic public sunlight/zombie/plant events, not learned
  skip gates, periodic updates or the paper's training mechanism. Current board
  features remain available at every decision through a separate fusion path.

This index records sources used by the current implementation and the particular
ideas adopted. Project choices such as the entity cap, embedding width, rewards
and mastery gates are not established by these papers. Release history and local
measurements belong in [iteration history](iteration.md).

## Entities and recurrent decisions

- Hester et al., [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732):
  accepted-demonstration action ranking alongside value regression. The project uses
  only its supervised preference idea, not its replay or DQN implementation.
- van Hasselt et al., [Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714):
  reward/target scale sensitivity motivated separating raw ledger values from the
  development multiplier; PopArt is not used.

- Vinyals et al., [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z):
  entity observations, structured action arguments and recurrent processing.
  No model weights, game features, league training or reported performance are adopted.
- [Official AlphaStar unit encoder and Transformer](https://github.com/google-deepmind/alphastar/blob/700b1e74364ed5dfc66f6cd2574c5ffac2fa474e/alphastar/architectures/components/units.py):
  shared numerical/categorical entity embeddings, absent-unit masks and contextual
  attention. [mini-AlphaStar §3.2](https://arxiv.org/html/2104.06890v2#S3.SS2)
  informed the distinction between raw entity lists and learned vectors.
- [AlphaStar Unplugged](https://arxiv.org/abs/2308.03526): inspected offline
  demonstration and return-learning motivation. The local one-game initialization
  does not reproduce its dataset, algorithm or generalization evidence.
- [SB3 2.7.0 HER replay buffer](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/her/her_replay_buffer.py):
  inspected reward recomputation through the environment reward function before
  tensor conversion. Demonstration reuse applies that separation to verified
  replay observations/events and current reward coefficients; it does not use
  hindsight goals or the HER algorithm.
- Hausknecht and Stone, [Deep Recurrent Q-Learning](https://arxiv.org/abs/1507.06527):
  abstract inspected for memory under partial observation. The local objective
  uses complete returns for selected actions and one-step EMA bootstrap for isolated
  alternative-action probes; it makes no claimed recurrence advantage.
- [SB3-Contrib 2.7.1 recurrent policies](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/common/recurrent/policies.py)
  and [buffers](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/common/recurrent/buffers.py):
  inspected sequence processing, episode-start resets, environment boundaries and
  padding. These informed state ownership, not PPO/GAE learning. Inspected source
  SHA-256 values: `7b2314050c1325f2ebbd051b570e5811eb1e0219ebeabf91cb5bca621b88e6df`
  and `b1c5632149bfe8db3472f8608ed0f28e0686d8bb937aec05890ca2cee57ef75c`.
- Metz et al., [Discrete Sequential Prediction of Continuous Actions for Deep RL, §2.2](https://arxiv.org/html/1705.05035v3):
  assemble action components before executing one command. Its off-policy Bellman
  backups are not used. [SB3 2.7.1 QNetwork](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.1/stable_baselines3/dqn/policies.py)
  supplied the inspected raw-value/argmax reference; greedy normal branches,
  tile sampling and accepted-plant commitments are local decisions.

## Precision and device execution

- [PyTorch 2.8 gradient clipping](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/clip_grad.py):
  `clip_grad_norm_` uses one total parameter-gradient norm and a clamped scaling
  factor. Both fitting paths supply the same configured limit after accumulation.
- [PyTorch 2.8 AMP examples](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/amp_examples.rst):
  accumulation and selective autocast, used with FP32 master weights and BF16
  feature computation. No FP16 scaler is used. Inspected SHA-256:
  `ca4e1c814873b1191b2ea7aae208d2be27e355d2e221f59c10e9f6afa8b874e4`.
- [PyTorch 2.8 SDPA](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/functional.py):
  efficient backend selection and numerical differences; one full call with an
  exact query-chunk fallback. Inspected SHA-256:
  `2b6eb4aff305990cff9e32c6ff40ff82c1fec4b5ceba1f43540146a8eb64e952`.
- [PyTorch pinned/nonblocking transfer tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/pinmem_nonblock.py):
  buffer reuse, copy completion and stream dependencies for pinned staging.
  Inspected SHA-256: `c19188e197f5bccb7c26be751ce894c71091c02470164a5431f3829bad604c33`.
- Rabe and Staats, [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682):
  abstract inspected to distinguish arithmetic from storage cost; no published
  speed or memory ratios are transferred to this project.
- [CuPy 13.6 kernel guide](https://raw.githubusercontent.com/cupy/cupy/v13.6.0/docs/source/user_guide/kernel.rst)
  and [interoperability guide](https://raw.githubusercontent.com/cupy/cupy/v13.6.0/docs/source/user_guide/interoperability.rst):
  RawModule/NVRTC and DLPack/ExternalStream ownership for CUDA adapters.
- [WarpDrive: Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU](https://jmlr.org/papers/v23/22-0185.html)
  and its [project README](https://github.com/salesforce/warp-drive/blob/master/README.md):
  inspected device-resident simulation/learning and reduced CPU/GPU round trips.
  These inform independent batched scratch execution, device packing and compact
  live-game/valid-probe inference through existing handoffs. The host active plan,
  compact scratch prefix views over shared allocation, canonical scatter,
  deferred bootstrap patching and graph lifetime rules are local choices; no code
  or published throughput claim is adopted.

## Simulator, evaluation and presentation

- Pinned game **1.7.0**, commit `1424b4d802e783a36772091c49d96aa7c6036d7a`:
  public observation formulas, species constants, validation reasons, per-tick
  execution, CPU/CUDA state schemas and damage events define the adapter contract.
  The [engine lock](../src/pvz_rl/data/engine-lock.json) is authoritative.
- [Patoke reconstruction](https://github.com/Patoke/re-plants-vs-zombies/tree/c4692036c5e11d227c8fb7c593b734dac96da028)
  and [Bamcane animation reference](https://github.com/Bamcane/re-plants-vs-zombies/tree/0f6bbd39302acf69484ba8b3e071724e35cfba17/pak/reanim):
  attack/body rectangles, movement cadence, explosions, projectile and mower
  contact informed the pinned game's mechanics audit. Reconstruction is not
  proof of original-binary equivalence; no artwork is distributed here.
- [Gymnasium environment API](https://gymnasium.farama.org/api/env/): reset/step
  contracts and explicit termination versus truncation.
- Cobbe et al., [Procedural Generation and RL Generalization](https://arxiv.org/abs/1912.01588):
  abstract inspected for separated training, validation and held-out scenarios.
- Agarwal et al., [Statistical Precipice](https://arxiv.org/abs/2108.13264) and
  [rliable](https://github.com/google-research/rliable/tree/3ccd9f4dea577a04d3d2b557f259aac08badbd81):
  independent runs and interval reporting; the crossed paired bootstrap is local.
- [SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
  and [logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html):
  collection/update reporting and serialization infrastructure.
- [pygame-ce events](https://pyga.me/docs/ref/event.html): bounded event queues,
  wheel/resize handling and continuous pumping in a separate viewer process.
- [pygame-ce Surface API](https://pyga.me/docs/ref/surface.html): rendered text
  widths and blitting for the responsive, read-only learning-settings strip.
- [FFmpeg](https://ffmpeg.org/), inspected local 8.1 help and encoders:
  optional RGB-to-H.264/yuv420p export with explicit dimensions and frame rate.

## Fused recurrent fitting

- [PyTorch `torch.compile` documentation](https://pytorch.org/docs/stable/torch.compiler.html):
  fixed-shape encoder compilation with a bounded eager fallback. The recurrent
  loop and ragged storage remain outside the compiled region.
- [PyTorch 2.8 CUDA-graphs backend](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/backends/cudagraphs.py):
  inspected AOT forward/backward compilation and `cudagraphs_inner`'s static
  input copies, side-stream warmup and output cloning. The local backend extends
  ownership to every saved activation and returned gradient, retaining exact
  strides for SDPA and copying updated weights before replay. It does not use
  CUDA-graph tree lifetime inference for delayed recurrent backward.
- [PyTorch LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html):
  independent sequence batching with one hidden/cell state per sequence. Fused
  groups preserve chronological order and reset boundaries inside each slot.

These references support execution controls only; the fused-group arithmetic,
whole-pass gradient accumulation and one-event timing boundary are local
implementation decisions. No learning-quality result is inferred from the
throughput measurements.

## Collection and terminal reporting

- [PyTorch 2.8 CUDA semantics and graphs](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/cuda.rst):
  inspected static-shape/control-flow constraints, stream synchronization and
  long-lived static input/output buffers with stable addresses across replay.
  These inform direct no-grad collection graphs, the owned AOT fitting backend on
  all CUDA platforms, separate phase owners and handoff-safe release of retired
  captures. Outputs are copied before reuse; timing uses the existing host boundary
  rather than waiting per event. LRU limits and headroom values are local controls.
- [NVIDIA CCCL CUB `DeviceSelect`](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_select.cuh):
  inspected selected-item compaction as inspiration for canonical gather/scatter.
  This header explicitly does not support NVRTC, which the local CuPy adapter uses;
  it is not imported, compiled or adopted as an implementation dependency. The host
  planner uses published live flags/counts; GPU probe representatives cross only
  the existing handoff.
- [PyTorch pinned/nonblocking tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/pinmem_nonblock.py):
  inspected pinned transfer ownership and copy-completion requirements. The single
  reusable handoff waits before CPU consumption and before a named buffer is reused.
- The compact terminal cadence, duplicate keys and separation from detailed JSON
  snapshots are local implementation decisions, not research-derived claims.
