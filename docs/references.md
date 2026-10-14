# Sources actually used

## Tile-to-species target reliability — research inspection, 2026-10-13

The current objective adopts online selection/EMA evaluation and robust Huber
consistency below. Other inspected directions remain research comparisons, not
supported runtime alternatives. None establishes target correctness or a guaranteed
learning improvement; implementation verification is tracked in iteration history.

- van Hasselt et al., [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/html/1509.06461v3):
  inspected equation 4 and the Double DQN selection/evaluation separation. A local
  adaptation selects a geometry-candidate tile using detached online tile values,
  then evaluates that tile with frozen EMA weights for detached species consistency
  targets. Existing online tables and the EMA forward are reused; the architecture,
  complete returns, exploration and probes are unchanged. Correlated estimators can
  still share errors; this is not a correctness guarantee or adoption of DQN.
- [PyTorch 2.8 `HuberLoss` implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py):
  inspected its piecewise half-squared/linear formula. Species consistency uses
  twice unreduced Huber with delta 1.0 to retain MSE loss/gradients for small errors
  and bound large-error influence. Global group/outcome balancing remains unchanged.
  Delta is in training-return units; targets/rewards are not clipped and samples
  are not gated or excluded. Robust gradients do not establish target correctness.
- Lan et al., [Maxmin Q-learning: Controlling the Estimation Bias of Q-learning, section 4](https://arxiv.org/html/2002.06487v2):
  inspected the minimum across estimators for each action before maximization,
  and random data subsampling to reduce estimator dependence. Two independently
  trained tile estimators could supply a conservative species target. Additional
  memory/inference and underestimation are tradeoffs; the paper's tabular results
  do not guarantee accuracy for this project's Transformers.
- Anschel et al., [Averaged-DQN, section 3](https://arxiv.org/html/1611.01929v4):
  inspected averaging previous value predictions to reduce target variance.
  Averaging tile predictions before selecting a maximum is an alternative to a
  hard minimum, not a cure for shared systematic error. Averaging network weights
  is not generally equivalent to averaging nonlinear network predictions.
- Feinberg et al., [Model-Based Value Estimation, definition 3.1](https://arxiv.org/html/1803.00101v1):
  re-inspected bounded rewards plus a tail value. A project-specific adaptation
  could reserve same-species probes for predicted winner tiles and use completed
  rollout evidence to check species targets. Rotating four tiles does not verify
  the maximum over all candidates; actual exploratory returns and frozen-policy
  probe returns also have different continuation policies. With gamma one, a
  longer rollout does not discount away endpoint value error. The planting trigger
  and winner verification are local proposals, not claims of the paper.
- [Explorer project README](https://github.com/qlan3/Explorer): inspected its
  documented Double DQN, Maxmin DQN, Averaged DQN and ensemble implementations as
  implementation references. The project is no longer actively maintained; no
  dependency, code, replay algorithm or benchmark claim is adopted.

## Stateless two-stage attention — 2026-10-13

- Metz et al., [Discrete Sequential Prediction of Continuous Actions for Deep RL, §2.2](https://arxiv.org/html/1705.05035v3):
  inspected equation 7's internal conditional-max consistency and equation 8's
  final-action consistency. The species network learns a stopped-gradient target
  from EMA evaluation of the online-selected best geometry candidate, rather than
  a sampled poor tile's return. The two local networks, complete-return regression, EMA balancing
  and tile-probe schedule are project choices; SDQN's off-policy algorithm is not
  implemented. This is a learned consistency target, not an exact species maximum
  substituted into normal branch selection.
- Zambaldi et al., [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830):
  inspected spatial/entity self-attention for reasoning about relations in current
  observations. This informs separate current-board species and conditional spatial
  tile encoders. No agent algorithm, recurrence, dependencies or performance claims
  are adopted.
- Lee et al., [Set Transformer](https://arxiv.org/abs/1810.00825) and the
  [official project](https://github.com/juho-lee/set_transformer): inspected
  permutation-invariant attention over sets and query-based readouts. Public entity
  tokens plus global/coordinate queries use these spatial set-attention ideas;
  learned parameters remain independent between stages. Inducing-point attention,
  the paper's algorithm and project dependencies are not used. Network widths,
  depths, heads and the 256-entity cap are local choices, not proven optimal.

Current sources below are distinguished from explicitly historical inspections.
Historical references preserve attribution only; they are not runtime alternatives.
Measurements and version narratives belong in [iteration history](iteration.md).

## Persistent exploration

- Dabney et al., [Temporally-Extended epsilon-Greedy Exploration](https://arxiv.org/abs/2006.01782):
  inspected temporal persistence as a way to reduce dithering. This project commits
  to an alternative species until it can be planted; it does not sample random
  repetition durations or claim the paper's performance results.
- Bacon et al., [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
  and its [project](https://github.com/jeanharb/option_critic): inspected temporal
  abstraction and explicit option termination. The local controller has deterministic
  readiness/episode termination, not learned options, option critics or policy gradients.

## Historical selective-memory inspection

- Campos et al., [Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/abs/1708.06834)
  and the [official project](https://imatge-upc.github.io/skiprnn-2017-telecombcn/):
  inspected for selective recurrent-state updates on low-information sequences.
  The former event-memory implementation used deterministic public events instead
  of learned skip gates. That implementation and its fusion/history paths have
  been removed; the current two-stage policy is stateless. This reference records
  the prior inspection, not a supported memory mechanism.

This index records adopted ideas and explicitly labelled historical/comparative
inspections. Entity caps, embedding widths, rewards and mastery gates are not
established by these papers.

## Entities, values and demonstration supervision

- Hester et al., [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732):
  accepted-demonstration action ranking alongside value regression. The project uses
  only its supervised preference idea, not its replay or DQN implementation.
- van Hasselt et al., [Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714):
  reward/target scale sensitivity motivated separating raw ledger values from the
  development multiplier; PopArt is not used.

- Vinyals et al., [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z):
  entity observations and structured action arguments. Its recurrence is not used
  by the current stateless model.
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
  historical abstract inspection for memory under partial observation. Recurrence
  has been removed; this source supplies no current runtime mechanism or claim
  that statelessness improves learning.
- [SB3-Contrib 2.7.1 recurrent policies](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/common/recurrent/policies.py)
  and [buffers](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/common/recurrent/buffers.py):
  historically inspected sequence processing, episode-start resets and padding.
  Those recurrent paths are removed; PPO/GAE was never adopted. Inspected source
  SHA-256 values: `7b2314050c1325f2ebbd051b570e5811eb1e0219ebeabf91cb5bca621b88e6df`
  and `b1c5632149bfe8db3472f8608ed0f28e0686d8bb937aec05890ca2cee57ef75c`.
- Metz et al., [Discrete Sequential Prediction of Continuous Actions for Deep RL, §2.2](https://arxiv.org/html/1705.05035v3):
  assemble action components before executing one command. Its off-policy Bellman
  backups are not used. For the 2026-10-09 species/tile credit discussion, inspected
  equation 7's internal maximization backup and equation 8's terminal-component
  consistency, rather than assigning the sampled placement return to both heads.
  These conditional-max ideas now inform stopped-gradient, online-selected/EMA-evaluated tile targets for
  independent species predictions; normal inference does not derive species Q
  values from maxima. The current inspection/adoption is detailed above; the
  paper's off-policy algorithm is not adopted.
  [SB3 2.7.1 QNetwork](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.1/stable_baselines3/dqn/policies.py)
  supplied the inspected raw-value/argmax reference; greedy normal branches,
  tile sampling and accepted-plant commitments are local decisions.

## Species/tile credit research — 2026-10-09

This earlier comparison motivated conditional species/tile attribution and bounded
greedy-EMA probes. The current objective uses double-estimator two-stage consistency, described
above, not exact derived branch maxima. COMA and action
branching remain comparisons only, not adopted actors, advantages or privileged inputs.

- Foerster et al., [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/html/1705.08926v3)
  and [PyMARL's COMA learner](https://github.com/oxwhirl/pymarl/blob/c971afdceb34635d31b778021b0ef90d7af51e86/src/learners/coma_learner.py):
  inspected the conditional counterfactual baseline, detached advantage and actor
  loss. Holding species fixed while comparing tiles can inform placement-specific
  credit. This is an adaptation of the comparison idea: action components are not
  independent agents, and advantages cannot replace absolute Q-return targets
  without a matching objective. The paper's privileged critic inputs are not
  proposed as policy inputs here.
- Feinberg et al., [Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning](https://arxiv.org/html/1803.00101v1)
  and its [project README](https://github.com/vlad17/mve/blob/19835bba87a2e5abc9bca653d011bf2eed68dd62/README.md):
  inspected definition 3.1's bounded rollout rewards plus terminal value estimate.
  Bounded isolated probes expose delayed placement effects before EMA
  bootstrap. The project already has a simulator; this implementation does not use
  learned dynamics or old dependencies, nor does it treat one-step neural
  estimates as observed counterfactual outcomes. Re-inspected definition 3.1 and
  the project README on 2026-10-10 for accepted-plant 30-second probes. The
  planting trigger and same-species alternative-tile coverage are local design
  choices, not claims from the paper. The current species-tail bootstrap retains
  the bounded reward-plus-tail idea, not a learned dynamics model.
- Tavakoli et al., [Action Branching Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1711.08946)
  and the [authors' project](https://github.com/atavakol/action-branching-agents/tree/1e026011505779e0ad30060ab3422f97eff08300):
  inspected the abstract and project README as a contrasting factorized-action
  design. Independent branches alone do not establish which component caused a
  poor result; species-dependent placement interactions must remain represented.

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

## Batched encoder execution

- [PyTorch `torch.compile` documentation](https://pytorch.org/docs/stable/torch.compiler.html):
  fixed-shape encoder compilation with a bounded eager fallback. Flat transport,
  accounting and ragged storage remain outside the compiled region.
- [PyTorch 2.8 compile API/source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/__init__.py)
  and [Dynamo cache limits](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/cache_size.py):
  inspected code-object cache ownership and the per-cache recompilation limits.
  Species/tile fitting caches must not share their shape allowance; quiescent
  phase releases renew the owner's cache instead of raising global limits.
- [Python 3.12 code objects](https://docs.python.org/3.12/reference/datamodel.html#codeobject.replace):
  inspected `code.replace()` to create an independent code object while preserving
  encoder bytecode, globals, defaults and bound model parameters. This is an
  execution-cache boundary, not a model or checkpoint-format change.
- [PyTorch 2.8 CUDA-graphs backend](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_dynamo/backends/cudagraphs.py):
  inspected AOT forward/backward compilation and `cudagraphs_inner`'s static
  input copies, side-stream warmup and output cloning. The local backend extends
  ownership to every saved activation and returned gradient, retaining exact
  strides for SDPA and copying updated weights before replay. It does not use
  CUDA-graph tree lifetime inference for delayed whole-pass backward.
- [PyTorch LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html):
  historical inspection for independent sequence/state ownership. LSTM and
  sequence fitting have been removed; current fitting uses active flat batches.

These references support execution controls only; the flat-group arithmetic,
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
  planner uses published live flags/counts; probe endpoint facts cross only
  the existing handoff.
- [PyTorch pinned/nonblocking tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/pinmem_nonblock.py):
  inspected pinned transfer ownership and copy-completion requirements. The single
  reusable handoff waits before CPU consumption and before a named buffer is reused.
- The compact terminal cadence, duplicate keys and separation from detailed JSON
  snapshots are local implementation decisions, not research-derived claims.

## Independent background tile probes

- [Sample Factory paper](https://arxiv.org/abs/2006.11751) and
  [project README](https://github.com/alex-petrenko/sample-factory):
  inspected the abstract and synchronous/asynchronous sampling and multi-policy
  descriptions. They inspire separating runnable environment work from a long
  counterfactual task. No PPO, off-policy correction, dependency or claimed
  throughput is adopted; both local policies remain frozen within each cohort.
- [EnvPool paper](https://arxiv.org/abs/2206.10558) and
  [project README](https://github.com/sail-sg/envpool):
  inspected the abstract and batched pool/asynchronous API descriptions.
  Reusable independent lanes and stable environment identities inform the local
  scheduler; EnvPool itself is not integrated. GPU contention, local FIFO source
  backpressure, retained decisions, exact-once endpoint patches and drain-before-save
  are project choices, not performance guarantees from these sources.
# Current implementation sources

- [NVIDIA `nvidia-smi` documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html):
  inspected GPU utilization's kernel-active sample-window definition (product
  dependent, roughly 1 second to 1/6 second) and device/process queries. This
  informs device-wide utilization interpretation and UUID-targeted snapshots,
  rather than treating utilization as SM occupancy or achieved compute rate.
- [PyTorch profiler recipe source](https://github.com/pytorch/tutorials/blob/main/recipes_source/recipes/profiler_recipe.py)
  and [PyTorch 2.8 profiler implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/profiler/profiler.py):
  inspected CPU/CUDA activities, user ranges, finite warmup/active schedules,
  compressed Chrome trace export and profiler overhead. The fitting collector
  wraps the existing learner with host timings and enables one bounded trace
  only on request; shapes, allocation profiling and stacks stay disabled.
- [PyTorch 2.8 `nonzero` documentation source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/_torch_docs.py):
  inspected its explicit CUDA host/device synchronization note. Existing
  fitting boolean selections/nonzero and variable request sizes are profiling
  suspects, not a confirmed cloud bottleneck or a reason to change attribution.

- [pip local project installs](https://pip.pypa.io/en/stable/topics/local-project-installs/)
  and [Bash builtins / traps](https://www.gnu.org/software/bash/manual/html_node/Bourne-Shell-Builtins.html):
  inspected regular versus editable local installs and exit-trap cleanup. The Bash
  bootstrap uses the existing verified archive installer for a non-editable pinned
  game, an editable research checkout and temporary-source cleanup on exit.
- PyTorch, [BF16 CUDA semantics](https://pytorch.org/docs/stable/notes/amp_examples.html)
  and [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html):
  informed the BF16 runtime / FP32-master separation and explicit nonfinite
  handling. No INT8 or INT4 runtime path was adopted.
- The accepted-plant counterfactual rollout design was removed rather than
  retained as a zero-count compatibility path; earlier historical measurements
  remain evidence only and are not current performance claims.
