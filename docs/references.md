# Sources actually used

## Saving, complete-action exploration and hardware telemetry — 2026-09-25

- [Invalid Action Masking](https://arxiv.org/abs/2006.14171), abstract inspected:
  supports consistent legal masking in policy gradients. It does not justify making
  valid digging illegal or establish these initial PVZ weights. HTML SHA-256:
  `7610f59abcc3971e49b5188d189659175c2d72960f586a6b23fe81a35a5aa4b3`.
- [Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://jmlr.org/papers/v21/20-212.html),
  abstract inspected: task sequencing and transfer motivate checking a prerequisite
  task's value. The 100-sun saving design and removal of placement are local hypotheses,
  not recommendations demonstrated by this abstract. HTML SHA-256:
  `8101362af6763f61f7cbae0b887a2b492dcdd4abe425dae982a53ac96f6d4bc2`.
- [SB3 v2.7.1 evaluation callback source](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.1/stable_baselines3/common/callbacks.py):
  inspected EvalCallback's deterministic evaluation and after-evaluation event.
  The project extends event reporting to failed/partial attempts. Raw source SHA-256:
  `1659f783e53c00ee0f50d0666135f1e0d8febe23deb958dffb4dd9fe5f3bd32c`.
- [NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html):
  inspected GPU selection, utilization and memory fields. GPU utilization measures
  the fraction of its sampling period with kernels executing; memory utilization is
  memory-controller activity, not allocated VRAM. Sampling periods depend on device
  (roughly one sixth to one second); percent alone does not identify a bottleneck.
  HTML SHA-256: `017647ae72b332a94e01e6d9672d1d82367e56de1d0685132eb91a972035b17e`.
- [psutil 7.0.0 documentation source](https://github.com/giampaolo/psutil/blob/release-7.0.0/docs/index.rst):
  inspected nonblocking system/per-core/process CPU and RSS APIs, first-reading
  priming, minimum 0.1-second recommended spacing, and process CPU's one-core units.
  The first nonblocking CPU reading is meaningless and ignored. Source SHA-256:
  `6a4fdbf85348ebc983d809b2301e95ccb5cf8c04ff40f9f6cfd4dfe8fb972e08`.

Re-inspected installed game 1.4.0, simulation 1.1.0, commit
`1fc80386859087b9d715c4706b3f7875844430cc`: plant costs, sunflower payment/recharge,
200-HP basic movement/bites, and engine `_detonate`'s half-open mine tile interval.
The existing CPU/CUDA parity controls verify the unchanged dependency. The
[saving and probability derivations](math/saving-and-actions.md) are project-specific;
no published learning/speedup claim is transferred to this laptop.

During reproducibility verification, also inspected installed **PyTorch 2.8.0**
`torch/_dynamo/convert_frame.py`, `preserve_global_state` (lines 222–288).
Tracing saves and restores the global CPU/CUDA RNG states, which can interact with
a concurrent sampling thread even when compilation later fails. This environment
has no Triton backend. Checking that prerequisite before entering tracing restored
the strict same-seed output/no-output control here; this does not establish compiled
execution correctness on other environments. Installed file SHA-256:
`5890b891106ba8372c94e79a4f5d682587e07ae00850629cee96dba7bd8fac4a`.

## Exploration phases and CUDA throughput — 2026-09-25

- [PyTorch 2.8 CUDA semantics and CUDA Graphs](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/cuda.rst):
  rechecked stream ownership, static-address requirements, graph replay's CPU
  dispatch benefit and restrictions from dynamic control flow. The compiled and
  tensor-only paths preserve eager fallback and do not change the periodic PPO
  scheduler.
- [Sample Factory synchronous/asynchronous design notes](https://github.com/alex-petrenko/sample-factory/blob/master/docs/07-advanced-topics/sync-async.md):
  rechecked the warning that overlap often gives little benefit for GPU-resident
  environments. This supports keeping two slots and measuring contention instead
  of increasing pipeline depth.
- Installed **SB3-Contrib 2.7.1** masked distributions and PPO update code:
  rechecked legal-mask consistency, joint log probabilities and deterministic
  evaluation. The fast categorical path retains those equations while removing
  repeated device-dependent validation in hot minibatches.
- [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py):
  rechecked explicit entropy, likelihood-ratio and optimizer ordering used by the
  fixed-shape compiled learner kernels.

The separate warm-up/formal schedule, floors and deterministic validation rule are
project-specific responses to the stopped placement run. The run fell from 4,068
warm-up transitions/s to roughly 2,680 transitions/s during actor learning and
reached only 8–10% recent placement wins by game 3,781. These observations do not
establish that the schedule or compiled path improves later learning.

## Objective and exact PPO guard — 2026-09-24

- [Spinning Up: reward and return](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html):
  inspected finite-horizon undiscounted versus discounted objectives. Used to distinguish
  the selected gamma=1 outcome objective from the GAE trace decay.
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html):
  inspected clipping limitations and KL early stopping. Clipping does not bound all
  probability changes. Whole-window rollback and the hierarchical exact guard are
  project-specific engineering choices, not claims made by this source.
- [SB3-Contrib 2.7.1 MaskablePPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/ppo_mask/ppo_mask.py):
  inspected normalization, clipping, approximate KL, stopping, and gradient clipping.
  Independent numerical controls retain PPO likelihoods while explicitly changing
  normalization scope and adding the transactional window guard.
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/):
  re-inspected entropy and stopping discussion; [What Matters in On-Policy RL?](https://arxiv.org/html/2006.05990v1),
  timestep handling and regularization sections, motivates checking discount and
  entropy settings without importing continuous-control performance claims.
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438): abstract
  inspected for the estimator's bias/variance tradeoff. No new full-paper empirical
  claim is made; duration-based numerical controls verify this implementation.
- Pinned game 1.4.0 rules and shipped wave lists, source commit
  `1fc80386859087b9d715c4706b3f7875844430cc`: inspected costs, HP/armor, sun cap,
  sky timing and all three difficulty inventories to derive conservative bounds.

All arithmetic, counterexamples and assumptions are in
[the math record](math/training-objective.md). Learning rates and coefficient
calibrations remain experimental, and no training-strength claim follows from them.

## Periodic pipeline and accelerator overlap — 2026-09-24

- Espeholt et al., [IMPALA](https://arxiv.org/abs/1802.01561): inspected the
  actor–learner queue and V-trace discussion. Decoupling motivates bounded producer–
  consumer scheduling; V-trace is rejected because this project keeps standard PPO.
- Espeholt et al., [SEED RL](https://arxiv.org/abs/1910.06591): inspected centralized
  accelerator inference and trajectory batching. The project adopts frozen batched
  inference snapshots without RPC or a distributed service.
- Petrenko et al., [Sample Factory](https://arxiv.org/abs/2006.11751) and its
  [synchronous/asynchronous design notes](https://github.com/alex-petrenko/sample-factory/blob/master/docs/07-advanced-topics/sync-async.md):
  inspected double buffering, queue ownership and policy-lag tradeoffs. Its warning
  that asynchronous mode often gives little benefit for GPU-resident environments
  is retained as a limitation.
- Lu and Luo, [Periodic Asynchrony](https://arxiv.org/abs/2511.18871): inspected the
  producer–consumer mechanism and periodic weight-consistency argument. The fixed
  two-rollout window and boundary synchronization are adapted here; its LLM/NPU
  speed claims are not transferred.
- PyTorch 2.8, [CUDA Graphs and stream semantics](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/cuda.rst):
  inspected stream events, static-address requirements and CPU-synchronization
  constraints. Rechecked the v2.8.0 stream-synchronization examples during implementation:
  side-stream consumers must wait for producers, and allocations cannot be recycled
  before consumer work completes. Explicit completion events and stream synchronization
  enforce those rules here. CUDA Graph capture is not implemented.
- Weng et al., [EnvPool](https://arxiv.org/abs/2206.10558): reviewed the CPU
  environment execution bottleneck; it is not adopted because PVZ already runs its
  simulator on CUDA.

## Hierarchical action arguments and stopping modes — 2026-09-24

- [PySC2 action interface](https://github.com/google-deepmind/pysc2/blob/master/pysc2/lib/actions.py): inspected `ArgumentType` and `FunctionCall(function, arguments)` validation. Used as a reference for separating a command from its arguments, while retaining a compact engine transport. Retrieved source SHA-256: `c6ea590816349b18b2946d8d11068a8780a3c32946daba456617d37b27aebe62`; no unverified commit is claimed.
- Huang and Ontanon, [A Closer Look at Invalid Action Masking](https://arxiv.org/html/2006.14171v3), masking-gradient discussion and naive-masking counterexample: sampling and optimization must use the same legal distribution. The inspected HTML has SHA-256 `30e735c2330ef0607a16f63f5801e1c9d071c7fb31cc46b82e0aa98bc06c2d4b`. This does not validate the project's exploration coefficients or predict a PVZ win-rate gain.
- Installed **SB3-Contrib 2.7.1** maskable distributions and PPO learning loop: inspected categorical masking, entropy, joint log probabilities and callback/update boundaries. The project uses conditional arguments rather than independent MultiDiscrete probabilities. Stage-only stopping extends the existing mastery state machine and completed-update callbacks.

The SC2LE HTML URL attempted during planning returned 404; no new full-text
findings are attributed to that attempt. The previous SC2LE inspection remains
recorded below. Numeric action labels do not themselves create a large classifier;
the factorized history experiment is a separate representation hypothesis.

## Smaller observations and update throughput — 2026-09-24

- Andrychowicz et al., [What Matters in On-Policy RL?](https://arxiv.org/html/2006.05990v1),
  §§3.2 and 3.8: inspected initialization and regularization results. Their continuous-control
  evidence motivates testing initialization; it does not establish a PVZ dig bias or guarantee
  that less entropy improves play. The -12 bias is a project-specific 100 Hz experiment.
- Huang and Ontañón, [A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms](https://arxiv.org/html/2006.14171v3), masking justification and experimental setup:
  retain consistent legality at sampling and optimization. Legal but unproductive digging
  is not classified as invalid. Grouped probabilities already separate type mass from tile count.
- Inspected [PyTorch 2.8.0 Embedding source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/sparse.py):
  lookup weights, padding-gradient behavior, and default unscaled dense gradients. Small
  categorical one-hot multiplication supplies the same operation and zero padding gradients;
  independent output/gradient/Adam controls verify this project-specific implementation.
- Inspected PyTorch's [performance tuning guide](https://github.com/pytorch/tutorials/blob/main/recipes_source/recipes/tuning_guide.py),
  [SDPA tutorial](https://github.com/pytorch/tutorials/blob/main/intermediate_source/scaled_dot_product_attention_tutorial.py),
  and [2.8.0 Adam source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py).
  The guide informed phase profiling; attention replacement, fused Adam, mixed precision
  and compilation were not adopted. The documentation website returned 403, so repository
  sources were inspected. No published speedup is assigned to this laptop.

Read-only placement evidence: `runs/compact-stages-101/placement` recorded 696 early digs
in its first 1,024 completed games, with the actor frozen. Its final recorded phase totals
were 419.23 seconds collection and 2,336.83 seconds optimization. A constant 0.2225% dig
probability gives 67.17% cumulative probability over 500 choices; real states change, so
this is an explanatory control, not an exact prediction for each episode. Observation
removal is user-directed information reduction, not a paper-proven sufficient statistic.

## Regional simplification — 2026-09-24

- Inspected the pinned [PVZ engine](https://github.com/Comb1e/pvz-cuda-work/tree/1fc80386859087b9d715c4706b3f7875844430cc), public `ZombieView`, CPU vault transition, and CUDA storage schema. `has_pole` clears at vault start; behavior labels and countdowns need not be exposed by the research encoder.
- Reused the regional representation and bounded public-history basis recorded below. Removing explicit zombie behaviors and replacing pole counts with nearest unused-pole distance is a user-directed project hypothesis. No cited paper establishes that plant HP/history fully replaces biting or that this simplification improves learning.

## Event-memory Transformer — 2026-09-23

| Source inspected | Used here | Limitation |
|---|---|---|
| [GTrXL](https://arxiv.org/abs/1910.06764), methods pp. 3–5 | Gating, pre-normalization, identity-biased initialization | Its RL experiments use V-MPO; no PVZ performance claim |
| [Transformer-XL](https://arxiv.org/abs/1901.02860), pp. 3–4 | Bounded past context and relative positions | Raw public banks here replace learned recurrent activations |
| [Compressive Transformer](https://arxiv.org/abs/1911.05507), pp. 2–4 and RL memory experiment | Local/event/summary history | No learned compression or reconstruction objective copied |
| [Longformer](https://arxiv.org/abs/2004.05150), §3 and Fig. 2 | Local plus selected global context | No language-model speedup transferred |
| [BigBird](https://arxiv.org/abs/2007.14062), §2 | Bounded sparse context | No random attention or expressivity guarantee transferred |
| [S4](https://arxiv.org/abs/2111.00396), [Mamba](https://arxiv.org/abs/2312.00752), abstracts | Reviewed linear-time alternatives | Not adopted; abstract-level review only |
| [SB3-Contrib 2.7.1 recurrent PPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/v2.7.1/sb3_contrib/ppo_recurrent), installed collector/timeout code | Reset masks, terminal bootstrap and on-policy boundaries | No LSTM implementation added |

PDF SHA-256 values for GTrXL, Transformer-XL, Compressive, Longformer and BigBird:

- `98b64dec5b94f8b122a136cc0bdc6536958daf28e881b5f9b794375385386636`
- `6aa192b22820267309e8e7c501c514a3a3055256d9c8691b106c7f53f9098671`
- `0c650936864a36739c8c34f4860e64fa01ab146c21494e0ebd3a701b4f608768`
- `053756b54e2d3462d4690b3aed489c962f17df7c3531865e47374c4a651f526c`
- `9dc448464532d61f084de6a224b84acf3b421142e05a40ebfb982b4eca5834a2`

Admission, latest-state categorical summaries, raw-token deduplication and public
burn-in are project-specific choices. These sources do not establish that removing
PVZ timers improves learning. Engine source revisions and limits are in the lock
and validation record. No extra runtime dependencies are introduced.

Methods and adopted/rejected ideas are in [research design](research.md). This
catalog retains source revisions and the extent of the evidence inspected.

## Exploration and critic adaptation (2026-09-23)

- [Spinning Up PyTorch PPO](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py): inspected separate Adam optimizers, squared value loss, independent value iterations and actor-only KL stopping. This supports a simple independent critic schedule; it does **not** prescribe a stage warm-up or its duration.
- [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py): inspected categorical sampling/log probabilities and PPO ratio/value updates. The exploratory mixture must be the distribution used for both collection and loss calculations.
- [PPO](https://arxiv.org/abs/1707.06347) and [Curriculum Learning for RL Domains](https://arxiv.org/abs/2003.04960): abstracts inspected to contextualize on-policy updates and task transfer. No paper result establishes PVZ improvement or the selected exploration rate.
- [PyTorch DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), exponential epsilon schedule and action selection: a smooth rate approaching its endpoint is useful here. Only the schedule principle is adopted; PPO still trains on the actual mixed distribution, with no replay buffer or DQN loss. Source fetched 2026-09-23 has SHA-256 `d0513808527578cf77596c79a3b0df234e0994fbff9b583726d1831c76e4ccb8`; no unverified Git revision is claimed.
- Local source-checkpoint evaluation and legal one-action counterfactuals identified transferred-critic optimism, pessimism on planting states and action-type collapse. These motivate a wait/plant mixture and brief critic adaptation without extra reward terms, replay buffers or networks. The requested 10% initial rate, 0.1% milestone at stage game 3,000 and stage restarts are project-specific experimental choices, not established by these sources.

Fetched project source on this date. GitHub commit-API lookup was rate-limited and Git ref retrieval failed; no unverified revision is claimed. Inspected source SHA-256 values:
Spinning Up `4522d149d54d7d759c613775ab787eb7fa754fcbfd103f28d7c4003ee815c264`;
CleanRL `d8dfbb7ac0b21e747b77d6e6673e566db4d6cdacc83bcebcd19cf87ad0e459ae`.
Local copies are verification artifacts under `artifacts/v0111/sources/`.

## Independent policy/value learning (2026-09-22)

| Source inspected | Evidence used | Application and limits |
|---|---|---|
| [DeepSeek-V4.1-Flash report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf), repository revision `dba1be0a40aa45a94ad051997016db3960a90277` | Architecture/optimization and post-training sections; pages 9, 14, 30 and 31 visually inspected; benchmark/appendix overview | DSpark blocks auxiliary gradients into the backbone (§2.4.3); CED avoids unnecessary phase-specific work (§2.2); task verification (§5.1); length bias and staleness (§5.2). These inspire gradient isolation, deferred critic batches and task-specific diagnostics. It contains no PPO actor-critic separation experiment. |
| Andrychowicz et al., [What Matters in On-Policy RL?](https://arxiv.org/html/2006.05990), §3.2 and Appendix B.7 | HTML architecture results and shared-objective discussion | Independent actor/value encoders. Their reported benefit on four of five continuous-control environments is not evidence of a PVZ gain. |
| [SB3-Contrib 2.7.1 policy source](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/common/maskable/policies.py), installed source | Separate feature extractors, actor-only distribution and critic prediction paths; SB3 save/load and PPO code | Reuse one policy interface; independent optimizer ownership and checkpoint states; tensor-only conversion of the current shared checkpoint. |
| [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), 2022; [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | Shared/separate networks, clipping, bootstrap and KL stopping descriptions | Independent optimization controls and actor early stopping; no new algorithm or performance claim. |
| Cobbe et al., [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416), abstract only | Shared-representation interference motivation | Context for the hypothesis; PPG's phases/distillation are not implemented. |

The downloaded 51-page DeepSeek PDF has SHA-256
`ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.
Its speed and quality claims concern large language models. No published speedup
is transferred to this laptop. KV caches, MoE, FP4, speculative actions, Muon,
Sinkhorn optimization, teacher distillation and stale asynchronous samples are
not adopted. The §5.2 completed-trajectory length bias does not directly describe
our buffer: PVZ already trains on partial-game transitions. It does explain why
completed-game logs and actual rollout composition must be distinguished.

## Sources used for the compact method (2026-09-22)

| Source and inspected version | Evidence used | Decision |
|---|---|---|
| Engstrom et al., [Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729), 2020; [associated project](https://github.com/implementation-matters/code-for-paper/tree/094994f2bfd154d565c34f5d24a7ade00e0c5bdb), commit `094994f2bfd154d565c34f5d24a7ade00e0c5bdb` | Paper HTML methods and project README | Audit PPO reductions and independent loss/gradient/Adam controls; no code vendored |
| Andrychowicz et al., [What Matters in On-Policy RL?](https://arxiv.org/abs/2006.05990), 2020 | Sections 3.2, 3.3, 3.5 and 3.8 | Small configurable architecture; account for initialization, learning rate, batch size and regularization interactions |
| SC2LE, [arXiv:1708.04782](https://arxiv.org/abs/1708.04782), 2017 | PDF p.10 input preprocessing and FullyConv description | Categorical embeddings and spatial/scalar inputs; no additional privileged state |
| Ng, Harada and Russell, [Policy Invariance Under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), 1999 | PDF pp.2–4 visually inspected, definition and theorem | A discounted potential difference and independent telescoping/cycle/terminal controls |
| [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py), file revision `e421c2e50b81febf639fced51a69e2602593d50d` | Explicit rollout/update code | Ordinary minibatch reduction and optional KL stopping; design reference only |
| [SB3-Contrib v2.7.1](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.7.1/sb3_contrib/ppo_mask/ppo_mask.py), installed source and PPO docs | Update sequence, normalization, clipping and stopping before update at KL >1.5×target | Preserve true joint policy ratios, standard reduction, optimizer/checkpoint interfaces |

These sources guide implementation; none establishes an improvement in this PVZ
setup. Local archived episode/update records support investigating instability,
not a proven causal diagnosis. New bounded results are recorded separately in
[validation](validation.md). Earlier entries below describe historical inspections;
retired methods are not supported modes in 0.9.0.

## Papers

| Source and version | Evidence inspected | Decision informed |
|---|---|---|
| Bergdahl, Sestini, Gisslén, [Reinforcement Learning for High-Level Strategic Control in Tower Defense Games](https://arxiv.org/abs/2406.07980v1), CoG 2024 | Full HTML methods and results, including action masking and generalization experiments | Compare direct plant placement with a five-strategy hybrid and random strategy control. The reported 57.12% vs 47.95% concerns per-level trained agents, not a successful shared policy. |
| Dias, Foleiss, Lopes, [Reinforcement Learning in Tower Defense](https://doi.org/10.1007/978-3-030-95305-8_10), 2022 | All 13 pages of the user PDF `F:/chrome/978-3-030-95305-8_10.pdf`; pages 9–12/tables visually inspected | Regional metadata and simpler tasks; unspecified reward weights and scores do not establish DQN superiority. |
| Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017 | Abstract and algorithm description; library implementation | PPO as the initial algorithm, with rollout/update accounting. |
| Huang and Ontañón, [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171v3), 2020/FLAIRS 2022 | Abstract, theoretical rationale, experimental masking discussion | Shared legality mask at sampling, learning, and evaluation; unmasked ablation without changing rewards. |
| Ng, Harada, Russell, [Policy Invariance Under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), ICML 1999 | Bibliographic abstract and potential-based shaping result; source PDF availability confirmed | Discounted potential difference with terminal handling and independent telescoping tests. The chosen potential weights are this project's configuration. |
| Huang et al., [Gym-μRTS](https://arxiv.org/abs/2105.13807), CoG 2021 | Abstract and project README | Single-machine strategy-game experiments and controlled ablations. Its runtime/results are not predictions for this project. |
| Narvekar et al., [Curriculum Learning for Reinforcement Learning Domains](https://arxiv.org/abs/2003.04960), 2020 | Abstract and framework motivation | Fixed difficulty curriculum with a mixed-difficulty control. |
| Cobbe et al., [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://arxiv.org/abs/1912.01588), 2019/2020 | Abstract and generalization motivation | Separate learner seeds, training scenarios, validation, final seeds, and changed-wave families. |
| Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264), NeurIPS 2021 | Abstract, recommendations, rliable project metadata | Report independent training runs and intervals. The crossed paired bootstrap here is locally implemented and tested, not copied from rliable. |
| Towers et al., [Gymnasium: A Standard Interface for Reinforcement Learning Environments](https://arxiv.org/abs/2407.17032), preprint 2024 | Abstract and [environment API](https://gymnasium.farama.org/api/env/) | Standard reset/step contract and explicit termination versus truncation. |

## Projects

| Project | Inspected revision/evidence | Use and limitations |
|---|---|---|
| PVZ game, local `E:/Projects/pvz` | `b3cfbd886ab378313a1fdb57ee43a9a1b36a0793`; API, engine/config/types/replay, tests and controller | Original game source. The acceptance controller is frozen in `frozen_baseline.py`; strategy proposals adapt its preferences. Current engine pins are listed below. |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | Installed 2.7.1, base/on-policy collection code | PPO implementation and vector environment lifecycle. |
| [SB3-Contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) | Installed 2.7.1; documentation and collection code | MaskablePPO, in-environment masks for subprocess workers, explicit mask passing during evaluation. Recurrent MaskablePPO is not supported. |
| [greinermachine/PVZRL](https://github.com/greinermachine/PVZRL) | `ccef5e566d4fd03de201186b38b1c1a5097b1592`; README and adapter/reward declarations | Reference for structured PvZ observations, shared legality, and experiment artifacts. Different game/bridge; no comparable held-out performance reproduced. |
| [KXAND/RLPVZ](https://github.com/KXAND/RLPVZ) | `3dfae1d2bb2bc1c6ae197e8227db4810edde1a79`; README | Reference for PPO/DDQN training, curricula, and simulator integration. README explicitly disclaims reproduced stable win rates. |
| [ubaidill-chem/PvZ_RL](https://github.com/ubaidill-chem/PvZ_RL) | `166e1f121aa3df629fad7067ed8898cfa4bdb3b7`; README and repository metadata | Simulation design reference, not evidence of learned skill. |
| [MicroRTS-Py](https://github.com/Farama-Foundation/MicroRTS-Py) | `91ae6f54c92d5f97abb34bcaff3013adc51cb687`; README | Spatial representation and separate train/eval maps. README marks the wrapper deprecated; not a dependency. |
| [rliable](https://github.com/google-research/rliable) | `3ccd9f4dea577a04d3d2b557f259aac08badbd81`; repository metadata and linked paper | Statistical reporting inspiration. Archived repository; not a dependency. |

GitHub metadata did not identify standard licenses for the three external PvZ
repositories at inspection time. Their source was not vendored. The only frozen
controller comes from the user's existing local game project.

## StarCraft methods and projects

| Source/version inspected | Evidence inspected | Decision informed |
|---|---|---|
| Vinyals et al., [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://doi.org/10.1038/s41586-019-1724-z), 2019; DeepMind author PDF, 29 pages | Methods including structured actions, human-data dependence, undiscounted terminal objective, UPGO and league; relevant ablations, including visual inspection | Structured heads and a separate horizon experiment; omit imitation, league scale and privileged inputs |
| Mathieu et al., [AlphaStar Unplugged](https://arxiv.org/abs/2308.03526), 2023, 32-page PDF | Architecture, offline training/results, memory and scaling sections; Table 3 visually inspected | Separate modalities and start without memory; do not treat BC or offline results as online PPO evidence |
| Liu et al., [Revisiting of AlphaStar](https://doi.org/10.1109/TG.2023.3265975), online 2023, TG 16(2), 2024 | Verified abstract via Semantic Scholar and closed-access metadata via OpenAlex; full text unavailable | Motivate explicit difficulty and weakness analysis; no unverified detailed result adopted |
| Liu, [Rethinking of AlphaStar](https://arxiv.org/abs/2108.03452), v3, 23-page PDF | Interface/replay analysis and raw-versus-human action economy experiment, Appendix D | Inspect repetitive actions; distinguish empirical results from architectural criticism |
| Arulkumaran, Cully, Togelius, [AlphaStar: An Evolutionary Computation Perspective](https://arxiv.org/abs/1902.01724), 2019, complete 3-page PDF | PBT, coevolution and quality-diversity discussion of early AlphaStar | Retain diverse tasks; no claim that copying population training improves PVZ |
| Vinyals et al., [StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/abs/1708.04782), 2017, 20-page PDF | Mini-games, FullyConv architecture, section 4.4 separate argument entropy, full-game failures; section 4.4 visually inspected | Spatial placement head and separate exploration terms; preserve win rate as outcome |
| Chua et al., [Runtime Action Interference for AI Control of AlphaStar in StarCraft II](https://arxiv.org/abs/2608.21398), 2026, 22-page PDF | Post-inference admission/no-op wrapper, 32-person exploratory study, limitations and aggregate appendix | Audit execution and outcomes; do not adopt cooldown interference or claim training improvements |
| [google-deepmind/alphastar](https://github.com/google-deepmind/alphastar), `700b1e74364ed5dfc66f6cd2574c5ffac2fa474e` | README and architecture encoder/head definitions through GitHub API | Reference separate feature streams and action arguments; no dependency added. Repository does not supply online RL code |
| [liuruoze/mini-AlphaStar](https://github.com/liuruoze/mini-AlphaStar), `554206724da64b684308634fff54d3828a6114a2` | README requirements, imitation initialization, resource recommendations and experiment links | A design reference, not evidence of comparable pure-RL laptop performance |
| [Raw-vs-Human-in-AlphaStar](https://github.com/liuruoze/Raw-vs-Human-in-AlphaStar) | README identifying the Rethinking experiment and raw/human configuration switch | Confirm experiment scope; no code imported |
| SB3/SB3-Contrib 2.7.1 installed policy, distribution and PPO interfaces; existing project CUDA optimizer | Policy construction, optimizer parameter registration, joint action evaluation, update capture and checkpoint loading | Reuse the PPO objective, add one shared exploration-loss interface, and verify fresh/loaded spatial heads |

## GPU implementation sources

| Source inspected | Evidence used | Implementation decision |
|---|---|---|
| Lan et al., 2022, [WarpDrive: Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU](https://jmlr.org/papers/v23/22-0185.html), and [project README](https://github.com/salesforce/warp-drive) | GPU simulation/learning architecture and explicit distinction between kernels and scenario configuration | Keep simulation and PPO tensors on one device. No WarpDrive code copied and no published speedup transferred to this laptop. |
| CuPy **v13.6.0** [kernel guide](https://raw.githubusercontent.com/cupy/cupy/v13.6.0/docs/source/user_guide/kernel.rst) | RawKernel/RawModule, NVRTC compilation, launch configuration and cache | Optional pinned CuPy backend and fused ordered game kernels. |
| CuPy **v13.6.0** [interoperability guide](https://raw.githubusercontent.com/cupy/cupy/v13.6.0/docs/source/user_guide/interoperability.rst) | DLPack lifetime, contiguous arrays, ExternalStream ownership/device rules | Shared CuPy/PyTorch views on one explicit stream; pointer and write tests. |
| Installed **SB3/SB3-Contrib 2.7.1**: `common/buffers.py`, `common/on_policy_algorithm.py`, `ppo/ppo.py`, `ppo_mask/ppo_mask.py` | Collector, timeout bootstrapping, environment-major flattening, NumPy permutation, advantage normalization, PPO loss/gradient clipping/update sequence | Tensor collector/buffer and GAE, with independent fixed-data loss/gradient/optimizer comparisons; aggregate metric transfers after the final update. |
| NVIDIA runtime **12.8.90** and NVRTC **12.8.93** Windows wheels | Installed header/DLL layout and actual kernel compilation | Optional compiler/runtime wheels remove the requirement for global CUDA Toolkit or Visual Studio; process-local discovery. |
| Native game **1.3.0**, commit `8861824df6893a34c2cd4df7f9b68613376d7964` | Source and 214 game tests, including exact CPU/CUDA state/event/replay comparisons | The Python simulator remains the oracle. Compact facts preserve reward attribution; demonstrations require CPU replay agreement. |

## Reward, optimizer and presentation sources

| Source/version | Evidence inspected | Decision informed |
|---|---|---|
| Schulman et al., [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | Abstract; installed SB3 2.7.1 recurrence and independent controls | Bias/variance tradeoff; lambda 0.999 is a local experiment, not a published PVZ result |
| [SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) and [logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html), installed 2.7.1 | Documentation, collection/update loops, vector worker/buffer code | Capture final/post-update metrics, preserve TensorBoard, masks and terminal observations; optional runtime caches retain sample order |
| PyTorch 2.8.0+cu128 | Installed distribution validation, device/buffer code and local cProfile | Avoid repeated small transfers; keep precision and validation; VRAM use alone is not speed evidence |
| [FFmpeg](https://ffmpeg.org/), local 8.1 essentials | `-version`, `-encoders`, rawvideo demuxer and MP4 muxer help; libx264 confirmed | Stream RGB to H.264/yuv420p with explicit frame rate/size and fast-start; no binary vendored |
| Git, installed `archive` command | Pinned archive compared with the manifest | Stage/build the game inside research `build/`, never inside its source checkout |

The installed game `engine.py`, `types.py`, `config.py`, `replay.py` and CUDA event
schema were inspected for event ordering and attribution. Plant/projectile sources
are positive and mower sources negative; `PlantRemoved(reason=eaten)` differs from
digging/detonation. Damage uses starting base HP; bites use actual plant HP removed;
empty blasts match source and tick. These facts support exact CPU/CUDA accounting.
Historical event rewards and choice-state actor reduction were local experiments.
Version 0.9.0 removes them; only outcome, mower activation cost and one potential
remain. The initial dig bias is a configurable, trainable project-specific choice.

## Local engine and experiment provenance

| Version/source | Evidence inspected and use |
|---|---|
| Original `b3cfbd886ab378313a1fdb57ee43a9a1b36a0793` | API, rules, replay, tests and frozen baseline controller |
| Game 1.2.0, `a47056d8141ec635d3ff3f4d5561d6a75cfca2cc` | Renderer, compressed demos, metadata, seek/viewer tests, pin exporter; package/simulation identity separated |
| Game 1.2.1, `6fd1f54706369915013a49eab5c1790f8c55ab0a` | Source diff, native HUD and legality; unchanged combat; public events and lesson feasibility |
| Installed game 1.3.0, `8861824df6893a34c2cd4df7f9b68613376d7964` | CUDA kernel/batch sources, 214 engine tests and exact state/event/replay differential controls |
| Native CPU reader 1.2.2 / `a95524e`, CUDA reader 1.3.1 / `314528a` | Action-phase loader compatibility with research code at `665aec5`; retain installed simulator pin |
| `runs/masked-cuda-101` | 10,229 episode records, purchase totals and validation; historical exploration diagnosis |
| `runs/cuda-shared-101` | Interim 2,923-game snapshot and phase times; not a final result |
| `runs/sc2-shared-101` | Completed records, selected/final validation, plant/dig traces, three original demos; checkpoint `ce6b6fc3b480233c67b4009e6ce44d7412d2e2ef8fc7db8c047d34fe70fbb65e` |
| Committed `evidence/` and ignored local `artifacts/` | Tests, measured timings and bounded diagnostics; indexed in [validation](validation.md) |

Historical sources were inspected on 2026-09-20/21. The compact-method sources
above were inspected on 2026-09-22. Some online SB3/PyTorch retrievals failed, so those entries explicitly use
installed versioned source. Revisiting remains abstract-only. Repository claims,
published findings, proposed adaptations and local results are distinct. No paper
establishes these reward weights, human similarity or a two-hour win-rate gain.

For 0.8.0 (2026-09-22), inspected installed SB3 2.7.1 `common/base_class.py`
loading and `common/save_util.py` custom-object deserialization. Replace retired
buffer-class metadata during historical inference while retaining policy and
optimizer state. Inspected the local Git archive/staging implementation to stage
the pinned tree independently of the game checkout HEAD. This maintenance change
adopts no new learning method or performance claim.

For stage checkpoints (0.8.1, 2026-09-22), inspected the installed
[Stable-Baselines3 2.7.1](https://github.com/DLR-RM/stable-baselines3/tree/v2.7.1)
`common/base_class.py` load, parameter restoration and learning setup, alongside
the project's tensor PPO collector and mastery state machine. SB3 restores policy
and optimizer state, while resetting timestep counters alone does not reset the
project's curriculum, game count or wall allowance. This informed explicit stage
initialization separate from resume. No new paper-derived algorithm or performance
claim is introduced; the existing curriculum evidence remains as recorded above.

For the 0.8.1 mastery follow-up (2026-09-22), re-inspected this project's curriculum
state machine, post-update validation callback, checkpoint compatibility checks,
and existing curriculum/seed-separation references above. The 100/100 pass gate
is the user's requested acceptance rule; the 500/2,000-game intervals are local
configuration choices, not paper-derived optima. No new learning method or
performance claim is introduced.

For 0.10.1 (2026-09-23), inspected the installed pinned game's integer movement,
sky-income, sunflower timer, card cooldown and purchase/dig implementation, plus
the research scenario generator. These supply the saving-task necessity proof
and independent feasible/failing controls; the numeric lesson is a local design,
not a paper result. Reused the potential-difference formulation above when
reducing the economy coefficient, without introducing a purchase bonus.

Inspected installed **SB3 2.7.1** `ppo/ppo.py` (rollout size is `n_steps * n_envs`,
minibatch divisibility) and **PyTorch 2.8.0**
`torch/utils/benchmark/utils/timer.py` (warmup, CUDA synchronization, repeated
measurements and medians). These support the parallelism benchmark protocol,
not a claim that larger batches improve learning. Online retrieval of SB3's
versioned PPO page, PyTorch's benchmark recipe and Narvekar et al.'s curriculum
survey failed in this session; no newly read online findings are claimed.

For 0.10.2 (2026-09-23), searched and inspected installed **SB3 2.7.1**
`common/callbacks.py`, specifically `EventCallback` and `EvalCallback` event
handling, deterministic evaluation and strict best-score improvement, alongside
this project's post-update callback and curriculum state machine. This informs
an explicit stage-success evaluation event with resumable pending state. The
2,000-game mastery interval and evaluation-after-each-stage rule are user-directed
scheduling choices, not paper-derived optima or a new learning algorithm.

For 0.10.3 (2026-09-23), searched and inspected installed **SB3-Contrib 2.7.1**
`ppo_mask/ppo_mask.py` (`n_steps` and rollout collection), confirming that the
128-step setting bounds collection per update rather than an episode. Also read
the versioned [SB3 MaxAndSkipEnv implementation](https://github.com/DLR-RM/stable-baselines3/blob/v2.7.1/stable_baselines3/common/atari_wrappers.py):
it repeats actions and sums rewards until termination/truncation, which would
change per-tick PVZ choices if copied directly.

Read Sutton, Precup and Singh (1999), [Between MDPs and semi-MDPs](https://doi.org/10.1016/S0004-3702(99)00052-1),
especially pp. 189–191, equations 6–12, and the interruption discussion on pp.
196–197. The inspected author PDF (`http://incompleteideas.net/papers/SPS-aij.pdf`)
defines discounted cumulative rewards and duration-dependent bootstrapping for
options. Used to assess waiting compression and explain why dropping elapsed
discounting is incorrect; no options algorithm is added. The weighted-sampling
proposal in research.md is a local estimator design, not a result from this paper.
The archived 0.10.0 seed-101/102 comparison logs supply the waiting diagnosis.

For 0.10.4 (2026-09-23), retrieved and inspected the **abstract** of Narvekar et al.
(2020), [Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://jmlr.org/papers/v21/20-212.html).
It frames task sequencing and transfer as separate design questions. No full-text
claim or timing recommendation is attributed to it. Also inspected Farama
[MiniGrid 2.3.1 DynamicObstaclesEnv](https://github.com/Farama-Foundation/Minigrid/blob/v2.3.1/minigrid/envs/dynamicobstacles.py),
its constructor, configurable obstacle count/size and task limits. This supports
explicit challenge parameters in a small task definition; its rewards, action
restrictions and timeout formula are not copied. Neither source establishes that
tighter PVZ lessons cure early digging.

Actual timings derive from installed game 1.3.0's `config.py`, `engine.py`,
`cuda/backend.py` and `cuda/simulation.cu`: 120/480-tick sunflower payments,
150-tick recharge, 30-tick firing, 200-HP basics, movement on spawn ticks and
income-before-plant processing. Reference controls and committed independent
tests establish feasibility and specific delay counterexamples. The research
adapter preserves the installed source pin and records effective snapshot rules.
Numeric trial choices are local design, not published performance claims.


For **0.11.0 net realized value** (2026-09-23), reused the inspected Ng, Harada
and Russell (1999) shaping theorem and Sutton, Precup and Singh (1999)
semi-MDP return formulations cited above. The former distinguishes an invariant
potential difference from this deliberately changed accounting objective; the
latter motivates discounting continuation by elapsed duration. The implementation
uses `gamma^ticks` in TD targets and `(gamma * lambda)^ticks` in GAE. This trace
choice is project-specific; the options paper does not validate it as a PPO
performance improvement. Inspected installed SB3 2.7.1 rollout-buffer GAE and
SB3-Contrib timeout handling as the unit-duration numerical reference.

Inspected pinned game **1.3.0**, commit
`8861824df6893a34c2cd4df7f9b68613376d7964`, `engine.py` damage/income processing,
public `DamageApplied` / `SunProduced` events, and `cuda/simulation.cu` damage,
income, explosion, chomper and mower processing. Damage events expose actual
health/armor removed; positive sources identify plants/projectiles, negative
sources identify mowers. Income exposes actual capped amounts and sky/sunflower
sources. The basic's 200 HP supplies the common damage denominator. Read-only
counters are added through the existing research adapter, with no engine-file,
installed-package or state-order change. Source assertions intentionally fail if
the pinned kernel hook changes.

The 50-sun basic value, 600-sun mower cost, health-weighted assets and ungated
account are a **project-specific experimental hypothesis**. No paper supplies
these coefficients or establishes an early-digging cure. Historical peak/drawdown
is diagnostic because ordered loss/repayment controls invalidate peak gating.
The inspected RUDDER abstract (arXiv:1806.07857) motivates examining delayed credit;
its redistribution algorithm is not adopted and no performance claim is borrowed.
