# Sources actually used

Methods and adopted/rejected ideas are in [research design](research.md). This
catalog retains source revisions and the extent of the evidence inspected.

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
| Leafy's Lawn Lab, local `E:/Projects/pvz` | `b3cfbd886ab378313a1fdb57ee43a9a1b36a0793`; API, engine/config/types/replay, tests and controller | Original game source. The acceptance controller is frozen in `frozen_baseline.py`; strategy proposals adapt its preferences. Current engine pins are listed below. |
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
The reward coefficients, choice-state actor reduction and initial dig bias are
local changes implementing Leafy's preferences and diagnosed failure modes.
Potential-shaping invariance is not claimed for added event rewards.

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

Sources were inspected on 2026-09-20/21; this consolidation adds no new literature
claims. Some online SB3/PyTorch retrievals failed, so those entries explicitly use
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
is Leafy's requested acceptance rule; the 500/2,000-game intervals are local
configuration choices, not paper-derived optima. No new learning method or
performance claim is introduced.
