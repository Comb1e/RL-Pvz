# Sources actually used

## GPU simulator and tensor learning path — 0.6.0, 2026-09-21

| Source inspected | Evidence used | Implementation decision |
|---|---|---|
| Lan et al., 2022, [WarpDrive: Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU](https://jmlr.org/papers/v23/22-0185.html), and [project README](https://github.com/salesforce/warp-drive) | GPU simulation/learning architecture and explicit distinction between kernels and scenario configuration | Keep simulation and PPO tensors on one device. No WarpDrive code copied and no published speedup transferred to this laptop. |
| CuPy **v13.6.0** [kernel guide](https://raw.githubusercontent.com/cupy/cupy/v13.6.0/docs/source/user_guide/kernel.rst) | RawKernel/RawModule, NVRTC compilation, launch configuration and cache | Optional pinned CuPy backend and fused ordered game kernels. |
| CuPy **v13.6.0** [interoperability guide](https://raw.githubusercontent.com/cupy/cupy/v13.6.0/docs/source/user_guide/interoperability.rst) | DLPack lifetime, contiguous arrays, ExternalStream ownership/device rules | Shared CuPy/PyTorch views on one explicit stream; pointer and write tests. |
| Installed **SB3/SB3-Contrib 2.7.1**: `common/buffers.py`, `common/on_policy_algorithm.py`, `ppo/ppo.py`, `ppo_mask/ppo_mask.py` | Collector, timeout bootstrapping, environment-major flattening, NumPy permutation, advantage normalization, PPO loss/gradient clipping/update sequence | Tensor collector/buffer and GAE, with independent fixed-data loss/gradient/optimizer comparisons; aggregate metric transfers after the final update. |
| NVIDIA runtime **12.8.90** and NVRTC **12.8.93** Windows wheels | Installed header/DLL layout and actual kernel compilation | Optional compiler/runtime wheels remove the requirement for global CUDA Toolkit or Visual Studio; process-local discovery. |
| Native game **1.3.0**, commit `8861824df6893a34c2cd4df7f9b68613376d7964` | Source and 214 game tests, including exact CPU/CUDA state/event/replay comparisons | The Python simulator remains the oracle. Compact facts preserve reward attribution; demonstrations require CPU replay agreement. |

Local measurements are recorded separately in `gpu-performance.md`; neither GPU
memory allocation nor utilization alone is treated as evidence of training speed.


## Action timing, defense rewards and game budgets — 0.5.0

Inspected 2026-09-20. This revision implements Leafy's requested objective and
scheduling changes; its weights and budget sizes are not published results.

| Source actually used | Evidence inspected | Decision informed |
|---|---|---|
| Lawn Lab 1.2.1, commit `6fd1f54706369915013a49eab5c1790f8c55ab0a` | Installed `engine.py`: `step`, `_advance`, `_sun_income`, `_detonate`, `_advance_zombies`, `_advance_mowers`; active wall-nut HP and card costs | Reuse native validation and combat, isolate zero-time action application in the research subclass; use public event-time sunlight, actual wall-nut bites, and source/tick blast attribution. |
| Same game release, `replay.py` and `ui.py` | Recorder `_append`, compressed I/O, Playback initialization, `_checkpoint`, step/seek/cache, `_finish`, and App restart/seek integration | Explicit research replay version for zero-time actions; preserve native rendering, seek controls, outcomes and frame timing without editing upstream. |
| Stable-Baselines3 / SB3-Contrib 2.7.1 | Installed/on-project rollout-start/end and training-end callback integration; VecEnv automatic reset behavior; existing PPO update capture and checkpoint save/load interfaces | Count completed training episodes, keep updates unchanged, check game targets after optimization, broadcast curriculum counts for subsequent resets, and persist game counters in checkpoints. |
| Ng, Harada, Russell (1999), linked below | Previously implemented telescoping control `-Phi(initial)+gamma^T Phi(final)` retained and independently verified with immediate actions | Keep discount per policy decision and distinguish potential transformation from the new event objective. Do not claim preserved equivalence to win-only rewards. |
| Local 0.5.0 tests and reports | Independent event arithmetic, actual engine scenarios, zero-time and native replay comparisons, game-count CPU/CUDA/Windows integration and resume checks | Establish correctness and availability only; no learning efficacy claim and no new pilot. |

All relevant upstream source was read from the installed pinned dependency. No
external controller or demonstration was introduced into learning. Existing PPO,
masking and tower-defense literature remains the algorithmic basis.

Reviewed on 2026-09-20. Source evidence is distinct from local availability tests.
No full RL research result has been reproduced in this repository yet.

## Reward revision — 0.4.1

Inspected 2026-09-20. The reward weights implement Leafy's explicit request,
including the corrected mower penalty −2/N; they are not paper-derived weights.

| Source | Evidence actually used | Decision informed |
|---|---|---|
| Lawn Lab 1.2.1, commit `6fd1f54706369915013a49eab5c1790f8c55ab0a` | Installed `engine.py` spawn/damage/death/mower ordering; `types.py` public events/cards/zombie views; `config.py` active health/cost rules | Attribute only the killing hit, credit preceding base-HP loss, use public type plus active starting HP, and match empty activations by engine tick/lane without modifying game sources. Immediate mower kills explain why real empty-activation counts normally stay zero. |
| Ng, Harada, Russell (1999), already catalogued below | Existing potential-difference result; independent discounted-sum and plant/dig controls in this project | Retain `gamma*Phi(next)-Phi(current)` and zero terminal potential while changing economy features. Do not extend its invariance claim to newly added damage/kill rewards. |
| Local 0.4.1 regression and integration suite | Public-event numerical fixtures, real one-tick/batched runs, custom-rule cases, legacy checkpoint tests and generated reports | Verify formulas and compatibility. These checks do not establish improved learning; the previous 0.4.0 pilot used another reward. |

The combined economy scale/weight, full purchase-cost valuation, and exclusion of
armor and lethal damage are explicit implementation choices recorded in
`reward-design.md`. No new algorithm or external source code was adopted.

## Papers

| Source and version | Evidence inspected | Decision informed |
|---|---|---|
| Bergdahl, Sestini, Gisslén, [Reinforcement Learning for High-Level Strategic Control in Tower Defense Games](https://arxiv.org/abs/2406.07980v1), CoG 2024 | Full HTML methods and results, including action masking and generalization experiments | Compare direct plant placement with a five-strategy hybrid and random strategy control. The reported 57.12% vs 47.95% concerns per-level trained agents, not a successful shared policy. |
| Dias, Foleiss, Lopes, [Reinforcement Learning in Tower Defense](https://doi.org/10.1007/978-3-030-95305-8_10), 2022 | Publisher abstract and text/conclusion | Use structured state and headless simulation. |
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
| Leafy's Lawn Lab, local `E:/Projects/pvz` | `b3cfbd886ab378313a1fdb57ee43a9a1b36a0793`; API, engine/config/types/replay, tests and controller | Sole game dependency. The acceptance controller is frozen in `frozen_baseline.py`; strategy proposals adapt its preferences. Installed Python/TOML files are verified against the pinned manifest. |
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

## Visualization and progress implementation — 0.2.0

| Source | Evidence inspected | Decision informed |
|---|---|---|
| [SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) and [logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html); installed SB3/SB3-Contrib 2.7.1 | Official callback/logger documentation and installed `on_policy_algorithm.py` / `ppo_mask.py` update loops | Collect optimizer metrics at the next rollout start and training end, after updates. Keep TensorBoard and add a separate throttled operational log. |
| [FFmpeg project](https://ffmpeg.org/), local FFmpeg 8.1 essentials build | `-version`, `-encoders`, `-h demuxer=rawvideo`, `-h muxer=mp4`; confirmed `libx264`, explicit raw frame rate/size/pixel format, and `faststart` | Stream RGB frames to H.264/yuv420p MP4, without real-time display capture or keeping all frames in memory. Binary is an external tool, not vendored. |
| Leafy's pinned game, same commit as above | `Playback.step`, final/intermediate hash checks, public observation types, offscreen art functions | Render replay ticks and retain wrapper truncation information in research-owned sidecars. No game source change. |

These are implementation references. Local smoke exports verify availability;
they provide no new evidence that the policy wins reliably.

## PVZ 1.2.0 integration — 0.3.0

| Source | Evidence inspected | Decision informed |
|---|---|---|
| Leafy's Lawn Lab, `a47056d8141ec635d3ff3f4d5561d6a75cfca2cc`, package 1.2.0 | `docs/api.md`, `docs/demos.md`, `rendering.py`, `replay.py`, `ui.py`, `tools/export_engine_pin.py`, and rendering/demo/seek tests | Adopt BoardRenderer/RGBFrame, native metadata conventions, compressed `.pvzdemo` I/O, and the seekable viewer. Record package and simulation versions separately. |
| Git project, installed `git archive` command | Verified pinned commit archive and compared Python/TOML hashes with the source manifest | Stage packaging inputs in the research workspace so building/installing cannot write into the game checkout. |

The new game version's engine, rules, scenario-generation code, and preset wave
files are unchanged from the previous pin. Known gameplay regression cases were
checked independently during integration. This is compatibility evidence, not new
evidence of learned performance. The original frozen baseline retains its original
source attribution. No external RL algorithm or library version changed.

## CUDA default — 0.3.1

Inspected the installed Stable-Baselines3 2.7.1 implementation in
`common/base_class.py` (`get_device`) and `common/on_policy_algorithm.py`
(`_setup_model`, policy device transfer, and `_maybe_recommend_cpu`). These support
using the existing configurable device interface and retaining the README's
whole-pipeline benchmarking guidance: CUDA support alone does not imply that this
MLP trains faster. Local PyTorch 2.8.0+cu128 availability and saved GPU tensors were
verified on the RTX 4070 Laptop GPU; the evidence is recorded in `validation.md`.
No algorithm or dependency version changed for this device-default update.

## Runtime transport and game patch integration — 0.3.2

Inspected on 2026-09-20. These sources inform implementation; timing measurements
are local evidence and do not imply improved learned skill.

| Source | Evidence actually inspected | Decision informed |
|---|---|---|
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), installed 2.7.1 | `common/vec_env/subproc_vec_env.py`, `base_vec_env.py`, `common/buffers.py`, callback/logger and PPO collection code | Reuse spawned workers; carry masks in existing reset/step responses; retain terminal observations; subclass sampling without changing GAE or NumPy permutation order. Time phases through callbacks and explicitly separate benchmark warmup. |
| [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib), installed 2.7.1 | `common/maskable/utils.py`, `buffers.py`, `policies.py`, `distributions.py`, `ppo_mask/ppo_mask.py` | Keep mask-aware collection/update implementations intact. Serve mask `env_method` calls locally and reuse CUDA rollout tensors across epochs. Preserve mask/sample numeric validation. |
| [PyTorch](https://github.com/pytorch/pytorch), installed 2.8.0+cu128 | Installed `torch/distributions/distribution.py` validation and SB3's `to_torch` calls; local cProfile of copies, distributions, and multiprocessing | Optimize repeated transfer/indexing rather than enlarge the policy to consume VRAM. Keep validation and floating-point precision unchanged. |
| Leafy's Lawn Lab package 1.2.1, commit `6fd1f54706369915013a49eab5c1790f8c55ab0a` | API and iteration docs, complete source diff from 1.2.0, `engine.py` legality, renderer, source-pin exporter | Pin the new package; reuse defeated/total HUD. Cache engine legality only while all public legality inputs remain equivalent. Verify simulation/rule sources unchanged and run upstream tests without checkout writes. |

Attempts to fetch current online SB3/PyTorch documentation encountered HTTP/DNS
errors. The table describes the installed versioned source actually read, not an
unverified online document. Existing PPO/masking literature above remains the
algorithmic basis; no new training algorithm was adopted.

## Evidence boundaries (research claims)

### Paper adaptation and pure-RL profiles — 0.4.0

Inspected 2026-09-20. Source evidence and implementation decisions:

| Source | Evidence actually inspected | Decision informed |
|---|---|---|
| Dias, Foleiss, Lopes (2022), DOI `10.1007/978-3-030-95305-8_10`; user PDF `F:/chrome/978-3-030-95305-8_10.pdf` | All 13 pages; visual inspection of pages 9–12; Table 4 representations and Table 5 scores | Test regional metadata and simpler introductory tasks. Do not infer DQN superiority, copy unspecified reward weights, or claim recurrent results. Detailed evidence boundaries are in `paper-adaptation.md`. |
| SB3-Contrib 2.7.1, `common/maskable/distributions.py` and `policies.py` | Installed distribution interfaces, mask handling, `_build`, policy inference and action evaluation | Implement a joint type/tile distribution through existing policy interfaces, preserving the 406-action engine contract and PPO algorithm. |
| SB3/SB3-Contrib 2.7.1 callback and on-policy update loops | Installed rollout start/end and training-end hooks | Read entropy components after optimization; stop timed pilots at completed-update boundaries; keep the same optimizer across curriculum stages. |
| Local `runs/masked-cuda-101` episode/validation JSON | 10,229 episodes and accepted-purchase counts; selected and final validation summaries | Identify absent sustained attackers, inspect savings/exploration signals, and define focused ablations rather than changing algorithms without evidence. |
| Lawn Lab 1.2.1, existing public scenario/observation/rule API | `LevelSpec`, `Spawn`, card costs/recharge and projectile firing intervals; independent lesson controls | Build seeded lessons without engine changes or hidden observations, and derive explicit scales from pinned rules. |

Grouped action selection, lane summaries, mastery thresholds, and pilot screening
rules are project-specific design decisions. No external project code was copied.
The engine pin and library versions remain those recorded for 0.3.2.

For the user-requested kill reward, inspected the same pinned engine's `_damage`,
`_clear_dead`, projectile movement, explosion/swallow paths, and `_advance_mowers`.
`DamageApplied.source` is positive for plant/projectile entities and negative for
mower rows; deaths emit `ZombieDefeated` within the same tick. This public contract
supports killing-blow attribution without modifying the engine. The +1/N and -1/N
weights are a local configurable choice implementing the user's preference, not a
published finding. Tests independently cover all damaging plants, mixed batches,
nonlethal damage, and mower kills after plant damage.

The existing game is a daytime PvZ-style clone with automatic sun collection and
documented custom balance. Its results are not directly comparable to commercial
PvZ or the 2024 paper's level suite. Winning through legal game actions does not
establish human behavioral similarity. The literature motivates this experiment;
only independent held-out evaluations can establish performance in this environment.


## SC2-inspired pure-RL work — 0.7.0 (2026-09-21)

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

Full-text findings, repository descriptions, local observations and new design
choices remain separate in [sc2-adaptation.md](sc2-adaptation.md). The normalized
unweighted tile bonus and exact architecture sizes are project adaptations. No
AlphaStar speedup or win-rate claim is transferred to this laptop.
