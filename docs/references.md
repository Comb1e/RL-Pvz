# Sources actually used

Reviewed on 2026-09-20. Source evidence is distinct from local availability tests.
No full RL research result has been reproduced in this repository yet.

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
