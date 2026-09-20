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

## Evidence boundaries

The existing game is a daytime PvZ-style clone with automatic sun collection and
documented custom balance. Its results are not directly comparable to commercial
PvZ or the 2024 paper's level suite. Winning through legal game actions does not
establish human behavioral similarity. The literature motivates this experiment;
only independent held-out evaluations can establish performance in this environment.
