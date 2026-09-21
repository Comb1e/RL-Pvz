# Selected StarCraft methods for PVZ learning

## Evidence and research limits

2026-09-21. Six full papers were inspected, including the relevant methods,
results and limitations. *Revisiting of AlphaStar* was accessible only through
its verified publisher/metadata abstract. No detailed results are attributed to
that unavailable full text. Sources and inspected repository revisions are in
[references](references.md).

| Source | Adopted insight | What the evidence does not establish |
|---|---|---|
| Vinyals et al., Nature 2019 | Structured actions and a sufficiently long return horizon; diverse situations | Its human initialization, supervised-policy KL, human strategy rewards, privileged critic and huge league budget are not available here |
| Mathieu et al., AlphaStar Unplugged, 2023 | Separate scalar and spatial encoders; measure whether memory pays for itself | The offline results require a large human replay dataset; the 90% headline compares against a behavior-cloning reference, not final Grandmaster AlphaStar |
| Liu et al., Revisiting, online 2023 / volume 2024 | Define difficulty and inspect specific weaknesses | Abstract-only evidence; no independently verified ablation details |
| Liu, Rethinking, 2021 | Audit repetitive ineffective actions and interface assumptions | Replay criticisms do not prove that a planning module would fix PVZ |
| Arulkumaran et al., 2019 | Preserve task diversity | A perspective on the January 2019 demonstration, not a PVZ population-training experiment |
| Vinyals et al., SC2LE, 2017 | Resolution-preserving spatial heads, mini-games, independent entropy terms for action arguments | Mini-game improvement does not demonstrate full-game success; some agents optimized mining scores without learning to win |
| Chua et al., Runtime Action Interference, 2026 | Separate proposed actions, execution and observed outcomes | A runtime controller and exploratory perception study, not a convergence result; no throttling wrapper is added |

SC2LE section 4.4 uses separate entropy penalties for action function and
arguments. Our normalized, unweighted tile bonus is a project-specific
adaptation, not a reproduction of that paper's objective. AlphaStar Unplugged
Table 3 reports 70% for its LSTM behavior-cloning variant and 84% without memory;
its authors note tuning and training costs. These results motivate starting small,
not a general claim that memory is harmful. Nature AlphaStar uses undiscounted
terminal outcomes; our gamma=0.9999 variant is a conservative, separate experiment.

## Local diagnosis

An interim, read-only snapshot of `runs/cuda-shared-101` at 2,923 completed games
showed zero sustained-attacker purchases in its rolling 100 episodes, mean maximum
sun 50.5, and best macro validation win rate 8%. Approximately 416.6 of 1,496.8
elapsed seconds were validation. These are historical observations from an active
run, not final results or a controlled baseline comparison.

Earlier experimental grouped policies also planted and immediately dug up plants.
At the initial normal-game state there are 45 legal placements for each of three
affordable plant types, plus wait. Flat uniform exploration gives wait probability
1/136. Grouped equal type logits initially give 1/4, but maximizing true joint
entropy favors the groups with more legal tiles and returns the optimum to 1/136.
An independent analytic control gives the joint-entropy gradient of the wait
logit as `-3*log(45)/16` at equal type logits. This motivates changing exploration;
it does not prove entropy is the only cause of poor learning.

## Implemented experimental method

The exploration bonus is `0.01 H(type) + 0.001 mean_available[H(tile|type) /
log(max(2, legal_tiles))]`. There is no type-probability factor in the second
term. Wait-only states get zero tile bonus. PPO log probabilities, ratios and
clipping still use the real joint distribution. True joint entropy, weighted
conditional entropy, type entropy and the exploration bonus remain separate logs.
There is no scripted savings or plant-retention rule.

The spatial policy unpacks all 1,140 tactical values through the shared observation
layout. Shared 64-unit lane and global encoders feed two 64-channel 3x3
resolution-preserving convolutions on the 5x9 board. A 1x1 head emits nine tile
maps; pooled board features and global features feed separate 256x256 type/value
heads. There are no new policy-visible seeds, task identifiers or future schedules.

The existing placement/saving/easy/standard/shared curriculum preserves its lesson
rules and rehearsal mixture. It probes every 100 completed games, requires two
consecutive passes, and requires 100 completions from episodes that actually
started in the current stage. Old-stage completions after promotion do not count
toward new-stage residency. Transitions only affect resets.

The principal candidate retains gamma=0.999. Profile F changes only gamma to
0.9999, consistently in shaping, GAE and timeout bootstrap. At one wait per tick,
the discount half-life changes from 34.64 to 346.56 simulated seconds. Immediate
placements/digs also count as decisions, so this is not simulated-time discounting.
All user-selected reward coefficients remain unchanged, including loss=-2,
plant/mower kill attribution, damage, defense and economy terms.

## Scheduling and comparisons

New configurations validate every 1,000 completed games on all 50 fixed seeds
per difficulty. Validation is after optimization; multiple crossed milestones
coalesce into one evaluation. Matching normal-game curriculum probes reuse the
same checkpoint's results. A complete final validation is required for final-run
comparison; partial evaluations never select a checkpoint.

`--max-minutes` is cumulative across resume and includes startup, validation and
presentation. The 120-minute candidate reserves 15 minutes for finalization.
Shorter availability runs reserve at most one eighth of their allowance. The
runner estimates the next complete rollout/update duration before starting it,
then stops only between complete updates. Evaluation and export check deadlines
between operations; an in-flight kernel, save or encoder flush must finish safely.
The limit is cooperative, not an operating-system process kill. Pending results
are labeled explicitly and can be regenerated. This finalization detail is the
practical adjustment needed to preserve useful checkpoints.

| Profile | Recipe |
|---|---|
| A | Flat spatial-v1 baseline |
| B | Tactical-v2, grouped policy, original joint-entropy bonus |
| C | B plus balanced action-head exploration |
| D | C plus mastery curriculum |
| E | D plus spatial policy: `sc2-inspired` |
| F | E plus gamma=0.9999: `sc2-long-horizon` |

`compare-sc2` is user initiated. It first runs learning diagnostics for seeds 101
and 102, capped at five minutes each; success requires passing placement and
saving mastery gates. The A-F runs use 101 in forward order and 102 in reverse
order, with the same game ceiling, per-run time allowance and full validation
cases. Reports show actual games/decisions/time and nominal milestones without
requiring identical rollout overshoot. Paired bootstrap intervals resample learner
runs and common scenario seeds, preserving method and difficulty pairing. Two
learners support development screening, not strong statistical conclusions.

A candidate is recommended only for further study when diagnostics pass and
normal-game macro validation improves over A for both learners. Profile identity,
engine pin, reward coefficients and resource controls are checked before combining
results. Gamma is an explicitly named exception for F. Final-test seeds are never
used by this command. The old 30-minute `pilot` command retains its original scope.

## Verification and results

See [validation](validation.md) for measured checks. The implementation task does
not launch the comparison matrix or formal training. Short integration runs with
artificial cutoffs establish availability, not learned competence. Profiles E/F
remain experimental; no claim of improved win rate has been established.

Two five-minute profile-E diagnostics were run on 2026-09-21. Each used the normal
128×128 collection geometry, the unchanged lesson definitions, and the actual
five-minute cumulative deadline. They finished the last update before the reserve.

| Learner seed | Completed lesson games | Decisions | Elapsed seconds | Placement probes | Early digs/game |
|---|---:|---:|---:|---|---:|
| 101 | 768 | 851,968 | 300.03 | Six probes, all 0/20 wins | 3.000 |
| 102 | 768 | 835,584 | 300.03 | Six probes, all 0/20 wins | 2.909 |

Neither advanced beyond placement or won a training game. Both purchased three
peashooters per game on average, but almost all were dug up within five simulated
seconds. Purchasing attackers in a lesson that restricts plant choice to
peashooters does not establish better normal-game exploration. The short
diagnostics are **unsuccessful**, not evidence that two-hour training cannot work.
They do show that the proposed exploration/architecture changes alone have not
yet resolved the observed plant/dig behavior under this budget.

Final normal-game validation completed only 128/150 cases before each deadline.
It remains explicitly pending; neither partial evaluation selected `best.zip`.
The 37.5-second reserve in a five-minute diagnostic was insufficient for those
validations. Reports can be rebuilt without training, but the missing validation
is not filled in or treated as a zero win rate. The two-hour profile retains its
15-minute reserve. No A–F comparison was launched and no candidate is recommended
over baseline. Further comparison is user initiated.

The compact [diagnostic evidence](evidence/sc2-v070/diagnostics.json) records the
development source hash used by both runs. Detailed episode/probe logs and final
checkpoints remain under `artifacts/sc2-diagnostics-v070`. The later export-time
accounting fix was independently checked without changing the learning objective.

LSTM/transformer memory, imitation, offline replay learning, UPGO, self-play leagues,
population training, privileged critic inputs and inference-time throttling are
outside this change. Existing observations, rewards, direct controls and shared
policy make narrower controlled experiments more useful first steps.
