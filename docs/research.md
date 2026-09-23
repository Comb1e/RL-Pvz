# Research design

## Current method and hypothesis

Research 0.10.3 exposes one configurable CUDA MaskablePPO method, one compact
observation and one spatial grouped policy. The question is whether this shared
policy can learn plant selection, timing and placement across easy, standard and
hard within a practical laptop budget. Smaller input/network size, ordinary PPO
reductions and potential shaping are hypotheses to test, not established PVZ gains.

The current hypothesis is that shared actor/critic features contributed to rapid
forgetting after saving-to-easy transfer. In the inspected `compact-stages-101`
run, normal easy validation fell from 26% to 2% on the same 50 seeds. Across stage
games 101–200, easy games won 18/74 with 14.2% early digs per accepted planting;
games 201–300 won 6/85 with 25.3%. Placement and saving rehearsals later collapsed
too. The initial 100% rolling win rate contained only completed placement lessons.
These observations establish regression, not its sole cause.

The actor and critic now own independent copies of the compact encoder and
separate Adam states. The 0.10.0 comparison changed no reward, exploration,
learning-rate, discount, curriculum or rollout defaults. Independent gradient
clipping and actor-only KL stopping accompany the split; the critic continues
its epochs. Thus the comparison tests this optimization package, not encoder
separation in perfect isolation. Critic predictions are deferred to larger frozen
batches before GAE; no stale-policy experience is introduced.

## Research choices

| Evidence | Adopted decision | Limits |
|---|---|---|
| Implementation Matters; What Matters in On-Policy RL | Audit PPO numerically; expose ordinary hyperparameters; use a smaller configurable network | Their benchmarks are not this discrete strategy game |
| SC2LE categorical/spatial preprocessing | Embed plant categories; combine spatial and scalar inputs | No imitation, league, privileged critic or recurrent architecture is copied |
| Potential-based shaping theorem | Use one discounted potential difference with correct terminal handling | Function approximation and finite training do not guarantee learned invariance or success |
| CleanRL and SB3-Contrib PPO | Standard full-minibatch reduction; optional KL stopping before optimizer steps | Correct PPO arithmetic alone does not establish good exploration |
| Local easy-stage regression; What Matters §3.2; DeepSeek DSpark gradient isolation | Separate learned actor/value features and optimizer states, retaining the existing reward | Interference remains a hypothesis; learning must improve on paired development checks |
| DeepSeek phase-specific execution and task-quality/length-bias discussion | Batch critic inference, audit task-level data composition, preserve successful and failing controls | No LLM architecture or reported speedup is copied |

Actual inspected versions and sections are in [references](references.md).
Alternative algorithms, recipes and policy factories are removed. A narrow tensor
conversion accepts only 0.9.0 compact weights for initialization; it does not restore
shared-encoder training or resume. CPU reference controls and non-learning baselines remain. The older
committed implementation is used only as temporary comparison evidence.

## Reward objective

Only terminal outcome, mower activation cost and potential shaping contribute:

`reward = outcome - mower_activation_cost * new_activations + gamma * Phi(next) - Phi(current)`.

Defaults are +1 for victory, −2 for defeat and −0.2 once per activated mower.
`Phi = 0.5 * defeated / max(1, initial_zombies) + 0.1 * (sun + living_plant_costs) / 300`.
Sun above 300 is retained; that denominator scales units. Full purchase value is
retained while a plant lives, including at low health. Buying preserves resource
value, and digging/death removes it. Damage, source of kills, wall-nut bites and
failed explosions have no extra coefficients. Early digging is measured, not
forbidden or directly penalized.

For T transitions, discounted shaping telescopes to
`-Phi(initial) + gamma^T * Phi(final)`. Natural terminal potential is zero;
external truncation keeps potential and bootstraps the value. Buying and immediately
digging cannot create positive shaping profit. The mower activation cost intentionally
changes the task objective; shaping provides feedback for that objective. Discounting
remains per decision, including zero-time actions. The resource potential can exceed
terminal rewards in magnitude, so value scale and gamma still matter.

## Learning and curriculum

The 500 inputs retain exact plant coordinates and aggregate enemies/projectiles
into three regions per lane. Counts are never clipped. IDs, seeds, task names and
future schedules never reach the network. Aggregation loses exact positions and
future information; partial observability remains a limitation.

The grouped distribution selects wait, one of eight plants, or dig, then a tile.
PPO ratios use the true joint probability. Exploration is
`0.01 * H(type) + 0.001 * mean_available_groups(H(tile|group) / log(max(2, legal_tiles)))`.
Tile entropy is not weighted by group probability. This avoids the joint-entropy
incentive to prefer types with many possible tiles. The type/tile coefficients,
trainable initial dig bias and architecture widths are configurable.

Defaults: 256 environments ×128 transitions per update, minibatches 1,024, four epochs,
learning rate 3e-4, gamma 0.999, lambda 0.999, clipping 0.2, value coefficient 0.5,
gradient norm 0.5 and target KL 0.01. Advantages normalize over all transitions;
singletons skip normalization. No choice-only reweighting remains.

### Waiting efficiency: investigated, not yet adopted

The 128-transition rollout horizon is an update interval, not a per-game action
limit. A learning transition includes waiting; the separate `agent_actions`
diagnostic includes only accepted planting/digging. Waiting cannot simply be
discarded: income, damage, mower activation and outcomes often occur during waits.

A read-only audit on 2026-09-23 of the archived 0.10.0 separated-method comparisons
found 2,256,780 waits in 2,274,354 transitions for seed 101 (99.23%), and 2,449,913
in 2,469,404 for seed 102 (99.21%), across 719 and 854 completed games respectively.
These are older easy-stage transfer experiments with the earlier saving lesson,
not measurements of current 256-environment training. Their mean recorded
per-update legal-choice fractions were 99.78% and 62.22%; that metric is computed
over executed actor minibatches, not a census of completed episodes. It nevertheless
shows that many waiting states allow alternatives, often digging. Skipping only
forced waits cannot be assumed to remove most redundant work.

The recommended first experiment preserves every-tick decisions and dense reward,
timeout and GAE calculations. Keep every planting/digging sample for optimization,
but uniformly sample a configurable fraction of waiting samples. If there are
`N` total samples, `W` waits and `K` sampled waits, estimate each full-rollout loss as
`(sum(nonwait losses) + (W/K) * sum(sampled wait losses)) / N`. Handle `W=0`
explicitly and require `K>=1` when `W>0`. Normalize advantages from the full
rollout before sampling; correct actor, exploration and critic losses consistently.
Retain a representative full-rollout KL audit. This estimates the existing
objective without deliberately rewarding busier behavior. Finite-sample variance,
gradient clipping and Adam still mean optimizer trajectories will differ.

Separately, skip actor inference where waiting is the only legal action, retaining
its reward/value information. This changes execution cost without changing choices;
its GPU indexing overhead needs measurement. Do not wait for 128 plant/dig actions
before updating: an agent can legitimately finish a game with far fewer actions.

The larger alternative is temporal abstraction: combine waits into a transition
with duration `k`, discounted reward `sum(gamma**j * reward[j])` and bootstrap
discount `gamma**k`. The corresponding potential difference is
`gamma**k * Phi(next) - Phi(current)`. Sutton, Precup and Singh supply that return
formulation, not a PPO speed or PVZ win-rate guarantee. Dense GAE with lambda below
one is not automatically reproduced by substituting a duration discount; it needs
an explicit estimator and independent controls. Repeating voluntary waits also
removes intermediate action opportunities. Ordinary fixed frame skipping is
therefore not a drop-in optimization under the current per-tick controls.

Neither sample thinning nor temporal abstraction is enabled in this release.
Compare learning and wall time before adoption: normal-game/lesson wins, attacker
purchases and early digs per planting matter alongside optimizer throughput.

The curriculum is placement → saving → easy → standard → shared. It keeps the
full board and lessons' configured plant restrictions. Rehearsal and final
20/40/40 distribution remain. Mastery defaults to 100/100 per required task, at
least 100 completed games started under that stage, one passing check and probes
every 2,000 games. Normal validation runs once after each individual stage passes,
and alone selects `best.zip`. Failed/incomplete stages and game/time limits do not
trigger unrelated normal-game evaluation. The 100/100 requirement and separate
100-case mastery / 50-case-per-difficulty normal validation pools are unchanged.
Standalone stage handoffs copy compatible weights into fresh optimizers; automatic
promotion preserves both current optimizers. All these numeric settings are adjustable.

### Saving lesson

The task requires income investment by construction, using unchanged engine
rules. Start with 50 sun; allow sunflower and peashooter, ordinary waiting and
digging, but no mowers. Three basic zombies spawn simultaneously at tick 1000
in three distinct seeded lanes. There are ten lane combinations. The 100 mastery
seeds cover all ten, so this remains a small introductory task rather than broad
economic generalization evidence.

**Necessity proof without sunflowers.** The only remaining plant is a 100-sun
peashooter, which shoots and blocks in its own lane. Digging neither moves a plant
nor refunds its cost. Zombies start at integer x=9500 and reach the house at
x=-500. Basic speed is 200 units/s at 20 ticks/s, exactly 10 units per tick.
Movement occurs on the spawning tick. An entirely unblocked lane therefore
loses at `1000 + (9500 - (-500))/10 - 1 = 1999` ticks (99.95 seconds).
Sky income is 25 at multiples of 200 ticks. Before this loss, at most
`50 + floor(1999/200) × 25 = 275` sun is available: fewer than three purchases.
With three threatened lanes and at most two plants ever purchased, at least one
lane is entirely unblocked, proving the loss regardless of placement or digging.
The tenth sky payment would give 300 at tick 2000, one tick too late. Spending
zero-time actions forever cannot produce a victory or additional income.

**Feasibility witness, used only in tests.** Plant sunflowers at ticks 0 and 200
in rows 0 and 1, column 1. Each costs 50, first produces 25 after 120 ticks, then
every 480 ticks. At tick 1000 their four total payments plus five sky payments,
minus purchase costs, leave `50 + 100 + 125 - 100 = 175` sun. Observe the spawned
lanes and buy peashooters at column 0 in those lanes, in ascending row order, at
ticks 1000, 1150 and 1560. A flower pays at 1080, funding the second purchase.
The first two spend 200; sky at 1200/1400 and flower
payments at 1280/1560 supply another 100. The shared 150-tick card cooldown is
respected. The reference engine wins at tick **1831 (91.55 seconds)** in every
lane combination. The sky-only control loses at **1999** in every combination.
The analytical proof establishes necessity; these controls establish feasibility,
not that PPO will learn the strategy. No control is used for learning.

Version 0.10.1 also reduces `economy_weight` from 0.5 to 0.1. A living 100-sun
plant contributes 0.03333 potential instead of 0.16667; 25 new sun contributes
0.00833 instead of 0.04167 before discounting. Purchasing transfers equal value
from sun to plants and never creates a positive purchase bonus. Defeated weight,
terminal rewards and mower cost are unchanged. Reduced economic shaping also
reduces the shaping loss from destruction or digging; its learning effect requires
future measurement. Archived results use their recorded weights and lesson.

## Evaluation and evidence

Separate learner seeds from game seeds. Development includes 0–9 and 42; training
uses 1,000–99,999; normal validation uses 100,000–100,049; mastery uses
100,050–100,149. Final-test seeds 200,000–200,299 and changed-scenario seeds
300,000–300,099 are reserved for user-initiated evaluation. The changed families
redistribute later rosters, shorten wave spacing or concentrate lanes.

Primary outcome is normal-game win rate per difficulty and the equally weighted
macro average. Also measure lesson wins, accepted plant purchases, early digs per
accepted planting, sustained attackers, first-attacker time, sun, mower use,
truncation and throughput. Zero planting makes the digging ratio unavailable.
Fewer digs alone do not establish progress. Interpret lessons separately from
normal-game ability, and development validation separately from held-out evidence.
Normal validation now samples stage-success checkpoints, so its curve is sparse
and conditional on passing stages; an absent point is not a zero win rate. This
scheduling change saves evaluation work but delays detection of normal-game
regressions within a stage. It does not establish better learning or measured
wall-time savings. Archived runs retain their recorded schedules.

The 0.10.0 comparison starts both methods from the same mastered saving checkpoint
(SHA-256 `3bc3d252e3107c34c26c0f5623ddaf57eb3bac0d62fb74f54d5362e4b9d364ab`),
with fresh optimizers/counters and learner seeds 101/102. The reference is commit
`1ab8c59`; the candidate uses independent encoders. Five-minute learning windows
include startup and scheduled probes; normal and lesson endpoint evaluations are
bounded separately, with total learning/comparison evaluation below 30 minutes.
Order is reference101, separated101, separated102, reference102. All use the same
engine, reward, curriculum distributions/gates, parallelism and evaluation cases.
Actual game and decision counts are reported, including the first 200 completed
games, without interpolation. This tests continuation from one upstream trained
model, not two independent full curriculum training runs.

Claim the collapse resolved only if both seeds improve normal easy validation,
reduce early digs per planting and retain lesson competence. Fewer digs without
planting/attacker activity are not progress. Stronger generalization claims require
independent runs, paired scenario differences and uncertainty. Two short transfer
pilots are development evidence. Final-test seeds remain untouched. Results and
missing evaluations are recorded in [validation](validation.md).

The bounded test improved easy validation from 2% to 66% and from 0% to 28%,
with fewer aggregate early digs per planting for both seeds. However, saving
retention was 65/100 and 0/100, and seed 102 regressed after roughly 500 games.
The acceptance condition fails. This is an experimental implementation of the
requested separation, not evidence that feature interference was the sole cause
or that the collapse is fixed. No reward or optimizer tuning followed these results.

The earlier 0.9.0 fresh-start comparison failed its adoption criterion; those
historical results remain in validation and do not describe the new transfer test.
