# Research design

## Current method and hypothesis

Research 0.10.0 exposes one configurable CUDA MaskablePPO method, one compact
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
separate Adam states. The first comparison changes no reward, exploration,
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
`Phi = 0.5 * defeated / max(1, initial_zombies) + 0.5 * (sun + living_plant_costs) / 300`.
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

Defaults: 128 games ×128 decisions per rollout, minibatches 1,024, four epochs,
learning rate 3e-4, gamma 0.999, lambda 0.999, clipping 0.2, value coefficient 0.5,
gradient norm 0.5 and target KL 0.01. Advantages normalize over all transitions;
singletons skip normalization. No choice-only reweighting remains.

The curriculum is placement → saving → easy → standard → shared. It keeps the
full board and lessons' configured plant restrictions. Rehearsal and final
20/40/40 distribution remain. Mastery defaults to 100/100 per required task, at
least 100 completed games started under that stage, one passing check and probes
every 500 games. Normal validation every 2,000 games alone selects `best.zip`.
Standalone stage handoffs copy compatible weights into fresh optimizers; automatic
promotion preserves both current optimizers. All these numeric settings are adjustable.

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
