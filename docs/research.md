# Research design

## Current method and hypothesis

Research 0.9.0 exposes one configurable CUDA MaskablePPO method, one compact
observation and one spatial grouped policy. The question is whether this shared
policy can learn plant selection, timing and placement across easy, standard and
hard within a practical laptop budget. Smaller input/network size, ordinary PPO
reductions and potential shaping are hypotheses to test, not established PVZ gains.

Local historical records motivated the change: a saving-stage run fell from 78/100
to 0/100 mastery, early digging increased and some updates had large approximate
KL (maximum about 0.575). Its last choice-only actor fractions were roughly 3–6%.
The two similarly named archived folders both actually used learner seed 101;
they are not evidence from two independent seeds.

## Research choices

| Evidence | Adopted decision | Limits |
|---|---|---|
| Implementation Matters; What Matters in On-Policy RL | Audit PPO numerically; expose ordinary hyperparameters; use a smaller configurable network | Their benchmarks are not this discrete strategy game |
| SC2LE categorical/spatial preprocessing | Embed plant categories; combine spatial and scalar inputs | No imitation, league, privileged critic or recurrent architecture is copied |
| Potential-based shaping theorem | Use one discounted potential difference with correct terminal handling | Function approximation and finite training do not guarantee learned invariance or success |
| CleanRL and SB3-Contrib PPO | Standard full-minibatch reduction; optional KL stopping before optimizer steps | Correct PPO arithmetic alone does not establish good exploration |
| Local digging/instability observations | Remove event-reward complexity and choice-only reweighting; retain trainable initial dig bias | Several changes occur together, so the short comparison cannot isolate causes |

Actual inspected versions and sections are in [references](references.md).
Alternative algorithms, recipes, policy factories and legacy model loaders are
removed. CPU reference controls and non-learning baselines remain. The older
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
Standalone stage handoffs copy compatible weights into a fresh optimizer; automatic
promotion preserves the current optimizer. All these numeric settings are adjustable.

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

The release comparison uses fresh starts from previous commit `0dbe95e` and the
compact method, learner seeds 101/102 and equal five-minute allowances. Method order
reverses for the second seed. Game pin, default task distributions, mastery gates,
parallelism and evaluation cases match. Report actual games, decisions and elapsed
time; do not fabricate interpolation across rollout overshoot. This compares bundles
of changes, not isolated ablations. No previous trained weights are reused.

Keep the new method experimental unless both seeds reduce destructive digging
without worse lesson performance. Stronger normal-game claims need independent
runs, paired scenario comparisons and uncertainty intervals. Bootstrap reporting
resamples learner runs and common seeds; two short pilots cannot establish robust
performance. Formal suites and comparison matrices are never started automatically.
Results, including missing evaluations and negative outcomes, live in
[validation](validation.md).

The release comparison failed that adoption criterion: both compact runs lost all
20 deterministic placement cases, while both reference runs won all 20. Normal
validation also provides no evidence of improvement. The compact implementation
remains experimental; the stochastic-training/deterministic-evaluation gap needs
diagnosis. Reduced late digging alone is insufficient.
