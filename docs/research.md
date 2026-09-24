# Training method

One CUDA MaskablePPO method learns a shared policy for easy, standard and hard.
There is no imitation, scripted placement, savings rule or plant-retention rule.
The timer-free Transformer is **experimental**. Correct memory does not establish
stronger play. See [validation](validation.md) and [sources](references.md).

## Observation and memory

The `event_v6` layout has 281 values: 135 plant values (type, health, behavior per
tile), 135 regional zombie values and 11 globals. Each of
three regions per lane stores five type counts, total health, total armor, nearest
zombie distance and nearest unused-pole distance. Counts retain crowds without
clipping. With the default local scale of five, health is divided by five times
maximum zombie HP, armor by five times maximum armor, and counts by five.
Distances use the full house-to-spawn interval;
-1 means absent, including when zombies exist but none has an unused pole.

Walking, carrying-pole, vaulting, biting and dead counts, plus the separate pole
count, are removed. `has_pole` identifies unused poles and clears at vault start.
Plant HP/history can reveal damage but cannot perfectly reconstruct active biting;
nearest-pole distance cannot recover how many unused poles remain. This intentional
information loss is an experimental simplification, not proof of redundancy.
The 11 globals are sun, elapsed time, current/total waves, initial/defeated zombie
counts and five mower-spent flags. Projectile inputs, spawned count and mower
position/ready/moving flags are absent; ready and moving mowers therefore alias.
These reductions deliberately lose information. Full simulator/accounting data remains.
All plant, zombie and card countdowns remain absent. Legal masks reveal current
legality. Seeds, schedules, entity IDs and task names are excluded.

Each independent actor/critic uses embeddings 8/4, scalar encoder 64, two
32-channel convolutions, two gated attention blocks (128 width, four heads,
feed-forward width 256), and 128×128 heads. Together they have 1,128,060 parameters.
A shared actor head outputs three kind logits (wait/dig/plant) and eight conditional
species logits. Nine spatial maps supply legal tiles. Planting selects species before
tile; digging only selects tile. The simulator and rollout buffer carry the executed
command as one compact integer. All 406 possible commands remain available through
legal masks; they are not 406 independent top-level policy choices.

Eight local tokens, 32 event tokens and eight summaries bound history. Public
plant health/state, zombie counts/health/armor, sun/wave, mower-spent
and mask changes retain events. A region gaining/losing unused-pole presence also
retains an event. Movement of either nearest distance alone does not; the latest
distance is still visible for action selection. Zombie state changes alone supply
no event signals. Every decision sees current state. Summaries retain latest
public state, span and count; categories are never
averaged. Memory clears at every game boundary. Compression can lose timing.

Current queries attend only to present/past tokens. Relative elapsed distances
are positional features, not entity timers. Raw histories are re-encoded with
current weights. Sixteen-transition chunks use an eight-transition public-history
burn-in prefix, excluded from losses. It reconstructs admission/compression, not
learned hidden activations. Archived contexts supply the remainder of each chunk.

GTrXL motivates gated pre-normalization; Transformer-XL motivates recurrence and
relative positions; Compressive Transformer motivates summaries; Longformer/BigBird
motivate local plus selected global context. This project-specific combination
does not reproduce their algorithms or performance. S4/Mamba are not adopted.

## Reward and time

Living plant value is cost multiplied by remaining health fraction. Net value is
the change in sun plus living assets, minus actual sky income, plus **0.25 times
effective plant damage**, minus **600 per new mower activation**. Reward is outcome
plus **0.1 times net value divided by 300**. Victory gives +1, defeat −2. The
denominator is a scale, never a sun cap.

Purchases conserve value. Damage reduces asset value; digging/death loses only
the remainder. Actual sunflower income earns credit. Overkill and mower damage
earn none; plant damage includes HP, armor, explosions and consumption. Assets are
not zeroed at termination. Peak/drawdown remain diagnostics. This changes the
objective intentionally; it is not policy-invariant shaping.

Game 1.4.0 / simulation 1.1.0 runs at **100 Hz**. Combat definitions retain seconds,
but finer collision rounding changes exact outcomes and hashes. Accepted plant/dig
operations are instantaneous; waits/rejections advance one tick. Videos sample
25 FPS independently while replay verification checks every tick.

Duration k uses gamma^k for bootstrap and (gamma*lambda)^k for GAE. Both remain
**0.999 per tick**, as requested. Gamma's physical half-life changes from 34.6
seconds at 20 Hz to **6.93 seconds** at 100 Hz. This materially shortens credit and
confounds comparisons. Natural endings disable bootstrap; cutoffs use terminal
observations and histories before reset. Zero-time actions cannot postpone a loss
in the discount clock.

## Optimization and exploration

Defaults: 128 environments ×128 transitions, batch 1,024, four epochs, learning
rate 3e-4, clip 0.2, value coefficient 0.5 and gradient norm 0.5. Waiting remains
a transition; games continue across updates. Separate Adam states and clipping
isolate actor/critic. Target KL 0.01 stops actor updates above 0.015 approximate
KL; critic epochs continue. Zero disables this gate. Optional critic rate defaults
to actor rate.

PPO uses actual joint kind/species/tile probabilities. Balanced entropy coefficients are
0.01 for action-kind entropy, 0.001 for normalized conditional species entropy,
and 0.001 for mean normalized tile entropy. Conditional exploration terms are not
weighted by how often their parent kind is chosen; unavailable branches contribute zero.
Initial dig bias −12 is trainable. Sampling differs from greedy
evaluation: with equal other logits and 10% added noise, wait/dig-only states initially
give dig probability 0.00000553 per choice. Over 500 choices the illustrative chance
of any dig is 0.28%. This is not a probability cap; positive
advantages can increase digging. The 1,024-game warm-up remains unchanged.

Plant/state embeddings use equivalent one-hot matrix products during backpropagation
and ordinary lookups during inference. The optimization avoids sorting repeated
category indices across raw history banks. No network width, precision, PPO reduction
or action timing changes are used to obtain speed.

Added wait/plant-type noise starts at 10%, held during 1,024 stage-resident games
of critic adaptation, then decays exponentially to 0.1% at stage game 3,000 and
toward zero afterward. Digging has no random floor. Rates change between rollouts.
Validation is greedy. At 100 Hz this aggressive noise has more opportunities per
second; no improvement is assumed.

## Curriculum

| Stage | Training mixture | Default mastery |
|---|---|---|
| placement | Placement | 100/100 placement |
| saving | 80% saving, 20% placement | 100/100 saving |
| easy | 80% easy, 10% each lesson | 100/100 easy |
| standard | 45% easy, 45% standard, 10% saving | 100/100 each normal task |
| shared | 20% easy, 40% standard, 40% hard | 100/100 each difficulty |

Probes run every 2,000 completed games by default, with at least 100 stage-resident
completions and one consecutive pass. Cases are 100050–100149. Stage changes affect
future resets, preserving optimizer identity. Parameters are configurable.
The stage option normally stops at mastery or its configured budget. Add
`--until-stage-complete` with `--stage` (or set `training.until_stage_complete = true`
and `curriculum.run_stage`) to continue until that stage passes. Both training time
and game ceilings are inactive in this mode. Probes, minimum residency, per-game
cutoffs, warm-up and exploration clocks remain active. Failed probes continue the
same stage; a passing probe stops after a complete update without automatic promotion.
Explicit `--games`, `--steps` and `--max-minutes` conflict with this mode.

Ctrl+C saves an interruption checkpoint with both optimizers and schedules. Resume
keeps the stopping mode, cumulative elapsed time and mastery progress; use a fresh
output directory. The mastered `final.zip` can initialize the next stage with fresh
optimizers through `--init-from`. An already mastered resume finalizes pending outputs
without collecting another rollout. Missing targets and ETA are reported as null,
not as a synthetic large training budget.

Lessons have no sky income or mowers. Placement starts with 100 sun and basics in
one lane at ticks 5/105/205. Saving starts with 150 sun and basics in two lanes at
4300/5500/6700. Placement permits peashooter; saving also permits sunflower. Full
board and legal digging remain available.

### Lesson pressure trial

Without flower income, 150 sun cannot fund two 100-sun shooters. The tested saving
control invests at ticks 0/750 and buys shooters at 4300/6150, winning at 10501.
Flower first/interval payments are 600/2400 ticks. Waiting loses at 9299. Investment
delay 347 wins and 348 loses; placement delay 502 wins and 503 loses. These are
control-specific boundaries, not universal limits. Controls never supply learning actions.

Normal validation follows each passed stage on 50 seeds per difficulty,
100000–100049. Macro win rate selects best; ties retain the earliest checkpoint.
Mastery does not select best. Budget exhaustion alone does not force validation.
Matching checkpoint/case results are reused; final-test seeds stay untouched.

## Transfer, budgets and outputs

One recipe ships. Weights-only initialization requires compatible architecture and
starts fresh optimizers, counters and memories. Resume restores the experiment and
remaining allowance but restarts interrupted games with empty history. Switching
stopping modes uses compatible weights with `--init-from`, rather than changing a resumed experiment. Older
observation or action-head signatures and 20 Hz pins cannot load. No conversion paths ship.

The default 120 minutes reserves 15 for finalization. Stops occur at complete
updates; evaluation/export checks preserve incomplete status. Suites repeat this
method; benchmarks change parallelism only. Formal training is user initiated.

Runs record settings, source/structural signatures, checkpoint identity, episode
and optimizer diagnostics, compression and attention allocation. Diagnostics
use the final collection batch for attention and the currently retained
bank for compression; event-admission frequency accumulates over collection.
Early digging means removal within **500 ticks/five seconds**, including same-tick removal.
Track it per planting alongside attackers and wins; fewer digs alone does not
establish improvement. Ledger metrics include discounted/undiscounted returns.

One selected actor supplies all three demos. CPU replay must match GPU outcome
and hash before publication. Without validated best, final checkpoint/report still
save but there are no selected-policy demos. Reports need no model. Old 20 Hz
recordings were removed; historical report data remains preserved.
