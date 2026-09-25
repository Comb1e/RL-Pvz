# Training method

## Periodic on-policy execution

A bounded
two-slot producer–consumer window is the only training path. At its start, the
collector receives frozen actor and critic copies. Both rollouts are generated with
the same policy version; the learner updates the first slot while the second is
collected. Live weights synchronize to the collector only after both slots are
drained. A final partial window may contain one slot so decision budgets do not
overshoot unnecessarily.

The collector and learner use separate CUDA streams and explicit events. Each slot
stores its own event-memory archive, terminal context, behavior log-probabilities,
values, advantages and returns. Curriculum transitions, probes, validation and
checkpoint writes wait for an empty window. Ctrl+C completes that window before
saving; failed partial windows cannot be saved. Resume starts a fresh window with
saved optimizers and counters.

The scheduler preserves temporal memory, masks, 128 environments, 128 transitions per environment and the
two independent Adam optimizers. PPO uses the saved behavior log probabilities
for its usual ratios, with no V-trace or additional policy-lag correction. Measured
overlap and critical-path throughput are recorded in run metrics; published results from other systems are not transferred to this GPU.

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
effective plant damage**, minus **200 per new mower activation**. Reward is outcome
plus **0.01 times net value divided by 300**. Victory gives +1, defeat −2. The
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

Gamma defaults to **1**: outcome rewards keep their value regardless of completion
time. Duration k still uses gamma^k for bootstrap and (gamma*lambda)^k for GAE.
Lambda defaults to **0.9995998799359583 per tick**, giving a 17.32-second trace
half-life. Natural endings disable bootstrap; cutoffs use terminal observations
and histories before reset. Truncated returns are incomplete. Discounted and
undiscounted episode returns coincide with gamma 1; both fields remain available.

The [checked derivations](math/training-objective.md) establish outcome separation
for shipped normal levels completed naturally before the 1200-second cutoff.
They do not establish learning improvement or apply to arbitrary custom settings.

## Optimization and exploration

Defaults: 128 environments ×128 transitions, batch 1,024, four epochs, actor rate
1e-4, critic rate 3e-4, clip 0.2, value coefficient 0.5 and gradient norm 0.5.
Normalize advantages once per rollout, not separately within each sequence minibatch.
Targets and diagnostics retain unnormalized advantages. Separate Adam states and
clipping isolate actor/critic learning.

Target KL 0.01 combines sampled early stopping with an exact hierarchical KL check
against the frozen window behavior. Check each available slot after updates and
new slot-two observations before further updates. Only genuine-choice states enter
the guard average. Either slot exceeding the target, or non-finite candidate KL,
restores the window-start actor and its Adam state. Actor stopping persists across
both slots; critic epochs and collection continue. Zero disables KL guarding.
No retry, adaptive learning rate, action restriction or alternative scheduler is used.
The guard bounds collected-state average movement, not all unseen behavior.

PPO uses actual joint kind/species/tile probabilities. Balanced entropy coefficients are
0.01 for action-kind entropy, 0.001 for normalized conditional species entropy,
and 0.001 for mean normalized tile entropy. Conditional exploration terms are not
weighted by how often their parent kind is chosen; unavailable branches contribute zero.
Initial heads have zero final weights. Wait has weight 1.2, each available species
weight one, and legal digging weight exp(−12), shared across its tiles. A log(number
of available species) correction to the plant-kind logit keeps initial species mass
equal as affordability changes. Saving's initial board gives wait 23.08% and each
of sunflower/peashooter/wall-nut/mine 19.23%. Initial legal tiles are uniform.
The injected prior also uses wait weight 1.2 and uniform species/tiles, excluding dig.
Mix complete action probabilities and refactor their conditional heads so collection,
PPO, entropy and exact KL all describe the same distribution. Digging stays trainable.
The [mathematical controls](math/saving-and-actions.md) cover empty, partial and forced
masks, epsilon zero/one, and unused-argument gradients. Greedy conditional evaluation
is distinct from choosing the most likely complete command.

Plant/state embeddings use equivalent one-hot matrix products during backpropagation
and ordinary lookups during inference. The optimization avoids sorting repeated
category indices across raw history banks. No network width, precision, PPO reduction
or action timing changes are used to obtain speed.

Exploration now has separate warm-up and formal phases. It is 10% injected
wait/plant-type noise with full entropy during the first 1,024 stage-resident games.
Actor learning starts at 5% injected noise and full entropy, then decays over the
next 3,000 stage games to a 0.1% injected floor and 10% entropy floor. Both floors
remain until mastery and reset when a new stage begins. Digging remains legal and
has no injected random prior. The schedule is a project-specific stability
hypothesis; it does not guarantee a win-rate improvement. Rates change only between
complete windows and are recorded with phase/progress/floor diagnostics.
Validation is explicitly deterministic: injected epsilon is set to zero and greedy
learned heads are used, regardless of the training schedule.

The current CUDA path uses device-resident sequence index templates, inference-mode
frozen evaluation, a tensor-only masked categorical fast path, and optional fixed-shape
`torch.compile` wrappers that do not alter checkpoint parameter keys. The compiled
path falls back to eager execution if the local compiler rejects a graph. Window
records expose compilation status and host/device phase timings; no environment,
rollout, PPO, reward, observation or action semantics change.

## Curriculum

| Stage | Training mixture | Default mastery |
|---|---|---|
| saving | Saving only | 100/100 saving |
| easy | 80% easy, 20% saving | 100/100 easy |
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

Saving starts with 100 sun, no sky income or mowers, and all eight species subject
to ordinary legality. Three randomly chosen lanes each receive basics at 75, 87 and
99 seconds: nine zombies, covering all ten lane combinations. Saving has no rehearsal
of another lesson. Easy retains 20% saving rehearsal; later mixtures are unchanged.

The [no-income proof and feasibility witness](math/saving-and-actions.md) explain
why a win requires actual sunflower production despite allowing mines and every
other plant. Five sunflowers followed by shooters in publicly visible lanes win all
ten combinations at 136.52 seconds in CPU and CUDA controls. Controls provide no
learner actions. Configurable timings and budgets are experimental lesson design,
not evidence of learnability or a cure for collapse.

Normal validation follows each passed stage on 50 seeds per difficulty,
100000–100049. Macro win rate selects best; ties retain the earliest checkpoint.
Mastery does not select best. Budget exhaustion alone does not force validation.
Matching checkpoint/case results are reused; final-test seeds stay untouched.

## Transfer, budgets and outputs

One recipe ships. Weights-only initialization requires compatible architecture and
starts fresh optimizers, counters and memories. Resume restores the experiment and
remaining allowance but restarts interrupted games with empty history. Switching
stopping modes uses compatible weights with `--init-from`, rather than changing a resumed experiment. Older
observation, action-head or action-distribution signatures and 20 Hz pins cannot load. No conversion paths ship.

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

Pipeline telemetry measures host collection/update intervals after their CUDA work
completes. Overlap is the intersection of those intervals, not an estimate from
phase totals and not proof that GPU kernels execute simultaneously. Window time
includes weight synchronization and hashing; complete benchmark time also includes
callbacks. Collection and update totals overlap and must not be added to estimate
wall time. Deadline prediction uses the last completed window. Ctrl+C is deferred
to that boundary; an unexpected failed window cannot produce a resume checkpoint.

## Optimizer compatibility and diagnostics

Checkpoints identify optimizer protocol `periodic_exact_kl_v1`. Resume requires
that protocol and identical experiment settings. Compatible architecture weights
require the current `balanced_species_tiles_v1` action distribution for inference,
transfer or resume. Previous distribution checkpoints require fresh training; files
remain preserved. Same-protocol resume restores the experiment exactly. Snapshot construction preserves RNG state.

Current-window diagnostics include exact KL, attempted and retained actor steps,
rejected windows, effective entropy coefficients, and signed raw advantages,
positive-advantage fractions and target errors by action category with counts.
Sparse statistics combine sums/counts across slots; absence in one slot does not
hide evidence in the other. Completed-game metrics describe earlier trajectories
and must not be read as a fresh deterministic evaluation. All diagnostics remain
outside policy inputs. No learning comparison is launched during implementation.

## Hardware telemetry and reporting

Training and benchmarks share a background psutil/nvidia-smi sampler. It is enabled
by `training.performance.telemetry`, with a one-second target interval configured by
`logging.hardware_sample_seconds`. The sampler primes and ignores the first CPU
reading, uses nonblocking CPU/RAM queries, and bounds hidden nvidia-smi calls. It
selects CUDA's actual GPU UUID, respecting device remapping, and never synchronizes
a training transition. Raw samples append and flush to `hardware-metrics.jsonl`;
only ten minutes of recent readings are kept in memory.

Samples record UTC, elapsed wall time, session, stage, warm-up/formal phase and
training/validation/reporting activity, plus the latest complete-window throughput,
collection/update/overlap and critical-path times. GPU measurements are device-wide,
including other programs. Memory-controller busy percent is activity, whereas VRAM
MiB is allocated capacity. System/busiest logical CPU percentages range 0–100;
process CPU uses one-core units and can exceed 100. Missing readings are null with
diagnostics, never manufactured zeros. GPU sampling windows are device-dependent.

The report's hardware panels refresh after every probe and validation attempt,
including interrupted/partial attempts, and on completion or graceful interruption.
Images replace atomically; plotting failure leaves checkpoints valid. A forced kill
cannot run plotting but flushed raw samples survive. `visualize --report-only` loads
no model and generates no demonstrations. A partial final hardware-log record is
ignored; malformed earlier records remain an error. Resume ancestry is limited to
the recorded checkpoint boundary, with separately labeled session time axes.
Historical utilization cannot be reconstructed where it was never recorded.

The 15-second terminal block distinguishes recent task statistics, latest-window
throughput, sampled hardware and whole-run averages. Checkpoints, mastery results
and report paths appear as separate events. Missing values show n/a; until-mastery
runs have no fabricated percentage or ETA.
