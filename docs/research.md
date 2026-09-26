# Research protocol — 0.22.0

The sole method is complete-game CUDA collection, Monte Carlo action-value
regression and conditional plant PPO. See [architecture](architecture.md) for
the exact 286 public inputs and workflow, and [mathematics](math/complete-game-learning.md)
for targets, weighting, mixtures and outcome bounds. The game package is pinned
to merged commit `092841aa0dc7dcb1f2c4e92bfbb1ba63e0adff8d` (1.5.0, simulation 1.2.0).
Its production, recharge, health and headless rules are derived from inspected
PC reconstruction sources with documented geometric/animation approximations.

## Collection, objective and evaluation

Each default cohort contains one complete game in each of 32 environments.
Finished games pause. The frozen collection controller chooses maximum legal
critic values; only planting species/tiles are sampled. After returns are
finalized only the selected network trains, then updated weights apply to a new
cohort. Roles alternate 256 critic games and 256 actor games, with eight updates
at 32-game cohort boundaries per phase. Stages start with critic fitting. An
actor cohort without planting samples trains neither network and still receives
role credit; it cannot silently fall back to critic fitting. The 6 GiB host trajectory budget overflows to disk;
bounded minibatches move to CUDA through two pinned staging slots. Raw tokens
may be cached up to 2 GiB while leaving a 2 GiB free-VRAM reserve; the cache is
disposable and CPU/disk storage remains authoritative.

Reward remains outcome (+1 win, -2 loss) plus net value/30,000. Net value uses
actual production, health-weighted living assets, effective plant damage and
mower cost. Head loss neutralizes once; autonomous decay earns no plant-damage
credit. Gamma is 1. At 1,200 seconds, a cutoff is an explicitly labelled failure
with the defeat reward once and no bootstrap. An external stop is not a loss.

Critic errors are measured against newly completed actual returns before fitting,
with separate wait/plant/dig coverage. Balanced group MSE prevents abundant
waiting samples from drowning sparse groups. Plant advantages retain the frozen
collection Q(plant) baseline. Conditional PPO ratios/KL exclude greedy decisions.
The critic and actor never share parameters or Adam states.

Exploration starts at 10% in planting arguments and decays to 0.1% over 3,000
stage games. Species/tile entropy coefficients decay to 10% of their initial
0.001 values. Floors hold until stage completion; values are fixed within each
cohort. Evaluation disables injected exploration and takes greedy arguments.
Greedy controller values do not guarantee useful coverage or successful learning.

## Curriculum and checkpoints

Stages are saving → easy → standard → shared. Saving has 100 initial sun, no
sky income/mowers, all species subject to normal legality, and three selected
lanes with basics at 75/87/99 seconds (nine threats). Easy mixes 80% easy with
20% saving. Later mixtures, held-out probes and residency remain configured in
the single `configs/train.toml`; defaults require 100/100 on each required task,
one passing probe, minimum residency and probes every 2,000 games.

Stage selection with `--until-stage-complete` disables run time/game ceilings;
explicit ceilings conflict with it. It stops after the selected stage passes
and its cohort update finishes, then performs normal-game stage validation.
It does not advance automatically. Bounded runs remain available.

Resume uses one atomic checkpoint archive, which preserves
unfinished physical games, gameplay and sampling RNG, token banks, cohort phase,
optimizer cursor, targets and rollback state. No in-flight game is silently
restarted. Stage transfer loads only compatible weights into a fresh experiment.
The new optimizer protocol rejects full 0.21.0 resume. Compatible 0.21.0
inference and weights-only initialization remain available.
Existing run files remain intact and can be read with their original versions.

## Interpretation

The old saving witness (five flowers and three shooters) failed after mechanics
corrections. A revised public-only witness adds an affordable mine while waiting
for a shooter; lesson difficulty is not changed. Controls and their measured
results are recorded in validation. They demonstrate feasibility, not PPO mastery.
A proof for every randomized speed/timing realization is not claimed.

The viewer and hardware monitor are outputs only. Reports refresh after evaluation
attempts and graceful stops; flushed samples allow offline regeneration. CPU/GPU
utilization cannot be reconstructed for old runs that never recorded samples.
No formal learning experiment is part of this release verification.
