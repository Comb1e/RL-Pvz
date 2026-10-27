# Training objective

The six signed learning components are global terminal, global asset value,
action damage, action blocking, action sun and action mower. All component
accounting and return accumulation use FP64; trajectory targets are stored in
FP32 and cast to BF16 for model fitting. Legacy combat, rejection and proximity
fields remain diagnostics; completion time changes the terminal component only
after a cohort is ranked:

\[
r^{physical}=r^{global\ asset}+r^{action\ damage}+r^{action\ blocking}
             +r^{action\ sun},
\qquad r^{raw}=r^{global\ terminal}+r^{action\ mower}+s r^{physical},
\qquad r^{train}=r^{global\ terminal}+r^{action\ mower}+10s r^{physical}.
\]

Gamma is one; selected-action targets sum the complete episode without bootstrap.
The base terminal rewards are +1 for victory and -1 for defeat or cutoff
(`reward.loss_penalty=1.0`). After each completed cohort, normally 128 games,
terminal rows are ranked together by completion time, including losses. A signed
quadratic adjustment with maximum magnitude `reward.win_time_weight=0.3` gives
terminal wins in [+0.7,+1.3] and losses in [-1.3,-1]. This adjustment is applied
only to the terminal row; physical rewards and mowers additionally affect totals.
Here `s=progress_weight/value_scale=0.01/2000` in the current profile and the
multiplier is `training.objective.dense_multiplier=10`; scaling is applied once
to the combined physical terms, giving /20,000 for fitted physical rewards.
The same interface builds demonstration and autonomous targets.
Reward `action_global_value_v5` and objective `complete_return_action_global_v5`
identify the quadratic terminal and wait-representative contract. Trajectory and
autonomous recovery are v5; demonstration initialization is v11. Old
optimizer/recovery states cannot resume;
structurally compatible weights still initialize fresh runs. Verified historical
demonstrations are repriced under the current contract.

The native and browser live game windows display accumulated **training** reward
for that environment and episode, including causal damage/blocking/sun credits
and direct mower penalties. The display resets with the episode and is captured
with the same board snapshot; it adds no policy inputs or per-step CUDA readbacks.
The endgame time adjustment is not available until cohort finalization, so a
finished board initially shows its base terminal outcome with a pending label.
Finalization replaces the HUD total by environment/episode identity. Delayed
captures cannot undo that replacement; native rendering refreshes its reward
overlay and caption. Gym RGB rendering shares the component conversion.
[Sequential Q learning](sequential-training.md) specifies
the selected-action, consistency and demonstration losses. Maximum net value
and drawdown remain diagnostics.

## Annotation convergence and L2 phase

Tile and action annotation fitting keeps the grouped train/validation/test split
fixed. The top-1 feasible and bottom-k absolute-infeasible objectives use only
the training partition for gradients. Fitting first runs until both exact
training metrics reach 100%. If either validation metric is still below 100%,
the fitter enters a second phase without changing the split or adding validation
records to the training source.

In that phase it adds an explicit penalty to the selected head's FP32 master
parameters:

\[
L_{L2}=\frac{\lambda}{2}\sum_{\theta\in H}\theta^2,
\]

where `H` is the trainable tile or species/action head and `lambda` is
`training.annotation.l2_strength` (overrideable with `--l2-strength`). The
penalty is absent during initial training convergence. Publication requires
exact 100% feasible and absolute-bottom accuracy on both train and validation;
the held-out test split is evaluated only after that gate passes. Every phase,
metric, and regularizer value is recorded in the annotation progress, history,
and report artifacts.

Annotation `--resume-from` restores the last durable model, Adam, Python/NumPy/
PyTorch/device RNG state, cumulative pass count, metrics, and L2 phase. A partial
gradient pass is discarded; an optimizer commit finishes evaluation and atomic
progress saving before Ctrl+C takes effect. The dataset, grouped split, kind,
objective, schema, and model structure must match. Omitted fitting options inherit
saved values; conflicting explicit options require a fresh `--init-from` run.
The pass budget is an execution limit outside that compatibility identity:
`--passes N` permits N additional commits from the restored count, and zero is
unlimited. Reports record the starting count, cumulative count, invocation count,
and effective limit. Resume repairs missing, duplicate, or torn history tails from
the durable progress metrics without repeating committed passes.

Self-contained annotation progress v4 stores configuration/digest, model storage
and precision identities, source provenance, and convergence status. All four
initialization paths (tile, action, demo, autonomous) accept it through `--init-from`,
including unfinished and zero-pass snapshots. They transfer the complete model's
weights and create fresh optimizer/counters/RNG and, where applicable, EMA state.
The source pass, kind, convergence status, and SHA-256 of the exact loaded file
version are recorded. Concurrent replacement is safe: deserialization and hashing
use the same immutable bytes, with bounded retries for Windows sharing conflicts.
This weight handoff does not certify convergence or run held-out test evaluation.
The final annotation initialization artifact remains gated by exact train and
validation verification. Progress v3 can resume with its obsolete total-pass cap
ignored, then writes v4; direct downstream loading of v3 is rejected with upgrade
guidance. Converged annotation artifacts remain supported. Annotation progress is
separate from autonomous `train --resume` recovery.

Assets are available sun plus each living plant's purchase cost times its
remaining-health fraction. Purchase is neutral. Effective damage includes
actual body and armor damage from plants/projectiles, capped at remaining HP;
autonomous headless decay and mower damage do not earn plant-damage value.
With basic body HP 270, damage valuation is `(2000/900)*30/270` per effective HP.

For asset value A, actual sky income S, actual plant-produced income P, effective
damage D, mower activations M, and eligible early Sunflower income E, define
plant asset loss `L=P-(A_after-A_before-S)` and `k=2000/900`. The physical ledger is

\[
R_{physical}=(P-kL)+k(30/270)D+B+(P+2E),\qquad r^{action\ mower}=-0.1M.
\]

Here `B` is per-plant blocking credit, with its coefficients multiplied by `k`.
`reward.asset_loss_multiplier=k` prices health loss and plant removal separately
from production income. Its default of one for older saved configurations
reproduces the original asset delta. The current damage, blocking, and plant
loss prices compensate for the change from value scale 900 to 2000, preserving
their fitted rewards. Both production-sun contributions use the new lower
scale. `reward.mower_penalty=0.1` is a final-unit penalty per activation. It
bypasses `value_scale`, `progress_weight`, and `dense_multiplier` in both raw
and fitted rewards. Mowers are excluded from physical net value/development;
`mower_expenditure` remains zero as a legacy diagnostic.
Subtracting actual sky income prevents free ambient sun from
earning global asset reward. Plant-produced sun remains valuable only when it
fits below the sun cap. Asset losses and mower expenditure are each charged
once; diagnostic kill counts are not another reward term. Damage and delayed
source events are assigned to their placement action by the causal ledger; no
separate kill bonus is added.

The blocker coefficients are priced value per health point: ordinary plants
are approximately `k*10/max_health`, while Wall-nut is `k*100/max_health`.
Blocking itself has no negative action term. The global asset value still records
the plant's health loss, so ordinary plants remain economically costly while a
Wall-nut can receive a positive combined local signal. A terminal defeat still
contributes the global loss penalty.

Eligibility is evaluated at each production event: total roster must be positive
and `3 * cumulative_spawned < total`. Defeat/removal never reopens eligibility;
spawns earlier in the same tick count before production. The configurable exact
fraction is `reward.early_sun_spawn_fraction=[1,3]`, with
`reward.early_sun_extra_multiplier=2`. Count only actual credited sunlight, not
requested production, and never sky income. `early_sun_bonus` is a diagnostic
subset of action sun, not a seventh component. The dense multiplier applies once
to the four physical terms; source sunlight and the production part of global assets
are intentionally both retained.

| Control | Physical ledger effect | Fitted reward contribution |
|---|---:|---:|
| Remove 270 effective HP | action damage `k*30` | +0.0033333333 |
| Full Wall-nut bites | blocking `k*100` | +0.0111111111 |
| Full ordinary-plant bites | blocking approximately `k*10` | +0.0011111111 |
| Lose a healthy 100-sun shooter | global asset `-k*100` | -0.0111111111 before bite credit |
| Produce 25 uncapped early Sunflower sun | global asset `+25`, action sun `+75` | +0.005 |
| Produce 25 uncapped late Sunflower sun | global asset `+25`, action sun `+25` | +0.0025 |
| Sky produces 25 sun | excluded from global assets | 0 |
| Buy a healthy plant | sun converts into equal plant value | 0 |
| Activate one mower | final-unit action mower `-0.1`, no physical cost | -0.1 |
| Empty cherry | no combat credit; remaining plant asset value is lost | negative asset contribution only |

Head loss can stop a game before all body HP disappears. Damage credits use
actual damage and never award the unremoved remainder. Combined signals
include both blocking and lost assets: a fully eaten 50-sun Wall-nut contributes
`k*(100-50)/20000 = +0.0055555556`; a fully eaten 100-sun shooter contributes
approximately `k*(10-100)/20000 = -0.01`, before its damage credit. Historical
net-value bounds from other reward profiles do not bound the current six-component
fitting return. These physical prices are independent of terminal ranking and
the fitting-only invalid-action target.

## One-time house entries

Across all five lanes, living non-headless zombies enter stage one at `x < 2000`
and stage two at `x < 1000`. Boundaries derive from the pinned 1,000-unit tile width.
`reward.home_entry_penalties=[0.005,0.015]` charges each previously uncharged stage:

| Movement | Additional component |
|---|---:|
| 2001 → 2000 | 0 |
| 2000 → 1999 | -0.005 |
| 1000 → 999 | -0.015 after stage one |
| 2000 → 999 | -0.020 |
| Remain inside, or retreat and return | 0 |

The ledger remembers the deepest stage by zombie ID. Initial entities start at
their initial stages; new spawns start uncharged. Only threats surviving the
transition are considered. Death never refunds a charge; headless zombies cannot
breach and incur none. Maximum cost is -0.020 per zombie, or -0.30 for fifteen.
IDs remain private accounting metadata. Recovery restores the ledger.

## Ranked endgame time

After all actual games finish, sort their completion durations from fastest to
slowest, including losses. Equal durations share their average zero-based rank.
For rank r in the actual cohort size n > 1, define

```text
u = 1 - 2r/(n-1)
a = w * u * abs(u), where w = reward.win_time_weight = 0.3
win terminal  =  1 + a
loss terminal = -1 + min(a, 0)
```

`w` is the maximum absolute rank adjustment. The signed piecewise quadratic
makes smaller changes near the median position.

| Normalized rank u | Win terminal | Loss terminal |
|---:|---:|---:|
| +1, fastest | +1.300 | -1.000 |
| +0.5 | +1.075 | -1.000 |
| 0 | +1.000 | -1.000 |
| -0.5 | +0.925 | -1.075 |
| -1, slowest | +0.700 | -1.300 |

A single-game or entirely tied cohort receives zero adjustment. With 128 distinct
times, the central two games are on opposite sides of the median position. The
shared helper serves trajectory finalization, episode reports, standalone
evaluation and verified demonstrations. A single demonstration is a singleton
cohort and therefore retains its base outcome. Rank never enters policy inputs.

`endgame_time_adjustment` reports the applied terminal value minus the base
outcome, so an early loss reports zero. `terminal_reward` and `global_terminal`
are authoritative; legacy `victory_time` stays zero. Repeated finalization
replaces the terminal value and applies only its difference to episode totals.

Collection marks completed rewards provisional. `FINALIZE_REWARDS` writes terminal
adjustments before `RETURNS` and `FIT`. Saved finalization
state prevents applying adjustments or publishing episode records twice on resume.
Finalized HUD overrides are cleared on episode reset, with episode-serial fences
rejecting delayed captures and finalization messages from the previous game.
Curriculum counts victories independently of shaped rewards.

These shaping terms intentionally change the objective and do not claim to preserve
every ordering induced by terminal rewards. [Invalid proposals](invalid-action-penalties.md) execute ordinary wait with zero
rejection charges. Their fixed -1 targets apply only to fitting and do not alter
this reward ledger or earlier chronological returns.

## Physical-return variation diagnostic

Before fitting compaction, bounded FP64 centered moments measure population
standard deviations over all actual decisions. Dense return sums asset, damage,
blocking and produced-sun returns, converted once with
`dense_multiplier * progress_weight / value_scale`. Taking moments of that sum
retains covariance. Terminal outcomes, time ranking and direct mower penalties
are excluded from dense variation but included in total-return variation. Reports
contain `dense_return_std`, `target_std`, `dense_to_target_std_ratio` and
`low_dense_return_variation`.

A ratio below 5% emits one cohort-identified warning event; routine progress does
not repeat historical warnings. Effectively zero total variance has a null ratio
and no warning. An all-wait cohort with no physical rewards can legitimately warn
when outcomes or mowers vary. This diagnostic never changes targets.
