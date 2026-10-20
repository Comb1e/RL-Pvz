# Training objective

The six signed learning components are global terminal, global asset value,
action damage, action blocking, action sun and action mower. All component
accounting and return accumulation use FP64; trajectory targets are stored in
FP32 and cast to BF16 for model fitting. Legacy combat, rejection and proximity
fields remain diagnostics; completion time changes the terminal component only
after a cohort is ranked:

\[
r^{physical}=r^{global\ asset}+r^{action\ damage}+r^{action\ blocking}
             +r^{action\ sun}+r^{action\ mower},
\qquad r^{raw}=r^{global\ terminal}+s r^{physical},
\qquad r^{train}=r^{global\ terminal}+10s r^{physical}.
\]

Gamma is one; selected-action targets sum the complete episode without bootstrap.
The base terminal rewards are +1 for victory and -1 for defeat or cutoff
(`reward.loss_penalty=1.0`). After each 128-game cohort, terminal rows are
ranked by completion time. With `reward.win_time_weight=0.5`, the earliest
winner receives +1.5 and the latest winner +0.5. Losses before the median stay
at -1; losses after the median become progressively more negative, reaching
-1.5 for the latest rank. This adjustment is applied only to the terminal row.
Here `s=progress_weight/value_scale` and the multiplier is
`training.objective.dense_multiplier=10`; scaling is applied once to the
combined physical terms. The same interface builds demonstration and
autonomous targets. [Sequential Q learning](sequential-training.md) specifies
the selected-action, consistency and demonstration losses. Maximum net value
and drawdown remain diagnostics.

Assets are available sun plus each living plant's purchase cost times its
remaining-health fraction. Purchase is neutral. Effective damage includes
actual body and armor damage from plants/projectiles, capped at remaining HP;
autonomous headless decay and mower damage do not earn plant-damage value.
With basic body HP 270, damage valuation is 30/270 per effective HP.

For asset value A, actual sky income S, effective damage D, mower activations M,
and eligible actual Sunflower income E, the physical ledger is

\[
R_{physical}=(A_{after}-A_{before}-S)+(30/270)D+B+C-500M,
\]

where `B` is the signed per-plant blocking credit and `C` is produced sunlight
credit (including the configured early-sun source bonus). The current profile
values 270 effective basic-zombie damage at 30 sunlight, a full Wall-nut block
at 100 sunlight, a full ordinary-plant block at 10 sunlight, and one mower at
500 sunlight. Subtracting actual sky income prevents free ambient sun from
earning global asset reward. Plant-produced sun remains valuable only when it
fits below the sun cap. Asset losses and mower expenditure are each charged
once; diagnostic kill counts are not another reward term. Damage and delayed
source events are assigned to their placement action by the causal ledger; no
separate kill bonus is added.

The default blocker coefficients are raw sun-equivalent value per health point:
ordinary plants are `10/max_health`, while Wall-nut is `100/max_health`.
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
to all five physical terms; source sunlight and the separate global asset delta
are intentionally both retained.

| Control | Physical ledger effect |
|---|---:|
| Lose a healthy 100-sun shooter | global asset `-100` plus its signed bite credit |
| Produce 25 uncapped early Sunflower sun | global asset `+25` plus source action sun |
| Produce 25 uncapped late Sunflower sun | global asset `+25` plus source action sun |
| Remove 20 effective HP | action damage `(20*30/270)` |
| Activate one mower | action mower `-500` |
| Empty cherry | no combat credit |
| Cherry against three full basics | action damage from actual body/armor damage |
| Cherry against one full conehead | action damage from actual body/armor damage |

Head loss can stop a game before all body HP disappears. These explosion controls
use actual damage, and never award the unremoved remainder. Bounds for normal
levels are conservative; they are not claims of training success or optimal values.

For the pinned normal levels, 50 initial sun and the 120,000-tick cutoff,
independent controls bound final assets by 18,990 and actual sky income by 3,525.
The largest roster has 38,130 body-plus-armor HP. Cumulative net value therefore
lies between -4,575 and 26,001.111111 sun-equivalents. Outcome plus physical development
is at least 0.8475 on a natural win and at most -0.1332962963 on a natural loss.
These bounds cover only terminal plus **unscaled physical development**, excluding
the early Sunflower bonus, proximity, time shaping, custom scenarios and incomplete games. They do not
bound the full raw or fitting return.

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

After all actual games finish, sort the 128 completion durations from fastest to
slowest. Equal durations share their average rank. For rank q in a cohort of n
games, the adjustment is

a_q = w(1 - 2q/(n-1)), with w=reward.win_time_weight=0.5.

A winning terminal row receives 1+a_q. A losing row receives -1+min(a_q,0).
Thus winners before the median improve while early-loss penalties stay at -1;
late winners decline and late-loss penalties increase, ending at +0.5 and -1.5
for the last rank. A single-game or equal-duration cohort has zero adjustment.
The rank is cohort context and never a policy input.

Collection marks completed rewards provisional. `FINALIZE_REWARDS` writes terminal
adjustments before `RETURNS` and `FIT`. Saved finalization
state prevents applying adjustments or publishing episode records twice on resume.
Curriculum counts victories independently of shaped rewards.

These shaping terms intentionally change the objective and do not claim to preserve
every ordering induced by terminal rewards. [Invalid proposals](invalid-action-penalties.md) execute ordinary wait with zero
rejection charges. Their fixed -1 targets apply only to fitting and do not alter
this reward ledger or earlier chronological returns.
