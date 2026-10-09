# Training objective

The six signed components are terminal, development, invalid planting, empty
digging, home proximity and victory time. All component accounting and return
accumulation use FP64; finalized fitting targets use FP32. Reporting sums the six
components. Fitting changes only the development coefficient:

\[
r^{raw}=r^{terminal}+r^{development}+r^{invalid\ plant}+r^{empty\ dig}
       +r^{home}+r^{time},\qquad
r^{train}=r^{raw}+9r^{development}.
\]

Gamma is one; selected-action targets sum the complete episode without bootstrap.
Terminal rewards remain +1 for victory and -2 for defeat or cutoff. The multiplier
is `training.objective.dense_multiplier=10`. The same interface builds demonstration
and autonomous targets. [Recurrent training](recurrent-training.md) specifies the
selected-action, counterfactual and demonstration losses. Maximum net value and
drawdown remain diagnostics.

Assets are available sun plus each living plant's purchase cost times its
remaining-health fraction. Purchase is neutral. Effective damage includes
actual body and armor damage from plants/projectiles, capped at remaining HP;
autonomous headless decay and mower damage do not earn plant-damage value.
With basic body HP 270, damage valuation is 50/270 per effective HP.

For asset value A, actual sky income S, effective damage D, mower activations M,
and eligible actual Sunflower income E,
the development reward is

\[
R_{development}=\frac{A_{after}-A_{before}-S+(50/270)D-200M+2E}{30000}.
\]

Subtracting actual sky income prevents free ambient sun from earning development
reward. Plant-produced sun remains valuable only when it fits below the sun cap.
Asset losses and mower expenditure are each charged once; diagnostic kill counts
are not another reward term.

Eligibility is evaluated at each production event: total roster must be positive
and `3 * cumulative_spawned < total`. Defeat/removal never reopens eligibility;
spawns earlier in the same tick count before production. The configurable exact
fraction is `reward.early_sun_spawn_fraction=[1,3]`, with
`reward.early_sun_extra_multiplier=2`. Count only actual credited sunlight, not
requested production, and never sky income. `early_sun_bonus` is a reported subset
of development, not a seventh component. The dense multiplier applies once:
25 eligible sun earns 75/30,000 raw development and 0.025 training reward;
late income earns 25/30,000 and approximately 0.00833 respectively.

| Control | Development reward |
|---|---:|
| Remove a healthy 100-sun shooter | -100/30,000 |
| Produce 25 uncapped early Sunflower sun | 75/30,000 |
| Produce 25 uncapped late Sunflower sun | 25/30,000 |
| Remove 20 effective HP | (20*50/270)/30,000 |
| Activate one mower | -200/30,000 |
| Empty cherry | -150/30,000 |
| Cherry against three full basics | 0 |
| Cherry against one full conehead | (640*50/270-150)/30,000 |

Head loss can stop a game before all body HP disappears. These explosion controls
use actual damage, and never award the unremoved remainder. Bounds for normal
levels are conservative; they are not claims of training success or optimal values.

For the pinned normal levels, 50 initial sun and the 120,000-tick cutoff,
independent controls bound final assets by 18,990 and actual sky income by 3,525.
The largest roster has 38,130 body-plus-armor HP. Cumulative net value therefore
lies between -4,575 and 26,001.111111 sun-equivalents. Outcome plus physical development
is at least 0.8475 on a natural win and at most -1.1332962963 on a natural loss.
These bounds cover only terminal plus **unscaled physical development**, excluding
the early Sunflower bonus and rejection
costs, proximity, time shaping, custom scenarios and incomplete games. They do not
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
IDs remain private accounting metadata. Recovery restores the ledger, and each
isolated probe copies it before execution.

## Cohort victory time

After all actual games finish, compute the median simulation duration \(m\),
including victories, losses and timeouts. For 128 games it is the average of the
64th and 65th sorted durations. Partial cohorts report their actual reference count.
Probes never contribute durations. With `reward.win_time_weight=0.1`:

\[
r^{time}(T)=\begin{cases}0.1(m-T)/(m+T)&\text{victory and }m+T>0,\\
0&\text{otherwise}.\end{cases}
\]

At a 160-second median, victories lasting 80, 160, 240 and 320 seconds receive
+0.03333, 0, -0.02000 and -0.03333. A faster defeat receives no bonus. A single-game
demonstration has \(m=T\) and zero adjustment. The same median finalizes terminal
probes; nonterminal probes bootstrap separately. Median is context, never an input.

Collection marks completed rewards provisional. `FINALIZE_REWARDS` writes terminal
adjustments and fixed probe targets before `RETURNS` and `FIT`. Saved finalization
state prevents applying adjustments or publishing episode records twice on resume.
Curriculum counts victories independently of shaped rewards.

These shaping terms intentionally change the objective and do not claim to preserve
every ordering induced by terminal rewards. Reduced [rejection costs](invalid-action-penalties.md)
still penalize invalid proposals without multiplying them by the development scale.
