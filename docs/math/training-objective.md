# Training objective

The implemented complete-game targets, sequential Q values and balanced loss are
specified in [recurrent training](recurrent-training.md). Gamma is one;
cutoff episodes receive defeat reward once and never bootstrap. Maximum net value
and drawdown remain diagnostics.
Total reward additionally includes the explicit
[rejected-action penalties](invalid-action-penalties.md). Economic
outcome-separation bounds do not apply to totals with repeated penalties.

Assets are available sun plus each living plant's purchase cost times its
remaining-health fraction. Purchase is neutral. Effective damage includes
actual body and armor damage from plants/projectiles, capped at remaining HP;
autonomous headless decay and mower damage do not earn plant-damage value.
With basic body HP 270, damage valuation is 50/270 per effective HP.

For asset value A, actual sky income S, effective damage D and mower activations M,
the development reward is

\[
R_{development}=\frac{A_{after}-A_{before}-S+(50/270)D-200M}{30000}.
\]

Subtracting actual sky income prevents free ambient sun from earning development
reward. Plant-produced sun remains valuable only when it fits below the sun cap.
Asset losses and mower expenditure are each charged once; diagnostic kill counts
are not another reward term.

| Control | Development reward |
|---|---:|
| Remove a healthy 100-sun shooter | -100/30,000 |
| Produce 25 uncapped sun | 25/30,000 |
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
lies between -4,575 and 26,001.111111 sun-equivalents. Outcome plus development
is at least 0.8475 on a natural win and at most -1.1332962963 on a natural loss.
These bounds exclude rejection penalties, custom scenarios and incomplete games.
Repeated rejected proposals can make a win's total reward negative.
