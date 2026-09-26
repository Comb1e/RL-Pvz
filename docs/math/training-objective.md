# Training objective

The implemented complete-game targets, sequential Q values and balanced loss are
specified in [sequential Q control](sequential-q-control.md). Gamma is one;
cutoff episodes receive defeat reward once and never bootstrap. Maximum net value
and drawdown remain diagnostics. The economic accounting below is unchanged.
Total reward additionally includes the explicit
[rejected-action penalties](invalid-action-penalties.md). Earlier
outcome-separation bounds do not apply to totals with repeated penalties.

Assets are available sun plus each living plant's purchase cost times its
remaining-health fraction. Purchase is neutral. Effective damage includes
actual body and armor damage from plants/projectiles, capped at remaining HP;
autonomous headless decay and mower damage do not earn plant-damage value.
With basic body HP 270, damage valuation is 50/270 per effective HP.

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
