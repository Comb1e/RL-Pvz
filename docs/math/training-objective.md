# Training objective

The implemented complete-game objective, target derivation, conditional planting
likelihood/KL and balanced critic weighting are specified in
[complete-game learning](complete-game-learning.md). Gamma is 1; cutoff episodes
receive defeat reward once and are never bootstrapped. Plant advantages are
fixed before critic fitting. Maximum net value and drawdown are diagnostics.

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
