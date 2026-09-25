# Saving feasibility and conditional actions

Saving has 100 sun, no sky income/mowers and nine basic threats in three lanes.
Costs imply that cherry/chomper/snow/repeater cannot be the first purchase.
A single shooter exhausts the initial funds and cannot damage other lanes.
Buying a 50-sun flower or wall-nut without collecting income leaves funds for
at most two mines; buying only mines permits four. At least one attacking lane
then receives at most one mine. Blast width, speed variation and rising timing
must be checked explicitly rather than assuming the old constant-speed proof.
The maintained no-income controls cover all ten lane combinations; they are not
an exhaustive theorem about every randomized realization and placement sequence.

First flower production is 3–12.5 seconds, then every 23.5–25 seconds. Original
slow-card delays also apply. The old fixed purchase-time witness is invalid.
The maintained public-only witness grows five flowers, chooses shooting lanes
from visible headed zombies and uses an affordable mine while waiting for a
shooter. It has no access to future lanes, production countdowns or RNG state.
Game settings are unchanged. Measured CPU/CUDA results belong in validation.

For a mathematical worst-production check, starting with 100 sun, the first
five flowers can be purchased at ticks 0, 751, 2001, 3750 and 5000. Counting each
previous flower's payments at t_i+1250+2500*k gives enough funds for every
purchase. This establishes a possible investment schedule before tick 7500;
it does not by itself establish a winning defense.

Initial actor conditionals are uniform over legal species, then uniform over
that species' legal tiles. There is no initial stochastic wait/dig/plant mixture.
The critic starts wait=plant=0 and dig=-2 and resolves the tie as wait. See
[complete-game learning](complete-game-learning.md) for normalization, exact KL,
actual targets and the limitation of greedy action coverage.
