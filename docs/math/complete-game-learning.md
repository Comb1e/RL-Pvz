# Complete-game controller and conditional PPO

The release uses completed episodes, including explicitly labelled cutoff failures.
An external interruption is not a terminal event. With gamma 1, the target at
decision t is the sum of actual rewards from t through the episode's final
decision. Zero-duration planting and digging decisions are included once.

The critic has 47 outputs: wait, plant, and 45 digging tiles. Illegal choices
are excluded before argmax. Equal values prefer wait, then plant, then the
first row-major digging tile. Q(plant) estimates the return under the frozen
plant-argument policy used to collect the cohort; it is not a maximum over
species. Initial outputs are 0, 0, and the defeat reward for every digging tile.
A negative all-wait episode supplies a negative wait target while leaving the
unobserved plant output without a direct regression target. This allows, but
does not guarantee, subsequent planting; shared features can affect both.

For each populated action group g (wait, plant, dig), compute its mean squared
selected-value error. Average these group means. Thus 10,000 waiting decisions
and 10 planting decisions each receive half the aggregate critic weight when
digging is absent. Absent groups contribute neither loss nor denominator.
Minibatches must use cohort group counts, not their own group proportions.
For N total decisions and K populated groups, a sample in group g has weight
N/(K*n_g). This makes the full-batch weighted MSE equal to the average of group
means. The final short minibatch retains the same per-sample denominator.

For planting decisions only, freeze A = actual return - collection Q(plant)
before fitting the critic. PPO uses the conditional species-and-tile probability
p(s,x). It never assigns a likelihood to the greedy controller or digging tile.
The exploratory prior is uniform over legal species and, within a species,
uniform legal tiles. The collection distribution is (1-epsilon)*p + epsilon*u.
The same mixture supplies saved log probabilities, PPO ratios, entropy and KL.
Normalize advantages once across collected planting decisions. The species
entropy is normalized by log(number of legal species); tile entropy by the
corresponding log(number of legal tiles), then averaged equally across legal
species. A singleton has zero entropy. Both bonuses start at coefficient 0.001.

Exact KL is the sum over legal planting pairs of old_p*log(old_p/new_p),
averaged over recorded planting decisions. No-plant cohorts skip actor fitting.
Rollback restores both actor weights and Adam moments; it cannot undo critic
fitting, completed games or RNG progress. Critic and actor phases never overlap.

The plant exploration schedule at stage progress g is
epsilon = 0.10 * (0.001/0.10) ** min(g/3000, 1).
The entropy factor is 0.10 ** min(g/3000, 1). Both values are resolved once at
cohort start. Evaluation uses epsilon zero and greedy plant arguments.

Greedy decisions provide no general coverage guarantee. A poorly estimated,
untried action may remain untried. Report pre-fit errors and sample counts for
each selected action group; reduced digging alone is not evidence of learning.

## Mechanics and accounting checks

The supported reconstruction gives ordinary body HP 270, pole body HP 500,
cone armor 370 and bucket armor 1,100. One point of effective plant damage is
therefore worth 50/270 sun-equivalents. Autonomous headless decay is excluded.
Head loss neutralizes a threat once; later body removal is not another defeat.

Asset accounting remains sun plus health-weighted living purchase value.
Cumulative net value is final assets minus initial assets minus actual sky
income plus effective plant-damage value minus mower expenditure. Reward is
outcome plus net value/30,000. Purchases exchange equally valued assets.

For the shipped normal levels and 1,200 seconds, the fastest sky schedule has
an initial 425-tick delay followed by min(950,425+10*n) ticks after drop n.
This permits 141 drops, or 3,525 sun before capping. Final assets are bounded by
9,990 sun plus 45*200 living purchase value = 18,990. Initial assets are 50.
Hard-mode spawned body plus armor totals 38,130, so plant damage is bounded by
38,130*50/270 = 7,061.111111 sun-equivalents, even when bodies decay or mowers
remove them without reward. Five mower activations cost at most 1,000.

These conservative inputs give net value in [-4,575, 26,001.111111], a natural
win reward at least 0.8475, and a natural loss reward at most -1.1332962963.
They do not assert outcome separation for arbitrary custom levels. Executable
controls must verify the level totals and schedule against the released rules.

Animation phases use inspected frame counts: mine rise 19 frames at 18 fps
(ceil(100*19/18)=106 ticks), chomper bite 25 at 24 fps (105 ticks, hit at 70),
and swallow 28 at 12 fps (234 ticks). The mine's underground delay is 1,500
ticks, separate from rising. Chomper digestion is 4,000 ticks, separate from
the bite and recovery. Fixed-point, animation-free movement is a reconstruction
approximation; these controls do not establish binary equivalence to PC/GOTY.
