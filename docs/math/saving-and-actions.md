# Saving feasibility and balanced action probabilities

These project-specific controls use the pinned 100 Hz game, commit
`1fc80386859087b9d715c4706b3f7875844430cc`. They establish feasibility and probability
identities, not a guarantee that PPO learns the lesson. The maintained independent
controls are in `test_saving_economy.py`, `test_sc2_policy.py`, and
`test_exploration_warmup.py`.

## Why a saving win requires sunflower income

The lesson has 100 initial sun, no sky income or mowers, and three attacking lanes.
Each lane receives a 200-HP basic at ticks 7500, 8700, and 9900. There are
choose(5,3) = 10 lane combinations. All eight species are legal when affordable.

Without actual sunflower production, expenditure is at most 100: there are no
refunds or kill payments. Cherry bomb and chomper cost 150, snow pea 175, and
repeater 200, so none is affordable. One peashooter costs all 100 and damages only
one lane. A sunflower or wall-nut costs 50, leaving at most two 25-sun mines;
one attacking lane then has no damage source. The remaining possibility is mines
alone: at most four, so at least one attacking lane receives at most one mine.

Basics move at 200 position units/second (0.2 tiles/second). Their 12-second
spacing is 2400 units. A mine has 300 HP and a basic bites for 100 every second.
Even conservatively allowing a full three seconds of blocking before a lone
unarmed mine is eaten, the gap stays at least 2400 − 3×200 = 1800 units.
The mine damages exactly its own tile's half-open 1000-unit interval, in its own
lane. It cannot catch multiple such basics. Once it explodes or is eaten, later
basics remain. Digging only removes defenses and budget; waiting cannot create
resources. These cases exhaust the affordable combinations without flower income.
The proof depends on the shipped budget, three lanes, spawn spacing and pinned
combat rules; arbitrary configurable lessons require a new check.

## Public-observation feasibility witness

Buy five sunflowers in column zero, rows zero through four, when affordable and
ready. Then buy a peashooter in column one for each visible, undefended zombie lane.
The controller sees current public zombies and legal masks, never the future lane
selection. Sunflowers are bought at ticks 0, 750, 1500, 3000, 3750; shooters at
7500, 8250, 9300. All ten lane combinations win at tick 13652 (136.52 seconds).

Flowers first pay at age 600 ticks, then every 2400 ticks. By victory, they have
made respectively 6, 6, 5, 5, and 4 payments: 650 sun. All eight plants remain
healthy and the nine basics lose 1800 effective HP. Thus cumulative net value is
650 + 0.25×1800 = 1100, and episode return is 1 + 1100/30000 = **1.03666667**.
Waiting loses with return −2. Immediate shooter purchase and removal loses with
return −2 − 100/30000 = **−2.00333333**. Mine-only and single-shooter counterexamples
also lose. Gamma remains one, so discounted and undiscounted returns agree.

## Initial branch weights and legality

Let n be the number of species with at least one legal tile, w=1.2, and
d=exp(−12). With indicators I_wait and I_dig for legal waiting and digging,

    Z = w I_wait + n + d I_dig
    P(wait) = w I_wait / Z
    P(species s) = I_s / Z
    P(dig) = d I_dig / Z

Each species' mass is divided uniformly among its legal tiles; the total dig mass
is divided among legal dig tiles. Zero final-head weights remove accidental
initial board preferences. Biases are log(w), −12, and zero. Adding log(n) to the
plant-kind logit supplies total planting weight n; the conditional species softmax
divides it by n. Unavailable heads carry no mass. All preferences remain trainable.

At saving's empty 100-sun board, sunflower, peashooter, wall-nut and potato mine
are affordable. Wait is 1.2/5.2 = **23.076923%**, and each species is
1/5.2 = **19.230769%**, divided over 45 tiles. Dig is illegal on the empty board.
In a wait/dig-only state, with warm-up epsilon 0.1, dig probability is
0.9 exp(−12)/(1.2 + exp(−12)) ≈ 0.00000460814. Its constant-probability
500-choice occurrence chance is approximately 0.2301%, not a permanent cap.

## One mixture for collection and optimization

The exploratory prior Q has wait weight w, one weight per legal species, uniform
legal tiles, and zero digging weight. Normalize Q by w I_wait + n. If there is
no eligible prior action (for example dig-only), use the learned distribution.
For all other states, mix complete actions:

    P_epsilon(a) = (1 − epsilon) P_learned(a) + epsilon Q(a)

Recover kind masses by summing their complete actions, species conditionals by
dividing their branch masses by total planting mass, and tile conditionals by
dividing each action's mass by its branch mass. This preserves the same joint
distribution in sampling, log probabilities, entropy, PPO ratios and exact KL.
Unlike independently mixing heads, it gives every legal species tile an explicit
epsilon Q(a) floor even after a strong learned tile preference. At epsilon one,
digging mass is zero unless digging is the only eligible action.

For a selected action, the negative log-mixture gradient is the learned
negative-log gradient multiplied by
(1−epsilon) P_learned(a) / P_epsilon(a). Independent NumPy enumeration checks this
identity and exact entropy. Unused species/tile arguments receive no likelihood
gradient. Deterministic evaluation sets epsilon to zero and greedily selects each
conditional head; this is not global argmax over the 406 complete actions.
