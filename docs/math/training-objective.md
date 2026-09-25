# Training objective and update controls

Verified against game commit `1fc80386859087b9d715c4706b3f7875844430cc`
(100 Hz). The accompanying independent controls live in the existing reward,
saving-economy, distribution, and periodic-pipeline tests. These calculations
establish accounting and update properties, not learned playing strength.

## Assets, net value, and reward

Let A = sun + sum(Cost × HP / maximum HP) over living plants. For a transition,

    ΔU = A_next − A_current − actual_sky_income + 0.25 × effective_damage
         − 200 × new_mower_activations
    r = outcome + ΔU / 30000

Outcome is +1 for victory, −2 for defeat, and zero otherwise. The scale is
progress_weight / value_scale = 0.01 / 300. It is not a sun cap.
The damage conversion is 50 / 200: one basic zombie's health is worth 50 sun.
Only actual plant/projectile health and armor removal counts, without overkill
or mower damage. Actual capped production enters A; actual sky income is removed.

A healthy purchase exchanges sun for an equal asset, giving zero. Digging loses
remaining value. Damage followed by removal loses the original value once.
Terminal assets are not cleared by accounting. Summing transitions gives

    U = A_final − A_initial − sky_total + 0.25 × damage_total − 200 × activations
    episode_reward = outcome + U / 30000

Production minus plant-value loss equals the resource term above. Cumulative
net value and episode net value are the same quantity. Historical maximum is
max(0, all prefix U); drawdown is maximum − current U. Neither enters reward.

| Event | Net sun-equivalents | Development reward |
|---|---:|---:|
| Healthy / half-health shooter dig | −100 / −50 | −0.00333333 / −0.00166667 |
| Sunflower payment | +25 | +0.00083333 |
| 20 effective damage | +5 | +0.00016667 |
| Mower activation | −200 | −0.00666667 |
| Empty cherry / cherry killing one basic | −150 / −100 | −0.005 / −0.00333333 |
| Cherry killing three basics or one conehead | 0 | 0 |
| Cherry killing one buckethead | +175 | +0.00583333 |

## Outcome separation for shipped normal games

This proof assumes natural completion within 1200 seconds, initial sun 50,
the pinned costs/HP, no zombie healing, five mowers, and the shipped wave lists.
It does not apply to arbitrary custom scenarios or incomplete truncations.

Final assets cannot exceed 9990 + 45 × 200 = 18990. Actual sky income cannot
exceed 120 × 25 = 3000. Total effective damage is bounded by spawned health and
armor. The easy/standard/hard totals are respectively 3000/14500/33000 HP,
worth 750/3625/8250 sun. Consequently, across these levels,

    −50 − 3000 − 5 × 200 ≤ U ≤ 18990 − 50 + 8250
    −4050 ≤ U ≤ 27190
    win_reward ≥ 1 − 4050 / 30000 = 0.865
    loss_reward ≤ −2 + 27190 / 30000 = −1.0936666667

Thus even these loose bounds separate all naturally completed wins from losses.
The development weight 0.01 is below both the negative-loss bound
300 × 2 / 27190 = 0.02206694 and the win/loss separation bound
300 × 3 / (27190 + 4050) = 0.02880922. These are sufficient bounds, not optimal
coefficient estimates. Mower cost 200 and the learning rates remain tuning choices.

All ten saving lane pairs were checked with public-state scripted controls:

| Control | Wins / 10 | Return range |
|---|---:|---:|
| Two flowers, two correctly placed shooters | 10 | +1.0175 |
| Wait | 0 | −2 |
| One shooter | 0 | −1.995 |
| Three flowers | 0 | −1.99416667 to −1.99083333 |
| Continue buying flowers | 0 | −1.9975 to −1.97583333 |
| Immediate shooter purchase/removal | 0 | −2.00333333 |

## Return clock and GAE

Let T_t be elapsed simulation ticks before transition t, and k_t its duration.
Recorded return is sum(gamma^T_t × r_t). Gamma = 1 makes this the ordinary
episode reward, so later outcome rewards are not weakened. Economic outcomes
can still differ: equal outcome rewards do not imply equal total rewards.

The old gamma 0.999 per 100 Hz tick has a 6.928-second half-life. Even restoring
the former 20 Hz time horizon with gamma = 0.999^(20/100) gives only 0.00045109
weight to a victory at 385 seconds. This motivates the undiscounted objective.

For value learning, with m_t = 0 at natural termination and 1 otherwise,

    δ_t = r_t + gamma^k_t × m_t × V_next − V_t
    GAE_t = δ_t + (gamma × lambda)^k_t × m_t × GAE_next
    target_t = V_t + GAE_t

Lambda = (0.999 × 0.999)^(20/100) = 0.9995998799359583 retains the former
combined trace decay with gamma now 1: half-life 17.320 seconds, weight 0.9500643
over 128 ticks. This is a trace calibration, not a proof of the best estimator.
Rollout boundaries bootstrap from the frozen critic. At a time cutoff, bootstrap
once from the terminal observation/history before reset; the reset stops GAE
from entering the next game. Natural termination never bootstraps. Zero-duration
actions have factors 1; inserting them cannot postpone discounting. Positive
truncated returns are incomplete and are not evidence of victory.

## Exact hierarchical KL and rare actions

For frozen behavior P and candidate Q with the same legal mask,

    KL(P || Q) = KL(P_kind || Q_kind)
      + P(plant) × KL(P_species || Q_species)
      + sum_g P(branch_g) × KL(P_tile|g || Q_tile|g)

Branches are eight planting species and digging. Unavailable branches have zero
mass. This identity equals enumerating the 406 transport actions. Use the actual
exploration mixture, not the unmixed logits. The guard averages over observations
with multiple legal actions and checks each available rollout separately.

For old dig probability 10^-6 and new probability 0.1, exact KL is 0.10534790.
If all sampled actions were waits, the estimator r − 1 − log(r), where
r = Q(wait)/P(wait), gives only 0.00536042. The chance of observing no digs in
32768 samples is (1 − 10^-6)^32768 = 0.96776304. The estimator is unbiased in
expectation, but this finite-sample failure matters for rare actions.

PPO clipping is not a hard KL constraint. Sampled stopping prevents later steps;
it cannot undo an excessive earlier step. A rejected window restores actor
parameters and Adam moments, but never critic state, collected games, or RNG
progress. This guarantees the checked average KL bound on collected choice
states after rejection, not pointwise or unseen-state behavior.

## Exploration and advantage interpretation

For binary dig probability p and its logit z, dH/dz = p(1−p) log((1−p)/p).
It is positive for p < 1/2: entropy encourages a rare legal dig without any
positive game reward. The injected exploratory prior separately excludes digs.

After warm-up w, define q = min(1, max(0, (games−w)/D)), where D is the formal
decay length. The shipped defaults use D=3,000, formal epsilon start/floor
0.05/0.001, and formal entropy factor start/floor 1.0/0.1. Thus

    epsilon = 0.05 × (0.001 / 0.05)^q
    entropy_factor = 1.0 × (0.1 / 1.0)^q

During warm-up epsilon is 0.1 and the entropy factor is 1.0. At q=1, both values
are held at their floors until stage mastery; a stage transition starts a fresh
warm-up clock. Resolve and freeze these values before each periodic window. The
injected prior still excludes digging. Deterministic validation sets epsilon to
zero and uses the greedy learned heads. These schedule values are project-specific
stability hypotheses, not a learning guarantee.

A negative immediate reward can have positive δ if V_next − V_current is large.
This is an estimator claim, not actual profit. Full GAE can reverse a one-step
error. Normalize advantages once across the entire rollout; retain raw advantages
and targets for diagnostics and MSE. Aggregate action-category sums/counts across
slots, never averages involving empty-category NaNs. Explained variance uses pooled
return and residual sums/squared sums before taking variances: averaging separate
rollout ratios would not describe the combined sample. It measures agreement with
bootstrapped targets, not independent Monte Carlo truth.

See [references](../references.md) for inspected source scope. No independent
mathematical control can establish that these changes improve learned win rate.
