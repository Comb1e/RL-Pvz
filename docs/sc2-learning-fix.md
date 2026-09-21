# SC2 run diagnosis and plant-role reward candidate — 2026-09-21

The implementation keeps one shared policy and the 120-minute ceiling. New methods
remain experimental until matched runs demonstrate normal-game improvement.

## Completed run inspected

`runs/sc2-shared-101` completed 9,084 games and 18,055,168 decisions in 6,387 seconds.
Its best checkpoint at 6,002 games scored 42% easy, 0% standard and 0% hard on
50 fixed validation cases per difficulty (14% macro win rate). Final weights scored
40% / 0% / 0%. The best checkpoint's demonstrations planted only wall-nuts and
potato mines. The last 100 training games averaged 0.35 sustained-attacker purchases
and 8.73 voluntary digs within five seconds of planting; mower kills averaged 6.35.

The initial zero kill logs were genuine: placement lessons disable mowers, and
early policies dug up their peashooters. CPU and CUDA kill attribution agree in
independent controls. Logs now show the current stage, recent task composition,
mower-free lesson context and kill means to two decimal places.

## Replay repair

The files declare `pvz-rl/actions-v1`, which allows several immediate plant/dig
operations at the same simulation tick. Standalone viewers only recognized ordinary
positive-time version-1 recordings. Renaming the version would change semantics.
Both native checkouts now dispatch the research format to an independent reader,
preserving native seek, rendering, provenance and state verification.

All three original demos verify, including rewind and seeking to completion:

| Difficulty | Outcome | Final tick |
|---|---|---:|
| Easy | Won | 4,226 |
| Standard | Lost | 4,184 |
| Hard | Lost | 5,074 |

They share checkpoint SHA-256
`ce6b6fc3b480233c67b4009e6ce44d7412d2e2ef8fc7db8c047d34fe70fbb65e`.
No recording or checkpoint was converted. Native CPU reader 1.2.2 is commit
`a95524e`; CUDA reader 1.3.1 is `314528a`. The research environment deliberately
retains pinned simulator 1.3.0 / `8861824`, so existing models remain evaluable.
`tools/watch_demo.ps1` in either game checkout loads its source reader with a
specified existing Python environment, without replacing installed training code.

## Learning changes

`configs/sc2-efficient.toml` adds choice-state actor normalization, GAE lambda
0.999 and 1,024-sample minibatches. A forced single legal action has no policy
choice; it still trains the critic and participates in GAE. The actor, entropy and
KL reductions use actual choice states in this profile. This changes optimization
weighting and is explicitly optional, not claimed to be algebraically identical
to the old objective. Learning rate, four epochs, gamma and rollout length remain
unchanged. Empty and singleton choice sets have independent finite-gradient tests.

GPU evaluation reuses finished slots immediately while retaining seed/case order,
deterministic actions and exact CPU replay verification. The original fixed-batch
path remains available with `runtime.refill_evaluation=false`.

At Leafy's request, `configs/sc2-plant-rewards.toml` adds:

| Component | Default | Interpretation |
|---|---:|---|
| Offensive plant potential | +0.1 per living listed plant | Peashooter, snow pea, repeater, chomper; configurable list |
| Eaten non-wall-nut plant | −0.02 per event | Only the engine's `eaten` reason counts |
| Empty ash explosion | −0.2 per event | Existing penalty retained; armor damage makes it a hit |

The offensive term is `gamma * Phi(next) - Phi(current)`, logged separately.
An immediate first placement contributes +0.0999, digging it contributes −0.1;
the discounted sum is zero. At actual terminal states the potential is zero;
at time limits it is retained. This supplies immediate planting feedback without
permanent bonuses for repeated plant/dig cycles. The eaten penalty is a deliberate
objective change. Existing economy, wall-nut, kill, damage and terminal terms remain.
Normal detonation and voluntary digging do not count as consumption. Wall-nuts
retain the original economy-potential and absorption terms, with no new eaten penalty.

The GPU path scans existing public diagnostic events entirely on-device, adding
about 252 MiB for 128 games. It never places those events, IDs or future schedules
in policy observations. Legacy configurations default the new weights to zero.
Changed profiles require fresh models; existing checkpoints keep their saved settings.

The delivered plant-reward profile is `sc2-plant-rewards-v2`. Its additional
`initial_dig_logit=-6` initializes the trainable dig-head bias at a lower value.
With otherwise equal wait/dig logits, a fresh policy digs with probability about
0.00247 rather than 0.5 each decision. This addresses the local observation that
stochastic policies removed almost every new attacker before experiencing useful
combat. It is a local exploration prior, not an AlphaStar method or a scripted
retention rule. Digging remains legal, its logit receives PPO gradients, and a
learned preference for digging overrides the starting bias. Loading a checkpoint
never reapplies it. The earlier reward-only `v1` trials remain separately labeled.

## Bounded checks and interpretation

The initial optimization-only comparison gave both methods five minutes, seed 101,
the same five normal validation cases per difficulty, and unchanged mastery probes.
Baseline: 1,024 games / 1,048,576 decisions, about 5,093 training decisions/s.
Efficient: 1,792 games / 1,818,624 decisions, about 10,007 training decisions/s.
Both scored zero on normal validation and remained in placement. This is throughput
evidence, not a demonstrated learning gain. The second seed's comparison was stopped
after the user requested reward changes; it is incomplete and must not be pooled.

The reward-only v1 checks each had five-minute allowances. Both normal validations
scored 0/15. Seed 101 briefly achieved 7/20 placement-probe wins; seed 102 stayed
at 0/20. Neither won a sampled training game. This motivated the trainable digging
initialization, rather than claiming the reward change alone solved exploration.

The final v2 checks each had **three-minute** allowances, including validation:

| Learner seed | Games / decisions | Actual elapsed | Last 100 lesson win rate | Early digs/game | Normal validation wins |
|---|---|---:|---:|---:|---:|
| 101 | 821 / 1,048,576 | 173.50 s | 64% | 0.01 | 2/15 (2/5 easy; 0/5 standard; 0/5 hard) |
| 102 | 1,030 / 1,015,808 | 174.98 s | 84% | 0.04 | 0/15 |

Seed 101 passed placement probes at 20/20 and 18/20 and entered saving. Its two
saving probes scored 20/20, but stage residency was insufficient for promotion.
Seed 102 reached 20/20 on its last placement probe, with no time for a second pass.
Both correctly report `curriculum_incomplete`. Across their whole training histories,
they won 144 and 318 lesson games respectively; the table's 64%/84% values refer
only to the last 100. Active training throughput was 9,775 and 9,602 decisions/s.

This is an improvement in early retention and lesson learning. It does **not**
meet the criterion of better normal-game performance for both learner seeds.
The v2 profile remains experimental, and the original SC2 configuration is preserved.
The three-minute results are not paired-budget estimates against the five-minute
controls or the user's longer run. Machine-readable evidence is in
[diagnostics.json](evidence/sc2-v071/diagnostics.json).

No two-hour run, complete profile matrix or held-out final-test evaluation was launched.
Five-minute results cannot establish two-hour success or generalization. Waiting
frequency alone is not a failure criterion, and shaped return does not select models.

## Verification and remaining limits

Complete research and both game suites passed; see `validation.md` for exact counts.
Independent controls cover actor gradients/Adam updates, forced/singleton choices,
CPU/CUDA eaten-plant attribution, wall-nut exemption, plant/dig cancellation, terminal
versus cutoff potential, corrupt/native/legacy replays, slot reuse and complete traces.
The original easy winning demo's native final frame was visually inspected.

The new profile has no verified normal-game advantage yet. Reward shaping may aid
credit assignment but cannot guarantee useful tactics. The mastery curriculum can
still finish incomplete. Only a user-initiated comparison under the same time limit
can establish whether the candidate improves the requested full-game win rates.
