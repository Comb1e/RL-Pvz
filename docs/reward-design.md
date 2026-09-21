# Reward and action specification — 0.5.0

Implemented 2026-09-20 for Leafy's requests. Game package 1.2.1, simulation
1.0.0, source `6fd1f54706369915013a49eab5c1790f8c55ab0a` remain unchanged.

## Exact objective

Let `N = max(1, initial zombie count)`. All shipped profiles use:

```text
reward = terminal + plant kills/N - 2*mower kills/N
       + sum(nonlethal plant HP damage / full starting zombie HP)/N
       - empty mower activations/5
       - sum(sun at each mower activation)/300
       + 0.2 * wall-nut bite HP damage / full wall-nut HP
       - 0.2 * explosions dealing no zombie HP or armor damage
       + gamma*Phi(next observation) - Phi(current observation)

terminal = +1 for victory, -2 for defeat, 0 otherwise
Phi(running) = 0.5*defeated/N + 0.5*(sun + living plant purchase costs)/300
Phi(genuine terminal) = 0
gamma = 0.999 per policy decision
```

The `sparse` condition disables only the potential term. Set event weights to zero
as well for terminal-only rewards. A loss's total reward can differ from −2 because
combat terms and the final potential correction are separate. Truncation is not a
loss and retains final potential for value bootstrapping.

## Event attribution and default magnitudes

- A death's last same-tick `DamageApplied` identifies its killing source. Positive
  source IDs are plants/projectiles; negative ones identify mower rows. IDs are
  temporary event joins, never policy features.
- Nonlethal damage rewards actual base HP removed, normalized by starting base HP
  from active rules and N. Armor-only removal and the lethal hit earn zero damage
  reward. Earlier hits count even if death happens within the same decision batch.
- Empty activation means no kill by that mower on that tick. A different lane or
  later kill does not cancel its −0.2. The pinned engine immediately kills the
  triggering zombie, so this term is normally inactive.
- **Every mower activation** additionally incurs `−sun/300` once. Read sun after
  all prior action costs and actual `SunProduced.amount` events, including capped
  income. Later income does not affect earlier activations. The penalty is not
  repeated while the mower travels and is separate from −2/N for each kill.
  At 0/50/300/600 sun, this extra penalty is 0/−1/6/−1/−2. It is uncapped up to
  the game's sun cap. `mower_sun_weight` and `mower_sun_scale` control the formula.
- Actual `PlantDamaged.damage` on a wall-nut earns
  `wall_nut_damage_weight * damage / rules.plants.wall_nut.health`. Default weight
  0.2 gives +0.005 for a 100-HP bite and +0.2 for a fully eaten 4,000-HP nut.
  Other plants, mere contact/biting state, and idle wall-nuts receive no credit.
  A final partial bite receives only the actual remaining HP. Costs and loss of
  the plant still affect potential separately.
- Each `PlantExploded` with no positive HP **or armor** removal by that exact
  source at that tick costs `empty_explosion_penalty`, default 0.2. Cherry bombs
  and potato mines are the explosive plants in this game. Another plant's hit,
  another explosion, a future zombie, or an earlier hit cannot rescue an empty
  blast. A mine waiting to trigger, digging, and being eaten do not count as blasts.
  A damaging explosion gets existing kill/damage credit and no extra blast bonus.

The sun, bite and missed-explosion defaults are project choices implementing the
user's preferences, not weights established by a paper. They are configurable.
They intentionally change the task objective; win rate remains the selection
metric. Absorption credit can favor longer blocking, and sun-weighted mower costs
can favor spending before a breach. Full-plant bite credit is bounded per nut;
its benefit to learned win rate has not been measured.

## Potential and independent mathematical controls

All living plants contribute their full public-card purchase costs, regardless of
health. Economy is uncapped. Buying a 100-sun plant preserves `sun + plant value`;
digging or losing it removes 100, even when total wealth exceeds 300. Bomb/mine
consumption also removes value. A constant positive potential contributes
`(gamma−1)*Phi` per decision.

Discounted shaping telescopes to `-Phi(initial) + gamma^T*Phi(final)`, including
zero-time actions. Natural terminals use zero final potential; external truncation
retains it. This invariance argument concerns the potential transformation of the
chosen combat objective; it does not turn combat bonuses into win-only training.
Gradual damage can earn more cumulative credit than an instant kill.

## Timing and game-count training

Place/Dig applies immediately without advancing the simulation. The agent may
choose again at that tick using updated observations and masks. Wait/rejection
advances one tick at 20 ticks/second. Ordinary costs, positive card recharges,
occupancy, and finite board size limit legal zero-time operations; no additional
per-tick cap is imposed. A research-local subclass suppresses the pinned engine's
advance hook during immediate operations. No game source files are modified.

PPO still discounts each policy decision by 0.999, including immediate operations.
An all-wait discount half-life is approximately 34.6 simulated seconds; additional
operations shorten it in simulated time. No discount, GAE, rollout, network, or
optimizer retuning is bundled into this change.

Completed training games drive the budget, curriculum, validation intervals, ETA,
and primary curve axis. Wins, losses and timeouts count; probes/evaluation/demos
do not. Stop after optimizing the final rollout and report any excess games.
Game counts and curriculum state survive checkpoints/resume. An unfinished game
at interruption is not completed; resume starts fresh episodes.

## Compatibility and verification

New timing, rewards and game budgets require fresh runs. Configurations without
new reward weights default those terms to zero; missing loss weight remains −1.
Missing action timing retains fixed ticks; missing budget unit retains decisions.
Resume requires the same research configuration; never reinterpret old weights.

Research-format `.pvzdemo` files explicitly mark `pvz-rl/actions-v1`; use
`pvz-rl replay` for verification, seeking, viewing or MP4 export. Legacy native
recordings remain readable. All outcomes/provenance stay outside policy inputs.

Tests combine synthetic arithmetic with actual engine events, same-source/tick
counterexamples, full/partial/custom-health bites, empty and damaging explosions,
spending/income/cap ordering, simultaneous kills, terminal/cutoff boundaries,
reset, batched totals, discount telescoping and action/dig cycles. Full suites,
short CPU/CUDA learning, checkpoints and presentation checks are in `validation.md`.
No new pilot or formal research training was launched for this release.
