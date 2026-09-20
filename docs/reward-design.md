# Reward specification — 0.4.1

Implemented 2026-09-20 for Leafy's requested reward changes. The game remains
package 1.2.1 at commit `6fd1f54706369915013a49eab5c1790f8c55ab0a`, simulation
1.0.0. No engine modification is required.

## Exact objective

Let `N = max(1, initial zombie count)`. Each decision advances up to ten engine
ticks. Sum its event contributions and then add terminal reward and, for shaped
conditions, the potential difference:

```text
reward = terminal + plant kills/N - 2*mower kills/N
       + sum(nonlethal plant HP damage / full starting zombie HP)/N
       - empty mower activations/5
       + gamma*Phi(next observation) - Phi(current observation)

terminal = +1 for victory, -1 for defeat, 0 otherwise
Phi(running observation) = 0.5*defeated/N
                        + 0.5*(sun + sum(living plant purchase costs))/300
Phi(genuine terminal observation) = 0
gamma = 0.999
```

The potential difference is disabled for the condition named `sparse`; event
rewards are still enabled. Turn all four event weights off as well to reproduce
terminal-only rewards. Normal-game checkpoint selection remains macro win rate,
not episode return. No survival, invalid-action, or waiting penalty was added.

## Damage and mower attribution

- The last damage event before a same-tick death is the lethal hit. Positive
  source IDs denote plants/projectiles; negative IDs denote mower rows in the
  pinned engine. Only transient event joins use these IDs.
- Reward actual base HP removed by nonlethal plant damage. Armor removal gives
  zero, and the lethal hit gives zero damage reward even if its damage is large.
  Earlier hits still count if death occurs later in the same decision batch.
- Use the zombie type's starting base HP from active rules, never remaining HP.
  Basic, flag, conehead, and buckethead zombies have 200 base HP; pole-vaulting
  zombies have 400. Armor is separate. Public observations and spawn events give
  the type, including zombies spawned and killed entirely within one batch.
- An activation is empty only when that mower kills nothing on that engine tick.
  Another lane's kill, or a later kill, does not cancel the 0.2 penalty. A mower
  kill on the activation tick incurs only its −2/N kill penalty. This definition
  is independent of the wrapper's action-batch size.
- In the pinned game, activation immediately kills the triggering zombie. Thus
  the empty-activation penalty is implemented and independently tested but is
  normally zero in real gameplay. No upstream changes are proposed for this rule.

For a 10-zombie episode, a nonlethal 20-HP basic-zombie hit earns 0.01 and a
plant kill earns 0.1. A plant kill after nine such hits earns 0.19 cumulatively;
an instant plant kill earns 0.1. A mower finishing the same nine-times-wounded
zombie gives 0.09−0.2 = −0.11. This is a consequence of the requested nonlethal
damage bonus: cumulative rewards depend on the path to a kill. These are event
totals before terminal reward, shaping, and discounting, not win probabilities.

## Economy potential and mathematical controls

All living plants contribute their purchase cost, regardless of remaining health.
Use costs from public cards so custom rules remain correct. Keep zombie-progress
weight 0.5 and combine the previous 0.3/0.2 economy weights into 0.5. Scale the
sum by 300 without capping it. This replaces the old separate capped sun and
sunflower terms; it does not replace the difference between consecutive potentials.

Buying a 100-sun plant from 200 sun gives `200 = 100+100`, preserving economy.
Digging that plant lowers total value by 100. This remains true above 300 sun,
so the old cap cannot hide the cost of a plant/dig cycle. Plant death also removes
value, including when a bomb or mine consumes itself; plant health loss alone
does not. A constant positive potential still contributes `(gamma−1)*Phi` per
decision. Planting is value-neutral but its discounted shaping term need not be zero.

The discounted shaping terms telescope to
`-Phi(initial) + gamma^T*Phi(final)`. Natural terminals use zero final potential;
external truncation retains it for value bootstrapping. The invariance argument
applies to this potential term relative to the chosen combat objective. It does
not make the added damage and kill rewards equivalent to win-only training.

## Compatibility and verification

All shipped profiles share this reward; they still differ in observation, action
distribution, and curriculum. Old configurations without `potential_mode` use
their original formula. Missing event weights default to zero; existing saved
kill weights remain authoritative. Changing reward settings requires a fresh run.
Resume checks the entire research configuration and rejects changed objectives.

Episode JSON records HP damage, empty activations, kill counts, and each reward
contribution. Progress and curves expose damage reward and empty activation counts;
evaluation summaries use null for metrics absent from archived records.

Independent controls cover health/armor boundaries, earlier and lethal hits,
spawn/death batches, lane/tick attribution, all eight purchases and digs, custom
costs/HP, terminal and truncation algebra, legacy configurations, and real-game
single-tick versus batched totals. Full validation is recorded in `validation.md`.

The 0.4.0 timed pilot predates this objective and supplies no evidence of its
learning performance. Only correctness and short integration checks were run
for this revision; no new pilot or formal training was launched.
