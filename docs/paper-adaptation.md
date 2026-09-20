# Learning from the tower-defense paper

## Evidence and scope

Source: Dias, Foleiss, and Lopes, *Reinforcement Learning in Tower Defense* (2022),
DOI [10.1007/978-3-030-95305-8_10](https://doi.org/10.1007/978-3-030-95305-8_10).
The user supplied `F:/chrome/978-3-030-95305-8_10.pdf`. All 13 pages were read;
pages 9–12, including the experimental tables, were visually inspected.

The paper experiments with structured game metadata and representations of
different sizes. Table 4 lists 16/20/65/83-input variants and includes enemy counts
in thirds of a path. The game is simplified and runs independently of rendering.
Its event queue skips empty simulation cycles. Its learner is DQN with a memory
of 2,000 transitions. Reward descriptions mention periodic, enemy-related, and
terminal signals, but do not provide sufficiently specified weights for a faithful
reward reproduction. Table 5 does not clearly associate every reported score with
the input configurations. Its best reported “maximum average” score is 2,180.65;
the reported human average is 18,680. LSTM/temporal modeling is proposed future work.

These observations motivate representation experiments and simpler introductory
tasks. They do not establish DQN superiority over PPO or reliable human-level play.
This project adapts ideas, rather than claiming to reproduce the paper's game or
its experiments. Checkpoints, logs, and local test results are separate evidence.

## Local problem

The archived `runs/masked-cuda-101` records contain 10,229 completed episodes.
Accepted purchases total 52,484 potato mines, 21,361 sunflowers, and 36,012 wall-nuts;
there are no peashooters or other sustained attackers. Its selected checkpoint at
2,252,800 decisions wins 6/50 easy cases and 0/50 standard and hard cases: 4% macro
validation win rate. The final validation is 0%. These are archived development
results, not held-out tests and not a direct budget-matched comparison with 0.4.0.

At 50 starting sun there are 45 legal placements for each of sunflower, wall-nut,
and potato mine, plus wait: 136 legal actions. Equal flat logits give waiting
probability 1/136. Initial sun was represented as 50/9990, approximately 0.005.
This suggests investigating exploration and economic information. It does not
prove a single cause, and the approximately 93% observed waiting rate is not itself
a failure metric: cooldowns and saving both legitimately require waiting.

## Adopted changes

### Tactical v2: 1,140 public numeric values

| Block | Shape / count | Meaning |
|---|---:|---|
| Plant grid | 5 × 9 × 17 = 765 | Type, health fraction, timer, behavior state |
| Zombie regions | 5 × 3 × 16 = 240 | Type counts, health, armor, position, slow/pole/timer/state aggregates |
| Projectile regions | 5 × 3 × 3 = 45 | Counts, damage sums, icy counts |
| Globals | 54 | Sun, elapsed time, wave/counts, cards, mower positions/states |
| Card economy | 8 × 2 = 16 | Interleaved recharge-ready and fractional sun shortfall |
| Lane summaries | 5 × 4 = 20 | Nearest zombie distance, sustained firepower, wall-nut health, sunflower count |

Three equal regions cover `house_x` through `spawn_x`. A coordinate on an internal
boundary belongs to the next region. Coordinates outside that extent go into the
nearest endpoint region; no entity is discarded. Integer sums precede scaling.
Counts and attributes can exceed one; crowds are never clipped. Reordering public
entities or changing IDs, task names, seeds, or hidden schedules cannot change
features when the remaining public values are equal.

Sun is divided by maximum card cost (200). Local counts use five entities; global
counts retain 75. Card cooldown uses its own recharge duration. Projectile damage
uses the largest projectile-plant damage (20), not cherry-bomb blast damage.
Fractional shortfall is `max(cost-sun,0)/max(cost,1)`; ready indicates recharge only,
so availability still depends on sun and the mask. Nominal sustained firepower is
damage × shots per cycle × tick rate / firing interval, divided by the configured
20 damage/s scale. Wall-nut health is expressed in full wall-nuts per nine tiles;
sunflower count is divided by nine. No-zombie nearest distance is one; type counts
distinguish that from a zombie at spawn. These summaries describe current visible
resources, not a scripted action recommendation.

The sun/count changes and lane summaries are project-specific additions. The
regional aggregation is the closest adaptation of the paper's representation work.
Spatial v1 remains available unchanged as the controlled baseline.

### Grouped action probabilities

The network predicts ten type logits and 9 × 45 conditional tile logits. The
environment still exposes Discrete(406): wait, eight plants on 45 tiles, and digs.
The existing legality mask determines available types and conditional legal tiles.

For a non-wait action, `log p(a) = log p(type) + log p(tile | type)`. For wait it is
only `log p(wait)`. The entropy used in PPO is
`H(type) + sum_type p(type) H(tile | type)`; waiting contributes zero conditional
entropy. Masked types have zero probability. Dummy conditionals for unavailable
types have no joint mass. Both entropy components are logged after updates.

Equal logits give equal probability to available types, regardless of tile count.
At the initial normal-game state, wait therefore has probability 1/4. This is a
change in inductive bias, not a scripted savings rule. Deterministic inference
chooses the highest-probability type and then its highest-probability legal tile;
it deliberately need not equal the argmax of the flat joint probabilities.
Independent NumPy calculations test normalization, entropy, log probabilities,
zero illegal mass, and finite gradients on CPU and CUDA.

### Learning lessons

One policy and optimizer are retained through placement, saving, easy, standard
introduction, and shared normal games. Stage distributions and mastery gates are
versioned in `configs/pure-rl.toml`. The full board is present throughout. Placement
permits peashooter/wait/legal digging; saving also permits sunflower. Both have
three basic zombies in a seeded lane and no mowers. Normal-game evaluation always
permits all plants. Lesson restrictions also reject an unmasked caller's prohibited
request while advancing time, rather than relying solely on cooperative policies.

Every 16,384 decisions, 20 fixed validation cases check the current requirement.
Two consecutive passes and minimum stage residency are necessary. Failing a probe
resets the pass streak. Stage changes take effect at the next reset; current games
finish unchanged. Checkpoints preserve stage, entry step, last probe, and streak.
Normal validation alone chooses `best.zip`, using the mean of easy/standard/hard
win rates and the earlier checkpoint on ties. An unfinished curriculum is recorded
as such when the run ends.

Independent controls verify solvability in all five lanes: wait until a visible
zombie and 100 sun, plant one peashooter at column zero, then wait. Placement wins
at tick 883, versus wait-only loss at 1000. Saving wins at 2074, versus wait-only
loss at 2199. These controls exist only in tests. Their trajectories are never
given to PPO; pure-RL diagnostic policies start from random weights.

## Deliberately not adopted

- DQN or an off-policy replay memory: the paper provides no controlled evidence
  that switching would address this policy's exploration problem. PPO retains its
  on-policy rollout buffer and current clipping, discount, GAE, and update settings.
- Unspecified periodic/enemy rewards or survival bonuses from the paper. The
  original pilot retained terminal rewards and potential shaping. The user's
  subsequent request explicitly adds plant-kill rewards and mower penalties;
  this is a separately recorded objective change, not a paper reproduction.
- Event skipping: this would require changing or carefully extending the engine's
  timing semantics. The game checkout is untouched. Headless simulation and CUDA
  data-transport optimizations are already in place.
- LSTM, screenshots, imitation, scripted warm starts, or scripted placement during
  learning: these are outside this controlled pure-RL comparison.

## Pilot decision rule

The authorized pilot has one 30-minute budget, including at most five minutes of
placement/saving diagnostics. Each diagnostic can use up to 131,072 decisions and
checks 20 validation cases every 16,384. Passing requires a validated checkpoint
with at least 18/20 wins for both lessons. The four normal profiles use seeds 101
and 102, at most 262,144 decisions each, and validation every 32,768 on seeds
100000–100004 for each difficulty. Seed 102 reverses configuration order.

Each remaining job receives a share of the remaining wall-time budget. Collection
stops at a completed-update boundary. In-flight validation/probing and report/save
cleanup may extend past a slot. Reports compare the largest validation budget
completed by every profile and both seeds, without selecting each profile's best
score. Missing comparisons are explicit. A recommendation for a larger development
study requires passing diagnostics and a positive candidate-minus-baseline
difference for both seeds. This is a development screening rule, not statistical
proof of improvement. Final-test seeds are never used.

## Adjustments and remaining limitations

Keep the new method experimental until the pilot decision rule is met. Introductory
tasks may learn slower than expected, and mastery gates can use the entire budget
without reaching hard games. Regional aggregation loses precise enemy distances
within each region, although summed position and nearest-lane distance retain some
location information. The true joint entropy still includes conditional spatial
uncertainty; a separate type-entropy bonus is intentionally not introduced.

No upstream game modification is required. If future evidence suggests event-driven
acceleration or additional public fields, propose those changes separately and
verify tick/hash equivalence before adopting them. Do not alter the PVZ checkout.

## User-requested kill-source objective

The user subsequently requested a reward for plant kills and a penalty for mower
kills. New configs add +1/N per plant kill and -1/N per mower kill, with configurable
weights and optional normalization. The original terminal and potential terms remain.
This is an intentional change in preferred behavior, outside potential-invariance
guarantees. It is not a reward formula claimed by the paper. Credit uses the killing
damage in public events, so a wounded zombie finished by a mower is penalized.
The game and observation schema remain unchanged. Missing weights retain zero event
reward for old configurations, and changed-reward resumes are rejected.

The first pilot was stopped when this request arrived. Its diagnostics used the
previous reward and must not be pooled with subsequent runs. Placement collected
114,688 decisions and achieved 0/20 best validation wins; saving collected 73,728
decisions and achieved 1/20. Their completed training episodes had zero wins.
Saving episodes never bought sustained attackers and their maximum observed sun
was 50. Planting/digging and savings exploration therefore remain useful failure
questions. These unsuccessful diagnostics justify keeping the new method experimental.

The interrupted results are preserved under `artifacts/paper-pilot-v040`. The
revised-reward pilot uses a separate directory and only the remaining portion of
the authorized 30-minute learning budget. It completed all profiles/seeds. At the
shared 65,536-decision validation budget, baseline/tactical/grouped score 0% for both
seeds; the full method scores 0% for seed 101 and 6.7% for seed 102. Diagnostics still
fail, so the method remains **experimental/inconclusive**. Both full-method runs
end in placement without training episode wins or plant kills. Detailed counts,
timing, and artifacts appear in `validation.md`.
No result establishes human behavioral similarity or held-out performance.

## Adjustments suggested by the diagnostic evidence

Inspecting the revised-reward **final** diagnostic checkpoints on the predetermined
validation seed 100000 reveals a concrete failure. Placement plants a peashooter
at tick 0 and digs it up at tick 10; it repeats at ticks 150/160 and loses. Saving
plants a sunflower at tick 0 and removes it at tick 10, repeating six plant/dig
pairs without a plant kill. These verified recordings and complete action traces
are in `artifacts/pure-rl-diagnostic-traces`. They establish the behavior for these
cases, not a cause for every failure or a general impossibility result.

One mathematically relevant limitation is that **true joint entropy favors action
types with more legal tiles**. Although grouped equal logits initially give normal
wait probability 1/4, maximizing joint entropy alone gives the uniform distribution
over 136 legal concrete actions, returning wait probability to 1/136. In the
placement lesson the corresponding values are 1/2 and 1/46. An independent gradient
control confirms that joint-entropy ascent decreases the initial wait logit. The
implemented probability/entropy calculations are correct; this is an exploration
tradeoff in the chosen objective, not a numerical bug.

The next controlled pure-RL experiments should therefore consider (separately):

- Temporarily exclude digging only in the first teaching lesson, restore it in a
  later stage, and retain all normal-game controls. This may allow early trajectories
  to contain surviving plants and positive kill outcomes. It changes the approved
  lesson design, so it is a documented follow-up rather than a silent change here.
- Compare an explicit type-level entropy or prior regularizer against true joint
  entropy. Keep PPO log probabilities joint and clearly label any changed entropy
  objective; do not describe a type-only bonus as the joint policy entropy.
- Run a longer diagnostic only after fixing the failure mechanism under a separate
  approved compute budget. More input features or more VRAM alone do not address
  immediate removal of plants.

No additional strategy change or game-checkout edit was made for these suggestions.
