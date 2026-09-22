# Research design

Current research version: **0.7.1**. This is the specification for learning and
evaluation; [README](../README.md) contains daily commands, [architecture](architecture.md)
describes implementation, and [validation](validation.md) records measured results.
Earlier results remain attached to their original profile, timing, reward and
engine pin. No candidate has demonstrated reliable shared-policy performance yet.

## Objective and evidence

Learn when, which and where to plant using one policy across easy, standard and
hard. Direct-placement candidates use pure RL: no human demonstrations, scripted
warm start or scripted placement during learning. The hybrid is an explicitly
separate baseline. Winning this daytime clone does not establish human-like
behavior or performance in commercial PvZ.

The literature informed these choices; full citations, revisions and inspection
limits are in [references](references.md):

| Evidence | Adopted method | Limit |
|---|---|---|
| Bergdahl et al. (2024), PvZ strategic control | Five-strategy hybrid plus random-strategy control | Reported 57.12% vs 47.95% used separate agents per level; shared agents struggled |
| Dias et al. (2022), user-supplied tower-defense paper | Regional enemy metadata and introductory tasks | Table 4 uses 16/20/65/83 inputs; best “maximum average” 2,180.65 vs human average 18,680 does not establish DQN superiority; reward weights are incomplete |
| AlphaStar and SC2LE | Structured spatial/action heads, diverse tasks and separate argument exploration | Human imitation, privileged critics and league-scale compute do not transfer directly |
| AlphaStar Unplugged | Separate spatial/scalar features; start without memory | Offline human-replay results are not evidence for online PPO; its memory ablation is task-specific |
| Rethinking / Revisiting / Evolutionary Perspective | Audit repetitive behavior, task difficulty and coverage | Revisiting was abstract-only; critiques and perspectives do not prove a PVZ solution |
| Runtime Action Interference | Log proposed/executed behavior and outcomes | Inference/perception study, not a learning result; action throttling was not adopted |
| PPO, masking, potential shaping, GAE, procedural evaluation | Existing algorithms and controlled comparisons | Local coefficients, architecture sizes and priors require local evidence |

Archived diagnosis motivates experimentation: `masked-cuda-101` bought no sustained
attackers over 10,229 games; the later completed `sc2-shared-101` selected 42% easy
and 0% standard/hard validation. Early grouped policies planted and immediately
dug up attackers. Waiting frequency alone is neutral: saving and cooldowns require it.
These observations suggest exploration/credit problems, not a proven single cause.

DQN, offline replay learning, LSTM/attention, screenshots, imitation, UPGO,
population/league training, privileged inputs and event-skipping are not implemented
by these adaptations. A previous suggestion to forbid early digging was not adopted;
the current candidate uses a trainable initialization instead.

## Observation and action contract

The environment has 406 actions: wait `0`, plant `1 + 45*p + 9*r + c`, dig
`361 + 9*r + c`. Plant order is sunflower, peashooter, wall-nut, cherry bomb,
potato mine, snow pea, chomper, repeater. Masks use ordinary legality plus explicit
lesson plant restrictions. Poor but legal moves stay available in normal games.

Legal planting/digging applies immediately; another action can occur at the same
tick after revalidation. Wait/rejection advances one tick at 20 ticks/s. No extra
action cap is imposed. Wins/losses terminate; the 1,200-second external cutoff
truncates, retains the final observation for bootstrapping and counts as unsuccessful
in evaluation. Legacy configurations retain their saved action cadence.

`spatial_v1` has 2,719 values. The optional `tactical_v2` has:

| Block | Values | Information |
|---|---:|---|
| Plant grid | 5 × 9 × 17 = 765 | Type, health, behavior and timer at every tile |
| Zombie regions | 5 × 3 × 16 = 240 | Type counts, HP/armor, position, slowing, pole and state aggregates |
| Projectile regions | 5 × 3 × 3 = 45 | Count, damage and icy count |
| Globals | 54 | Sun, time, waves, public counts, cards and mowers |
| Card economy | 16 | Recharge-ready flag and fractional sun shortfall per card |
| Lane summaries | 20 | Nearest zombie, sustained firepower, nut health and sunflower count |

Three equal regions cover house-to-spawn distance; a boundary belongs to the next
region, and outside coordinates enter the nearest endpoint region. Integer totals
precede scaling; crowd values may exceed one. Sun uses maximum card cost (200),
local counts five entities, global counts 75, cooldowns each card's recharge, and
projectile damage the largest projectile-producing plant damage (20). Shortfall
is `max(cost-sun,0)/max(cost,1)`. Recharge-ready does not imply affordability.
Configurable lane scales are 20 damage/s and nine tiles; no-zombie distance is one.

Entity order, IDs, seeds, scenario names and hidden future schedules cannot change
features when the remaining public state matches. Aggregation loses information;
this is not a fully observable MDP claim.

## Rewards

Let `N = max(1, initial zombie count)`. Current common coefficients are in
`[reward]`; old configurations keep their saved formulas.

| Component | Contribution | Attribution |
|---|---|---|
| Victory / defeat | +1 / −2 | Natural terminal event; other components still apply |
| Plant kill | +1/N | Killing hit from a plant or projectile |
| Mower kill | −2/N | Includes zombies previously wounded by plants |
| Nonlethal plant damage | `damage / (starting base HP × N)` | Actual HP removed; excludes armor and the lethal hit |
| Empty mower activation | −0.2 | No kill by that mower on the activation tick |
| Any mower activation | `−sun_at_activation / 300` | Once per activation; ordered costs/income included |
| Wall-nut bite absorption | `+0.2 × actual damage / full nut HP` | Final bite uses remaining HP; default full nut earns +0.2 |
| Empty ash explosion | −0.2 | No HP or armor damage by that source at that tick |
| Economy/progress shaping | `gamma*Phi(next) − Phi(current)` | Defined below; disabled by the sparse condition |

The engine normally kills the mower's triggering zombie immediately, so empty
activations are normally zero. A later kill or another lane cannot cancel an empty
activation. Mower sun cost is uncapped by the reward: at 0/50/300/600 sun it is
0/−1⁄6/−1/−2. **300 is a scaling value, not a sunlight cap**; the game cap is 9,990.
An untriggered mine, digging or being eaten is not an explosion. Cherry bombs and
potato mines that damage armor count as successful blasts.

```text
Phi(running) = 0.5*defeated/N + 0.5*(sun + sum(living plant purchase costs))/300
Phi(natural terminal) = 0
```

Every living plant keeps its full purchase value regardless of health. Buying
preserves sun-plus-plant value; digging, death or detonation removes it. Economy
is not clipped at 300. Constant positive potential contributes `(gamma−1)*Phi`
per decision. A truncation keeps its final potential for value bootstrapping.

`sc2-plant-rewards-v2` adds:

| Setting | Default | Meaning |
|---|---|---|
| `offensive_plant_weight` | 0.1 | Extra potential per living offensive plant |
| `offensive_plants` | peashooter, snow pea, repeater, chomper | Configurable damaging-plant list |
| `eaten_plant_penalty` | 0.02 | Deduct for `PlantRemoved(reason="eaten")`, except wall-nuts |

The extra potential is logged as `offensive_shaping`. At gamma 0.999 a first
immediate placement adds +0.0999 and removal subtracts 0.1. Their discounted sum
is zero, so planting/digging cannot farm this bonus. Eaten penalties do not apply
to voluntary digging or normal detonation; wall-nuts retain their economy/absorption
terms. Earlier profiles default the new weights to zero.

Discounted potential differences telescope to `−Phi(initial) + gamma^T*Phi(final)`.
This applies to potential shaping of the selected combat objective, not to the
additional event rewards themselves. Damage rewards may favor gradual kills,
absorption may favor longer blocking, and mower sun costs may favor spending before
a breach. Win rate, not shaped return, selects models. `sparse` disables potentials
only; set event weights to zero too for a strictly terminal-only objective.

## Policies and optimization

Grouped policies choose wait/eight plant types/dig, then one of 45 legal tiles.
Non-wait log probability is `log P(type) + log P(tile|type)`; wait has no tile term.
PPO ratios/clipping use this actual joint distribution. Deterministic inference
chooses the most likely type, then its most likely tile; this need not maximize
the flat joint probability.

At 50 sun there are 136 legal concrete actions. Equal grouped type logits give
wait probability 1/4, but maximizing joint entropy favors large tile groups and
returns the optimum to 1/136. The independent wait-logit gradient at equal logits
is `−3*log(45)/16`. SC2 profiles therefore use:

```text
exploration bonus = 0.01*H(type)
                  + 0.001*mean_available[H(tile|type)/log(max(2, legal_tile_count))]
```

The tile term is not weighted by type probability; it is zero with no available
non-wait type. True joint entropy, type/conditional entropy and this bonus are
logged separately. The same exploration interface serves CPU and CUDA.

The spatial policy uses shared 64-unit lane/global MLPs, two 64-channel 3×3
convolutions on the 5×9 plant grid, nine 1×1 tile maps and separate 256×256 type/value
heads after mean/max pooling. See the [network diagram](architecture.md#policy).

| Profile | Actor reduction | GAE lambda | Minibatch | Gamma | Initial dig bias |
|---|---|---:|---:|---:|---:|
| SC2 inspired | All decisions | 0.98 | 256 | 0.999 | Normal initialization |
| SC2 long horizon | All decisions | 0.98 | 256 | 0.9999 | Normal initialization |
| SC2 efficient | Choice states | 0.999 | 1,024 | 0.999 | Normal initialization |
| SC2 plant rewards v2 | Choice states | 0.999 | 1,024 | 0.999 | −6 |

All keep learning rate 3e−4, clipping 0.2, four epochs and 128×128 rollouts.
Choice-state normalization/loss/entropy/KL use states with more than one legal
action; all transitions train the critic and GAE. Empty/singleton reductions have
finite controls. This changes optimization weighting and requires masking.

`initial_dig_logit=-6` is a local exploration prior, applied only to fresh grouped
models. With equal wait/dig logits it reduces initial dig probability to about
0.00247. It remains nonzero and trainable; reload never reapplies the bias. It is
neither a scripted retention rule nor a method claimed by AlphaStar.

Gamma discounts decisions, including zero-time operations. At one wait per tick,
half-life is about 34.6 seconds for 0.999 and 346.6 for 0.9999. Shaping, GAE and
timeout bootstrapping use the same gamma; extra immediate actions shorten the
simulated-time horizon. The longer horizon remains a separate experiment.

## Curriculum and checkpoint selection

Placement: random lane, three basics at ticks 1/81/161, 200 sun, no mowers,
peashooter/wait/legal digging. Saving: 50 sun, three basics at 1200/1280/1360,
sunflower and peashooter plus wait/dig, no mowers. Both retain the full board.
Independent shooting controls win and wait-only controls lose; those trajectories
are tests, never training data.

| Stage | Episode distribution | Required wins per 20-case probe |
|---|---|---|
| Placement | Placement lesson | Placement ≥18 |
| Saving | 80% saving, 20% placement | Saving ≥18 |
| Easy | 80% easy, 10% each lesson | Easy ≥16 |
| Standard introduction | 45% easy, 45% standard, 10% saving | Easy ≥16 and standard ≥12 |
| Shared | 20% easy, 40% standard, 40% hard | Continue to budget |

SC2 profiles probe every 100 completed training games, require two consecutive
passes and 100 completions from games started in that stage. A failed probe resets
the streak. Transitions affect future resets only; policy and optimizer persist.
Budget exhaustion never forces promotion. Older saved residency/schedules remain
loadable. Fixed-schedule controls use easy for 10% of the budget, equal easy/standard
for 30%, then 20/40/40 for 60%; mixed conditions use the final mixture throughout.

Normal validation uses all plants, every 1,000 completed games by default.
After an update crosses several thresholds, one evaluation records the actual
count and nominal milestone. Matching probes reuse same-weight/case results.
The equal-weight mean over easy/standard/hard chooses one `best.zip`; earlier ties
win. Partial evaluations cannot select it. Final validation avoids duplicate weights.

## Evaluation protocol and commands

| Split | Scenario seeds | Use |
|---|---|---|
| Development | 0–9, plus regression seed 42 | Debugging and known controls |
| Training | 1,000–99,999 | Sampled training scenarios |
| Validation | 100,000–100,049 | Selection and development comparisons |
| Final test | 200,000–200,299 | Frozen-policy familiar-distribution evaluation |
| Changed scenarios | 300,000–300,099 per family | Standard/hard generalization |

Changed families are `redistributed` (shuffle later-wave roster, retaining opening
three waves, types and wave sizes), `faster` (25-to-20-second spacing with matched
jitter), and `concentrated` (three seeded lanes with unchanged spawns/types).
Keep every case. Standard and hard share openings, so their presets alone do not
test strong structural generalization.

Non-learning baselines: `wait`, `random_legal`, frozen `heuristic`, and
`random_strategy`. The hybrid chooses wait/economy/attack/defense/emergency;
deterministic scripts produce at most one legal action. Compare against random
selection of those same strategies. Use CPU for hybrid. Conditions are masked,
unmasked, sparse, mixed and hybrid; the unmasked comparison adds no invalid penalty.

Run these only after freezing settings; checkpoint evaluation restores its config:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl evaluate --checkpoint runs\MODEL\best.zip --split test --record --output runs\MODEL-test
.\.venv\Scripts\python.exe -m pvz_rl evaluate --baseline heuristic --split test --record --output runs\heuristic-test
.\.venv\Scripts\python.exe -m pvz_rl evaluate --checkpoint runs\MODEL\best.zip --split ood --family redistributed --record --output runs\MODEL-redistributed
.\.venv\Scripts\python.exe -m pvz_rl report --episodes runs\MODEL-test\episodes.jsonl runs\heuristic-test\episodes.jsonl --curves runs\MODEL\learning-curve.jsonl --output artifacts\comparison
```

Repeat OOD for `faster`/`concentrated`. Evaluation records the first ten wins,
losses and truncations per difficulty in seed order, with hashes. Targets are
95% easy, 90% standard and 75% hard, not achieved guarantees. Report all learner
seeds, per-difficulty outcomes, resource use and 95% bootstrap intervals over
learners and shared scenarios. Preserve pairing; a positive lower confidence bound
is required to claim improvement over the heuristic. Do not pool profiles, engine
pins, changed reward protocols or repeated checkpoints under one method name.

Explicit development commands:

```powershell
.\.venv\Scripts\python.exe tools\check_sc2_learning.py --output artifacts\sc2-check
.\.venv\Scripts\python.exe -m pvz_rl pilot --output runs\paper-pilot --minutes 30
.\.venv\Scripts\python.exe -m pvz_rl compare-sc2 --diagnostics-only --output runs\sc2-diagnostics
.\.venv\Scripts\python.exe -m pvz_rl compare-sc2 --max-minutes 120 --games 10000 --output runs\sc2-comparison
.\.venv\Scripts\python.exe -m pvz_rl compare-sc2 --report-only --output runs\sc2-comparison
```

`check_sc2_learning.py` defaults to four five-minute baseline/reward checks, seeds
101/102 with reversed order; combined allowances cannot exceed 30 minutes. `pilot`
retains its 30-minute four-profile scope, with game-based matched budgets and
explicit missing comparisons. `compare-sc2` runs diagnostics then cumulative
A: flat, B: tactical/grouped, C: balanced exploration, D: mastery, E: spatial,
F: longer gamma, reversing A–F for the second learner seed. Its two-hour cap is
**per run**: the matrix can take roughly 24 hours plus diagnostics. It does not
automatically include the later efficient/plant-reward profiles.

Recommend a candidate only after diagnostics pass and normal-game macro validation
improves over baseline for both pilot seeds under the common resource limits.
Show actual games/decisions/time and incomplete comparisons; do not interpolate
scores or reinterpret a partial evaluation as zero. Two seeds are development
screening, not strong statistical evidence. Final-test seeds stay out of pilots.

The separate formal suite runs five conditions × five learner seeds and then
held-out/OOD evaluation and reporting. It is long-running and user initiated:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl suite --games 10000 --max-minutes 120 --output runs\formal
# Repeat original settings to recover the journal:
.\.venv\Scripts\python.exe -m pvz_rl suite --games 10000 --max-minutes 120 --output runs\formal --resume
```

Completed jobs remain; attempts use new directories. `--max-minutes` caps each
training run, not the whole suite's later evaluations. The suite's CPU hybrid
uses at most eight workers under the promoted GPU profile, recorded separately.
No formal matrix or held-out study was launched by implementation checks.
