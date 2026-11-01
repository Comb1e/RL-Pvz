# Empty-board Dig collapse — 2026-10-28

Read-only inspection of `runs/compact-stages-101/easy`, using its saved
configuration, cohort summaries, episode records, checkpoint and action journal.
The inspected checkpoint contains 6,816,226 decisions and 32 completed cohorts;
later records can continue to arrive. No formal training or reward changes were
made for this investigation.

## Observed failure

| Completed cohort | Wait proposals | Plant proposals | Dig proposals | Wins / 128 |
|---:|---:|---:|---:|---:|
| 4 | 14,622 | 198,928 | 0 | 55 |
| 9 | 6 | 3,082 | 201,635 | 3 |
| 22 | 5,940 | 199,501 | 23,602 | 30 |
| 23 | 0 | 0 | 202,187 | 6 |
| 32 | 0 | 0 | 205,649 | 1 |

Cohorts 23–32 all contain only rejected Digs. Cohort 32 has zero species/EMA
fitting rows and 118,747 retained tile-invalid rows. Its mean game reward is
−1.451567489, comprising −1.035161239 terminal reward and −0.41640625 mower
penalties. All four physical components are zero.

The checkpoint's 205,649 journal entries all propose Dig, receive `empty_tile`,
and execute wait. Mean species Q is −0.777355 for Dig and −1.233085 for Wait.
Dig exceeds every other branch in every entry, by 0.046875–0.12109375.
Species parameters have 88 Adam steps, while tile parameters have 116: the
species stage stops receiving gradients after cohort 22. Three earlier wait-only
cohorts account for the missing tile updates relative to 128 total passes.

Independent CPU inference on the checkpoint at the empty easy board, seed 1:
Dig species Q = −0.765625; conditional Dig tile values = [−6.6875, −6.65625].
Lowering tile values cannot change the independently selected species branch.
The requested minimum-Q-minus-0.1 rule keeps moving the tile target downward;
its near-0.01 selected MSE does not indicate improved game outcomes.

## Reward settings actually in this run

The physical conversion factor is `10 * 0.01 / 2000 = 0.00005`.

| Event | Fitted reward |
|---|---:|
| 270 effective plant/projectile damage, `basic_zombie_value=50` | +0.0025 |
| 25 ordinary Sunflower sun, both contributions | +0.0025 |
| 25 early Sunflower sun, extra multiplier 4 | +0.0075 |
| Remove a healthy 50-sun plant | −0.005555556 |
| Remove a healthy 100-sun plant | −0.011111111 |
| Activate a mower | −0.1 |
| Terminal loss | −1 to −1.3 |

A mower penalty equals 40 basic-zombie-equivalent damage rewards. These scales
make local physical improvements relatively small, but cannot explain or fix
the missing species supervision after collapse. The simulator does not award
positive digging rewards: a short real CPU sequence confirmed empty Dig returns
zero immediately, placement is asset-neutral, and digging that plant loses its
remaining asset value. Later loss/mower rewards are recorded in complete returns
but replaced by the fitting-only target on invalid rows.

Time shaping additionally favors an earlier loss over a later loss in the same
cohort (−1 versus up to −1.3). This is the configured economics, not proof that
time shaping caused this branch collapse. Prior documentation described an old
damage price and early-sun multiplier; current tables now match the actual TOML.

## Correction and limits

Objective v7 makes Dig unavailable when the pre-action public plant count is
zero. Selection still compares all ten branches, but an empty-board Dig skips
tile inference and fits only its species output to the current species minimum
minus 0.1. On a populated board, an empty selected tile remains tile-invalid.
The simulator continues reporting `empty_tile` and executing ordinary wait.
The reward ledger, coefficients, observations and model architecture are unchanged.

Temporary controls verify CPU/CUDA transition/availability agreement, no tile
forward for empty-board Dig, species-only versus tile-only gradients, no invalid
EMA forward, unchanged complete-return fields, full-board plant availability,
inactive/permission gates, contradictory-evidence rejection, old optimizer/recovery
rejection and compatible old weight loading. This establishes the routing fix;
policy recovery and convergence require a subsequent user-run experiment.
