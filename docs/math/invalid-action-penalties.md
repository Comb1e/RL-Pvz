# Invalid-action execution and selective fitting

The controller compares all ten species outputs. Every active plant and dig
branch exposes all 45 tiles. After species selection, unavailable plants and
disabled dig skip tile inference. Usable plants and enabled dig score every tile, then the simulator
validates occupancy. A full board never makes an affordable ready plant branch
unavailable, and an empty board never makes enabled dig branch-invalid.

Every rejected proposal executes exactly one ordinary wait tick. The original
proposal, actual reason and executed action `0` are retained separately.
Global terminal/asset and action rewards remain those of that
tick. `invalid_plant_penalty` and `empty_dig_penalty` are zero audit fields;
there are no active rejection-price settings.

The shared classifier requires explicit **pre-action** branch availability and
the actual outcome; legal masks only check accepted placements. It rejects
contradictory availability/acceptance/reasons instead of silently changing them.
Geometry cannot distinguish a full board from an unusable card. Inactive scheduler
slots are excluded from classification.

After malformed-request, game-state and range guards, the simulator checks the
selected branch's task permission, cooldown, sunlight, then occupancy. Thus an
occupied tile with insufficient sunlight reports `insufficient_sun`; simultaneous
cooldown reports `card_recharging`. Disabled digging reports `task_restriction`
before empty-tile validation. A task's other disabled branches do not affect the
selected allowed branch's reason. Rejection events and action results agree.

| Proposal/outcome | Species supervision | Tile supervision | EMA tiles |
|---|---|---|---|
| Ordinary wait | Complete return | None | None |
| Accepted plant/dig | Detached consistency | Complete return at proposed tile | Yes |
| Unavailable/cooldown/unaffordable/restricted plant | Proposed branch to -1 | None | None |
| Disabled dig | Proposed dig branch to -1 | None | None |
| Occupied plant tile with usable branch | None | Proposed tile to -1 | None |
| Empty dig, including an empty board | None | Proposed tile to -1 | None |

`training.objective.invalid_action_target=-1.0` supplies the squared-error target.
It is fitting-only: chronological complete returns still include actual tick
rewards, and earlier decisions never receive an artificial -1 reward. Invalid
proposals do not train the wait output just because their execution was wait.

Wait/plant/dig groups and their accepted/rejected strata retain whole-cohort
balancing. EMA consistency uses accepted plant/dig denominators only. CPU row
plans and route-separated width buckets skip irrelevant online stages and EMA.
Verified demonstrations use the same classifier after reconstructing pre-action
legality; historical penalties are verified as recorded facts, then repriced to
zero in memory without modifying recording files.

Both autonomous and demonstration fitting use the shared first-of-window
compactor. Genuine wait proposals and identical invalid proposals retain rows
1, 11, 21, etc. under `training.objective.invalid_fit_group_size=10`; the
historical setting name now controls both. Invalid proposals executed as wait
retain their original route/action/reason and never merge with genuine waits.
Accepted non-wait rows are preserved. Adjacency is per environment, independent
of interleaving or observation changes; episodes, missing decisions, inactive
gaps, resets and explicit run breaks split runs. A terminal row closes a run
after itself. Chunk/restart carry preserves the same representatives.

Returns, causal credits and raw metrics still use every decision. Fitting counts
and whole-cohort coefficients use representatives without frequency weights,
intentionally reducing long-wait weighting. Reports separate wait/invalid raw
and retained rows, runs, windows and reduction ratios.

The saved-weight contract and action dimensions are unchanged. The objective,
trajectory and recovery versions prevent resuming old optimizer/reward semantics.
The engine 1.7.1 source pin additionally rejects previous simulator recoveries and
demonstration archives before simulation/replay. No historical-reason migration
is provided. The previous exact pin remains allowlisted for weight-only loading
and matching annotation resume, preserving original source provenance.
