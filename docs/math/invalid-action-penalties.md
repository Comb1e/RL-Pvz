# Invalid-action execution and selective fitting

The controller compares all ten species outputs. Every active plant and dig
branch exposes all 45 tiles when available. After species selection, unavailable plants,
disabled Dig and Dig on a board with no plants skip tile inference.
Usable plants and enabled Dig on a populated board score every tile, then the simulator
validates occupancy. A full board never makes an affordable ready plant branch
unavailable. An empty board makes Dig branch-invalid, determined from the
pre-action public plant count before any conditional tile forward.

Every rejected proposal executes an ordinary wait for the configured decision
interval. The original
proposal, actual reason and executed action `0` are retained separately.
Global terminal/asset and action rewards remain those of that
interval. `invalid_plant_penalty` and `empty_dig_penalty` are zero audit fields;
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
| Unavailable/cooldown/unaffordable/restricted plant | Proposed branch to species minimum Q + offset | None | None |
| Disabled Dig or Dig on an empty board | Proposed dig branch to species minimum Q + offset | None | None |
| Occupied plant tile with usable branch | None | Proposed tile to conditional tile minimum Q + offset | None |
| Empty Dig tile on a populated board | None | Proposed tile to conditional tile minimum Q + offset | None |

The simulator retains its `empty_tile` reason for an empty-board Dig and
executes wait normally. Pre-action availability distinguishes this species-invalid
case from a tile-invalid empty Dig on a populated board. It uses the complete
public plant count, independent of encoded-entity truncation, and respects Dig
permission and running/active state. Only the species stage is forwarded/fitted
for empty-board Dig rows; there is no tile or EMA forward. The branch is still
selectable, so it receives its existing minimum-Q-minus-offset supervision.

`training.objective.invalid_action_q_offset=-0.1` supplies a finite non-positive
fitting-only offset. At the current row's observation, let `m` be the minimum
of all ten online species outputs for a branch-invalid row, or all 45 online
tile outputs conditioned on the proposed branch for a tile-invalid row. The
target is `y_invalid = stopgrad(m) + offset = stopgrad(m) - 0.1`.
Minima -1.2, 0, and +0.8 give targets -1.3, -0.1, and +0.7 regardless of which
invalid output was selected. All outputs participate, including invalid ones;
there is no legality mask, EMA estimate, return baseline, or cross-head minimum.
Accepted actions and genuine waits still target the complete return `G_t`.

The shared target helper serves autonomous/demo regression and prefit errors.
Fitting recomputes the detached minimum from the existing forward on each pass.
No gradient follows the minimum into another output; only the selected output
has direct regression supervision. Collection carries the corresponding head
minimum through its existing handoff into trajectory metadata for prefit errors.
The helper does not mutate stored returns or components. Complete returns include actual
interval rewards, and earlier decisions never receive the fitting offset as a
reward. Invalid proposals do not train the wait output just because execution
was wait. No species/tile/EMA forwards are added.

The old `invalid_action_target` setting is rejected rather than reinterpreted.
Objective/trajectory/recovery v7 prevents resuming older empty-board routing or fixed-target optimizer states;
compatible model weights remain valid for fresh initialization. Reward/annotation
identities and the demonstration archive format are unchanged. This is a moving
target: if the selected output is already the minimum, its target is another
0.1 lower. No floor or clamp is introduced.

Wait/plant/dig groups and their accepted/rejected strata retain whole-cohort
balancing. EMA consistency uses accepted plant/dig denominators only. CPU row
plans and route-separated width buckets skip irrelevant online stages and EMA.
Verified demonstrations use the same classifier after reconstructing pre-action
legality; historical penalties are verified as recorded facts, then repriced to
zero in memory without modifying recording files.

Both autonomous and demonstration fitting use the shared first-of-window
compactor. Genuine wait proposals and identical invalid proposals retain rows
1, 1+k, 1+2k, etc. for `k=training.objective.invalid_fit_group_size`; the
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
