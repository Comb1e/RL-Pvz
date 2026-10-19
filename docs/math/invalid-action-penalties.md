# Invalid-action execution and selective fitting

The controller still compares all ten species outputs. Plant proposals consider
empty tiles; digging considers all 45 tiles. Sun and cooldown do not hide a
branch. A full-board plant uses tile zero so rejection remains observable.

Every rejected proposal executes exactly one ordinary wait tick. The original
proposal, actual reason and executed action `0` are retained separately.
Global terminal/asset and action rewards remain those of that
tick. `invalid_plant_penalty` and `empty_dig_penalty` are zero audit fields;
there are no active rejection-price settings.

The shared classifier uses full **pre-action** public legality and actual outcome.
Policy geometry alone cannot distinguish an occupied tile from an unusable card.
Branch impossibility takes precedence, including cooldown plus occupancy.

| Proposal/outcome | Species supervision | Tile supervision | EMA tiles |
|---|---|---|---|
| Ordinary wait | Complete return | None | None |
| Accepted plant/dig | Detached consistency | Complete return at proposed tile | Yes |
| Unavailable/cooldown/unaffordable/restricted/full-board plant | Proposed branch to -1 | None | None |
| Occupied plant tile with usable branch | None | Proposed tile to -1 | None |
| Empty dig with plants elsewhere | None | Proposed tile to -1 | None |
| Dig with no plants | Dig branch to -1 | None | None |

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

The saved-weight contract and action dimensions are unchanged. The objective,
trajectory and recovery versions prevent resuming old optimizer/reward semantics.
