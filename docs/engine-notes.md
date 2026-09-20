# Notes for the separate PVZ game

Date: 2026-09-20. Inspected engine commit:
`b3cfbd886ab378313a1fdb57ee43a9a1b36a0793` in `E:/Projects/pvz`.

The existing engine is sufficient for the shared-policy training, logging, reports,
and replay videos in research version 0.2.0. This work changes only the research
project. The installed engine remains checked against its pinned source manifest.

## Optional future adjustments

| Observation | Possible future engine improvement | Current research solution |
|---|---|---|
| A replay's final engine status remains `running` when the research wrapper reaches its time limit. | Allow optional non-simulation metadata in replay files, such as external termination reason and policy identifier. Keep it outside simulation hashes and policy observations. | Adjacent replay metadata records the wrapper outcome and shared checkpoint hash. The renderer uses that metadata to display `TRUNCATED`. |
| The full interactive UI creates a display window, while the reusable plant/zombie art can draw onto a surface. | Expose a documented offscreen board/HUD rendering interface if several external consumers need it. | The research package owns its offscreen public-observation renderer and adds the research HUD. |
| Playback verifies state while advancing individual ticks, but standalone raw engine replays do not identify the learner checkpoint. | Consider optional replay provenance fields or a documented sidecar convention for external tools. | Demo manifests record checkpoint, replay, video, and final-state hashes. |

These are optional maintenance suggestions, not prerequisites or requested engine
patches. Any future engine revision should receive a new pin and repeat action,
legality, observation, deterministic playback, and baseline regression checks.
Keep seeds, private snapshots, and future schedules excluded from policy inputs.
