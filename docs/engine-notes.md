# Notes for the separate PVZ game

Date: 2026-09-20. Adopted package 1.2.0, commit
`a47056d8141ec635d3ff3f4d5561d6a75cfca2cc`, from `E:/Projects/pvz`.

The game now provides the offscreen rendering and replay provenance interfaces
previously suggested here. Research version 0.3.0 uses these native APIs and the
compact demo format. No changes to the game checkout are needed or made.

| Capability supplied by the game | Research integration |
|---|---|
| Recorder metadata and verified display outcomes | Embed policy/checkpoint identity and external cutoff reasons; retain legacy sidecars as read-only fallbacks. |
| BoardRenderer, RenderContext, and packed RGBFrame | Reuse the game's board/HUD for Gym rendering and optional MP4 export. |
| Compressed .pvzdemo files | Default to compact timed-operation recordings; retain JSON/gzip reader compatibility. |
| Playback seeking and operation descriptions | Use the native viewer, speed controls, inspection, and action feedback. |
| Separate package and simulation versions | Pin package 1.2.0 while preserving simulation compatibility identifier 1.0.0. |

No additional engine adjustment is currently required. Rendering still depends on
pygame, and video encoding belongs to the external research package. Metadata is
caller-supplied, so the research manifest hashes whole recordings in addition to
checking simulation hashes. Seeds, private snapshots, and future schedules remain
excluded from policy inputs.

Any future engine revision should receive a new source pin and repeat action,
legality, observation, deterministic playback, and known baseline checks before use.
Keep experimental results separated by source pin even if combat rules are identical.
