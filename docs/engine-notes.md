# Notes for the separate PVZ game

Date: 2026-09-20. Adopted package 1.2.1, commit
`6fd1f54706369915013a49eab5c1790f8c55ab0a`, from `E:/Projects/pvz`.

The game now provides the offscreen rendering and replay provenance interfaces
previously suggested here. Research version 0.3.2 uses these native APIs and the
compact demo format. No changes to the game checkout are needed or made.

| Capability supplied by the game | Research integration |
|---|---|
| Recorder metadata and verified display outcomes | Embed policy/checkpoint identity and external cutoff reasons; retain legacy sidecars as read-only fallbacks. |
| BoardRenderer, RenderContext, and packed RGBFrame | Reuse the game's board/HUD for Gym rendering and optional MP4 export. |
| Compressed .pvzdemo files | Default to compact timed-operation recordings; retain JSON/gzip reader compatibility. |
| Playback seeking and operation descriptions | Use the native viewer, speed controls, inspection, and action feedback. |
| Separate package and simulation versions | Pin package 1.2.1 while preserving simulation compatibility identifier 1.0.0. |
| Defeated/total zombie HUD | Reused automatically by research replay frames and the native viewer; numeric policy inputs stay unchanged. |

No additional engine adjustment is currently required. Rendering still depends on
pygame, and video encoding belongs to the external research package. Metadata is
caller-supplied, so the research manifest hashes whole recordings in addition to
checking simulation hashes. Seeds, private snapshots, and future schedules remain
excluded from policy inputs.

Any future engine revision should receive a new source pin and repeat action,
legality, observation, deterministic playback, and known baseline checks before use.
Keep experimental results separated by source pin even if combat rules are identical.

## Optional future speed improvements upstream

The current engine constructs candidate action objects and runs the validator for
each `legal_actions()` query. Research 0.3.2 caches that authoritative result while
status, occupied tiles, affordability, and cooldown availability are unchanged.
The cache is bounded to one state and queries again after relevant changes.
It does not require an engine modification or use private state.

If future profiling still identifies legality as a bottleneck, an upstream cached
query or public numeric mask could avoid repeated object construction for every
consumer. It should preserve the existing action order, validate against every
action at exact sun/cooldown/death/reset boundaries, and keep returned data detached.
This is a suggestion only; no game files were edited. Larger policies, increased
worker counts, mixed precision, and changed PPO batches are separate experiments,
not part of this runtime optimization.
