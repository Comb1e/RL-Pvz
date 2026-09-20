# Notes for the separate PVZ game

Date: 2026-09-20. Adopted package 1.2.1, commit
`6fd1f54706369915013a49eab5c1790f8c55ab0a`, from `E:/Projects/pvz`.

The game now provides the offscreen rendering and replay provenance interfaces
previously suggested here. Research version 0.4.0 uses these native APIs and the
compact demo format. No changes to the game checkout are needed or made.

| Capability supplied by the game | Research integration |
|---|---|
| Recorder metadata and verified display outcomes | Embed policy/checkpoint identity and external cutoff reasons; retain legacy sidecars as read-only fallbacks. |
| BoardRenderer, RenderContext, and packed RGBFrame | Reuse the game's board/HUD for Gym rendering and optional MP4 export. |
| Compressed .pvzdemo files | Default to compact timed-operation recordings; retain JSON/gzip reader compatibility. |
| Playback seeking and operation descriptions | Use the native viewer, speed controls, inspection, and action feedback. |
| Separate package and simulation versions | Pin package 1.2.1 while preserving simulation compatibility identifier 1.0.0. |
| Defeated/total zombie HUD | Reused by research replay frames and the viewer; the engine's public numeric fields remain unchanged. |

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

## Kill attribution and introductory tasks

Version 0.4.0 uses existing public `LevelSpec` and `Spawn` for teaching scenarios.
No task-specific combat changes are needed. Public `DamageApplied` source IDs and
`ZombieDefeated` events support plant/projectile versus mower killing-blow credit.
The research wrapper joins events transiently; entity IDs are not policy inputs.

An optional future API improvement would be an explicit `cause`/`source_kind` on
`ZombieDefeated`. That would make attribution self-contained instead of depending
on the pinned convention that negative damage-source IDs identify mowers. This is
only a suggestion. Current attribution is verified against projectile, explosion,
mine, chomper, mixed-source, and mower controls, and the game checkout is unchanged.

Version 0.4.1 also uses `DamageApplied.health_damage` for nonlethal reward, public
spawn/type information plus active rule health for normalization, and public card
costs for plant value. Existing events are sufficient. Mower activation immediately
kills its trigger in this version, so the requested empty-activation penalty is
normally inactive; it is tested with independent event fixtures. No upstream change
is needed to implement these reward rules.
