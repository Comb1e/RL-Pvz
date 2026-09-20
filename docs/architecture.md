# Current architecture

The research project depends on a pinned, separately installed game. Only the game
owns simulation state. A manifest verifies package version 1.2.1, simulation version
1.0.0, the Git source pin, source hashes, and rules hash before experiments. The
installer stages a verified Git archive in this project's build directory and
installs non-editably from there, without writing to the game checkout.

```mermaid
flowchart LR
    Config[TOML configuration] --> Runner[CLI and experiment runner]
    Runner --> Env[Gymnasium environment]
    Env --> Game[Pinned game engine]
    Game --> Public[Public observation and events]
    Public --> Encode[Versioned spatial or tactical features]
    Game --> Legal[Legal actions]
    Legal --> Mask[406-action or 5-strategy mask]
    Encode --> Policy[PPO or baseline]
    Mask --> Policy
    Policy --> Action[Shared action decoder]
    Action --> Env
    Public --> Reward[Terminal, economy potential and combat rewards]
    Reward --> Learn[PPO update]
    Learn --> Policy
    Runner --> Logs[Metadata, episode records, and progress log]
    Env --> Replay[Verified compact demo with metadata]
    Logs --> Report[Offline HTML and PNG curves]
    Replay --> Viewer[Native viewer with seeking]
    Replay --> Report
    Replay --> Video[Optional native frames to FFmpeg MP4]
    Video --> Report
    Learn --> Shared[One best checkpoint across all difficulties]
    Shared --> Demos[Fixed easy, standard, hard validation demos]
    Demos --> Replay
```

## Ownership and interfaces

| Part | Responsibility |
|---|---|
| Configuration and provenance | Constants, seed splits, source verification, hashes, versions |
| Actions and encoding | Stable public schemas, no private state or future schedule |
| Environment | One game per worker, one action per 10 ticks, explicit episode lifecycle |
| Rewards | Plant-plus-sun potential, terminal handling, shared public-event damage/kill accounting |
| Controllers | Frozen original heuristic, shared public-state facade, strategy proposals |
| Training | PPO updates, curriculum progress, validation-only checkpoint selection |
| Grouped policy | Masked action type and conditional tile; joint PPO probability and entropy |
| Curriculum state | Mastery probes, next-reset stage changes, persisted advancement counters |
| Pilot | Timed diagnostics, alternating profile order, matched-budget comparison |
| Runtime transport | Masks carried with observations, local mask access, reusable CUDA rollout tensors |
| Timings/benchmark | Separate collection and update costs, warmup excluded from measurements, paired data-path comparisons |
| Evaluation | Same cases and timing, independent policy RNG, complete episode records |
| Statistics/reporting | Run/scenario bootstrap, paired comparisons, figures and report |
| Progress | Shared phase names and throttled console/file events; no per-episode printing |
| Visualization | Offline run report, optional metrics, resume segments, checkpoint-identified demos |
| Recording compatibility | Native metadata, legacy sidecar fallback, natural outcome precedence |
| Rendering/video | Thin native-renderer adapter, verified playback, optional MP4 encoding |
| Suite | Restart journal for attempts, fixed settings, training before final testing |

Direct policies see a versioned float32 vector. Spatial v1 has 2,719 values:
plants occupy a 5×9×17 grid, zombies a 5×20×16 grid, projectiles a 5×20×3 grid,
and globals have 54 values. Tactical v2 has 1,140 values: the same plant grid and
globals, three distance regions per lane for zombies/projectiles, 16 card economy
features, and 20 lane summaries. Fixed scales live in the profile configuration.
Integer sums precede scaling, so tuple order does not alter aggregated observations.
IDs and scenario labels are not inputs. The encoding is intentionally partially observed.

Actions are wait 0, 360 plant/tile combinations, and 45 digs. The hybrid has five
strategy choices, each proposing at most one placement. Invalid direct actions
follow the engine contract; unavailable hybrid strategies wait and are logged as
rejected choices. Normal masks use only current legality. Placement/saving lessons
add explicit permitted-plant restrictions; prohibited requests wait, consume time,
and are recorded as rejected. Lessons still allow legal digs on the full board.

```mermaid
flowchart LR
    FlatMask[406 legal flags] --> Types[Available action types]
    FlatMask --> Tiles[Legal tiles for each type]
    Features[Public numeric features] --> Network[Policy MLP]
    Network --> Types
    Network --> Tiles
    Types --> Choice[Choose wait, plant type, or dig]
    Choice --> Location[Choose conditional tile unless waiting]
    Tiles --> Location
    Location --> Index[Original action index]
```

Grouped PPO uses `log P(type) + log P(tile|type)`, with no tile term for waiting.
Joint entropy is type entropy plus type-probability-weighted conditional entropy.
Unavailable types have zero mass. Deterministic evaluation greedily chooses type
then tile. Policy and distribution subclasses retain SB3's collector, PPO loss,
GAE, optimizer, and optimized transport buffers.

```mermaid
stateDiagram-v2
    [*] --> NeedsReset
    NeedsReset --> Running: reset
    Running --> Won: Engine victory
    Running --> Lost: Engine defeat
    Running --> Truncated: External time cutoff
    Won --> Running: reset
    Lost --> Running: reset
    Truncated --> Running: reset
```

The engine may still be running when the adapter is truncated. Vector wrappers
retain the terminal observation for PPO bootstrapping before resetting that worker.
True terminal potential is zero; truncated potential is preserved.

Public `DamageApplied`, `ZombieSpawned`, `ZombieDefeated`, and `MowerActivated`
events pass through one combat-accounting interface. A transient ID join identifies
the killing hit; IDs never enter policy observations. Plant kills earn +1/N,
mower kills −2/N, and earlier nonlethal plant hits earn removed base HP divided
by the type's full starting HP and N. Armor damage and the lethal hit earn no
damage reward. Type comes from the prior public observation or public spawn
event; maximum health comes from active rules. Empty activations cost 0.2,
matched by mower lane and engine tick even within multi-tick decisions.

```mermaid
flowchart LR
    Events[Public events] --> Join[Match damage and deaths]
    Join --> Kills[Plant or mower kill counts]
    Join --> Hits[Nonlethal plant HP damage]
    Join --> Empty[Activation with no same-tick lane kill]
    Rules[Active zombie health rules] --> Hits
    Board[Sun and public plant purchase costs] --> Potential[Economy and progress potential]
    Kills --> Reward[Separate reward parts]
    Hits --> Reward
    Empty --> Reward
    Potential --> Reward
    Reward --> Metrics[Episode records, progress, curves]
```

The potential is `0.5 * defeated/N + 0.5 * (sun + living plant costs)/300`.
Economy is uncapped and independent of plant health; purchases preserve value,
digging and plant deaths remove value. Configurable weights and scales live in
TOML. The shaping difference retains gamma and terminal/truncation handling.
Event rewards explicitly change the task objective; checkpoint selection uses
win rate. Legacy configs without `potential_mode` retain the original capped
sun/sunflower formula; absent event weights mean zero. Changed reward settings
require a fresh run and cannot be pooled under the same research configuration.

## Training and evaluation

The installed game is package 1.2.1 at source pin
`6fd1f54706369915013a49eab5c1790f8c55ab0a`, simulation version 1.0.0.
The native HUD displays defeated/total zombies. The research adapter does not
duplicate the counter or change the numeric observation fields.

The bundled configuration uses CUDA for policy inference during training and PPO
updates. Each spawned game worker simulates on the CPU. `--device cpu` selects CPU
training; CUDA requests fail early when unavailable. Device settings are retained
in experiment metadata and must match when resuming. Standalone evaluation and
demo generation load the shared checkpoint on the CPU by default.

```mermaid
flowchart LR
    Worker[CPU game worker] --> Response[Observation and legal mask together]
    Response --> Cached[Local mask cache]
    Response --> Collect[Stock SB3 rollout collector]
    Cached --> Collect
    Collect --> GAE[Stock returns and advantages]
    GAE --> Copy[One completed rollout copied to CUDA]
    Copy --> Batch[Original NumPy minibatch order]
    Batch --> Update[Stock PPO update on device tensors]
    Update --> Reset[Release cached tensors at next rollout]
```

Transport wrappers retain SB3's spawned-worker implementation. Masks are metadata,
not observation features. On automatic reset, the mask comes from the new episode;
the old terminal observation remains available for time-limit bootstrapping. Mask
queries use a local copy and unknown worker mutations invalidate cached entries.
The adapter reuses the engine's legal-action query until public legality inputs
change: status, occupied tiles, and each card's affordability/recharge availability.
It queries the engine again on changes or reset. Strategy candidates still update
each decision because threats and plant health matter to placement. This bounded
single-state cache applies to training, validation, and demos, with unchanged masks.

The device buffer subclasses retain SB3's GAE calculation, flattening, and NumPy
permutation generator. Only minibatch indexing moves onto the GPU. CPU sampling
uses the original implementation. Loss functions and validation stay unchanged.
Resuming may recreate an empty runtime buffer, preserving policy and optimizer.

Runtime flags live in `[runtime]`, have defaults for older configurations, and are
recorded in full provenance. They do not affect research compatibility. Networks,
precision, worker counts, rollout size, minibatches, rewards, curriculum, and
checkpoint selection are still governed by the same research settings.

```mermaid
flowchart TD
    Start[Read and validate configuration] --> Verify[Verify engine source and rules]
    Verify --> Workers[Create independent spawned workers]
    Workers --> Collect[Collect one PPO rollout]
    Collect --> Update[Optimize policy and value networks]
    Update --> Due{Validation due?}
    Due -->|yes| Eval[Evaluate fixed validation seeds]
    Eval --> Best[Keep highest macro win rate; earliest tie]
    Best --> Budget{Budget reached?}
    Due -->|no| Budget
    Budget -->|no| Collect
    Budget -->|yes| Final[Final validation and final checkpoint]
    Final --> Close[Close simulation workers]
    Close --> Demo[Load one best checkpoint and play fixed validation demos]
    Demo --> VerifyDemo[Verify and save compact demos]
    VerifyDemo --> Requested{Video requested?}
    Requested -->|yes| Export[Native RGB frames to MP4]
    Requested -->|no| Report[Write offline report with demo links]
    Export --> Report
```

Each worker tracks global collection progress in increments of the worker count,
so curriculum changes apply at episode reset without querying workers every action.
Resume restores weights and optimizer, sets curriculum progress before reset, and
starts fresh episodes. It preserves earlier eligible best checkpoints.
Checkpoints with a different engine source pin are rejected before deserializing
model weights. Archived reports and recordings can be read without loading a model;
archived checkpoints cannot generate new gameplay. Comparisons retain distinct
protocol hashes for each engine source pin, even when combat rules match.

Each run owns one policy and optimizer throughout all curriculum stages. One
`best.zip` maximizes the equal-weight easy/standard/hard validation win rate;
earlier checkpoints win ties. Every difficulty demo records the same checkpoint
hash. The independent seeds and ablation conditions in the suite are separate
shared-policy experiments, not models selected by difficulty.

```mermaid
stateDiagram-v2
    [*] --> Placement
    Placement --> Saving: 18/20 placement wins twice
    Saving --> Easy: 18/20 saving wins twice
    Easy --> Standard: 16/20 easy wins twice
    Standard --> Shared: 16/20 easy and 12/20 standard twice
    Shared --> Shared: 20/40/40 normal games
```

Teaching gates run every 16,384 decisions and require minimum stage residency.
Failures reset the consecutive-pass counter. Distribution changes are sent to all
workers for their next resets; active games finish under their original task.
Checkpoint attributes persist stage, entry step, last probe, and pass counters.
Resume restores this state before workers reset. Lesson probes live in separate
files and never choose `best.zip`. Budget exhaustion leaves the actual stage intact.
Baseline profiles retain the fixed progress schedule instead of mastery gates.

The timed pilot first runs two pure-RL diagnostics, then four profiles at two seeds
in alternating order. A deadline check at an update boundary prevents partial PPO
updates from becoming comparison checkpoints. Completed evaluations have separate
learning-profile and game-protocol hashes. Cross-profile reports require the same
engine, game rules, timing, and reward objective; they retain distinct method names.
Only a budget completed by all profiles and both seeds enters the pilot comparison.

```mermaid
stateDiagram-v2
    [*] --> Starting
    Starting --> Collecting
    Collecting --> Updating: Rollout ends
    Updating --> Collecting: Update finishes
    Updating --> Validating: Validation due or final update
    Validating --> Collecting: More decisions remain
    Validating --> Exporting: Checkpoints saved and workers closed
    Exporting --> Complete: Artifacts ready or export error recorded
    Collecting --> Interrupted: User interruption
    Updating --> Failed: Learning error
    Validating --> Failed: Evaluation error
```

The progress reporter shares a wall-clock throttle across training, evaluation,
and export. Routine messages default to every 15 seconds; important events print
immediately. Collection/update phase changes update status without printing every
rollout. Aggregates cover the last 100 completed training games. PPO statistics are
read after updates, at the next rollout start and at training end, including the
last optimized policy. Training ETA excludes validation and report time; export
time is recorded separately. TensorBoard remains controlled by SB3.
Collection/update timing uses rollout boundaries and captures the final update;
validation and report gaps are excluded. Completed phase totals and latest phase
durations enter the aggregate metrics and progress log. The benchmark warms up
one full rollout and reports setup/warmup separately. Optional paired measurement
alternates reference/configured order and compares final policy hashes.

## Result artifacts

```mermaid
flowchart LR
    Metrics[Post-update aggregate JSONL] --> Charts[Training and optimizer curves]
    Validation[Validation JSONL] --> Curves[Per-difficulty and mean win-rate curves]
    Best[Selected checkpoint and hash] --> Cases[First validation seed in each difficulty]
    Cases --> Replay[Compressed .pvzdemo with outcome and provenance]
    Replay --> Viewer[Native viewer with seeking]
    Replay --> Playback[One engine tick per frame]
    Playback --> Frames[Native BoardRenderer and RenderContext]
    Frames --> Encoder[Optional FFmpeg RGB stream]
    Encoder --> MP4[H.264 video and verification metadata]
    Charts --> HTML[Offline index.html]
    Curves --> HTML
    Replay --> HTML
    MP4 --> HTML
```

Report assets use relative links and need no network or server. Charts refresh
after validation and at completion. Compact demos are generated after training by
default; videos require configuration or an explicit `--videos` flag. Raw episode
records remain the research evidence. Missing metrics from old runs are shown as
unavailable. Resume metadata supplies an exact boundary for including ancestor
measurements; each segment retains its own wall time.

Video output uses temporary files and is published only after replay hashes and
encoder completion pass. The adapter records truncation and provenance in native
replay metadata, outside simulation state. The engine can correctly remain `running`.
The shared loader fills missing metadata from legacy sidecars but gives embedded
fields precedence. `Playback.display_outcome` controls labels at the verified end,
and natural outcomes take precedence over annotations. The encoder holds one frame at
a time and stores its error output in a temporary file. The final outcome is held
for two seconds by default. Export status is independent of completed training,
so dependency/encoder failures preserve checkpoints and support regeneration.

`visualize --run` rebuilds reports and missing/changed demo assets without training.
`replay --video` exports an individual recording. Configuration separates logging
and visualization preferences from research compatibility checks, while full
resolved settings remain in run provenance. The pinned game is not modified.

The Gym renderer retains a writable `(600, 1000, 3)` NumPy RGB array. MP4 output
defaults to native 1280×820 dimensions, configurable with two positive even numbers.
Rendering uses only public observations and explicitly provided presentation context.
The game owns drawing, action descriptions, seek controls, and compressed-file I/O;
the research project owns experiment selection, provenance, reports, and FFmpeg.
Compact recording and reports do not require pygame or FFmpeg.

## Formal suite

```mermaid
stateDiagram-v2
    [*] --> Training
    Training --> Evaluating: All 25 runs finish
    Evaluating --> Reporting: All fixed evaluations finish
    Reporting --> Complete: Statistics and figures written
    Training --> Interrupted: User interruption
    Training --> Failed: Error
    Evaluating --> Interrupted: User interruption
    Evaluating --> Failed: Error
    Reporting --> Failed: Error
    Interrupted --> Training: Resume journal
    Failed --> Training: Resume journal
```

Resume walks the journal and skips completed jobs. Every retry uses a new attempt
directory. Formal test cases do not enter learning or checkpoint selection.
Replays contain privileged snapshots for verification, but policy code only receives
encoded public observations and masks. Human-readable reports consume evaluation
records, never reward curves as a substitute for win rates.
