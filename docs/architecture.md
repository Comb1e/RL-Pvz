# Current architecture

The research project depends on a pinned, separately installed game. Only the game
owns simulation state. A manifest verifies the installed engine before experiments.

```mermaid
flowchart LR
    Config[TOML configuration] --> Runner[CLI and experiment runner]
    Runner --> Env[Gymnasium environment]
    Env --> Game[Pinned game engine]
    Game --> Public[Public observation and events]
    Public --> Encode[2719 numeric features]
    Game --> Legal[Legal actions]
    Legal --> Mask[406-action or 5-strategy mask]
    Encode --> Policy[PPO or baseline]
    Mask --> Policy
    Policy --> Action[Shared action decoder]
    Action --> Env
    Public --> Reward[Terminal reward and potential shaping]
    Reward --> Learn[PPO update]
    Learn --> Policy
    Runner --> Logs[Metadata, episode records, and progress log]
    Env --> Replay[Verified replay]
    Logs --> Report[Offline HTML and PNG curves]
    Replay --> Video[Offscreen frames to FFmpeg MP4]
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
| Rewards | Potential from public observations, true terminals versus truncations |
| Controllers | Frozen original heuristic, shared public-state facade, strategy proposals |
| Training | PPO updates, curriculum progress, validation-only checkpoint selection |
| Evaluation | Same cases and timing, independent policy RNG, complete episode records |
| Statistics/reporting | Run/scenario bootstrap, paired comparisons, figures and report |
| Progress | Shared phase names and throttled console/file events; no per-episode printing |
| Visualization | Offline run report, optional metrics, resume segments, checkpoint-identified demos |
| Rendering/video | Public-state HUD, tick-by-tick verified playback, bounded-memory MP4 encoding |
| Suite | Restart journal for attempts, fixed settings, training before final testing |

Direct policies see a flat 2,719-element float32 vector. Plants occupy a 5×9×17
grid, zombies a 5×20×16 grid, projectiles a 5×20×3 grid, and globals have 54 values.
Integer sums precede scaling, so tuple order does not alter aggregated observations.
IDs and scenario labels are not inputs. The encoding is intentionally partially observed.

Actions are wait 0, 360 plant/tile combinations, and 45 digs. The hybrid has five
strategy choices, each proposing at most one placement. Invalid direct actions
follow the engine contract; unavailable hybrid strategies wait and are logged as
rejected choices. Masks use only current legality.

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

## Training and evaluation

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
    Demo --> Export[Verify replays and stream frames to MP4]
    Export --> Report[Write offline HTML report and final export status]
```

Each worker tracks global collection progress in increments of the worker count,
so curriculum changes apply at episode reset without querying workers every action.
Resume restores weights and optimizer, sets curriculum progress before reset, and
starts fresh episodes. It preserves earlier eligible best checkpoints.

Each run owns one policy and optimizer throughout all curriculum stages. One
`best.zip` maximizes the equal-weight easy/standard/hard validation win rate;
earlier checkpoints win ties. Every difficulty demo records the same checkpoint
hash. The independent seeds and ablation conditions in the suite are separate
shared-policy experiments, not models selected by difficulty.

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

## Result artifacts

```mermaid
flowchart LR
    Metrics[Post-update aggregate JSONL] --> Charts[Training and optimizer curves]
    Validation[Validation JSONL] --> Curves[Per-difficulty and mean win-rate curves]
    Best[Selected checkpoint and hash] --> Cases[First validation seed in each difficulty]
    Cases --> Replay[Verified replay JSON and outcome sidecar]
    Replay --> Playback[One engine tick per frame]
    Playback --> Frames[Public board and HUD]
    Frames --> Encoder[FFmpeg stdin RGB stream]
    Encoder --> MP4[H.264 video and verification metadata]
    Charts --> HTML[Offline index.html]
    Curves --> HTML
    MP4 --> HTML
```

Report assets use relative links and need no network or server. Charts refresh
after validation and at completion; videos are generated after training. Raw episode
records remain the research evidence. Missing metrics from old runs are shown as
unavailable. Resume metadata supplies an exact boundary for including ancestor
measurements; each segment retains its own wall time.

Video output uses temporary files and is published only after replay hashes and
encoder completion pass. The adapter's truncation outcome lives in a sidecar;
the engine replay can correctly remain `running`. The encoder holds one frame at
a time and stores its error output in a temporary file. The final outcome is held
for two seconds by default. Export status is independent of completed training,
so dependency/encoder failures preserve checkpoints and support regeneration.

`visualize --run` rebuilds reports and missing/changed demo assets without training.
`replay --video` exports an individual recording. Configuration separates logging
and visualization preferences from research compatibility checks, while full
resolved settings remain in run provenance. The pinned game is not modified.

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
