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
    Runner --> Logs[Metadata and episode records]
    Env --> Replay[Verified replay]
    Logs --> Report[Statistics and figures]
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
```

Each worker tracks global collection progress in increments of the worker count,
so curriculum changes apply at episode reset without querying workers every action.
Resume restores weights and optimizer, sets curriculum progress before reset, and
starts fresh episodes. It preserves earlier eligible best checkpoints.

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
