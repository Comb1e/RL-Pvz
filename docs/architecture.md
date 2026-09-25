# Architecture

One CUDA MaskablePPO learner controls one shared playing policy. The separately
pinned game package owns combat rules. Its Python simulator verifies GPU traces;
training and optimization run on CUDA. Model inputs contain public information only.

```mermaid
flowchart LR
    Config[Single resolved TOML] --> Scenarios[Seeded CPU scenario queue]
    Scenarios --> Sim[100 Hz ordered CUDA simulator]
    Sim --> Public[281 public values and legal masks]
    Public --> Memory[Bounded causal public history]
    Memory --> Actor[Independent actor Transformer]
    Actor --> Kind[Wait / dig / plant]
    Kind --> Action[Conditional species and tile]
    Action --> Transport[One integer command on CUDA]
    Transport --> Sim
    Sim --> Reward[Net realized value and elapsed ticks]
    Memory --> Buffer[CUDA token archive and rollout sequences]
    Reward --> Buffer
    Buffer --> Window[Two-slot periodic on-policy window]
    Window --> Critic[Independent critic Transformer]
    Critic --> GAE[Timeout bootstrap and duration-aware GAE]
    GAE --> PPO[Contiguous chunks with public-history burn-in]
    PPO --> Actor
    PPO --> Critic
    PPO --> Schedule[Completed-game curriculum and validation]
    Schedule --> Save[Checkpoints and run records]
    Save --> Replay[GPU action traces verified by CPU]
    Hardware[Background CPU / GPU sampler] --> Samples[Flushed hardware JSONL]
    Schedule --> Refresh[Probe / validation completion]
    Samples --> Output
    Refresh --> Output
    Replay --> Output[Compact demos and offline HTML]
```

## Game and observations

Package 1.4.0 / simulation 1.1.0 advances 100 ticks/second. Durations originate in
seconds; integer positions, remainders and processing order are shared by CPU and
CUDA. Accepted plant/dig operations take zero ticks; waits/rejections take one.
Scenario names, seeds, entity IDs, private schedules and snapshots never enter a
policy token. Snapshots and hashes remain available only for verification.

The `event_v6` observation has 281 values: 135 plant values (type, health, behavior
per tile), 135 regional zombie values and 11 globals.
Each of five lanes has three regions. Each region contains five zombie-type counts,
total health, total armor, nearest zombie distance and nearest unused-pole distance.
Both distances use `(x - house_x) / (spawn_x - house_x)`; absence is -1, distinct
from a carrier at either endpoint. Counts and asset totals are scaled but not clipped.
Zombie behavior labels and pole counts are absent. `has_pole` supplies the unused-pole
distance; it becomes false when vaulting begins. The schema defines feature offsets
once for CPU encoding, CUDA constants and memory admission. Globals contain sun,
elapsed time, current/total waves, initial/defeated zombie counts and five mower-spent flags.
Projectiles and spawned counts are absent. Ready and moving mowers are indistinguishable
from mower input alone; only spent differs. The simulator and accounting retain full state.
No plant, zombie or card countdown is encoded. Legal masks still reveal immediate
action legality. Plant/state category embeddings have zero padding for empty tiles.
During gradient calculation, small one-hot matrix products accumulate their gradients without repeated
index sorting; inference uses ordinary lookups. The learned tables and optimizer semantics
are identical. Previous-action embeddings keep the ordinary implementation.

## Actions and legality

One shared 128×128 actor head feeds three action-kind logits and eight conditional
plant-species logits. Nine board maps supply the planting and digging positions.
Wait has no arguments; dig selects a tile; plant selects a legal species and then
its tile. The three heads are produced in one forward pass. Masks exclude unavailable
branches. Harmless dummy conditionals keep unavailable branches finite but contribute
no joint mass or action-likelihood gradient. Deterministic evaluation selects each
learned head greedily, without injected exploration.

The existing simulator command remains one integer per game, with a centralized
kind/species/tile codec. A 406-entry action-history embedding remembers the complete
executed command. This compact transport does not require a 406-way classifier.
The game package and replay API are unchanged.

Final action heads initialize to zero weights, with wait bias log(1.2) and dig bias
−12. The plant-kind logit includes log(number of available species): each affordable
species therefore has equal initial branch weight, independent of tile count.
Initial tiles are uniform. Injected exploration mixes complete actions with a prior
of wait weight 1.2, one weight per species and uniform legal tiles; it excludes dig.
Factoring this joint mixture preserves collection, log probabilities, entropy and KL.
All heads remain trainable. See [probability controls](math/saving-and-actions.md).

## Episode memory

Both encoders consume the same deterministic public history, with independent
learned representations and Adam states. A bank contains eight recent decision
tokens, 32 retained event tokens, and eight compressed summaries. Tokens contain
the observation, previous accepted action, reset marker and public tick. Rejected
actions have wait as their executed-action marker. No activations cross game boundaries.

Plant health/behavior changes, zombie counts/health/armor,
sun/wave changes, mower-spent flags and legal-mask changes admit events. A region gaining
its first or losing its last unused pole also admits an event, determined from the
distance sentinel alone. Changes to the clock or either nearest distance within a
region alone do not admit events. Removed projectiles, spawned counts, mower
positions/ready/moving states and zombie behaviors never supply hidden event flags.
Every decision still sees the current public state.
Local eviction sends events to a FIFO, and quiet tokens
or older events to summaries. A summary keeps the latest public state, earliest
represented tick and count; it never averages category identifiers. Summary stride
and all capacities are configurable. Finite history can lose timing information.

The current query attends to valid retained tokens at or before its own time.
Relative elapsed time, summary span and count describe temporal context; no hidden
entity countdown is reconstructed from engine fields. Two gated pre-normalized
attention blocks default to width 128, four heads and feed-forward width 256.
History is re-encoded using current weights, avoiding stale learned key/value caches.
This is a project-specific bounded Transformer, not a reproduction of GTrXL.

## Learning data and transitions

Each rollout contains 128 environments × 128 transitions (16,384 total).
At a synchronization boundary, the collector receives frozen actor and critic copies
and a policy-version hash. It fills two bounded rollout slots with that snapshot; the
learner optimizes the first ready slot while the collector fills the second. The next
snapshot is not created until both slots are drained, so all data in one window has
one behavior-policy version. No V-trace, replay, policy-lag correction or asynchronous
policy update is used.

Actor-only inference supplies actions and true mixed kind/species/tile log probabilities. The buffer archives each raw token once
and stores compact context references. Terminal contexts are captured before resets;
truncated games bootstrap from these, natural outcomes do not. The unchanged critic
evaluates stored causal contexts in batches before duration-based GAE.

Optimization shuffles contiguous per-environment chunks (default 16 transitions),
never the transitions within them. CUDA index templates preserve environment-major
ordering without rebuilding full Python index lists for every chunk. At each chunk
boundary it restores retained
public history and replays up to eight prefix transitions without loss. The prefix
reconstructs event admission/compression, not learned hidden state. Both encoders
then re-encode this history; the remaining chunk contexts use the exact archive.
Deterministic boundary banks are reconstructed once per rollout and reused across
epochs and shuffled environment groups. Learned features are never cached across updates.
No old rollout is reused after optimization. Slot ownership is transferred with CUDA
events, and allocations are reused only after the learner stream finishes. The
ready queue defaults to one entry (configurable up to two); the free queue owns
the two reusable slots. The collector blocks when the ready queue is full. All
tensors stay on CUDA apart from batched diagnostics and completed-game records.

```mermaid
stateDiagram-v2
    [*] --> Boundary
    Boundary --> Collecting: freeze weights and exploration; start window
    Collecting --> Overlapping: slot 1 ready; learn slot 1 and collect slot 2
    Overlapping --> Draining: learn slot 2; collector idle
    Collecting --> Draining: final one-slot decision budget
    Draining --> Boundary: both updates complete; commit metrics and release slots
    Boundary --> Probe: scheduled mastery/validation
    Probe --> Boundary: save progress; stage affects future resets
    Boundary --> Saved: budget, mastery or deferred Ctrl+C
    Collecting --> Failed: collector error
    Overlapping --> Failed: collector or learner error
    Draining --> Failed: learner error
    Failed --> [*]: drain worker; reject partial checkpoint
    Saved --> [*]
```

A game ending inside a slot saves its terminal context and resets only that game's
history; unfinished games retain raw event memory across slots and windows. Ctrl+C
is deferred through collection, both updates and metric commit, then saved at the
empty boundary. Unexpected failures drain the worker and preserve previously saved
checkpoints; incomplete-window weights are never serialized. Resume starts new games
and an empty window, restoring both optimizers, schedules and policy-version count.

Independent clipping and optimizers isolate policy and value learning. Rollout-wide
advantage normalization uses fixed statistics for all minibatches. Frozen collector
logits and epsilon are copied into each slot so the learner can calculate exact
hierarchical KL without reading the collector's mutable distribution.

```mermaid
stateDiagram-v2
    [*] --> Active: snapshot actor and Adam at window start
    Active --> Stopped: sampled KL stop
    Active --> Rejected: exact KL exceeds target or is non-finite
    Stopped --> Rejected: exact check fails on either slot
    Rejected --> Drained: restore actor and Adam; finish critic and collection
    Stopped --> Drained: finish critic and collection
    Active --> Drained: both slots pass exact checks
    Drained --> [*]: commit counters and retain or reject actor steps
```

Checks average only genuine-choice observations and test each available slot;
new second-slot observations are checked before more actor updates. Critic state,
RNG progress and environment history never roll back. No automatic retries occur.
Snapshot construction preserves CPU/CUDA RNG state. Warm-up and formal injected
noise are separate: formal exploration decays to floors by stage-resident games,
resets at a stage boundary, and stays fixed throughout each window. Evaluation
temporarily sets injected epsilon to zero and uses greedy actor inference with
separate episode banks.

The CUDA learner uses a tensor-only masked categorical fast path and inference-mode
frozen evaluation. Fixed-shape actor and critic feature paths may be compiled with
`torch.compile`; wrappers stay outside the module tree so checkpoint keys and
optimizer ownership remain unchanged. Compiler failures fall back to eager execution
and are recorded in window metrics.
An absent Triton CUDA backend is detected before tracing begins, so unavailable
compilation cannot enter global RNG restoration contexts beside the collector.

Reward is net realized value plus an outcome, with gamma 1 retaining late outcomes.
GAE keeps a time-based trace decay. Income, effective damage and remaining asset
losses share sun-equivalent units. Historical peaks/drawdown are diagnostics only.
See [objective derivations and limits](math/training-objective.md).

The curriculum is saving → easy → standard → shared. Saving uses only saving;
easy samples 80% easy and 20% saving. Standard is 45% easy, 45% standard and 10%
saving; shared is 20% easy, 40% standard and 40% hard. Saving has 100 initial sun,
all eight species, no sky or mowers, and three selected lanes each receiving a basic
at 75, 87 and 99 seconds. Public-state feasibility and the no-income impossibility
argument are [checked separately](math/saving-and-actions.md); they are not controllers
used by the learner. Stage changes affect future episode resets only.

Selected-stage runs can enable `until_stage_complete`. The game and wall-clock
ceilings become inactive; elapsed time and completed games still drive diagnostics,
exploration and probes. Failed probes continue the same stage. A complete passing
probe marks mastery, saves its checkpoint and triggers normal-game validation before
finalization. The selected stage never promotes automatically. Bounded training is
the default; conflicting explicit ceilings fail before output creation.

## Storage, compatibility and failure paths

Each run records resolved settings, exploration phase/floor state, source and structural
signatures, periodic pipeline depth, policy-version hashes, overlap timings, reward and
discount settings, seeds, elapsed allowance, curriculum state, episodes and optimizer
metrics. Checkpoints include both optimizers, optimizer protocol
`periodic_exact_kl_v1`, exploration protocol `phase_floor_v1`, and action-distribution
signature `balanced_species_tiles_v1`. Older distribution checkpoints cannot load
for inference, transfer or resume; their files and model-free reports remain available.
Same-protocol stage transfers copy compatible weights only; same-protocol
resume restores the saved experiment, starts fresh games and empty memory, and
preserves cumulative schedules. Resume is not a bitwise continuation of partial games.
All earlier observation/policy signatures are rejected before model loading.

The engine is installed non-editably from a commit-verified archive and complete
source manifest. Missing CUDA, changed pins, unsupported settings and incompatible
weights fail before a new run is created. Capacity exhaustion fails rather than
clipping entities. Deadline checks stop at completed updates and preserve explicitly
incomplete evaluation/export state. Validation alone selects best checkpoints.

Demos replay GPU actions through the CPU engine and require identical final outcome,
decisions and hash before publication. Native 100 Hz recordings remain model-free.
Video rendering samples a configurable lower rate while stepping/verifying every
tick. Offline reports read stored metrics, not models. Historical reports survive;
20 Hz recordings and incompatible models are not accepted by the current engine/policy.

## Hardware observations and report lifecycle

A shared background monitor serves ordinary training and benchmarks. It primes
psutil's interval counters, samples CPU/RAM without blocking, and makes bounded,
hidden nvidia-smi queries against the training GPU's CUDA UUID. It never issues
CUDA work or synchronizes a transition. Sampler failure is a diagnostic, not a
reason to reject a checkpoint. Missing fields stay null. GPU activity is device-wide;
memory-controller busy percent differs from VRAM capacity used. Process CPU uses
one-core percentage units and can exceed 100; system/core CPU remains 0–100.

Each sample appends and flushes to `hardware-metrics.jsonl` with UTC, session and
elapsed time, curriculum/exploration phase, activity, and latest complete-window
timings. The in-memory cache is bounded to ten minutes. Probes and normal validation
use one shared evaluation-event wrapper: enter validation, attempt evaluation, then
refresh the report in a finally path. Completion and graceful interruption also
refresh it. Plotting failures are recorded without invalidating saved models.

```mermaid
stateDiagram-v2
    [*] --> Training: start sampler and prime CPU counters
    Training --> Validation: complete window / evaluation due
    Validation --> Reporting: attempt ends, succeeds or fails
    Reporting --> Training: atomically replace plots
    Training --> Reporting: complete or graceful interruption
    Reporting --> Stopped: close sampler and logs
    Stopped --> Reporting: offline report-only command
```

Report panels show CPU, GPU activity, RAM/VRAM, power/temperature and window
throughput/timings on elapsed wall-time axes with phase backgrounds. Resume sessions
have separate labeled axes/lines, and ancestor logs stop at the saved checkpoint
boundary. Hardware logs tolerate only an incomplete final record after a hard stop;
malformed middle records raise an error. `visualize --report-only` needs metadata and
logs, never a model or demonstration. No samples means historical utilization is unknown.
The terminal emits aligned 15-second blocks and labels recent, window and run-average
measurements separately; unavailable values are n/a.
