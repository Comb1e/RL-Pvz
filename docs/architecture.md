# System architecture

This project learns plant-placement decisions from complete games on a pinned
100 Hz Plants vs. Zombies simulator. One Transformer–LSTM model supports human
demonstration initialization, autonomous collection, evaluation and replay.
The game package remains separate and pinned; this repository adapts its public
observations, action timing and reward accounting.

```mermaid
flowchart LR
    Human[Human easy game] --> Archive[Transitions + native replay]
    Archive --> Verify[Verify hashes and reconstruct every decision]
    Verify --> Init[Complete-return demonstration fit]
    Init --> Weights[initialization.pt]
    Weights --> Inspect[Checkpoint inspection and compatibility]
    Saved[Autonomous ZIP] --> Inspect
    Inspect --> Policy[Transformer–LSTM]
    Policy --> Simulator[CUDA cohort]
    Simulator --> Public[Public observations and action outcomes]
    Public --> Policy
    Simulator --> Store[Bounded host trajectories / disk overflow]
    Store --> Fit[Complete returns / chronological fitting]
    Fit --> Policy
    Policy --> Saved
    Saved --> Evaluate[CPU or CUDA evaluation / verified replay]
```

## Components and ownership

Configuration defines model dimensions and the protocol. The single bundled
`src/pvz_rl/data/train.toml` file contains the shared parameters plus sparse
`train` and `demo` overlays. Installed CLI use and source-tree commands resolve
the same values. There is one entity Transformer–LSTM architecture.
Shared checkpoint inspection validates format, model/optimizer protocols,
configuration identity and weight structure before environment creation. Loading
for evaluation retains saved learning settings. Fresh initialization uses current
configuration; autonomous resume automatically applies current execution and
logging settings without replacing saved learning parameters.
Engine, complete observation encoding, timing, action algebra
and network structure must match for autonomous weight transfer; demonstration
weights require the public input/output model interface. Reward/exploration prices
do not constrain weight-only reuse. Budgets and output preferences
are execution choices. Invalid input raises an error; there is no random fallback.

Environment adapters encode only public state: one int32 row per mower, plant,
zombie and projectile, with 11 fields per row. State categories preserve public
behavior phases, and numerical fields keep individual coordinates, health,
armor, timers, slowing, pole/headless flags and projectile damage together.
Eighteen globals contain sun, elapsed time, wave/total waves, initial/defeated
counts, eight cooldowns and four full-list entity counts. No IDs, allocation
slots, schedules, RNG state or private movement fields enter the policy.

The configured cap retains mowers, plants, zombies and projectiles in that order.
It omits projectiles first, then zombies, then plants. Zombies nearest the house
come first; public fields break ties. All five mowers remain, including spent
ones. Canonical ordering makes storage reproducible; attention does not depend
on row order. Global entity counts are measured before truncation. Diagnostics
record retained and omitted counts; omitted details cannot be recovered from counts.

CPU encoding consumes public observation objects. CUDA encoding derives equivalent
public countdowns and records directly from device arrays, including projectiles.
Only occupied simulator slots become records. Shared field definitions, categorical
maps and pinned-rule normalization describe both paths. Batching pads to its
largest retained list and carries a boolean validity mask.

```mermaid
flowchart LR
    Public[Public entity lists] --> Cap[Mowers / plants / zombies / projectiles; cap]
    Cap --> Raw[Integer entity matrix n × 11]
    Raw --> Embed[Shared learned embedding n × 32]
    Globals[18 globals] --> Readout[Global token + 45 tile queries]
    Embed --> Attention[2 masked bidirectional attention layers]
    Readout --> Attention
    Attention --> Summary[Global summary + tile features]
    Summary --> Fusion[256-wide current-state and memory fusion]
    Summary -->|context at event writes| LSTM[256-unit event LSTM]
    Events[Gross public events and elapsed ticks] -->|32-wide projection| LSTM
    LSTM --> Fusion
    Fusion --> Heads[Branch and conditional tile Q heads]
```

Each shared entity embedding sums type and state lookups and a projection of nine
normalized numerical values, then applies layer normalization. Coordinates use
pinned units, health uses species maximum health, armor uses maximum armor,
durations use seconds, and damage uses normal-pea damage. Valid values are never
clipped. The default width is 32, with two four-head attention layers, feed-forward
width 64 and no dropout. A global readout token and 45 coordinate tile queries
join the sequence. Queries let empty tiles receive features. Padded keys are
masked in every layer and padded entity outputs are zero. No entity-order or
causal attention mask is added; the LSTM supplies chronological memory.

Every decision encodes the current board. Gross sunlight gain/spend, zombie
spawn/defeat/removal and plant addition/removal counters determine whether memory
writes before the next decision, using the resulting board as context.
Initial entities are not additions. Elapsed ticks accumulate since the previous
write, but time, movement, damage, cooldown and rejection alone never write.
Sunlight gains are actual capped gains; simultaneous additions/losses remain
separate. Separate zero-time transitions retain their order.

The 32-wide entity summary and 64-wide scalar features combine with a 32-wide
event projection at writes through a 256-unit LSTM. Events use existing resource/
count scales and log1p(elapsed seconds), without clipping. Every decision fuses
its current summary/scalars with the latest event-memory output into 256 features
for both Q heads. Quiet decisions still respond to current board changes.
The shared Q selector compares wait, eight species and dig greedily, then
explores the selected branch's tile target. It chooses a tile for a
non-wait branch. Occupancy limits plant tiles; dig can target every tile.
A full board gives a deterministic tile-zero plant proposal that the simulator
may reject. Affordability and cooldown never remove a plant branch from that
comparison. The CPU adapter and CUDA batch pass the selected proposal unchanged
to validation, so a rejected plant advances exactly one tick, carries the pinned
reason into the transition, and receives `invalid_plant_penalty`. The proposal
is retained for rewards, Q fitting, journals and viewers; the resolved execution
action is exposed separately as wait.

Autonomous collection adds a separate tensor-based committed-plant controller.
After an accepted normal planting, a species coin may select uniformly from the
other seven species. The controller issues ordinary waits until that species is
affordable, off cooldown and has an empty tile, then selects its tile from the
current Q features. There is no timeout, digging fallback or defensive cancellation;
full-board commitments wait for a tile to become empty or the episode to end.
Successful exploratory placements do not retrigger exploration. Every waiting
decision still encodes the board and consumes genuine memory events.

```mermaid
stateDiagram-v2
    [*] --> Normal
    Normal --> Committed: accepted normal planting / species coin
    Committed --> Committed: unavailable / execute one-tick wait
    Committed --> Normal: exploratory planting accepted
    Committed --> Committed: unexpected planting rejection
    Committed --> [*]: episode ends
```

Both probabilities resolve from completed stage games and freeze for each cohort:
tile 50% to 1%, species 10% to 1%, exponentially over 10,000 games. Stage changes
restart progress. Evaluation, replay and counterfactual probes never run this
controller. Selected-action fitting uses actual controller proposals and their
Q-values; original policy proposals and commitment provenance remain diagnostics.

Reward accounting credits actual capped Sunflower income at its production event.
While three times cumulative spawns is less than the public total roster, its
development value triples. Sky income, discarded sun and zero-total scenarios
earn no bonus. CPU processes ordered public events; CUDA extends read-only counters
without detailed event buffers. Early sun and its bonus are reported separately;
the bonus enters development once and never changes physical net-value metrics
or recurrent event fields. Replay reconstruction reprices the same facts.

```mermaid
flowchart LR
    Q[Ten branch Q comparison] --> Tile[Wait or selected branch's tile]
    Tile --> Proposal[Selected proposal]
    Proposal --> Validate[CPU or CUDA validation]
    Validate -->|accepted| Execute[Execute proposal]
    Validate -->|rejected| Wait[Execute wait / advance one tick]
    Wait --> Penalty[Rejection reason + applicable penalty]
    Execute --> Facts[Gross public transition events]
    Wait --> Facts
    Facts --> Pending[Pending event and elapsed ticks]
    Pending -->|qualifying event| Memory[One LSTM write with resulting board]
    Pending -->|quiet| Frozen[Preserve hidden and cell exactly]
    Memory --> Fusion[Current board plus event memory]
    Frozen --> Fusion
```

The shared event-memory state owns hidden/cell, pending facts and elapsed ticks.
Consumption clears pending facts once and resets timing only on a write.
Reset clears all fields for new episode slots; inactive slots are frozen.
Accepted plants and digs take zero ticks; waits and rejections advance one tick.
A rejected attempt concurrent with a world event still queues a write. The same
runner serves collection, evaluation and EMA history.

```mermaid
stateDiagram-v2
    [*] --> Quiet: zero memory at reset
    Quiet --> Quiet: no qualifying transition event
    Quiet --> Pending: public transition event
    Pending --> Quiet: consume once before next decision
    Pending --> Pending: snapshot / restore retains facts and timing
    Quiet --> Frozen: episode finishes
    Pending --> Frozen: episode finishes
    Frozen --> Quiet: new episode reset
```

## Recording and initialization

A human session owns one easy attempt. The recorder checks unused output paths,
records action-phase order, and streams every proposal and its public observation,
outcome, duration, reward and episode boundary into JSONL. Native replay and
manifest describe completion; closing early leaves an incomplete archive.
Compact viewer history never replaces the complete fitting sequence.

Initialization rejects missing manifests, unsupported protocols, replay-hash or
engine/schema mismatches, incomplete episodes and reconstruction differences. Only entity-v1
archives and weights are accepted; earlier archives require a new recording.
Verification compares recorded facts rather than the full configuration digest;
changing optimizer or reward settings does not invalidate a recording. The loader
checks public reward facts and ledger arithmetic against replay, then replaces
rewards in its in-memory transitions using the current configuration. Both
verification and fitting share this loader; fitting consumes the verified data
without rereading historical reward totals from disk. The verification report
records the current reward settings, changed decision count and old/new totals.
Historical prices cannot be authenticated from old archives' configuration hashes;
they never become fitting targets. Observation, outcome, terminal and replay-hash
checks remain exact, and recording files remain unchanged.

```mermaid
flowchart LR
    Archive[Archive and pinned replay] --> Verify[Verify public facts and ledger arithmetic]
    Verify --> Price[Recompute rewards from replay events]
    Config[Current reward settings] --> Price
    Price --> Returns[Verified transitions and complete returns]
    Returns --> Fit[Whole-pass demonstration fit]
```

The generated `initialization.pt` stores weights and source metadata; it omits
encoder batching and performance settings. Fresh `--init-from` resolves the
current configuration before weight transfer. Source learning/reward metadata
is used for provenance, never to override the new run's settings. Compatibility
checks validate the engine, input/action semantics and network dimensions;
reward coefficients are excluded from weight compatibility.
Complete gamma-one returns supervise both Q heads. Whole-pass gradients are
clipped once using the same `training.max_grad_norm` limit as autonomous fitting;
only completed passes atomically replace `initialization.pt`.
`training.demo.passes` independently controls initialization (20 by default);
autonomous fitting uses `training.n_epochs` (4 by default). The initialization
`--passes` override does not change autonomous fitting.

## Autonomous lifecycle and recovery

```mermaid
stateDiagram-v2
    [*] --> Inspect
    Inspect --> Rejected: incompatible or corrupt checkpoint
    Inspect --> Idle: fresh / weights-only / resumed state
    Idle --> Collect: schedule remaining games
    Collect --> Collect: decision / store public transition
    Collect --> FinalizeRewards: every active game finishes
    FinalizeRewards --> Returns: median and terminal time rewards fixed
    Returns --> Fit: complete gamma-one targets
    Fit --> Fit: chronological chunks / detached state
    Fit --> Synchronize: all whole-pass updates committed
    Synchronize --> Idle: callbacks and next cohort
    Collect --> Saved: interrupt at decision boundary
    FinalizeRewards --> Saved: preserve pending publication and median
    Fit --> Saved: interrupt with current pass uncommitted
    Saved --> Collect: resume saved collection
    Saved --> FinalizeRewards: resume idempotent finalization
    Saved --> Fit: restart current pass only
    Idle --> Saved: budget or mastery reached
```

A cohort holds at most 128 games at fixed weights. Completed slots remain inactive
until the next cohort, and the final cohort uses only the remaining game count.
The collector builds a host `ActiveBatchPlan` from `enabled_envs` and the latest
published public retained-entity counts in `summary_host`. It groups live slots
by entity width and power-of-two batch size, within encoder microbatch/token
budgets. Canonical slot IDs gather observations, hidden/cell, pending events and
elapsed ticks; masked padding never becomes a game. Outputs scatter back to those
same IDs, preserving inactive memory exactly. Empty active sets do no inference.
Online and actual-history EMA inference use this plan; the tile-head wrapper
evaluates active rows only. Selection/exploration RNG still draws in the original
slot-shaped dimensions and order, so compaction does not reassign a live game's
random draw, probe cursor or journal identity.

Trajectories retain fixed transition metadata and globals, entity offset/count,
and contiguous append-only int32 entity slabs containing only real rows. Metadata
and entities share one RAM budget with disk overflow. Proposals, executed actions,
acceptance, durations, event facts/write masks/timing, omission counts, rewards,
controller mode, pending/next species, original policy action, cohort probabilities
and boundaries remain explicit. Group/outcome and probe head/branch/outcome counts
cover the full cohort, including spilled blocks.
Only recovery state stores private simulator snapshots and scenario RNGs.

At each decision, two scratch lanes independently restore pre-decision simulator,
RNG, accounting and proximity-ledger state. The branch pair and tile pair execute
in two simulator passes through shallow batch/feature prefix views of the existing
scratch allocation. Source-ID remapping copies original live slots into each
lane's compact prefix; simulator/feature work uses that prefix rather than all
original slots. Scheduled-role masks still distinguish eligible probes. Public
features and metadata scatter back to canonical game/probe rows before packing,
deduplication and bootstrap. The views do not allocate a smaller simulator or
shrink private storage bounds, and source IDs/private scratch state never become
policy inputs. Allocation failure releases the failed scratch allocation
and selects one lane, executing all four scheduled probes sequentially. It never
changes probe coverage. The frozen FP32 EMA teacher advances only on actual history;
probe next states receive complete independent event-memory copies, including
pending facts and timing. Probe events update only these forks. Only valid,
nonterminal representatives enter next-state inference after exact deduplication;
their values scatter back to every original probe row. Terminal bootstraps are zero.
Probe states never enter behavior history, viewer history or journals.

Persistent device feature buffers retain each compact pass in its own canonical
rows after scatter. Canonical records
are packed on device; duplicate probes share a slab offset only when entities,
globals and gross event/timing inputs match within the same actual game.
Identical resulting boards alone are insufficient; the forked hidden/cell
state is shared only within that game's decision. Proposal, rejection reason,
reward components and bootstrap remain separate records even when offsets alias.
Storage consumes packed offsets directly rather than repeating host deduplication.

Online observations use fixed 32/64/128/256 padding buckets. Probe padding uses a
conservative bound from already-published public entity counts, pending-zombie
counts, at most two shots per existing plant and one accepted new plant. No future
schedule or RNG determines policy input. The configured entity cap still limits
tokens, not simulator capacity; a bound violation fails instead of dropping records.
Collection runs direct no-grad CUDA graphs on every CUDA platform, while fitting
uses the custom AOT owned-output forward/backward backend on every CUDA platform.
Both copy outputs out of reusable capture buffers. Each online/EMA encoder owns
its own collection LRU cache, configured by
`training.performance.collection_graph_cache_entries` (8 live entries) and
`collection_vram_headroom_mib` (512). Shape, stride, dtype/device and computation
mode distinguish captures. Eviction fences a capture's last replay/output copies;
the existing handoff polls those fences and releases completed retired buffers
without another wait. Cache misses can also poll completed fences. The retired
list is separately bounded by the same entry limit; resident metrics include
these not-yet-released buffers. No queued replay loses its capture storage.
Admission checks VRAM headroom before and after capture; capacity pressure,
insufficient headroom and capture allocation failure fall back to exact eager
inference. Other capture failures record an explicit eager fallback.

Collection and fitting captures have separate phase owners. Entering fitting
drains pending collection work and releases collection captures and probe
workspaces. Entering collection drains delayed backward/prefetch work and releases
fitting captures before building collection workspaces. Fitting retains its own
bounded four-shape compilation set; collection shapes do not consume that set.
Phase memory and cache hit/miss/capture/eviction/fallback counters expose execution
cost without changing observations, rewards, exploration or optimizer updates.

Behavior globals transfer only active rows and rebuild zero-filled canonical host
rows. Probe globals use the same live-row transfer/canonical scatter contract;
inactive globals are not copied merely to preserve host shape. Small fixed
metadata can retain original slot-shaped staging.
Entity slabs use summed public bounds for active games, not full original capacity.
This does not promise exact deduplicated-byte DMA: a bounded unused tail can still
cross the handoff without adding another GPU count wait.

One reusable pinned handoff queues behavior inputs, probe metadata/slabs, dedup
representatives, execution results and episode headers/totals on the collection
stream. A single existing stream wait makes those representatives available to
the host planner. After consuming the previous decision's pending bootstrap copy,
the collector appends the current canonical rows, runs compact next-state EMA
inference and queues its bootstrap CPU copy. The CPU consumes that copy at the
next existing collection handoff and patches the appended rows through
`buffer.update_probe_bootstrap` before targets or fitting can use them. This
pipeline adds no ordinary per-decision wait for GPU dedup counts. The last copy
is explicitly drained at quiescent finalization, checkpoint, interruption and
shutdown boundaries; it cannot be serialized as a half-updated trajectory. Terminal
metrics need no separate header/totals read. Viewer staging remains read-only and
asynchronous; reset, checkpoint and shutdown explicitly drain it. Device timing
events are read after the existing handoff and reused, not individually waited on.

```mermaid
flowchart LR
    Source[Actual pre-decision state] --> Plan[Host active-slot / public-width plan]
    Plan --> Online[Compact online inference / canonical proposal]
    Plan --> EMA[Compact actual EMA history once]
    Online --> Scratch[Remap live source IDs / compact scratch prefix views]
    Scratch --> Branch[Branch pair / first pass]
    Scratch --> Tile[Tile pair / second pass]
    Branch --> Next[Scatter owned public features / canonical probe rows]
    Tile --> Next
    EMA --> Fork[Independent probe histories]
    Next --> Pack[Exact device deduplication / packed records]
    Online --> Actual[Execute actual proposal]
    Actual --> Handoff[Queue compact public evidence / one stream wait]
    Pack --> Handoff
    Handoff --> Previous[Patch previous bootstrap copy / release retired captures]
    Previous --> Store[Append canonical RAM/disk rows]
    Store --> Bootstrap[Valid unique nonterminal EMA inference]
    Fork --> Bootstrap
    Bootstrap --> Copy[Queue bootstrap CPU copy]
    Copy --> Boundary[Next existing handoff / quiescent drain]
    Boundary --> Patch[Patch appended rows before targets / fitting]
    Handoff --> Journal[Actual journal / viewer]
```

Game and win counts update immediately; completed rewards remain provisional until
the cohort median is known. Finalization includes all actual durations, applies time
shaping only to victories, fixes terminal probe rewards with that same median and
publishes finalized records once. Saved state includes median/reference count,
publication state, pending event/timing state, committed species/cohort probabilities,
EMA weights/version/history,
probe cursors, proximity ledgers and online/EMA compilation diagnostics.
Active execution maps, captures and cache entries are not serialized; resume
rebuilds them from restored canonical state. This execution change leaves the
recovery/trajectory contracts unchanged. Protocol/schema inspection precedes the
CUDA availability probe,
collector allocation and model transfer.
Autonomous recovery and trajectories use the current combined exploration contract;
missing or malformed controller state rejects recovery. Weight-only inspection
accepts full structurally compatible event-memory weights without deserializing
the old optimizer or collector. It validates the complete parameter dictionary
against the current model and starts fresh; the demo event-memory contract is unchanged.
Selected complete returns and outcome-balanced probe Huber errors share each whole-pass
optimizer update. EMA advances once after each successful commit, never after a failed
or interrupted attempt. Demonstrations add accepted-action ranking instead of probes.

Fitting visits episodes chronologically from zero recurrent state each pass.
The learning batch remains 1,024 decisions and each recurrent boundary remains
256 decisions, but the execution path groups four independent batches into a
single 4,096-decision transfer and forward/backward call when the environment
count permits. State is reset between slot groups, so no episode can observe
another episode's hidden state. Short episodes are padded; padding contributes
no target or loss. Only genuine event rows enter the recurrent sequence; latest
memory is gathered onto every current-state prediction. Nonempty action groups
have equal weight, with accepted/rejected outcomes sharing 50/50 when both exist. Encoder work uses at most 128 observations and a 65,536-token
budget. Allocation fallback lowers the encoder microbatch to 64, 32 and 16,
then enables one outer checkpoint without dropping entities.

Full-sequence SDPA selects the available efficient backend; an exact 64-query
fallback attends to all keys. When compilation is enabled, fixed-shape CUDA encoder
calls use the owned graph backends on all platforms; a failed
backend is recorded once and the exact eager path remains active. CPU preparation
caches sequence index templates,
gathers entity slabs into reusable buffers and fills metadata in bulk. Two
pinned buffers and one ordered worker prepare the next fused batch; transfer
events prevent reads or host-buffer reuse before the copy completes.

The fitting AOT backend captures forward and backward separately. Every output,
including internal activations saved for backward and returned parameter
gradients, is copied out of reusable graph storage. Copies retain tensor strides
and attention alignment. Consequently, replay cannot overwrite a preceding
microbatch's saved activations or accumulated gradients. Input buffers are
refreshed on every call, including updated FP32 weights. If backward capture
fails, the uncommitted pass discards gradients and restarts eagerly without
changing precision or counting an optimizer update. Allocation failures retain
the bounded memory fallback above.

CUDA fitting computes entity features, embeddings and auxiliary projections with
BF16 temporary values. Stored weights, normalization reductions, residual sums,
LSTM state/computation, Q heads, loss, gradients and Adam state remain FP32.
Collection, evaluation and demonstration fitting use FP32. Demonstration
initialization settings are independent of autonomous execution settings;
precision does not change raw observations, parameter shapes, or the learning
objective.

Each pass is an uncommitted transaction until finite gradients are clipped and
Adam updates once. Nonfinite BF16 computation discards the pass and retries in
FP32, retained for the rest of the cohort. An FP32 failure stops before updating.
CUDA timing records one fitting event per pass instead of one event per chunk;
intermediate progress uses cached metrics and does not drain the device. Exhaustion fails explicitly;
entity caps and recurrent sequence lengths never shrink. Equal weighting of nonempty wait/plant/dig groups is computed
across the entire cohort. Every pass accumulates chunk gradients before one
clipped Adam update; four passes are the default. Collection states are never
reused as fitting states after weights change.


```mermaid
stateDiagram-v2
    [*] --> Prepare
    Prepare --> Accumulate: ordered transfer completed
    Accumulate --> Accumulate: next chronological chunk
    Accumulate --> Prepare: allocation failure / smaller encoder microbatch
    Accumulate --> Prepare: nonfinite BF16 / discard gradients and select FP32
    Accumulate --> Commit: complete pass and finite gradients
    Accumulate --> Failed: nonfinite FP32 or exhausted memory fallback
    Commit --> Prepare: next pass
    Commit --> [*]: all passes committed
```

Execution precision and fallback status accompany recovery metadata. Resume applies
current precision, prefetch, encoder batching, attention fallback size,
instrumentation and logging settings automatically. Missing execution metadata
hydrates from the current profile, rather than historical FP32 defaults.
Learning parameters and input/output schemas are excluded from that refresh.
Fresh weight initialization uses current learning and execution settings.
Recovery retains saved rewards and clipping because collected returns and
optimizer state belong to that saved configuration.

Atomic ZIP replacement couples model, optimizer, counters, curriculum, exploration,
RNGs, trajectory/entity slabs and collection states. Schema metadata records
field order, categories, normalization and cap; configuration records dimensions.
CUDA recovery allocates the proved future projectile bound plus an equal headroom
for active shots already present in a mid-game snapshot. Recovery validates slab
dtype, shape and contiguous offsets/counts before reuse. A failed save leaves the previous
ZIP intact. Resume restores a decision boundary during collection. Fitting
restores completed updates and recomputes only the uncommitted pass with cleared
gradients. The network has no stochastic dropout, and fitting does not sample
minibatches. Run metadata is embedded so checkpoints can be moved independently
of adjacent JSON reports. In-place resume appends history.

## Curriculum, evaluation and presentation

```mermaid
stateDiagram-v2
    [*] --> Easy
    Easy --> Standard: easy gate and residency pass
    Standard --> Shared: easy and standard gates pass
    Shared --> Mastered: easy / standard / hard gates pass
```

Easy contains easy games only; standard mixes easy and standard equally; shared
uses 20%/40%/40% easy/standard/hard. Probes use held-out curriculum seeds. Progress
changes future resets, never a running game. Single-stage runs stop at that
stage's mastery; ordinary runs promote. Normal validation follows successful
stages.

Evaluation uses public observations through the same runner. CUDA evaluation can
refill finished slots or hold a fixed batch; only replaced episodes reset memory.
Selected action traces are replayed on the CPU reference before recordings are
published. Evaluation never uses exploration or updates weights.

The viewer receives Q values captured during actual decisions, plus bounded
history pages and retained/omitted entity counts. Generation and episode identities reject stale responses. **F**
clears selection/paging and follows the latest action while preserving horizontal
scroll. Presentation runs separately and supplies no simulation time or policy
inputs. Callback reports retain progress, coverage, probes and checkpoint status.
At viewer startup, the environment supplies a small immutable snapshot of the
resolved learning configuration: optimizer settings, sequence length, rewards
and tile exploration. The settings strip shows checkpoint-retained values on
resume, independently of refreshed execution settings. It is available before
the first board and during fitting; **S** changes visibility only. The viewer
never reloads defaults or reads GPU tensors to display these values.

Terminal output and `train.log` contain compact event records, not a dashboard.
Phase boundaries, completed cohorts, checkpoint saves, warnings and final states
print immediately. Routine output requires both material progress and the separate
60-second terminal interval. Interactive output rewrites one line; redirected
output uses timestamped lines without ANSI controls. Consecutive duplicates and
repeated identical report-refresh notices are suppressed. Current-cohort done/win
counts and completed-game mean raw/training returns precede throughput and hardware;
`n/a` means no completed games and `~` means provisional time shaping. Statistics
come from host-side pending episode records, persist through fitting/recovery and
clear before the next cohort starts. Interactive routine lines omit the timestamp/
phase prefix to retain these fields at narrow widths; durable logs remain complete.
Detailed `status.json` snapshots retain their independent 15-second cadence, alongside training and
hardware JSONL streams. Formatting uses host-side snapshots and cached fitting
metrics; it never reads active device accumulators.

The collection interval monitor reports current active games, decisions/game/s
and active transitions/s separately from lifetime totals. For interval active
transitions `D`, collection seconds `T` and active-game seconds
`A = sum(active_games_before_callback * elapsed_seconds)`, its rates are `D/A`
and `D/T`. Completed slots, padding and counterfactuals are not transitions.
Phase/cohort changes and restored counters reset the reporting interval;
fitting, validation and recovery downtime do not enter collection rates. Missing
or zero-duration intervals display `n/a`, not zero. Sampling uses host counters
only and never changes the detailed JSON or terminal cadence.

Mathematical controls and verification limits are in
[entity inputs](math/entity-inputs.md) and [recurrent training](math/recurrent-training.md); chronological release evidence
belongs in [iteration history](iteration.md). Short integration checks establish
mechanics and recovery, not win-rate improvement or formal training success.
