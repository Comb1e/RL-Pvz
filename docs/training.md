# Recording and training

Fresh weights are required; existing verified archives/native replays remain reusable.
Refit into a new output directory, never overwrite recordings or previous runs.
The current objective is `complete_return_joint_tile_probe_v1`. Rejected plants and empty digs
use the small shared penalties in `train.toml`; development is multiplied only in
learning targets. A zombie entering either of the two columns nearest the house is
charged once per boundary. Victory time is shaped only for wins, relative to the
median duration of the current completed cohort.

Every accepted planting, including committed exploration, queues up to four other
tiles of its species. Real games continue alongside forks retaining the pre-plant
board and complete EMA history; actual acceptance authorizes execution.
Each valid probe runs a frozen greedy EMA policy for up to 30 simulated seconds or 4096 transitions,
then bootstraps unless truly terminal/cutoff. Longer probes increase collection cost.
Auxiliary games retain independent simulator state and never alter the behavior game.
Cohort reward finalization occurs before complete
returns and fitting, and is recoverable without applying time rewards twice.

Probe settings live under `training.objective`: `probe_trigger = "accepted_plant"`,
`tile_probes = 4`, `probe_rollout_seconds = 30`,
`probe_rollout_max_decisions = 4096` and 64-transition heavy control reporting.
Only same-species placements on different empty tiles are probed; counts can be
1–44, limited by available alternatives. Branch probes are removed.
Real games, including the planting slot, continue while isolated auxiliary games
advance. Execution defaults allow 256 active auxiliary lanes and 128 outstanding
source jobs, with one auxiliary transition per scheduler round.
A full source bank pauses only another eligible planting decision, retaining its
proposal/draws until FIFO admission. No waits or discarded probes are substituted.
Saving stops new decisions, finishes issued tickets and drains all probes before
writing the checkpoint; drain latency is reported separately.

Install with the [quick start](../README.md). Run commands from the project root.
The default model throughout recording, initialization, training and evaluation
is the entity Transformer–LSTM (`entity_v1`, `transformer_lstm_q_v4`).
A public observation contains an integer `[n,11]` entity matrix and 18 global
values; each retained entity becomes a learned 32-dimensional vector.

## Record and initialize

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human-1000.jsonl --replay runs\human-1000.pvzdemo `
  --output runs\human-init-tile
```

Recording uses one easy-stage attempt, default seed 1000. Choose unused replay,
archive and sidecar paths. Pause retains queued actions; restart and stage
switching are disabled. Closing early leaves an incomplete recording.

Initialization checks the manifest, native replay hash, every observation,
proposed action, acceptance result, duration and terminal state before fitting.
It verifies the archived reward ledger's public facts and accounting identities,
then recomputes every reward from replay observations/events using current reward
settings. Complete returns use these recomputed rewards. Archived reward prices
are historical diagnostics; changing penalties, terminal rewards or development
weights does not require rerecording. The archive/replay stay unchanged, and
`replay-verification.json` reports changed reward counts and old/new totals.
Missing manifests, unsupported protocols, engine/schema mismatches and
reconstruction failures remain errors. Optimizer settings and the retired demo
clipping field do not affect recording verification.
Old checkpoints are rejected for both resume and weight initialization before
simulator allocation; there is no migration or partial-weight transfer. Existing
verified entity_v1 archives/native replays remain reusable: initialization
reconstructs events from native transitions under current settings. Use a new
output directory; existing recordings, runs and weights remain on disk.

Preserve the recording's input/action schema when using `--config`; learning and
reward settings may change. `--device cpu` is the
initialization default; `--device cuda` changes execution without changing archive
verification. `--passes`, `--learning-rate` and `--seed` override fitting settings.
Both fitting paths read the single `training.max_grad_norm` limit (5 by default);
there is no demo-only clipping setting or CLI override.
Time budgets are configuration settings only: initialization
uses `training.demo.time_budget_minutes` (30 by default). Defaults are 20 passes,
learning rate 0.0003 and 256-decision chunks. Checkpoints are published
only after complete passes, alongside curves, coverage and verification reports.

## Autonomous training

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --init-from runs\human-init-tile\initialization.pt --output runs\tile-trained
```

`--init-from` accepts demonstration `.pt` or compatible autonomous `.zip` weights.
It starts fresh Adam state, counters, curriculum and RNGs. Omitting both
`--init-from` and `--resume` starts a fresh recurrent model. No incompatible source
silently falls back to random initialization. Metadata records model family,
initialization type, source SHA-256 and parameter differences.

Fresh `--init-from` runs take all settings from the current `train.toml` (or
explicit `--config`), including rewards, clipping, learning rate and performance.
The source checkpoint supplies weights only. Demonstration weight transfer
validates the pinned engine, observation encoding, action semantics and network
dimensions; reward, objective, optimizer, return, recurrent chunk and performance
settings are supplied by the new run. Within the current contract, changed numerical
settings do not require another initialization. A new objective contract requires
fresh weights. Autonomous weight transfer keeps its stricter
saved-protocol checks.
Without `--config`, autonomous resume uses the checkpoint's saved configuration
plus missing execution defaults. Resume retains learning settings and rewards
to preserve unfinished trajectories and optimizer state. CPU autonomous training
is unsupported.

The curriculum is **easy → standard → shared**. Easy uses only easy games;
standard mixes easy/standard equally; shared mixes easy/standard/hard at
20%/40%/40%. Mastery requires the configured held-out gates and residency; promotion changes future
resets only. `--stage easy --until-stage-complete` requests an unbounded run
until that stage passes. Use it only intentionally.

Up to 128 games form a cohort at fixed weights. Finished slots stay inactive;
the final cohort is limited to the remaining requested games. Online and actual
EMA inference process live slots only, with public entity-width and power-of-two
batch buckets. Canonical slot identities, inactive memory and original slot-shaped
selection RNG draws are preserved. Padding is execution-only, not another game.
The recurrent collector retains the selected proposal for the trajectory and complete-return
target. History retains only gross sunlight gain/spend, zombie spawn/defeat/removal
and plant addition/removal events, plus resulting board context and time since the
last write. Rejection alone never advances memory. Current board features still
reach wait/joint-tile values every decision. Species values are best candidate-tile
maxima. Only the actual complete action receives return regression. Proposal, execution, acceptance, duration and
rejection reason remain learning/journal evidence.

Each of four default fitting passes recomputes recurrent states from episode
starts. State carries across 256-decision chunks with gradients detached at
chunk boundaries. `--batch-size` is the decision budget: 1,024 means four
256-decision sequences; it must be a positive multiple of chunk length, at most
1,024. Padding and inactive slots have no loss. Chunk gradients accumulate into
one clipped Adam update per whole-cohort pass, with equal total weight for each
nonempty wait/plant/dig group. Within each group, accepted and rejected decisions
receive 50% each when both exist; the sole present outcome otherwise receives
full group weight. Probe losses apply the same split within each represented
branch, with one initial complete-action error per probe.

Tile exploration decays from 50% to 1% over 10,000 completed stage games and stays
fixed during each cohort. Normal ten-way branch choice (wait, eight plants and dig)
remains greedy. After accepted normal planting, a separate 10% to 1% species coin,
also decaying over 10,000 stage games, may choose uniformly from the other seven
plants. The agent then waits until sunlight, cooldown and an empty tile permit
that plant; there is no timeout or defensive cancellation. The world and event
memory continue advancing. Its tile is chosen from the current board with tile
exploration. Exploratory planting does not retrigger the species coin. Both
schedules reset on stage promotion. Evaluation disables both exploration behaviors.
Plant tiles use occupancy only;
affordability and cooldown do not hide a branch. Dig tiles are unrestricted; a
full-board plant proposal targets tile zero and receives the simulator's
rejection outcome and configured invalid-plant penalty. The viewer and action
journal show that proposal while labeling its resolved execution as an automatic
wait.

Actual Sunflower sunlight earns triple development value before cumulative
zombie spawns reach one-third of the public total. The exact default comparison
is `3 * spawned < total`; cap-discarded sunlight, sky income and zero-total
scenarios receive no bonus. See [reward accounting](math/training-objective.md).

Compact progress shows completed games and wins out of the current cohort size,
plus mean raw return (`Rraw`) and training return (`Rtrain`) of completed games.
They are `n/a` before any finish; `~` marks provisional victory-time shaping.
These are not the rolling 100-game statistics. Finalized means remain visible
during fitting and recovery; the next cohort starts empty.

The default budget is 10,000 games with `training.max_minutes = 120` and a
15-minute finalization reserve. `--games`, `--n-envs`, `--batch-size` and
`--no-live-view` are available execution overrides. Change time allowances in
TOML. The viewer retains Q values from the actual decision and existing history,
progress, coverage and checkpoint reporting. Press **F** to follow latest actions.

## Entity inputs and memory

The single `src/pvz_rl/data/train.toml` controls both profiles. The configured
`encoding.max_entities` cap is 256. Overflow omits projectiles, then zombies,
then plants; all five mowers remain. Nearest-house zombies take priority.
Global counts include omitted entities, and the journal exposes omission counts.
The cap loses individual information above the limit. See
[entity fields, normalization and attention](math/entity-inputs.md).

`training.storage.ram_gib` covers ragged trajectory slabs and pinned staging;
excess slabs spill to disk. The current execution profile uses an encoder
microbatch of 128 observations, a 65,536-token budget and a fallback query
chunk size of 64 under `policy`. Allocation retries reduce only encoder work
size (128 → 64 → 32 → 16) and then enable one outer checkpoint; they never
shrink the entity cap or recurrent sequence. Exhausted retries fail explicitly.

Fitting groups four independent learning batches for execution only
(`training.performance.fit_sequence_groups = 4`). Each group keeps complete
256-step sequences and separate recurrent states, so chronological order,
loss normalization and the single whole-pass optimizer update are unchanged.
Partial groups and inactive environment slots are masked.

For a short device comparison of entity caps:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl.monitoring.entity_benchmark `
  --output artifacts\entity-benchmark.json
```

This measures synthetic inference/fitting and Torch memory, excluding simulation
and learning quality. Run resource measurements only when the device is available.

## Resume and evaluate

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\human-trained\latest.zip --output runs\human-trained
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\human-trained\latest.zip --split validation `
  --output runs\human-trained\validation --record
```

An autonomous ZIP embeds configuration and run metadata. Resume restores model,
optimizer, counters, curriculum, exploration, RNGs, unfinished trajectories and
collection state. Collection resumes at its saved decision boundary. An
unfinished plant commitment resumes with the same species and frozen probabilities.
The autonomous recovery/trajectory contracts are versioned; older recovery state
is rejected before allocating a simulator. Only current objective-v2 checkpoints remain
reusable through `--init-from` under current defaults, with a new output directory,
fresh optimizer/counters and no pending commitments. For example:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --init-from runs\joint-stages-101\easy\latest.zip --stage easy `
  --output runs\joint-restart-101\easy --seed 101 --games 128
```

All previous checkpoints, including joint objective-v1 weights, are incompatible; no migration
or legacy collector is provided. An
interrupted fitting pass restarts from its beginning with cleared gradients;
completed optimizer updates are retained. `interrupted.zip` is the recovery
artifact after interruption; `latest.zip` and `final.zip` are written on normal
completion. In-place resume appends training history. Budgets may be extended;
changing model structure or the configured simulator slot count is rejected.
Live-slot bucket shapes may change as games finish. Active maps and graph caches
are rebuilt on resume, not serialized; active-only execution does not change the
canonical episode identities. Checkpoint/finalization drains finish issued decisions
and apply every endpoint/bootstrap patch before saving or constructing fitting targets.

Evaluation accepts either checkpoint format and uses the shared stateful runner.
CUDA batches reset only replaced episode slots. A configuration with
`simulation.backend = "cpu"` selects CPU evaluation while preserving the model
contract. Recording traces are checked against the CPU simulator before export.
Choose a fresh evaluation directory. Validation and test seeds remain separate.

## Training throughput and precision

### Active-only collection

The host planner uses `enabled_envs` and already-published `summary_host` entity
counts, never future schedules or a new GPU count read. It gathers each live slot's
board and complete event memory, then scatters outputs to its canonical identity.
The tile-head wrapper evaluates live rows while selection/exploration random
draws retain their original slot-shaped dimensions and order. Entity caps, action
timing, rewards, scheduled probes and loss denominators are unchanged.

Auxiliary simulation/encoding uses occupied shallow prefix views over a bounded
allocation. Immutable sources map to independent lanes; endpoint evidence attaches
to stable canonical source rows. A one-lane/source fallback keeps every probe.
Behavior metadata remains original-slot-shaped while execution/global transfers
follow ready/issued slots, not terminal rows. Capacity-blocked decisions retain
their proposals and random draws without artificial waits or simulation time.

Endpoint deduplication includes entities, globals, hidden/cell and gross event/
timing inputs within one source job. Ongoing games never merge. Only unique
nonterminal endpoints need EMA-tail inference; terminal bootstrap is zero.
Actual rows append immediately; subsequent handoffs patch complete owned endpoints
exactly once before fitting, including spilled storage. Interruption stops new
selection, finishes issued decisions, then drains all jobs and transfers before
saving. No transient source or unfinished fork is serialized. Forked history
never replaces actual EMA history. See [execution details](math/training-throughput.md).

Each online/EMA encoder has a separate collection LRU graph cache. Defaults under
`training.performance` are `collection_graph_cache_entries = 8` live entries and
`collection_vram_headroom_mib = 512`. Cache eviction fences retired replay/output
copies; the existing handoff releases completed retirees without a new wait.
Retirement is separately bounded by the entry limit. Capacity pressure, insufficient headroom
before/after admission or allocation failure retains exact eager inference.
Setting cache entries to zero disables collection captures without disabling
the fitting backend. Collection/fitting phase owners release outgoing captures
and workspaces at quiescent boundaries; this avoids treating stale buffers as
needed by the next phase. These settings refresh on resume as execution controls,
not learning parameters. Cache counters and phase memory explain cost; they are
not evidence of a speedup or improved policy quality.

### Fitting precision

The `training.performance.fit_precision` setting is `features_bf16` for new CUDA
runs; use `fp32` for a reference comparison. BF16 applies to temporary encoder,
embedding and auxiliary-projection computation. Parameters, optimizer state,
LSTM, Q heads, losses and accumulated gradients remain FP32. No loss scaler is
used. Collection/evaluation and demonstration initialization remain FP32;
`training.demo.passes = 20` and autonomous `training.n_epochs = 4` are independent.

Two pinned staging buffers use the same `training.storage.ram_gib` allowance as
trajectory slabs, spilling resident slabs when necessary. `training.performance.prefetch`
controls ordered preparation/transfer overlap. Telemetry reports preparation,
transfer wait/device time, fitting, and optimizer time separately; overlapped
phase durations must not be added to infer wall time. Fitting synchronizes
timing and device summaries at pass boundaries, checkpointing, interruption or
shutdown; intermediate progress uses cached host metrics. Fixed-shape encoder
compilation is optional and records its backend plus `compiled`, `fallback` or
`unavailable` before reverting to eager execution. On every CUDA platform,
no-grad collection uses direct `CUDAGraph` captures and fitting uses the custom
AOT `cudagraphs_owned` backend: activations and gradients are copied out of
reusable graph buffers, preserving them through delayed backward and whole-pass
accumulation. It does not trace simulator pybind objects or invoke Inductor's
max-autotune SM check. There is no Windows/Inductor platform split. A backward
capture failure records `encoder_compilation`, discards uncommitted gradients
and retries the pass eagerly at the same precision; optimizer counters advance
only after a successful pass.

Entity checkpoints with matching objective and reward protocols can resume. Old
objective checkpoints can transfer compatible weights into a fresh cohort, but
cannot resume unfinished trajectories under changed targets. Resume automatically
uses current execution and logging defaults while keeping learning parameters:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\human-trained\interrupted.zip --output runs\human-trained
```

`--init-from` applies all current settings to a fresh run. Execution refresh on
resume cannot change passes, epochs, batch size, recurrent chunk length,
learning rate, `max_grad_norm`, rewards, curriculum or architecture. The live
window's read-only learning settings show the active values retained on resume;
press **S** to hide or show them. Missing execution metadata uses current defaults;
saved in-cohort numerical fallbacks remain when the execution profile is unchanged.
Precision changes begin when
the unfinished pass restarts. A nonfinite BF16 pass retries wholly in FP32;
only successful whole passes count as updates. Effective precision and fallback
reasons are saved. A nonfinite FP32 pass or exhausted allocation fallback fails
explicitly. See [precision and throughput controls](math/training-throughput.md).

For a bounded four-pass throughput check with fixed model dimensions:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl.monitoring.throughput_benchmark `
  --output artifacts\throughput-synthetic.json
.\.venv\Scripts\python.exe -m pvz_rl.monitoring.collection_benchmark `
  --output artifacts\throughput-collection.json
```

The default collection check executes no optimizer steps. It covers
128/64/32/16/4/1 live games in fixed 128 original simulator slots on sparse, mixed
and crowded snapshots. Live IDs are noncontiguous, including the one-game arm;
these are not six differently sized simulator allocations. It separates cold
measurements, eight warmup decisions and three warmed 32-decision trials.
`--cases` can also select rejection-heavy and accepted-action boundaries;
`--live-games` (alias `--env-counts`) and `--probe-capacities` select a smaller matrix.
It compares bounded background pools with the one-lane memory fallback, not a
historical trainer. For an explicitly requested post-fit check, add `--fit-cycle`:
each selected cycle arm collects complete one-second-cutoff games, performs four
fitting passes, then
measures collection again. This opt-in cycle does update diagnostic weights;
it is not formal training, mastery evaluation or a saved production checkpoint.
`--cycle-seconds` bounds each cycle (120 by default, at most 300).
Cycle defaults select 128 live games on mixed boards; `--fit-live-games` and
`--fit-cases` choose other cycle controls without changing original slot capacity.
Cold, warmed and post-fit results must remain distinct. It records simulator, probe,
encoding, online/EMA inference, synchronization, transfer, storage and presentation
timings, separate memory pools and coarse hardware samples. Subsecond trials can
have no hardware sample; null is not zero utilization. Rates count committed
active transitions, not padded capacity or simulated probes. Matching completed trials
are retained if the same evidence command is continued. Use a fresh output path
after changing implementation or settings. Viewer startup/history belongs to the
viewer integration controls, not this headless benchmark.

### Compact progress

The single `[logging]` table separates `progress_seconds = 15` machine-readable
snapshots from `terminal_progress_seconds = 60` routine terminal updates.
`terminal_mode = "compact"` is the only formatter. Phase changes, completed
collection/fitting, checkpoint saves, warnings, errors and interruption/completion
print immediately; unchanged progress and repeated report notices do not.
The terminal summary shows cohort progress, games/transitions, throughput,
collection/fitting times, optimizer step, available GPU/CPU utilization and the
latest warning. Interactive terminals reuse one line; redirected stdout and
`train.log` receive timestamped compact lines. Detailed telemetry remains in
`status.json`, `training-metrics.jsonl` and `hardware-metrics.jsonl`.

Collection adds `active N/cohort`, `decisions/game/s` and `active transitions/s`.
The interval uses only collection time and time-weighted live games, so a shrinking
cohort is not divided by the original slot capacity. Lifetime transition/time
totals remain separate. Phase/cohort changes and restored counters restart the
interval and exclude fitting, validation and recovery downtime; an unmeasured
interval is `n/a`, not zero. These host-only rates add no device synchronization
and do not alter the existing reporting cadences.

Transfer/per-probe and multiline dashboard implementations are removed. The
explicit performance-refresh flag, historical execution defaults and replay
sidecar annotation reader are also removed; current replay metadata is embedded.
The training CLI uses game-count budgets and `--eval-games`; the archived training
`--steps`/`--eval-interval` options and PPO report panels are removed. Benchmark
decision windows and independent CPU mathematical controls remain separate.

For the bounded objective comparison, use a separate five-pass initialization from
an existing verified archive, then run:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl.monitoring.objective_diagnostic `
  --checkpoint runs\diagnostic-demo\initialization.pt --output runs\objective-diagnostic
```

This runs one 16-game cohort with four passes and a 30-second simulator cutoff.
The optional checkpoint must use the current event-memory framework; omitting it
uses fresh weights. It reports event-write frequency, target errors, probe coverage,
memory saturation, fixed-history current-state Q sensitivity and throughput.
Output directories must be unused. Truncated diagnostic games are mechanical
controls, not curriculum mastery or a promised win-rate improvement.
