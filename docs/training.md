# Recording and training

The current objective is `complete_return_probe_v3`. Rejected plants and empty digs
use the small shared penalties in `train.toml`; development is multiplied only in
learning targets. A zombie entering either of the two columns nearest the house is
charged once per boundary. Victory time is shaped only for wins, relative to the
median duration of the current completed cohort.

Each decision probes two unselected branches and up to two alternative tiles. The
EMA teacher supplies fixed one-step targets. Probes run from scratch simulator state
and never alter the behavior game. Cohort reward finalization occurs before complete
returns and fitting, and is recoverable without applying time rewards twice.

Install with the [quick start](../README.md). Run commands from the project root.
The default model throughout recording, initialization, training and evaluation
is the entity Transformer–LSTM (`entity_v1`, `transformer_lstm_q_v3`).
A public observation contains an integer `[n,11]` entity matrix and 18 global
values; each retained entity becomes a learned 32-dimensional vector.

## Record and initialize

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human-1000.jsonl --replay runs\human-1000.pvzdemo `
  --output runs\human-init-events
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
  --init-from runs\human-init-events\initialization.pt --output runs\human-trained
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
settings are supplied by the new run. Changed settings therefore do not require a
new demonstration initialization. Autonomous weight transfer keeps its stricter
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
the final cohort is limited to the remaining requested games. The recurrent
collector retains the selected proposal for the trajectory and complete-return
target. History retains only gross sunlight gain/spend, zombie spawn/defeat/removal
and plant addition/removal events, plus resulting board context and time since the
last write. Rejection alone never advances memory. Current board features still
reach both Q heads every decision. Proposal, execution, acceptance, duration and
rejection reason remain learning/journal evidence.

Each of four default fitting passes recomputes recurrent states from episode
starts. State carries across 256-decision chunks with gradients detached at
chunk boundaries. `--batch-size` is the decision budget: 1,024 means four
256-decision sequences; it must be a positive multiple of chunk length, at most
1,024. Padding and inactive slots have no loss. Chunk gradients accumulate into
one clipped Adam update per whole-cohort pass, with equal total weight for each
nonempty wait/plant/dig group. Within each group, accepted and rejected decisions
receive 50% each when both exist; the sole present outcome otherwise receives
full group weight. Probe losses apply the same split inside each head/branch.

Tile exploration decays from 50% to 1% over 5,000 completed stage games and stays
fixed during each cohort. The ten-way branch choice (wait, eight plants and dig)
is always greedy; the budget is spent only when choosing the selected branch's
tile. Evaluation disables the tile coin. Plant tiles use occupancy only;
affordability and cooldown do not hide a branch. Dig tiles are unrestricted; a
full-board plant proposal targets tile zero and receives the simulator's
rejection outcome and configured invalid-plant penalty. The viewer and action
journal show that proposal while labeling its resolved execution as an automatic
wait.

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
interrupted fitting pass restarts from its beginning with cleared gradients;
completed optimizer updates are retained. `interrupted.zip` is the recovery
artifact after interruption; `latest.zip` and `final.zip` are written on normal
completion. In-place resume appends training history. Budgets may be extended;
changing model structure or the active collection shape is rejected.

Evaluation accepts either checkpoint format and uses the shared stateful runner.
CUDA batches reset only replaced episode slots. A configuration with
`simulation.backend = "cpu"` selects CPU evaluation while preserving the model
contract. Recording traces are checked against the CPU simulator before export.
Choose a fresh evaluation directory. Validation and test seeds remain separate.

## Training throughput and precision

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
`unavailable` before reverting to eager execution. The Windows CUDA build uses
the `cudagraphs_owned` backend: activations and gradients are copied out of
reusable graph buffers, preserving them through delayed backward and whole-pass
accumulation. It does not trace simulator pybind objects or invoke Inductor's
max-autotune SM check. Other platforms use Inductor when available. A backward
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

The collection check executes no optimizer steps: 128 environments, eight warmup
decisions and three 32-decision trials on sparse, mixed, crowded, rejection-heavy
and accepted-action snapshots. It compares two-lane execution with the current
one-lane memory fallback, not a historical trainer. It records simulator, probe,
encoding, online/EMA inference, synchronization, transfer, storage and presentation
timings, separate memory pools and coarse hardware samples. Subsecond trials can
have no hardware sample; null is not zero utilization. Matching completed trials
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
