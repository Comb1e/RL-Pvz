# Recording and training

Install with the [quick start](../README.md). Run commands from the project root.
The default model throughout recording, initialization, training and evaluation
is the entity Transformer–LSTM (`entity_v1`, `transformer_lstm_q_v2`).
A public observation contains an integer `[n,11]` entity matrix and 18 global
values; each retained entity becomes a learned 32-dimensional vector.

## Record and initialize

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human-1000.jsonl --replay runs\human-1000.pvzdemo `
  --output runs\human-init
```

Recording uses one easy-stage attempt, default seed 1000. Choose unused replay,
archive and sidecar paths. Pause retains queued actions; restart and stage
switching are disabled. Closing early leaves an incomplete recording.

Initialization checks the manifest, native replay hash, every observation,
proposed action, acceptance result, duration, reward and terminal state before
fitting. Missing manifests, unsupported protocols and configuration mismatches
have separate errors. Record a new demonstration or initialize from scratch.
Old model weights and aggregate-observation archives require fresh initialization;
there is no migration. Existing run files and recordings remain on disk.

Use the recording's `--config` when it was customized. `--device cpu` is the
initialization default; `--device cuda` changes execution without changing archive
verification. `--passes`, `--learning-rate`, `--gradient-clip` and `--seed` override
fitting settings. Time budgets are configuration settings only: initialization
uses `training.demo.time_budget_minutes` (30 by default). Defaults are 5 passes,
learning rate 0.0003, clip 0.5 and 256-decision chunks. Checkpoints are published
only after complete passes, alongside curves, coverage and verification reports.

## Autonomous training

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --init-from runs\human-init\initialization.pt --output runs\human-trained
```

`--init-from` accepts demonstration `.pt` or compatible autonomous `.zip` weights.
It starts fresh Adam state, counters, curriculum and RNGs. Omitting both
`--init-from` and `--resume` starts a fresh recurrent model. No incompatible source
silently falls back to random initialization. Metadata records model family,
initialization type, source SHA-256 and parameter differences.

Without `--config`, autonomous resume uses the checkpoint's saved configuration
plus missing execution defaults. A demonstration initialization uses its saved
structural metadata only and always takes execution settings from the current
`train.toml`. An explicit configuration must preserve the engine,
observation encoding, action semantics, rewards and network structure. Budgets
and output settings can differ. CPU autonomous training is unsupported.

The curriculum is **easy → standard → shared**. Easy uses only easy games;
standard mixes easy/standard equally; shared mixes easy/standard/hard at
20%/40%/40%. Mastery requires the configured held-out gates and residency; promotion changes future
resets only. `--stage easy --until-stage-complete` requests an unbounded run
until that stage passes. Use it only intentionally.

Up to 128 games form a cohort at fixed weights. Finished slots stay inactive;
the final cohort is limited to the remaining requested games. The recurrent
collector retains the selected proposal for the trajectory and complete-return
target. The next decision receives the resolved previous action: the proposal
when accepted, or wait (`0`) with `accepted=false` and the simulator's duration
when rejected. The rejection reason remains diagnostic metadata.

Each of four default fitting passes recomputes recurrent states from episode
starts. State carries across 256-decision chunks with gradients detached at
chunk boundaries. `--batch-size` is the decision budget: 1,024 means four
256-decision sequences; it must be a positive multiple of chunk length, at most
1,024. Padding and inactive slots have no loss. Chunk gradients accumulate into
one clipped Adam update per whole-cohort pass, with equal total weight for each
nonempty wait/plant/dig group.

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
`training.demo.passes = 5` and autonomous `training.n_epochs = 4` are independent.

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

Older entity checkpoints can resume without retraining. To apply the current
execution defaults while keeping their learning parameters:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\human-trained\interrupted.zip --output runs\human-trained `
  --refresh-performance
```

The flag is for autonomous resume; `--init-from` already applies current
execution settings automatically because demonstration checkpoints are
weights-only artifacts. This refresh cannot change passes, epochs, batch size, recurrent chunk length,
rewards, curriculum or architecture. Without it, saved precision is retained;
checkpoints lacking precision metadata use FP32. Precision changes begin when
the unfinished pass restarts. A nonfinite BF16 pass retries wholly in FP32;
only successful whole passes count as updates. Effective precision and fallback
reasons are saved. A nonfinite FP32 pass or exhausted allocation fallback fails
explicitly. See [precision and throughput controls](math/training-throughput.md).

For a bounded four-pass throughput check with fixed model dimensions:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl.monitoring.throughput_benchmark `
  --scope synthetic --label current --output artifacts\throughput-synthetic.json
.\.venv\Scripts\python.exe -m pvz_rl.monitoring.throughput_benchmark `
  --scope collection --label current --output artifacts\throughput-collection.json
```

The collection check uses one-second cutoffs, three warmed repetitions, and
viewer-on/off runs. It is a mechanical throughput check, not formal training.
The [recorded comparison](evidence/training-throughput-v030.json) includes raw
trials, phases, memory and measurement limits.
