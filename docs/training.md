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
uses `training.demo.time_budget_minutes` (30 by default). Defaults are 20 passes,
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

Without `--config`, loading uses the checkpoint's saved configuration plus missing
execution defaults. An explicit configuration must preserve the engine,
observation encoding, action semantics, rewards and network structure. Budgets
and output settings can differ. CPU autonomous training is unsupported.

The curriculum is **easy → standard → shared**. Easy uses only easy games;
standard mixes easy/standard equally; shared mixes easy/standard/hard at
20%/40%/40%. The saving lesson, its stage and rehearsals are removed. Mastery
requires the configured held-out gates and residency; promotion changes future
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

The single `src/pvz_rl/data/train.toml` controls both profiles. `encoding.max_entities`
sets the entity cap; model width, encoder microbatch (16), token budget (4,096),
and attention query chunk (64) are configured under `policy`. Mowers precede
plants, then zombies, then projectiles. On overflow projectiles are omitted first,
then zombies, then plants; all five mowers remain. Zombies closest to the house
are retained first. Ordering uses only public fields, so changing entity IDs
cannot change the chosen records. The network is invariant to permutations of
those retained rows. Counts in the global vector include omitted entities.

The viewer's action history records retained entities and omitted plants/zombies/
projectiles for the decision input. CPU/CUDA step info also reports omission
counts. A cap discards individual information above that limit; total counts do
not reconstruct it. See [schema and normalization](math/entity-inputs.md).

Only collating adds padding. Every attention layer masks padded keys. The global
and 45 tile query tokens remain valid even on an empty board. Training stores
real entity rows in separate append-only slabs, with offsets/counts in transition
metadata. Both slabs share `training.storage.ram_gib`; excess blocks spill to disk.

Encoder microbatches shrink with sequence width. Attention uses scaled dot-product
attention and an exact query-chunk fallback; each query still sees all valid keys.
Activation checkpointing recomputes encoder activations during backward. These
settings bound encoder working memory without changing chronological LSTM order,
whole-cohort loss weights, or whole-pass optimizer updates. Allocation failures
remain explicit; the program never silently reduces the configured entity cap.

To measure 256/512 caps on a device (short synthetic fitting, no formal training):

```powershell
.\.venv\Scripts\python.exe -m pvz_rl.monitoring.entity_benchmark `
  --output artifacts\entity-benchmark.json
```

The report separates inference from fitting and reports Torch allocated/reserved
memory. These timings exclude simulation and do not measure learning quality.

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
