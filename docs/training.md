# Recording and training

Install with the [quick start](../README.md). Run commands from the project root.
The default model throughout recording, initialization, training and evaluation
is Transformer–LSTM with 294 public observation values (`event_v8`).

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
have separate errors. The existing default-profile recording remains verifiable
through an exact configuration-digest migration for curriculum removal; changes
to other settings are not covered by that migration.

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

The event-memory alternative is explicit:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --config configs\event-memory.toml --output runs\event-memory
```

Historical event-memory weights remain loadable. Retained curriculum stages are
translated by name on resume. A checkpoint inside the removed lesson (or with
unfinished lesson trajectories) must use `--init-from`; its private simulator
state cannot be resumed after lesson removal.

It uses `event_v7`, its own optimizer protocol and 10%→0.1% tile exploration
over 3,000 stage games. Cross-family weight transfers are rejected.

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
