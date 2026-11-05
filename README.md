# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer understands the public board, an action readout chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Every stage learns
absolute post-action field/probe outcomes alongside its placement or Q objective.
The policy receives public board facts;
exact simulated outcomes are auxiliary targets. Learned forecast summaries can
feed the Q heads. Labels never override actions; validation uses the greedy network.

## Requirements

Python 3.12, Git, the pinned game checkout, and an NVIDIA GPU with native BF16
support for training. Human recording and the native viewer also require pygame.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

Linux/WSL or Git Bash:

```bash
bash tools/bootstrap.sh --game-repo ../pvz-cuda-work --python python3.12
```

Installers verify the [engine lock](src/pvz_rl/data/engine-lock.json) and leave the
engine checkout unchanged.

## First useful command

Record human gameplay into unused paths and finish with a natural win or loss:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human.pvzdemo --archive runs\human.jsonl
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human.jsonl --replay runs\human.pvzdemo `
  --output runs\human-init --device cuda
```

Demo initialization verifies the replay before fitting. Manual tile annotation is
removed; existing datasets and recordings remain on disk. `--init-from` accepts
outcome-only checkpoints from this version and starts with fresh optimizer and
counters. All earlier checkpoints are incompatible. `--resume` preserves the
optimizer, pending labels and committed fitting passes of a same-version run.

Start autonomous training when ready:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --init-from runs\human-init\initialization.pt --output runs\my-run `
  --gpu 0 --gpu-profile rtx4070-laptop
```

Default progression is
`cherry_bomb → cherry_bomb_obstacle → attacker_pre → peashooter → easy → standard → shared`.
Use `--stage NAME --until-stage-complete` for one stage, or `--games N` for a
completed-episode budget. Explicit normal stages skip preliminary stages. Omit
`--init-from` to start with random weights.

After training the attacker stage with this version, initialize Peashooter from
its compatible weights:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage peashooter --until-stage-complete `
  --init-from runs\compact-stages\attacker_pre\final.zip --output runs\peashooter-new
```

No seed is required: initialization is random and every training cohort draws an
independent random seed. Pending seeds are saved before preparation so recovery
recreates the same unfinished boards. `--seed` optionally fixes initialization;
`training.random_cohort_seeds = false` also restores reproducible scenario streams
for controlled diagnostics. Existing recoveries retain their saved seed behavior.

Tune [train.toml](src/pvz_rl/data/train.toml): each stage independently sets
`static_field_prediction_weight` (field depth, breach time and lane loss) and
`probe_prediction_weight` (probe depth). Equal values of `0.1` preserve the prior
loss; coefficients scale terms before the existing known-term average and warmup.
Setting either to zero removes only its loss contribution; shared parameters
still receive the other losses. Saved outcome-only runs with one combined weight
retain it for both groups on resume.
The first three stages use `placement_outcomes`: each proposed placement receives
one exact field/probe auxiliary target alongside its existing placement teacher. Rejected placements
receive the engine's no-op continuation. They add no Wait target or alternatives;
outcomes guide shared feature learning without entering tile scores in these stages.
Their memory budget charges occupied proposal snapshots and workers; Wait-interval
boundary buffers are reserved only for Peashooter/normal collection.

Peashooter and normal stages use `action_outcomes`. Peashooter keeps its prioritized
lane teacher and zero alternatives. Normal stages retain the branch runner-up and
one random legal alternative; effective Waits sample them with probability `0.005`.
Four fitting passes, normal n-step Q targets and species exploration remain in use.

The model uses public entities and 18 global values. It predicts post-action field
depth, breach time, lane loss and probe depth. Exact simulations supply auxiliary
targets only. Factual prediction and separate private forecast pretraining are removed;
stage collection starts directly after ordinary preparation. Previous checkpoints
are incompatible with inference, `--resume` and `--init-from`; start a fresh run.
New outcome-only checkpoints support those loading modes within this version.

`collection_protocol = "stage_action_outcomes_v4"` routes single-placement exercises
and real Wait trajectories separately. In Peashooter/normal, uninterrupted Waits and
rejections supply field ticks. One probe is anchored at first clearance and reused
until planting/Dig, actual spawning or an episode boundary closes its interval.
Interrupted fields continue from owned roots. Field labels exclude earlier
penetration and rebase breach times for each decision. Labels finish before fitting.
Inference and greedy validation run no field/probe simulations.

`training.performance.interval_backend = "packed_async"` keeps roots packed on the
GPU and transfers small results asynchronously. Main decisions have GPU priority.
Snapshots decode/spill under storage pressure or during recovery. The Peashooter
teacher keeps its workflow; `auxiliary_hybrid = true` additionally routes it through
the shared CPU/CUDA queue. Both families respect the combined auxiliary memory cap.

Validation uses fresh random panels after fully fitted cohorts. Completed runs
retain portable `final.zip` weights and visualizations; unfinished runs retain
recovery checkpoints and their exact pending work.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
.\.venv\Scripts\python.exe -m pvz_rl replay runs\human.pvzdemo --watch --speed 2
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\my-run --report-only
```

The [collection benchmark](tools/benchmark_peashooter.py) compares execution
settings by total time per environment. The [workflow profiler](tools/profile_training_workflow.py)
records an actual CLI cohort; the [diagram tool](tools/draw_collection_process.py)
draws chronological CPU/GPU timelines with wait dependencies. Use each tool's
`--help` for bounded controls and trace options.

Exactly three smoke tests cover CLI startup, CPU inference and bounded CUDA
learning/reload. They establish workflow availability, not mastery. Report-only
visualization loads no policy and starts no training.

- [Code structure and test ownership](AGENTS.md)
- [Architecture and workflows](docs/architecture.md)
- [Research references](docs/references.md)
