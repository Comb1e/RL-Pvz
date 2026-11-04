# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer understands the public board, an action readout chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Every autonomous
stage also learns lane-threat predictions. Peashooter and full-game stages predict
absolute post-action field/probe outcomes. The policy receives public board facts;
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
removed; existing datasets and recordings remain on disk. `--init-from` starts with
fresh optimizer and counters. Legacy shared fork-model weights transfer their
public-input projection and compatible weights, with a fresh absolute-outcome
readout; earlier model families remain incompatible. `--resume` preserves the
saved architecture, optimizer, pending labels and committed fitting passes.

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

Start fresh Peashooter training from the compatible attacker weights:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage peashooter --until-stage-complete `
  --init-from runs\compact-stages\attacker_pre\final.zip --output runs\peashooter-new
```

No seed is required: initialization is random and every training cohort draws an
independent random seed. Pending seeds are saved before preparation so recovery
recreates the same unfinished boards. `--seed` optionally fixes initialization;
`training.random_cohort_seeds = false` also restores reproducible scenario streams
for controlled diagnostics. Existing recoveries retain their saved seed behavior.

Tune [train.toml](src/pvz_rl/data/train.toml): stage tables own their placement
settings and independent `threat_prediction_weight` / `effect_prediction_weight`.
Normal stages jointly fit n-step Q targets; preliminary stages preserve their
placement losses and freeze the action readout. Four fitting passes, greedy tiles
and normal species exploration remain in use.

The first three stages use snapshot threat constraints and their existing placement
objectives. Peashooter keeps its prioritized lane teacher and sets
`training.peashooter.extra_actions = 0`. Normal stages retain two alternatives:
the relevant branch's second-highest-Q legal tile and one uniformly selected
remaining legal action. Waits/rejections sample alternatives at probability
`0.005`; legal non-Wait proposals always receive supervision.

Fresh runs use `collection_protocol = "wait_trajectory_v3"` and
`effect_objective = "absolute_outcome_v3"`. Real uninterrupted Waits, including
rejections, supply the field trajectory for auxiliary labels. At first clearance,
one private probe is anchored and reused through later empty-board Waits.
Planting/Dig, actual spawning and episode boundaries close an interval; any
unfinished suffix continues from its saved state. Field targets remain specific
to each decision. Collection proceeds with pending labels, which finish before
fitting. Inference and greedy validation run no field/probe simulations.

Public-model training can run interval outcomes and the Peashooter teacher through a
shared hybrid queue with `training.performance.auxiliary_hybrid = true`.
The main environment has GPU priority; short asynchronous
CUDA chunks and persistent CPU processes finish exact labels in the background.
Both families share the auxiliary VRAM cap. Pending snapshots spill within the
configured storage limit, and fitting waits for every required label. Resume can
refresh these execution settings without changing saved objectives or targets.

Benchmark multiple execution settings on fixed cohorts before choosing tuning:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_peashooter.py `
  --output runs\diagnostics\hybrid.json --envs 128 --seconds 90 `
  --auxiliary-workers 4 8 12 --auxiliary-ms 1 2 --modes hybrid
```

Compare `--modes sync cuda hybrid deferred` for synchronous execution,
asynchronous CUDA, hybrid workers and a collection-latency control that defers
simulation until collection ends. Results report total seconds per environment,
spawn-decision latency and simulated work. `auxiliary_*` settings live under
`training.performance`; existing stage resource limits remain upper bounds.
The warmed 4/8/12-worker screen selected four CPU workers and 2 ms chunks among
asynchronous settings. Small cohorts did not beat synchronous execution, and the
5% decision-latency target was unmet; use total seconds/env to compare your workload.
The 128-env, 60-second diagnostic collected and labeled in 130.0 s with hybrid
execution versus 98.9 s synchronously. Hybrid remains opt-in; synchronous is the
default until the coordinator and label drain improve.

Record a real collection and draw its CPU/GPU states and wait causes:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_peashooter.py `
  --output runs\diagnostics\process.json --envs 128 --seconds 60 `
  --modes hybrid --auxiliary-workers 4 --auxiliary-ms 2 `
  --warmup 0 --repetitions 1 --eager --process-trace
.\.venv\Scripts\python.exe tools\draw_collection_process.py `
  --input runs\diagnostics\process.hybrid-4-0.process.json `
  --output runs\diagnostics\process.png
```

CPU call spans, GPU stream spans, summed worker/job times and collection wall time
are reported separately because concurrent work overlaps.

Private pretraining retains its sampling procedure and separate counters, using
public-only inputs and absolute targets. Saved recoveries retain their original
supervision and collection behavior. Reports separate borrowed real ticks,
private field/probe work, interval reuse, alternatives and labeling time. Tune
resource caps under `training.performance` / `training.forecasts`.

Validation uses fresh random panels, saved for interrupted retries. Full fitted
cohorts must strictly exceed the configured qualification threshold before a due
validation launches. Cherry Bomb and Peashooter use their configurable mastery
thresholds; other fresh stages require perfect validation. Completed runs retain
portable `final.zip` weights and visualizations; unfinished runs retain recovery.
Temporary cohort files are removed after fitting; interrupted work is preserved
in recovery checkpoints.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
.\.venv\Scripts\python.exe -m pvz_rl replay runs\human.pvzdemo --watch --speed 2
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\my-run --report-only
```

Exactly three smoke tests cover CLI startup, CPU inference and bounded CUDA
learning/reload. They establish workflow availability, not mastery. Report-only
visualization loads no policy and starts no training.

- [Code structure and test ownership](AGENTS.md)
- [Architecture and workflows](docs/architecture.md)
- [Research references](docs/references.md)
