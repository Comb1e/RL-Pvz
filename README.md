# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer understands the public board, an action readout chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Bomb/attacker
stages learn placement objectives; Peashooter and full-game stages also predict
absolute post-action field/probe outcomes. Fresh stages have no threat-prediction
supervision or threat-summary injection. The policy receives public board facts;
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

The first three stages use `forecast_mode = "placement_only"` and their existing
placement objectives. Fresh Peashooter and normal stages use `forecast_mode = "action_outcomes"`:
no factual threat labels, loss, private cutoff tail or threat-summary injection.
Saved factual/snapshot runs retain their original workflow on `--resume`; weights-only
initialization uses the current settings. Peashooter keeps its prioritized lane
teacher and sets `training.peashooter.extra_actions = 0`. Normal stages retain two alternatives:
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

Fresh public-model training uses `training.performance.interval_backend = "packed_async"`.
Interval roots stay packed on the GPU, workers receive private device copies, and
small results transfer asynchronously. The main environment has priority; fitting
waits for all labels. Snapshots decode/spill only under storage pressure or during
recovery. The Peashooter teacher keeps its existing workflow. Missing settings in
older recoveries preserve the legacy backend.

`auxiliary_hybrid = true` additionally routes the teacher through the shared CPU/CUDA
queue. Hybrid remains opt-in. Both families respect the combined auxiliary VRAM cap.

The first forecast-enabled stage performs private forecast pretraining before real
collection. Fresh runs use `training.forecasts.pretrain_execution = "device_pipeline_v1"`:
larger GPU simulation waves share a queue, exact targets stay on the GPU, and a
bounded CUDA graph reduces fitting overhead. The 2,048 roots, sampled actions,
32-root optimizer panels, four passes and 256 updates remain unchanged. Preparation
runs ahead on one CPU thread. Source roots, workers and label scratch share the
auxiliary VRAM cap; the policy reserve covers fitting. Unfinished legacy resumes
retain their original private-pretraining execution. Exact simulation labels are
unchanged; padded CUDA fitting can introduce BF16 rounding differences from eager fitting.

Measure the complete section, including allocation, graph capture and cleanup:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_pretraining.py `
  --weights runs\compact-stages\attacker_pre\final.zip `
  --output runs\diagnostics\pretraining.json
```

The benchmark retains the normal 128-environment allocation and reports progress
transfers, simulated work, memory bounds and whether the 10-second target was met.
Use `--settings PATH` with a JSON list of forecast overrides to compare
`pretrain_queue_panels`, `pretrain_simulation_lanes`, `pretrain_simulation_ticks`
and `pretrain_graphs`. `--audit` hashes every panel's physical state, sampled actions
and targets; its extra downloads are included in the reported time.

Benchmark multiple execution settings on fixed cohorts before choosing tuning:

```powershell
.\.venv\Scripts\python.exe tools\benchmark_peashooter.py `
  --output runs\diagnostics\hybrid.json --envs 128 --seconds 90 `
  --auxiliary-workers 4 8 12 --auxiliary-ms 1 2 --modes hybrid
```

Compare `--modes sync interval interval-deferred` for the legacy backend, packed
interval execution, and the same capture path with execution deferred until drain.
Use a JSON `--settings` list to sweep `auxiliary_interval_lanes` (32/64/128) and
`auxiliary_cuda_target_ms` (1/2). `--collect-only` includes every required label and
the full drain, skipping fitting. Results report total seconds per environment,
spawn-decision latency and simulated work. `auxiliary_*` settings live under
`training.performance`; existing stage resource limits remain upper bounds.
The full hybrid modes `cuda hybrid deferred` remain available for separate teacher
experiments. Compare total collection-plus-labeling time, including drain, and
report cold preparation separately from warmed execution.

Record a real collection and draw two chronological CPU/GPU timelines:

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
The diagram shares elapsed timestamps across its two lanes, marks each loop's
start/end once, and names wait dependencies. New collection traces record CUDA
event timestamps; older inputs mark unavailable device timing explicitly. The
script writes PNG, SVG and a `.timeline.json` with the displayed phase boundaries.
These fixed-cohort benchmarks bypass CLI startup and private pretraining, disable
the viewer, and use the requested cutoff and optional `--weights`. The 60-second
random-weight control does not estimate a full initialized training cohort.

Time the actual CLI path, retaining its weights, pretraining, viewer and game cutoff:

```powershell
.\.venv\Scripts\python.exe tools\profile_training_workflow.py `
  --output runs\diagnostics\workflow.json --collect-seconds 180 -- train `
  --until-stage-complete --stage peashooter --gpu 0 --gpu-profile rtx4070-laptop `
  --init-from runs\compact-stages\attacker_pre\final.zip `
  --output runs\diagnostics\workflow-run
.\.venv\Scripts\python.exe tools\draw_collection_process.py `
  --input runs\diagnostics\workflow.json --output runs\diagnostics\workflow.png
```

Use an unused diagnostic run directory. The profiler stops between collection operations
after its wall window, saves recovery, and reports startup/collection/recovery
separately; one bounded labeling unit can extend the window. `--cprofile` adds
Python call profiling during collection. Fresh fork-enabled runs first pretrain
the forecast on 2,048 private roots with four passes; weights-only initialization
starts those counters afresh. The RTX 4070 Laptop profile disables compilation.

For a complete production cohort, add `--full-cohort --process-trace` before
`-- train` and allow a sufficiently long collection window, such as
`--collect-seconds 7200`. This retains the configured cohort size and cutoff,
finishes all required labels and four fitting passes, then saves recovery and
stops. The same drawing command includes startup, collection, label drain and
fitting, with measured CUDA events for collection and fitting. Instrumented wall
times include tracing overhead; checkpoint/report cleanup is reported separately.

For bottleneck diagnosis, add `--kernel-profile` to capture bounded CUPTI kernel
windows and Python call profiles, then summarize exclusive host time and both
batch-call / board-decision costs:

```powershell
.\.venv\Scripts\python.exe tools\report_training_hotspots.py runs\diagnostics\workflow.json `
  --kernels runs\diagnostics\workflow.kernels.json --output runs\diagnostics\workflow.hotspots.json
```

Use separate unprofiled repetitions for performance comparisons. Kernel-busy ratios
describe the sampled windows; CUDA event spans include launch gaps. The tracer uses
weak ownership so released simulation scratch stays released during fitting.
An optional `--stop-file PATH` requests a checkpoint at the next bounded operation.
Fresh execution settings batch teacher uploads, poll pinned progress buffers while
preparing roots, batch exact factual targets, bucket decoder captures, and gather
tested outcome tokens in arrays. Their switches live in `training.performance`.
Set `collection_tile_buckets = false` for bitwise fixed-weight comparisons:
padding changes BF16 batch rounding, while exact target construction is unchanged.

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
