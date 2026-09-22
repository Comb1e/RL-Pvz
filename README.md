# PVZ plant-placement research

Train **one shared policy for easy, standard, and hard** using ordinary planting,
digging, and waiting actions. Research **0.7.1** uses the pinned Lawn Lab game
**1.3.0**, simulation **1.0.0**. CUDA simulation and learning run on the GPU;
Python simulation remains the reference and explicit fallback.

The latest plant-reward profile is experimental. Short checks improved lesson
learning, but reliable normal-game improvement has not been demonstrated. See
[measured results](docs/validation.md#learning-results) before comparing profiles.

## Install

Use this checkout, `E:/Projects/Tower-Defence-AI/PVZ-plant`, and its single `.venv`.
The existing environment already contains the pinned simulator. To update only
the research package and check availability:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

For a fresh Windows installation, install Python 3.12 and Git. The installer
requires game commit `8861824df6893a34c2cd4df7f9b68613376d7964`. The native viewer
checkouts contain newer reader fixes, so use a temporary source copy under
`build/` instead of switching either viewer checkout:

```powershell
git clone --no-checkout E:\Projects\pvz-cuda-work build\pvz-install-source
git -C build\pvz-install-source checkout --detach 8861824df6893a34c2cd4df7f9b68613376d7964
.\tools\bootstrap.ps1 -GameRepo .\build\pvz-install-source -Cuda
```

The installer stages a verified archive and installs the game non-editably.
It uses the same research `.venv`; `build/pvz-install-source` is only installation
input. CUDA wheels include PyTorch 2.8.0+cu128, CuPy 13.6.0, runtime 12.8.90 and
NVRTC 12.8.93; no global CUDA Toolkit or Visual Studio is required. For CPU-only
installation, omit `-Cuda` and use `--simulator cpu --device cpu` when training.
Missing CUDA produces an error rather than a silent fallback. FFmpeg/libx264 is
needed only for optional MP4 export; compact demos do not require it.

## Train

Start a fresh model with the latest experimental rewards:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --config configs\sc2-plant-rewards.toml --seed 101 `
  --games 10000 --max-minutes 120 --output runs\sc2-plant-rewards-101
```

This uses **128 parallel GPU games × 128 decisions per game per rollout**.
Completed games control progress and scheduling. Legal planting/digging is
immediate; waiting or a rejected action advances one tick. All three difficulties
share the same policy and optimizer throughout the curriculum.

| Recipe in `configs/` | Purpose |
|---|---|
| `cuda.toml` | Flat-policy baseline with GPU simulation; default for unconstrained new runs |
| `baseline.toml`, `tactical.toml`, `grouped.toml`, `pure-rl.toml` | Earlier observation/grouping/teaching controls; retain their explicit CPU simulation settings |
| `sc2-inspired.toml` | Spatial grouped policy, balanced exploration, mastery lessons |
| `sc2-long-horizon.toml` | SC2 profile with gamma 0.9999 instead of 0.999 |
| `sc2-efficient.toml` | SC2 plus choice-state actor updates, GAE 0.999 and minibatches of 1,024 |
| `sc2-plant-rewards.toml` | Efficient profile plus offensive/eaten-plant rewards and a trainable initial dig bias |

[Research design](docs/research.md) specifies rewards, lessons, comparisons and
profile limitations. Changed learning profiles require fresh models. Copy a TOML
file to customize it and pass `--config`; defaults and legacy settings are never
inferred from another profile's name.

Useful overrides: `--n-envs`, `--rollout-steps-per-env`, `--device`, `--simulator`,
`--games`, `--eval-games`, and `--max-minutes`. Total rollout size is their
parallel-game/step product; conflicting `--rollout-size` settings are rejected.
CUDA supports masked, unmasked, sparse and mixed direct policies. Grouped policies
require masking; the scripted hybrid and modified combat rules use CPU simulation.

Normal validation defaults to every **1,000 completed games**, with 50 fixed
seeds per difficulty. `best.zip` maximizes their equal-weight mean win rate;
ties keep the earlier checkpoint. Lessons and shaped returns never select it.
A rollout may exceed the game target; its final PPO update still completes.

The 120-minute allowance includes startup, validation and presentation, reserving
15 minutes for finalization. It is cumulative across resume. Stops occur between
updates; in-flight work finishes safely. Pending evaluation/export is recorded,
and an unfinished teaching schedule is labeled `curriculum_incomplete`.

For a small integration check, use a new output directory:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --family diagnostic `
  --simulator cpu --device cpu --games 2 --n-envs 1 --rollout-size 64 `
  --batch-size 32 --eval-games 1 --validation-count 1 --output runs\smoke
```

This checks the pipeline with a restricted task; its result is not a full-game
benchmark. Installation and availability checks never start formal training.

## Resume

Resume into a new directory from `interrupted.zip`, or `latest.zip` if necessary:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --seed 101 `
  --resume runs\sc2-plant-rewards-101\interrupted.zip `
  --output runs\sc2-plant-rewards-101-resumed
```

Without `--config`, the saved configuration is restored. Repeat the original
condition, family and validation-count overrides if used. The game/time targets
remain total allowances, not additional budgets. Policy, optimizer, curriculum
and counters are restored; episodes restart, so rollout/RNG continuation is not
bit-for-bit. Learning settings and engine source pins must match. Runtime and
presentation settings may change. Old-pin models require their original environment;
archived statistics and recordings remain readable. Do not edit run metadata to
bypass compatibility checks.

## Logs, curves and demos

Progress appears every 15 seconds in the terminal and `train.log`: phase, games,
throughput, elapsed time, training ETA, recent 100-game statistics, curriculum,
attacker purchases, digging, kills and reward components. Mower-free lessons are
labeled. Episode records are saved without printing every episode.

```powershell
Get-Content runs\sc2-plant-rewards-101\train.log -Wait
# In another terminal:
.\.venv\Scripts\tensorboard.exe --logdir runs
Start-Process runs\sc2-plant-rewards-101\visualizations\index.html
```

| Output within a run | Contents |
|---|---|
| `metadata.json`, `config.json`, `status.json` | Settings, source/rule hashes, seeds, progress and stop reason |
| `training-episodes.jsonl`, `training-metrics.jsonl` | Episode details, rolling behavior, throughput and post-update PPO metrics |
| `learning-curve.jsonl`, `validation/` | Fixed-case validation performance and episode records |
| `best.zip`, `best.json`, `latest.zip`, `final.zip` | Selected, latest validation, and final policy checkpoints |
| `visualizations/index.html`, `*-curves.png` | Offline report and charts; reload after validation |
| `visualizations/demos.json` | Demo paths, outcomes and shared checkpoint hash |
| `visualizations/games/attempt-*/replays/*.pvzdemo` | Three CPU-verified recordings from the same `best.zip` |
| `visualizations/status.json`, `visualizations/videos/*.mp4` | Export status and optional videos |

Demos use deterministic actions and validation seed 100000 for each difficulty,
including actual losses or truncations. Diagnostic runs produce only a diagnostic
demo. Report curves separate training from validation; missing data and resume
segments are explicit. Regeneration does not train:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\sc2-plant-rewards-101
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\sc2-plant-rewards-101 --videos
.\.venv\Scripts\python.exe -m pvz_rl replay PATH_TO_DEMO.pvzdemo --watch --speed 2
.\.venv\Scripts\python.exe -m pvz_rl replay PATH_TO_DEMO.pvzdemo --video artifacts\demo.mp4
```

Use a replay path from `demos.json` or the report. `--videos` and `--no-videos`
override configuration on `train`, `suite` and `visualize`. Default output is a
report plus compact demos; disabling videos retains demos. Video is H.264/yuv420p,
1280×820 at the engine's 20 ticks/s, with a two-second outcome hold. Configure
`[logging]` and `[visualization]` in TOML for cadence, output, dimensions or FFmpeg.
Export failures preserve checkpoints and can be retried.

For an incompatible-version error, use **`pvz-rl replay`**. It understands the
`pvz-rl/actions-v1` format for zero-time actions. Standalone readers CPU 1.2.2 and
CUDA 1.3.1 also support it; their `tools/watch_demo.ps1` can load source with this
existing Python environment. Do not replace the pinned training simulator just
to view a recording. Legacy JSON/gzip replays remain supported.

Viewer controls: Space pause, period single tick, arrows seek five seconds,
Home/End, R restart, I inspect, and draggable timeline. Supported speeds are
0.5, 1, 2, 4 and 8. Archived reports can reuse existing recordings but cannot
regenerate gameplay from incompatible checkpoints.

## Evaluate and develop

```powershell
.\.venv\Scripts\python.exe -m pvz_rl evaluate --baseline heuristic `
  --split development --count 10 --record --output runs\baseline-development
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\sc2-plant-rewards-101\best.zip `
  --split validation --count 10 --record --output runs\plant-reward-validation
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests tools
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build --no-isolation
```

| Document | Read it for |
|---|---|
| [Research design](docs/research.md) | Current rewards, policies, curriculum, seed splits and experiment commands |
| [Architecture](docs/architecture.md) | Data flow, state machines, CUDA and native game integration |
| [Validation](docs/validation.md) | Test records, learning results, GPU measurements and their limits |
| [References](docs/references.md) | Papers/projects actually inspected and source revisions |
| [Iteration history](docs/iteration.md) | Version changes, causes and remaining issues |

Use Conventional Commits and feature/fix branches; use a PR when a remote is
configured. Keep formal training and final-test evaluation explicit.
