# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer understands the public board, an action readout chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Every autonomous
stage also learns lane-threat predictions. Full-game stages learn baseline-relative
action effects. Labels never override actions; validation uses the greedy network.

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
removed; existing datasets and recordings remain on disk. Older architectures are
incompatible; current fork-model weights remain compatible. Transfer with
`--init-from`; interrupted runs
resume with `--resume` and preserve optimizer updates and unfinished forks, hindsight labels and fitting passes.

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

The first three stages use `forecast_mode = "snapshot_rules"`: once all scenario
zombies have appeared, empty lanes target zero threat. The zombie lane remains
unknown, except that a ready mower targets zero home danger. These stages use a
zero baseline-input sentinel, run no forecast forks or effect pretraining, and
freeze the effect/null heads. Their placement oracles and objectives are unchanged.

Peashooter and normal stages retain static field/probe forks, sampled effect
labels and factual threat targets. Waits and rejected actions still receive
hypothetical alternatives. Forks neither choose actions nor consume the real RNG;
full-game inference and validation also compute baseline inputs. Tune execution
limits under `training.performance` and pretraining/TD schedules under
`training.forecasts`. Private pretraining starts at the first fork-enabled stage.

Threat targets preserve exact windows and permanently safe lanes, including prior
penetration. Losses/cutoffs leave unresolved futures partially labeled; Peashooter
retains its private Wait tail. Effect fitting separates change occurrence from
conditional signed change. Tune `effect_objective` and
`effect_conditional_weight` under `training.forecasts`. Reports separate these
losses, detection/calibration, fork cost and outcomes. Viewers show raw factual
threat beside predicted static field/probe maps; unknown targets stay grey.

Saved recoveries retain their original supervision. Compatible `--init-from`
uses current settings with a fresh optimizer. For bounded seed-matched ablations,
see `python -m pvz_rl.monitoring.forecast_ablation --help`.

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
