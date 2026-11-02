# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer understands the public board, an action readout chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Normal training
also predicts how far the currently visible zombies will reach in the actual
continuation. Labels never override actions; validation uses the greedy network.

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
removed; existing datasets and recordings remain on disk. The shared-encoder
architecture starts fresh: old model weights and autonomous checkpoints are
incompatible. Compatible weights transfer with `--init-from`; interrupted runs
resume with `--resume` and preserve optimizer updates and unfinished reach labels.

Start autonomous training when ready:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --seed 101 `
  --init-from runs\human-init\initialization.pt --output runs\my-run `
  --gpu 0 --gpu-profile rtx4070-laptop
```

Default progression is
`cherry_bomb → cherry_bomb_obstacle → attacker_pre → peashooter → easy → standard → shared`.
Use `--stage NAME --until-stage-complete` for one stage, or `--games N` for a
completed-episode budget. Explicit normal stages skip preliminary stages. Omit
`--init-from` to start with random weights.

Tune [train.toml](src/pvz_rl/data/train.toml): stage settings live under
`training.cherry_bomb`, `training.cherry_bomb_obstacle`, `training.attacker_pre`,
`training.peashooter` and `training.normal`; architecture dimensions live under
`policy`. Preliminary stages retain their automatic placement targets and train
the shared encoder/decoder while leaving the action readout untouched. Normal
cohorts jointly fit action/tile returns and reach cross-entropy, using four passes.
`training.normal.reach_prediction_weight` defaults to 0.1; zero disables its
gradient. Reach categories are columns 1–9 plus “did not enter”; empty target
lanes have no auxiliary loss. Focused live viewing and reports distinguish
predicted penetration from factual outcomes.

Validation uses fresh random panels, saved for interrupted retries. Full fitted
cohorts must strictly exceed the configured qualification threshold before a due
validation launches. Cherry Bomb and Peashooter use their configurable mastery
thresholds; other fresh stages require perfect validation. Completed runs retain
portable `final.zip` weights and visualizations; unfinished runs retain recovery.

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
- [Detailed documentation](docs/architecture.md) (historical architecture notes)
- [Research references](docs/references.md)
