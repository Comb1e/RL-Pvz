# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer understands the public board, an action readout chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Every autonomous
stage also learns visible-zombie reach. Labels never override actions; validation
uses the greedy network.

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
the shared encoder/decoder and reach heads while leaving the action readout
untouched. Normal cohorts jointly fit action/tile returns and reach cross-entropy,
using four passes. Each stage table has its own `reach_prediction_weight`, default
0.1; zero disables its gradient. Reach categories are columns 1–9 plus “did not
enter”. With `training.reach_cleared_lanes = true`, once all zombies have spawned,
lanes without non-headless zombies receive zero-penetration targets. Other empty
target lanes have no auxiliary loss. Clearance uses the pre-action board, including
off-board zombies; it never changes actions or creates plants. Scenario stages
finish reach observation at natural engine victory or the first home entry. Bomb and attacker
exercises use Wait after their one placement; Peashooter uses Wait after its
gameplay cutoff. Private continuation has no simulated-time cutoff and supplies
observed reach without changing recorded task results, timeout outcomes or rewards.
Wall-time limits save unfinished work; old recoveries retain their saved endpoints
and cleared-lane setting (missing means disabled). Normal games keep their existing
factual reach. These continuations add no episodes
or policy calls. Native and browser game viewers show predicted penetration beside
each live board, including in grid mode. The map is timestamped to the latest tile
decision and stays visible during Wait; it needs no additional model calls. Focused
viewing and reports also show observed supervision targets and mark cleared lanes.

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
