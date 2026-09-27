# PVZ plant research

Release 0.27.0 records one human easy game, verifies its native replay, and fits a
Transformer–LSTM two-level Q policy from complete reward-to-go. The shipped
observation protocol is `event_v8` (294 public values, including eight card
cooldowns). Formal autonomous training is a separate explicit workflow.

## Requirements

Python 3.12, Git, and the pinned game package. The live demo needs pygame; CUDA
is required only for the later autonomous collector.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

Bootstrap installs the pinned game package. Source and hash details are in [the
engine lock](src/pvz_rl/data/engine-lock.json).

## First useful command

Open the normal-speed easy game with the default seed 1000:

Use unused replay/archive paths. Existing files (including empty archives and
manifest/history sidecars) are rejected. Each recording permits one stage attempt;
restart and stage switching are disabled.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
```

After the game reaches a natural win or loss, run the one-demonstration fit:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human-1000.jsonl --replay runs\human-1000.pvzdemo `
  --output runs\human-init
```

The command writes `initialization.pt`, curves, per-branch coverage and a replay
verification report. An interrupted recording stays incomplete and is rejected
by initialization; no terminal result is invented. The result fits one
demonstration and does not establish generalization or mastery. The demo profile
is [configs/demo.toml](configs/demo.toml); it retains the recurrent policy and
50%→1% tile exploration settings over 5,000 games. The existing CUDA collector
keeps its separate [training profile](configs/train.toml). Neither command above
starts that collector. Validation disables exploration.

In the training viewer, press **F** to follow the latest action in all visible
grid panels, or in the focused panel. Historical selection and paging are cleared
without resetting horizontal table scrolling.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pvz_rl replay runs\human-1000.pvzdemo
```

- [Architecture and workflows](docs/architecture.md)
- [Training, resume and evaluation](docs/training.md)
- [Viewer controls](docs/live-view.md)
- [Math controls](docs/math/complete-game-learning.md)
- [Collision mathematics](docs/math/collision-audit.md)
- [Inspected sources](docs/references.md)
- [Iteration history](docs/iteration.md)
