# PVZ plant research

Record a human Plants vs. Zombies game, initialize a Transformer–LSTM Q policy,
then train and evaluate the same model autonomously. The pinned simulator runs
at 100 Hz; policies receive only public observations.

Direct policy masks describe board occupancy only. The ten-way Q selector always
compares wait, all eight plant species and dig; an unaffordable or cooling-down
winner remains the submitted plant proposal. The pinned simulator rejects that
proposal for one tick, reports its reason and applies `invalid_plant_penalty`.
The proposal remains the trajectory and Q-learning target, while execution and
recurrent history receive wait (`0`) plus `accepted=false` and one tick.

During collection, the ten-way branch choice is always the highest-Q choice.
Exploration applies only to the selected branch's tile target: plant tiles use
occupancy masks and dig tiles remain unrestricted. A tile exploration coin can
therefore change the target without changing which branch was proposed.

## Requirements

Python 3.12, Git and the pinned game package. Recording needs pygame.
Autonomous training requires NVIDIA CUDA; demonstration fitting defaults to CPU.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

The [engine lock](src/pvz_rl/data/engine-lock.json) fixes source and package hashes.

## First useful command

Record one easy game (seed 1000), using unused output paths:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
```

After a natural win or loss:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human-1000.jsonl --replay runs\human-1000.pvzdemo `
  --output runs\human-init
.\.venv\Scripts\python.exe -m pvz_rl train `
  --init-from runs\human-init\initialization.pt --output runs\human-trained
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\human-trained\latest.zip --output runs\human-trained
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\human-trained\latest.zip --split validation `
  --output runs\human-trained\validation
```

Training commands explicitly start autonomous learning. Availability and replay
checks do not. Incomplete recordings are rejected. The curriculum is easy →
standard → shared; a demonstration fit alone does not establish mastery.

All profiles share one bundled configuration at
`src/pvz_rl/data/train.toml`; use `--profile event-memory` for the archived
event-memory model.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pvz_rl replay runs\human-1000.pvzdemo
```

- [Training, configuration, resume and evaluation](docs/training.md)
- [Architecture and workflows](docs/architecture.md)
- [Viewer controls](docs/live-view.md)
- [Recurrent objective and mathematical controls](docs/math/recurrent-training.md)
- [Inspected sources](docs/references.md)
- [Iteration history](docs/iteration.md)
