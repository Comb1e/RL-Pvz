# PVZ plant research

Record a human Plants vs. Zombies game, initialize a Transformer–LSTM Q policy,
then train and evaluate the same model autonomously. The pinned simulator runs
at 100 Hz; policies receive only public observations.

Each plant, zombie, projectile and mower has its own 11-field public record,
embedded into 32 values and processed by attention before the recurrent policy.
The ten action branches remain greedy; exploration only chooses tiles. Rejected
proposals receive their penalty and execute/feed back as waits.

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

CUDA fitting uses BF16 features with FP32 master weights, LSTM and Q heads.
To apply current execution settings to an existing run, add `--refresh-performance`
when initializing from or resuming an existing checkpoint. Demonstration fitting keeps the configured five passes.

All parameters live in `src/pvz_rl/data/train.toml` (train and demo overlays).
This architecture requires a newly recorded demonstration or a fresh model;
aggregate-observation checkpoints and archives cannot be loaded. Existing entity_v1
weights remain compatible with the throughput update.
Existing files are preserved. Choose new output paths.

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
- [Entity schema, cap and attention mathematics](docs/math/entity-inputs.md)
- [Recurrent objective and mathematical controls](docs/math/recurrent-training.md)
- [Inspected sources](docs/references.md)
- [Iteration history](docs/iteration.md)
