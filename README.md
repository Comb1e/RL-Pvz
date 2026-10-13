# PVZ plant research

Record a human Plants vs. Zombies game, initialize a stateless two-stage
Transformer Q policy, then train and evaluate it on a separate, pinned 100 Hz
simulator. One Transformer selects a species, wait or dig; an independent
Transformer scores 45 tiles for the selected plant/dig branch. Both read only
the current public board. Accepted plantings create same-species background
tile probes while real games continue.

Fresh weights are required. Existing verified recordings can produce a new
initialization; previous checkpoints cannot be reused. Parameters live in
[src/pvz_rl/data/train.toml](src/pvz_rl/data/train.toml).

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

Record one easy game using unused output paths; finish with a natural win or loss:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
```

Follow [recording and training](docs/training.md) to initialize, train, resume or
evaluate. Demonstrations use 20 fitting passes; autonomous cohorts use four.
Fresh runs use current settings. Current-version resume retains learning/reward
settings and refreshes execution/logging controls. Background probes and both
Transformers share GPU resources; no throughput or win-rate improvement is promised.

## Common checks

Static checks do not start the simulator or training:

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
```

- [Validation and test ownership](docs/validation.md)
- [Architecture and workflows](docs/architecture.md)
- [Viewer controls](docs/live-view.md)
- [Entity schema and attention](docs/math/entity-inputs.md)
- [Sequential Q objective](docs/math/sequential-training.md)
- [Reward accounting](docs/math/training-objective.md)
- [Execution and throughput](docs/math/training-throughput.md)
- [Sources used](docs/references.md)
- [Iteration history and measurements](docs/iteration.md)
