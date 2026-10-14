# PVZ plant research

Record a human Plants vs. Zombies game, initialize a stateless two-stage
Transformer Q policy, then train and evaluate it with the pinned simulator.
The species Transformer selects a plant, wait or dig; the independent tile
Transformer scores 45 tiles for the selected branch. Both read only the current
public board. Planting creates no counterfactual rollout.

Fresh weights are required for `complete_return_sequential_no_probe_v1`.
Existing verified recordings can produce a new initialization in an unused directory;
previous checkpoints, including earlier two-stage `human-init\initialization.pt`,
cannot initialize or resume this objective. Preserve their files. Parameters live in
[src/pvz_rl/data/train.toml](src/pvz_rl/data/train.toml).

## Requirements

Python 3.12, Git and the pinned game package. Recording needs pygame.
Autonomous training requires NVIDIA CUDA with BF16 support.

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

Use [recording and training](docs/training.md) to initialize, train, resume or
evaluate. Demonstration fitting uses 20 passes; autonomous cohorts use four.
Runtime inference and fitting use BF16 model parameters; optimization uses
separate FP32 master weights and FP32 Adam state. INT8/INT4 paths are not
implemented.

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
- [Execution, compiler troubleshooting and throughput](docs/math/training-throughput.md)
- [Sources used](docs/references.md)
- [Iteration history and measurements](docs/iteration.md)
