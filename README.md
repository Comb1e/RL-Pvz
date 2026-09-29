# PVZ plant research

Record a human Plants vs. Zombies game, initialize an entity Transformer–LSTM Q
policy, then train and evaluate it on a pinned 100 Hz simulator using public inputs.

## Requirements

Python 3.12, Git and the pinned game package. Recording needs pygame.
Autonomous training requires NVIDIA CUDA; demonstration fitting defaults to CPU.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

The [engine lock](src/pvz_rl/data/engine-lock.json) fixes source and package hashes.
All train/demo parameters live in [train.toml](src/pvz_rl/data/train.toml).
`training.max_grad_norm = 5` is the single clipping limit for both fitting paths.

## First useful command

Record one easy game using unused output paths; finish with a natural win or loss:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
```

Follow [recording and training](docs/training.md) to initialize, train, resume or
evaluate. That guide also explains checkpoint compatibility and performance settings.
The live window shows the run's retained learning settings, including gradient
clipping and rewards. Press **S** to hide or show them.

`--init-from` transfers weights into a new run using all current settings from
`src/pvz_rl/data/train.toml` without requiring `--refresh-performance`.
`--resume` retains checkpoint learning settings and rewards; the refresh flag
only updates execution settings.
With `compile_kernels = true`, the Windows CUDA path captures the fixed-shape
entity encoder with CUDA graphs and falls back to eager execution once if the
installed backend cannot compile it. Captured activations and gradients are copied
out of reusable graph storage before recurrent fitting consumes them.

## Common checks

Static code checks do not start the simulator or training:

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
```

[Validation](docs/validation.md) describes the test suites and their execution costs.

- [Architecture and workflows](docs/architecture.md)
- [Viewer controls](docs/live-view.md)
- [Entity schema, cap and attention](docs/math/entity-inputs.md)
- [Reward accounting](docs/math/training-objective.md)
- [Recurrent objective and exploration](docs/math/recurrent-training.md)
- [Precision and throughput](docs/math/training-throughput.md)
- [Sources used](docs/references.md)
- [Iteration history and recorded measurements](docs/iteration.md)
