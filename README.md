# PVZ plant research

Record a human Plants vs. Zombies game, initialize one entity Transformer–LSTM Q
policy, then train and evaluate it on a separate, pinned 100 Hz simulator.
The model reads the current board every decision; LSTM memory writes only on
sunlight, zombie and plant events. Species values are their best candidate-tile
values; each accepted planting queues up to four same-species alternative tiles,
using isolated greedy-EMA rollouts of up to 30 simulated seconds. Real games keep
advancing alongside background probes; only source-capacity-limited slots pause.
This framework requires fresh weights: refit an existing verified
recording into a new initialization directory. Only public observations enter the
model. Parameters live in
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
recover. Demonstration initialization defaults to 20 passes (`training.demo.passes`);
autonomous training stays at four (`training.n_epochs`). Fresh runs use current
settings; resume retains learning parameters and
rewards while applying current execution and compact-logging settings automatically.
Current defaults triple early Sunflower income value and add accepted-plant species
commitments. Tile (50%) and species (10%) exploration both reach a 1% floor after
10,000 stage games. See the training guide for waiting behavior and current-version recovery.
Longer probes increase collection cost. Collection uses active-only inference and
bounded auxiliary pools, public-count buckets and
phase-owned CUDA graph caches; these execution controls do not change learning settings.

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
- [Recurrent training objective](docs/math/recurrent-training.md)
- [Reward accounting](docs/math/training-objective.md)
- [Execution and throughput](docs/math/training-throughput.md)
- [Sources used](docs/references.md)
- [Iteration history and measurements](docs/iteration.md)
