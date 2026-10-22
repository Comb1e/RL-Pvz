# PVZ plant research

Train a stateless two-stage Transformer Q policy against a pinned Plants vs.
Zombies simulator. The species stage chooses wait, a plant, or dig; the tile
stage scores all 45 plots. Both read only the current public board.

## Requirements

Python 3.12, Git, and the pinned game checkout. Autonomous training and fitting
require an NVIDIA GPU with native BF16 support. Recording also needs pygame.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

For Linux/WSL or Git Bash:

```bash
bash tools/bootstrap.sh --game-repo ../pvz-cuda-work --python python3.12
```

Both installers verify the [engine lock](src/pvz_rl/data/engine-lock.json),
install PyTorch 2.8.0/CUDA 12.8, and leave the game checkout unchanged.

## First useful command

Record a human game into unused paths and finish with a natural win or loss:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human.pvzdemo --archive runs\human.jsonl
```

Or open the browser annotator and label plant cards, Wait/Dig, or board tiles:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl annotate --kind action --port 6006
.\.venv\Scripts\python.exe -m pvz_rl annotate --kind tile --port 6006
```

Dig starts as an editable absolute-infeasible action label. If next-map
generation fails, use **Retry generation** on the same server; saved labels remain persisted.

[Training guidance](docs/training.md) covers annotation fitting before demo
initialization, autonomous training, GPU selection, Jupyter proxies, and recovery.
The active-only trajectory/recovery format is v4; older runs support compatible
weight-only initialization, but cannot resume. Model weights and the learning
objective are unchanged by the pipeline optimization.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
```

Use [test ownership](docs/validation.md) to select focused suites. For bounded
production-cycle calibration with fresh optimizer and simulator state:

```powershell
.\.venv\Scripts\python.exe tools\tune_gpu.py --gpu 0 `
  --workload full-cycle --initial-weights runs\human-init\initialization.pt `
  --output artifacts\pipeline --minutes 60 --no-compile
```

This is a scratch performance rehearsal, not a formal training run. Incomplete
comparisons never promote a profile; existing 4070 microbatch defaults remain.

- [Architecture and state flows](docs/architecture.md)
- [Training, reward settings, and recovery](docs/training.md)
- [Execution controls and diagnostics](docs/math/training-throughput.md)
- [4070 pipeline measurements and limitations](docs/evidence/training-pipeline-optimization-20261021.md)
- [Learning objective](docs/math/training-objective.md)
- [Viewer controls](docs/live-view.md)
- [References](docs/references.md)
- [Iteration history](docs/iteration.md)
