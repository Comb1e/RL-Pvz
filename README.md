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

Annotation fitting first reaches exact training-set convergence. If validation is
still below 100%, it continues on the unchanged training split with L2 pressure
until both splits are exact. To inspect long histories without collapsing detail
into an unreadable static plot, generate the interactive min/max-envelope viewer:

```powershell
.\.venv\Scripts\python.exe tools\plot_annotation_loss.py `
  runs\tile-initialization.history.jsonl --output runs\tile-loss.html
```

The HTML keeps every committed pass, supports zoom and drag, and reports the
nearest raw pass on hover. Adjust the regularization strength with
`--l2-strength` on either annotation-fitting command when needed.

Both fitters automatically cache prepared inputs when GPU memory permits and
pack checkpoint transfers while saving every pass. Existing commands work
unchanged. See [annotation execution settings and measurements](docs/math/training-throughput.md#annotation-fitting)
for cache limits, the streaming fallback, and opt-in microbatch tuning.

Resume a stopped fit with its saved optimizer, RNG, split, and L2 phase:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl fit-tile-annotations `
  --annotations data\tile-annotations.jsonl --output runs\tile-init\tile.pt `
  --resume-from runs\tile-init\tile.progress.pt --passes 1000
```

This allows 1,000 additional passes; `--passes 0` is unlimited. Omitted fitting
options inherit their saved values. Use `--init-from` for changed hyperparameters.
Any saved progress, including an unfinished or zero-pass snapshot, can supply
weights to another annotation fit, `initialize-demo`, or `train`:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl fit-action-annotations `
  --annotations data\action-annotations.jsonl --output runs\action-init\action.pt `
  --init-from runs\tile-init\tile.progress.pt
```

The handoff reads the latest committed snapshot, even while its source fit runs,
and starts a fresh destination optimizer. The final `tile.pt`/`action.pt` still
requires convergence. Annotation progress v3 needs one annotation resume to write
v4 before downstream initialization; `train --resume` requires autonomous recovery.

The active-only trajectory/recovery format is v4; older runs support compatible
weight-only initialization, but cannot resume. Model weights and the learning
objective are unchanged by the pipeline optimization.

Reward settings live in `src/pvz_rl/data/train.toml`. At `value_scale=2000`,
25 Sunflower sun gives +0.0025 normally or +0.005 early. One mower activation
is a direct -0.1 reward penalty, independent of sunlight scaling and the dense
multiplier. Live native/browser boards show each environment's accumulated
training reward. Terminal boards show a pending label until cohort time ranking
finishes, then refresh with the finalized reward.
Fitting retains the first of each 100 adjacent waits or identical invalid
proposals; complete returns and reward accounting retain every decision.
Reward/objective v4 requires fresh `--init-from` for previous weights rather than
optimizer resume. See the [reward conversion table](docs/math/training-objective.md).

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
```

Default pytest runs two CPU smoke checks: CLI startup and public-board inference.
The explicit `-m learning` command runs the two-game CUDA learning/checkpoint smoke
check and fails if CUDA requirements are missing. These checks cover core workflow
availability. See [test ownership](AGENTS.md#test-ownership) for their limits.

For bounded production-cycle calibration with fresh optimizer and simulator state:

```powershell
.\.venv\Scripts\python.exe tools\tune_gpu.py --gpu 0 `
  --workload full-cycle --initial-weights runs\human-init\initialization.pt `
  --output artifacts\pipeline --minutes 60 --no-compile
```

This is a scratch performance rehearsal, not a formal training run. Incomplete
comparisons never promote a profile; existing 4070 microbatch defaults remain.

- [Architecture and state flows](docs/architecture.md)
- [Execution controls and diagnostics](docs/math/training-throughput.md)
- [4070 pipeline measurements and limitations](docs/evidence/training-pipeline-optimization-20261021.md)
- [Learning objective](docs/math/training-objective.md)
- [References](docs/references.md)
- [Iteration history](docs/iteration.md)
