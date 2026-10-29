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

Engine 1.7.1 checks branch availability before tile occupancy. Reinstall with the
bootstrap command after updating. Older engine pins require fresh autonomous runs
and newly recorded demonstration archives; compatible weights still work with
`--init-from`, and matching annotation progress can still resume.

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

The active-only trajectory/recovery format is v7. Existing v7 runs, including
those created before residual-sun shaping, remain resumable; their already
collected cohort keeps its saved reward settings and the new settings begin with
the next cohort. Older recovery formats support compatible weight-only
initialization. Model architecture, observation encoding and weight layout
remain unchanged.

Reward settings live in `src/pvz_rl/data/train.toml`; each run saves its effective
configuration. With `value_scale=500` and `early_sun_extra_multiplier=6`,
25 Sunflower sun gives +0.01 normally or +0.04 early. Removing 270 effective
zombie HP gives +0.01.
One mower activation is a direct -0.1 reward penalty, independent of sunlight scaling and the dense
multiplier. Live native/browser boards show each environment's accumulated
training reward. Terminal boards show a pending label until cohort time ranking
finishes, then refresh with the finalized reward.
Fitting retains the first row of each configured window of adjacent waits or
identical invalid proposals (`training.objective.invalid_fit_group_size`);
complete returns and reward accounting retain every decision. Invalid branch
and tile proposals fit below the responsible head's lowest current Q-value,
using `training.objective.invalid_action_q_offset = -0.1`. Unavailable branches
use the minimum of all ten species outputs; illegal tiles use all 45 outputs
for the selected branch. A minimum of -1.2 gives a -1.3 target regardless of
which illegal output was selected. The target is detached and recomputed every
pass. Only the responsible head is fitted; game rewards and returns are unchanged.
Selecting Dig on an empty board is branch-invalid: only the species/action
Transformer is fitted and tile inference is skipped. With at least one plant,
Dig scores all 45 tiles; selecting an empty tile then fits only the tile stage.
Completion-time ranking uses a signed quadratic adjustment capped at 0.3:
terminal wins range from +0.7 to +1.3 and losses from -1 to -1.3. Physical
rewards and mower penalties additionally affect total game reward. A losing
episode also applies residual-sun passivity shaping. A win receives
`+0.2 * clamp(remaining_sun / 1000, 0, 1)^2` as a terminal reward. A loss
uses `-loss_max_penalty * clamp((remaining_sun - 200) / 800, 0, 1)^2` as the
per-wait maximum. Every accepted ordinary wait in a consecutive run receives
its own penalty: the longest run receives the maximum, the shortest receives
one-fifth of it, and intermediate run lengths use a quadratic curve. Equal-
length runs are treated as longest runs. Commitment waits, rejected actions,
plants and digs break runs and receive no penalty; a loss with no ordinary
waits receives no such penalty. Total loss shaping can therefore grow with the
number of passive waits. The settings live under `[reward.passivity]` in
`train.toml`.
Objective v7 requires fresh `--init-from` for previous weights rather than
optimizer resume. The reward protocol stays v5; annotation resume is
unchanged. See the [reward conversion table](docs/math/training-objective.md).

The agent cadence is also configured in `train.toml`:

```toml
[environment]
action_frequency_hz = 10
```

The pinned simulator runs at 100 Hz, so this resolves to ten simulator ticks per
agent decision. `decision_ticks` and `action_timing` are derived and validated;
edit `action_frequency_hz` when changing the cadence. A frequency must divide the
engine tick rate exactly.

Plant exploration also checks after each 15 simulated seconds of continuous
ordinary waiting. Configure `training.exploration.plant_wait_check_seconds` in
`train.toml`; the same saved planting-exploration probability applies to each
check. Rejected actions break the streak, and commitment waits do not count.
See [exploration and recovery](docs/math/sequential-training.md#plant-exploration)
for species selection and cohort-boundary behavior.

Resume an interrupted compatible run with its optimizer, RNG and fitting progress:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\compact-stages-101-new\easy\interrupted.zip `
  --output runs\compact-stages-101-new\easy `
  --gpu 0 --gpu-profile rtx4070-laptop
```

Resume retains saved exploration rates, decay schedules and the active reward
settings for an already collected cohort. The current wait interval and
`[reward.passivity]` values take effect at the next collection cohort. A saved
3% rate means a 3% chance per check, not guaranteed planting.

## Common checks

Inspect one cohort's saved settings, event rewards, component breakdown,
warnings, action mix and fitting compaction while training continues:

```powershell
.\.venv\Scripts\python.exe tools\report_rewards.py runs\compact-stages-101-new\easy 6
```

The two arguments are the run folder and the **cohort number shown in the log**
(starting at 1). Redirect stdout to a text file to retain a report. The script
reads saved configuration/logs; unfinished data is marked provisional. See
[reward diagnostic definitions](docs/math/training-objective.md#physical-return-variation-diagnostic).

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
