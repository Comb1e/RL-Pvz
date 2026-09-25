# PVZ plant-placement research

Research **0.20.0** trains one shared CUDA PPO policy using a timer-free event-memory
Transformer. Training starts with saving, then easy, standard and shared difficulties.
All eight plants are available in saving. Results remain experimental. Compatible
0.19.0 checkpoints can resume; this release adds presentation without changing learning.

## Requirements

Python 3.12, Git, an NVIDIA CUDA GPU and compatible driver. The CPU simulator is a
correctness/replay reference. FFmpeg/libx264 is optional for video export.

## Installation

Run from the project root:

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

The installer reuses `.venv` and installs a verified, non-editable archive of
[pvz-cuda-work](https://github.com/Comb1e/pvz-cuda-work): package **1.4.0**, simulation
**1.1.0**, **100 Hz**. The [engine lock](src/pvz_rl/data/engine-lock.json) remains
unchanged. Neither game checkout is modified. Dependency locks include CUDA runtime,
PyTorch and the hardware sampler's psutil dependency.

## First training run

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --config configs\train.toml `
  --stage saving --until-stage-complete --seed 101 --output runs\saving-101
```

Choose an unused output directory. This trains only saving until its mastery gate
passes, with no training time/game ceiling. Per-game cutoffs and probes remain active.
For a bounded automatic curriculum, omit `--stage` and `--until-stage-complete` and
use `--games 10000 --max-minutes 120`. Formal training starts only when you run a command.

Ctrl+C finishes the current periodic window and saves `interrupted.zip`. Resume with
`--resume runs\saving-101\interrupted.zip --output runs\saving-resumed`. After mastery,
use `--stage easy --init-from runs\saving-101\final.zip` and a new output directory.
New-protocol resume restores the experiment; stage transfer copies weights into a
fresh experiment. Older checkpoints are rejected; existing runs and recordings are preserved.

Training opens a resizable **four-game live window**. Each panel follows a randomly
selected actual training game; **Switch** selects another environment without
changing any game. Finished games show their result for one second before switching.
Closing the window keeps training running. Add `--no-live-view` for headless training,
or `--live-view` to override saved output settings when resuming. The viewer uses
the optional `ui` dependency installed by bootstrap; unavailable displays disable
only the viewer. Configure `visualization.live_enabled`, `live_fps` (default 5),
and `live_window_size` independently of reports and video exports.

Defaults remain 128 environments × 128 transitions per rollout, two rollouts per
synchronization window. Waiting counts as a transition; 128 is not a game-length limit.
Validation disables injected exploration and uses deterministic conditional choices.
See [training and curriculum](docs/research.md) for schedules, gates and handoff details.
The terminal prints 15-second blocks with completed-game means for reward,
discounted return, net value and economy, alongside exploration and recent window
throughput. Cutoff counts identify partial episode returns; current gamma = 1 makes
discounted and undiscounted episode returns equal.

## Common checks and curves

```powershell
Get-Content runs\saving-101\train.log -Wait
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\saving-101 --report-only
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```

Training appends CPU/GPU/RAM/VRAM/power/temperature readings to
`hardware-metrics.jsonl`. Hardware panels refresh after every mastery probe,
normal-game validation attempt, completion and graceful interruption, under
`visualizations/index.html`. A forced stop leaves flushed samples available to the
report-only command, which requires no checkpoint. Older runs without samples
cannot reconstruct past hardware utilization. Sampling is controlled by
`training.performance.telemetry` and `logging.hardware_sample_seconds` (one second).

Omit `--report-only` to generate demonstrations from a validated shared checkpoint.
Open a generated 100 Hz recording with `pvz-rl replay FILE --watch`; optional videos
sample at 25 frames/second without changing simulation. No best checkpoint or demos
are generated before the first successful stage's normal-game validation.

- [Architecture, package layout and workflows](docs/architecture.md)
- [Training, reward, curriculum and limitations](docs/research.md)
- [Saving feasibility and action-probability calculations](docs/math/saving-and-actions.md)
- [Objective and PPO derivations](docs/math/training-objective.md)
- [Exploration schedule](docs/math/exploration-schedule.md)
- [Inspected research sources](docs/references.md)
- [Verification and measured results](docs/validation.md)
- [Iteration history](docs/iteration.md)
