# PVZ training research

Research **0.25.0** uses one CUDA Transformer Q network. It selects wait, a plant
species or dig, then a tile for non-wait commands. After 128 complete games it fits
both selected Q heads to actual remaining rewards. Learning remains experimental.

## Requirements

Python 3.12, Git, an NVIDIA CUDA GPU/driver, host RAM and disk space for complete
trajectories. Defaults are 128 games per cohort and a 6 GiB trajectory RAM budget
with disk overflow, plus up to 2 GiB of cached raw tokens on CUDA. FFmpeg is optional
for video; the live viewer uses pygame.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

Bootstrap reuses `.venv` and installs a verified non-editable archive of game
**1.6.0**, simulation **1.3.0**, at **100 Hz**. The exact source and hashes are
in [the engine lock](src/pvz_rl/data/engine-lock.json).

## First useful commands

A bounded saving experiment, started only when you execute it:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --config configs\train.toml `
  --stage saving --games 2000 --max-minutes 120 --seed 101 --output runs\saving-101
```

To continue until that stage passes, replace `--games` and `--max-minutes` with
`--until-stage-complete`. The 1,200-second per-game failure cutoff still applies.
Validation disables injected exploration. A four-game window shows actual
training behavior, estimated wait/plant/dig returns, and why the highest Q
value was chosen (including its lead or a tie). All ten first-level outputs
remain visible; rejected plants are shown as automatic waits with their reason,
and empty digs show their explicit penalty.
These are predicted future rewards. Focus opens a readable full-game action/Q
table; Switch chooses another unfinished game. Viewer controls and storage are
in [the live-view guide](docs/live-view.md). Add `--no-live-view` to close the window.

Ctrl+C saves at the next atomic simulation/ledger or optimizer-step boundary.
The atomic `interrupted.zip` includes unfinished games, optimizer progress and
RNG state. Resume with that archive:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\saving-101\interrupted.zip --output runs\saving-resumed
```

After mastery, start a new experiment with `--stage easy --init-from
runs\saving-101\final.zip` and a new output directory. Stage transfer uses
compatible Q weights with a fresh optimizer. Existing artifacts are preserved.
This release requires fresh experiments because the action-selection and reward
protocol changed. 0.24.0 checkpoints remain usable for inference with their
original semantics, but cannot initialize or resume new training. Same-release
interruption/resume and stage transfer remain supported. Historical curves can
be regenerated offline.

## Common checks and curves

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\saving-101 --report-only
```

Terminal blocks show completed-game reward, net value, discounted return,
Q-update steps, pre-fit errors/species coverage, and collection,
data preparation, transfer, cache and optimization timing. Missing planting
coverage remains visible; exploration cannot force planting. Gamma 1 makes
discounted and undiscounted reward equal. Cutoff failures receive defeat reward.
Hardware samples are flushed to `hardware-metrics.jsonl`; curves refresh after
probes, validation and graceful interruption. Offline reports need no model.

- [Exact inputs, network roles and training workflow](docs/architecture.md)
- [Curriculum, rewards, stopping and recovery](docs/research.md)
- [Mathematical controls and limitations](docs/math/complete-game-learning.md)
- [Rejection penalties and reward limits](docs/math/invalid-action-penalties.md)
- [Current mechanics and bounds](docs/math/mechanics-024.md)
- [Viewer controls and action history](docs/live-view.md)
- [Verified results](docs/validation.md)
- [Inspected sources](docs/references.md)
- [Iteration history](docs/iteration.md)
