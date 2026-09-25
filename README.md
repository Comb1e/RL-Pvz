# PVZ training research

Research **0.21.0** uses one timer-free CUDA Transformer controller across
saving, easy, standard and hard games. A greedy critic chooses wait/plant/dig;
conditional PPO learns planting species and tiles from complete games. Results
remain experimental. This release requires fresh models.

## Requirements

Python 3.12, Git, an NVIDIA CUDA GPU/driver, host RAM and disk space for complete
trajectories. Defaults are 32 games per cohort and a 6 GiB trajectory RAM budget
with disk overflow. FFmpeg is optional for video; the live viewer uses pygame.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

Bootstrap reuses `.venv` and installs a verified non-editable archive of game
**1.5.0**, simulation **1.2.0**, at **100 Hz**. The exact source and hashes are
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
training behavior; use its Switch buttons or add `--no-live-view`.

Ctrl+C saves at the next atomic simulation/ledger or optimizer-step boundary.
The atomic `interrupted.zip` includes unfinished games, optimizer progress and
RNG state. Resume with that archive:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\saving-101\interrupted.zip --output runs\saving-resumed
```

After mastery, start a new experiment with `--stage easy --init-from
runs\saving-101\final.zip` and a new output directory. Stage transfer uses
compatible actor/critic weights with fresh optimizers. Existing artifacts are
preserved; older models require their original software.

## Common checks and curves

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\saving-101 --report-only
```

Terminal blocks show completed-game reward, net value, discounted return,
pre-fit critic errors/coverage and collection/critic/actor timing. Gamma 1 makes
discounted and undiscounted reward equal. Cutoff failures receive defeat reward.
Hardware samples are flushed to `hardware-metrics.jsonl`; curves refresh after
probes, validation and graceful interruption. Offline reports need no model.

- [Exact inputs, network roles and training workflow](docs/architecture.md)
- [Curriculum, rewards, stopping and recovery](docs/research.md)
- [Mathematical controls and limitations](docs/math/complete-game-learning.md)
- [Verified results](docs/validation.md)
- [Inspected sources](docs/references.md)
- [Iteration history](docs/iteration.md)
