# PVZ training research

Research **0.22.0** uses one timer-free CUDA Transformer controller across
saving, easy, standard and hard games. A greedy critic chooses wait/plant/dig;
conditional PPO learns planting species and tiles from complete games. Results
remain experimental. Training alternates 256 critic games and 256 actor games,
updating the selected network after every 32 completed games.

## Requirements

Python 3.12, Git, an NVIDIA CUDA GPU/driver, host RAM and disk space for complete
trajectories. Defaults are 32 games per cohort and a 6 GiB trajectory RAM budget
with disk overflow, plus up to 2 GiB of cached raw tokens on CUDA. FFmpeg is optional for video; the live viewer uses pygame.

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
training behavior, estimated wait/plant/dig returns, and why the highest legal
value was chosen (including its lead or a tie). Unavailable choices are marked.
These are predicted future rewards; species probabilities are conditional on
planting. Click a species for its tile heatmap. Switch selects an unfinished
game; add `--no-live-view` to disable the window.

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
preserved. Version 0.21.0 supports inference and compatible `--init-from`; full
resume requires a 0.22.0 alternating-role checkpoint.

## Common checks and curves

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\saving-101 --report-only
```

Terminal blocks show completed-game reward, net value, discounted return,
the current role and progress, pre-fit errors/planting coverage, and collection,
data preparation, transfer, cache and optimization timing. No planting samples
means an actor cohort skips its update; the greedy controller is unchanged. Gamma 1 makes
discounted and undiscounted reward equal. Cutoff failures receive defeat reward.
Hardware samples are flushed to `hardware-metrics.jsonl`; curves refresh after
probes, validation and graceful interruption. Offline reports need no model.

- [Exact inputs, network roles and training workflow](docs/architecture.md)
- [Curriculum, rewards, stopping and recovery](docs/research.md)
- [Mathematical controls and limitations](docs/math/complete-game-learning.md)
- [Verified results](docs/validation.md)
- [Inspected sources](docs/references.md)
- [Iteration history](docs/iteration.md)
