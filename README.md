# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer encodes the public board, an action readout chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Exact simulated
post-action outcomes serve as auxiliary training targets only; validation uses
the greedy network on public inputs.

## Requirements

Python 3.12, Git, the pinned game checkout, and an NVIDIA GPU with native BF16
support for training. Human recording and the native viewer also require pygame.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

Linux/WSL or Git Bash:

```bash
bash tools/bootstrap.sh --game-repo ../pvz-cuda-work --python python3.12
```

Installers verify the [engine lock](src/pvz_rl/data/engine-lock.json) and leave the
engine checkout unchanged.

## First useful command

Record human gameplay into unused paths and finish with a natural win or loss:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human.pvzdemo --archive runs\human.jsonl
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human.jsonl --replay runs\human.pvzdemo `
  --output runs\human-init --device cuda
```

Demo initialization verifies the replay before fitting. `--init-from` starts fresh
optimizer and counters from same-version weights; `--resume` restores a
same-version run. Checkpoints from earlier model versions are rejected.

Start autonomous training when ready:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --init-from runs\human-init\initialization.pt --output runs\my-run `
  --gpu 0 --gpu-profile rtx4070-laptop
```

Default progression is
`cherry_bomb → cherry_bomb_obstacle → attacker_pre → prediction → attacker → peashooter → easy → standard → shared`.
Use `--stage NAME --until-stage-complete` for one stage, or `--games N` for a
completed-episode budget. Explicit normal stages skip preliminary stages. Omit
`--init-from` to start with random weights.

Each stage can start from the previous stage's weights;
[pre_train.ps1](tools/pre_train.ps1) chains them in order:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage prediction --until-stage-complete `
  --init-from runs\compact-stages\attacker_pre\final.zip --output runs\compact-stages\prediction `
  --gpu 0 --gpu-profile rtx4070-laptop
```

`prediction` fits the selected action's field and probe forecasts on randomized
current frames. The live viewer labels completed scenes `scored`. All 200
validation scenes must have a categorical loss of 0.05 or less. Use its
`final.zip` to initialize `attacker`.

Across stages, a probe starts once the field clears, queued shots fire and their
projectiles clear, moving mowers finish, and surviving Chompers are ready. The
static-field **H** cell shows the chance of reaching the house. Start fresh with
this model version and use its checkpoints for `--resume` and `--init-from`.

`attacker` mixes the three placement exercises in shuffled 640-board cohorts
(64 Cherry Bomb, 256 Obstacle, 320 Attacker Pre) and requires 200/200 on a fresh
validation panel with the same proportions. Peashooter can start from
`runs\compact-stages\attacker\final.zip`.

Resume an interrupted stage from its recovery checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage cherry_bomb_obstacle --until-stage-complete `
  --resume runs\compact-stages\cherry_bomb_obstacle\interrupted.zip `
  --output runs\compact-stages\cherry_bomb_obstacle
```

No seed is required; `--seed` optionally fixes initialization. Tune stages,
loss weights and execution settings in [train.toml](src/pvz_rl/data/train.toml);
each setting is commented there. Validation uses fresh random panels after fully
fitted cohorts. Completed runs keep portable `final.zip` weights and
visualizations; unfinished runs keep recovery checkpoints.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
.\.venv\Scripts\python.exe -m pvz_rl replay runs\human.pvzdemo --watch --speed 2
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\my-run --report-only
```

The smoke tests check that the workflow runs, not that the agent masters a stage.
For throughput work, the [collection benchmark](tools/benchmark_peashooter.py),
[workflow profiler](tools/profile_training_workflow.py) and
[timeline tool](tools/draw_collection_process.py) document their options in `--help`.

- [Code structure and test ownership](AGENTS.md)
- [Architecture and workflows](docs/architecture.md)
- [Research references](docs/references.md)
