# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer encodes the public board. An effect head predicts this frame's
untouched field, breach, and probe, and five detached lane tokens go to the
action decoder and the tile decoder. An action readout then chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Illegal cards
and cells are excluded before that choice: a plant needs sun, a finished
cooldown, and an empty cell; Dig needs a cell that already has a plant. Exact simulated
pre-action outcomes are auxiliary training targets. Validation uses the greedy
network on the public board.

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
`prediction → cherry_bomb → cherry_bomb_obstacle → attacker_pre → attacker → peashooter → easy → standard → shared`.
Use `--stage NAME --until-stage-complete` for one stage, or `--games N` for a
completed-episode budget. Explicit normal stages skip preliminary stages. Omit
`--init-from` to start with random weights.

Each stage can start from the previous stage's weights;
[pre_train.ps1](tools/pre_train.ps1) chains them in order:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage cherry_bomb --until-stage-complete `
  --init-from runs\compact-stages\prediction\final.zip --output runs\compact-stages\cherry_bomb `
  --gpu 0 --gpu-profile rtx4070-laptop
```

`prediction`, cherry bomb, cherry-bomb obstacle, attacker pre, and attacker
take one fitting pass per cohort. Each cohort is scored once before that pass,
and the pass still runs when the score is already perfect. The fitting line
shows that score as an integer ratio, such as `612/640`, and `streak k/10`.
The report charts the check from before the first fitting pass for these five
stages, redraws every 10 cohorts and when a stage advances, and does not chart
exploration for them. A checkpoint that omits that pair still resumes. When the
open cohort has not committed a pass, the finished first check is that point;
otherwise the next cohort records it. Changing `ema_decay` or the stage-gate
settings means the open prediction run does not resume. Prediction counts a
scene when every lane matches field depth, breach bin, and probe depth. The
four placement stages count a board when the greedy legal tile is in its
labeled successful set. A stage advances after 10 consecutive full cohorts meet
that bar: prediction, obstacle, attacker pre, and attacker at 100%, cherry bomb
strictly above 0.95. Peashooter uses the same 0.95 rule on games won. Easy,
standard, and shared require every game of each of their tasks to be won, and
every required task has to occur. Training does not run a held-out panel. The
live view draws the three bands and hides action scores that were not recorded.
Prediction trains the shared board encoder, the effect head, and the
forecast-input embeddings, and its forward stops after those three. Later
stages freeze those three and still run them. The teacher on every stage uses
EMA decay 0.99.
Placement and Peashooter train and run the tile decoder and tile head. Easy,
standard, and shared also train and run the action decoder and action head. The
forecast reads the current encoder memory before the action is chosen. Use the
prediction `final.zip` to initialize `cherry_bomb`.

Across stages, a probe starts once the field clears, queued shots fire and their
projectiles clear, moving mowers finish, and surviving Chompers are ready. The
static-field **H** cell is 1 when the predicted field class is home. Start fresh
with `shared_public_outcomes_q_v11` and use its checkpoints for `--resume` and
`--init-from`.

`attacker` mixes the three placement exercises in shuffled 640-board cohorts
(64 Cherry Bomb, 256 Obstacle, 320 Attacker Pre). Its stage bar is the same
labeled-set check as the other placement stages. Peashooter can start from
`runs\compact-stages\attacker\final.zip`.

Resume an interrupted stage from its recovery checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage cherry_bomb_obstacle --until-stage-complete `
  --resume runs\compact-stages\cherry_bomb_obstacle\interrupted.zip `
  --output runs\compact-stages\cherry_bomb_obstacle
```

No seed is required; `--seed` optionally fixes initialization. Start a new run:
each width is that group's heads times `head_dim`, and a previous flat policy does
not transfer. The live view's gradient field scales the next cohort's update and
still clips at `max_grad_norm`. Its `3×` control arms the next cohort at
`learning_rate_restart` times `learning_rate` for one pass. The cohort line
prints each pass's gradient norm before that clip. A checkpoint of this model
that omits the scale, those norms, or the restart flags stays at 1 and idle,
and records later passes. Peashooter and the three normal stages each have a
fitting pass count unless that restart is armed. Tune those passes, the loss
weights, and the execution settings in [train.toml](src/pvz_rl/data/train.toml);
each setting is commented there. Separate evaluation commands keep their panels.
Completed runs keep portable `final.zip` weights and visualizations;
unfinished runs keep recovery checkpoints. A run's JSONL logs roll at 5 MB
into a folder per log, such as `hardware-metrics/`. On the next open, a file
already larger than that is moved into the folder and later records go to a
new file there.

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
