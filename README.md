# PVZ plant research

Train a sequential Q policy against a pinned Plants vs. Zombies simulator. A
shared Transformer understands the public board, an action readout chooses
Wait/species/Dig, and a conditional decoder scores all 45 tiles. Every autonomous
stage also learns hindsight threats and baseline-relative action effects. Labels never override
actions; validation uses the greedy network.

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

Demo initialization verifies the replay before fitting. Manual tile annotation is
removed; existing datasets and recordings remain on disk. The shared-encoder
architecture starts fresh: old model weights and autonomous checkpoints are
incompatible. Compatible weights transfer with `--init-from`; interrupted runs
resume with `--resume` and preserve optimizer updates and unfinished forks, hindsight labels and fitting passes.

Start autonomous training when ready:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --seed 101 `
  --init-from runs\human-init\initialization.pt --output runs\my-run `
  --gpu 0 --gpu-profile rtx4070-laptop
```

Default progression is
`cherry_bomb → cherry_bomb_obstacle → attacker_pre → peashooter → easy → standard → shared`.
Use `--stage NAME --until-stage-complete` for one stage, or `--games N` for a
completed-episode budget. Explicit normal stages skip preliminary stages. Omit
`--init-from` to start with random weights.

Tune [train.toml](src/pvz_rl/data/train.toml): stage tables own their placement
settings and independent `threat_prediction_weight` / `effect_prediction_weight`.
Normal stages jointly fit n-step Q targets; preliminary stages preserve their
placement losses and freeze the action readout. Four fitting passes, greedy tiles
and normal species exploration remain in use.

`training.forecasts` controls isolated CUDA forks, sampled alternatives, private
random/scripted effect pretraining, auxiliary warmup, feature injection and TD
horizons. Every decision receives a static baseline: existing zombies finish with
no actions or future spawns, then Conehead probes test the damaged defenses. Lanes
continue independently after actual home entries. Executed and sampled actions
receive signed categorical changes relative to that baseline; forks never select
real actions or consume the real RNG. Baselines are required during inference and
validation too. This work can be substantial; resource limits bound resident
scratch lanes, not the simulated duration of a fork.

Threat targets summarize realized penetration at 2τ, 9τ and episode end, breach
time and per-lane loss. τ is the reference Basic-zombie tile-crossing time,
including its gait. Loss and
cutoff windows retain observed lower bounds instead of invented safe outcomes.
Peashooter privately Waits after its cutoff for hindsight labels; normal cutoffs
remain censored. The viewer shows the raw end-horizon penetration CDF and home
risk, with unknown targets grey. Head predictions never override actions.

Defaults pretrain on 2,048 private boards, warm auxiliary weights over 2,048 real
episodes, and enable detached forecast features after 4,096 normal games. The TD
schedule increases from 5 to 20 steps and from a 60-step to a 600-step discount
half-life. Potential shaping starts disabled. Reports separate primary/threat/effect
loss, calibration, gradient ratios, fork cost and real-game outcomes. Use
`python -m pvz_rl.monitoring.forecast_ablation --help` for seed-matched ablations;
these require their own explicit game and wall-time budgets. New fork-model weights
start fresh; older models and recoveries are rejected without changing user files.

Validation uses fresh random panels, saved for interrupted retries. Full fitted
cohorts must strictly exceed the configured qualification threshold before a due
validation launches. Cherry Bomb and Peashooter use their configurable mastery
thresholds; other fresh stages require perfect validation. Completed runs retain
portable `final.zip` weights and visualizations; unfinished runs retain recovery.
Temporary cohort files are removed after fitting; interrupted work is preserved
in recovery checkpoints.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
.\.venv\Scripts\python.exe -m pvz_rl replay runs\human.pvzdemo --watch --speed 2
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\my-run --report-only
```

Exactly three smoke tests cover CLI startup, CPU inference and bounded CUDA
learning/reload. They establish workflow availability, not mastery. Report-only
visualization loads no policy and starts no training.

- [Code structure and test ownership](AGENTS.md)
- [Architecture and workflows](docs/architecture.md)
- [Research references](docs/references.md)
