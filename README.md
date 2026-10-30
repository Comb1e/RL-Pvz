# PVZ plant research

Train a stateless two-stage Transformer against a pinned Plants vs. Zombies
simulator. One network selects species/Wait/Dig; another scores all 45 tiles.
Both consume only public board facts. Fresh autonomous runs use bounded simulator
search to teach action values and comparisons; validation uses the network alone.

## Requirements

Python 3.12, Git, the pinned game checkout, and an NVIDIA GPU with native BF16
support for training. Recording also requires pygame.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

Linux/WSL or Git Bash:

```bash
bash tools/bootstrap.sh --game-repo ../pvz-cuda-work --python python3.12
```

The installers verify the [engine lock](src/pvz_rl/data/engine-lock.json), install
the CUDA environment, and leave the engine checkout unchanged.

## First useful command

Record a game into unused paths and finish with a natural win or loss:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human.pvzdemo --archive runs\human.jsonl
```

Or label public boards in the browser:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl annotate --kind action --port 6006
.\.venv\Scripts\python.exe -m pvz_rl annotate --kind tile --port 6006
```

To start autonomous training when ready:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --seed 101 `
  --output runs\my-run --gpu 0 --gpu-profile rtx4070-laptop
```

Default progression is `cherry_bomb → peashooter → easy → standard → shared`.
Each stage can also run independently. Explicit normal stages skip tile training:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage cherry_bomb --until-stage-complete `
  --seed 101 --output runs\cherry --gpu 0 --gpu-profile rtx4070-laptop
.\.venv\Scripts\python.exe -m pvz_rl train --stage peashooter --until-stage-complete `
  --seed 101 --output runs\peashooter --init-from runs\cherry\final.zip `
  --gpu 0 --gpu-profile rtx4070-laptop
```

Without `--until-stage-complete`, `--games` limits completed real episodes.
Peashooter uses easy waves, 50 starting sun, disabled mowers and configurable
1.5× sky-sun frequency. Cherry Bomb uses its exact attributed-hit oracle with no
exploration. Each tile stage has its own episode budget and validation clock;
automatic progression preserves optimizer state, while separate `--init-from`
runs transfer weights into a fresh optimizer.
Normal fitting keeps three tile-only cohorts followed by one joint cohort, with
four passes each. Tile-only fitting freezes species parameters and species EMA.

Settings live in [train.toml](src/pvz_rl/data/train.toml). `[training.search]`
controls the public-board prior, query frequency and search budget. Defaults allow
128 terminal continuations per query and eight queries per real episode; search
can be expensive. Confirmation uses fresh sampled worlds before publishing labels.
Only the real games count toward training and validation schedules.

Search preserves reward economics and duration shaping, with the duration reference
frozen within each queried board. Its hidden-state prior is approximate. A longer
losing defense can still score worse because the configured duration penalty remains.

Every 2,000 completed episodes, rounded to a fitted cohort boundary, a full cohort
must exceed 90% success to launch validation (normally 116/128). Collection may be
search-assisted; mastery still requires 100/100 greedy network validation.
Each tile stage requires 100/100 on that skill alone, then completes or advances;
there is no joint probe or skill revisit in fresh runs. The combined `--stage pre`
is retired; old checkpoints retain their saved combined-stage workflow on resume.

Resume saved weights, optimizer, pending search, simulator and fitting progress:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --resume runs\my-run\latest.zip `
  --output runs\my-run --gpu 0 --gpu-profile rtx4070-laptop
```

Resume retains the saved search mode/settings and sky countdowns. Older currently
resumable checkpoints without search settings continue their original workflow.
Use `--init-from PATH` for a fresh optimizer and the current search configuration.
Annotation progress also supports compatible weight initialization; it is not an
autonomous recovery checkpoint.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\my-run --report-only
.\.venv\Scripts\python.exe tools\report_rewards.py runs\my-run 1
```

Default pytest runs two CPU checks. The explicitly selected learning check runs
two one-second CUDA games, one fitting pass and checkpoint reload, with search and
prerequisites disabled. These establish availability, not mastery. Bounded
diagnostic commands disable search; manual evaluation never uses it.

Reports separate Cherry Bomb, Peashooter and normal progress from oracle work, teacher losses and
greedy validation. `--report-only` loads no policy and starts no training. Reward
reports accept the one-based cohort number shown in logs.

- [Test ownership and code structure](AGENTS.md)
- [Architecture](docs/architecture.md)
- [Execution controls and diagnostics](docs/math/training-throughput.md)
- [Learning objective background](docs/math/training-objective.md)
- [Research references](docs/references.md)
- [Iteration history](docs/iteration.md)
