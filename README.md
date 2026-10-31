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
.\.venv\Scripts\python.exe -m pvz_rl annotate --port 6006
```

The studio labels positive tile sets only. Manual action labeling and explicit
absolute-infeasible labels are removed. Existing tile records reuse their positive
sets; legality still prevents labeling occupied cells as positive. Tile annotation
fitting keeps its top-1 ranking hinge. Both Cherry Bomb stages use a zero-margin
hit hinge plus expected Manhattan distance to the nearest successful tile, with
all 45 outputs competing. The distance weight defaults to 0.1 and softmax
temperature to 1.0 in `training.tile_scenarios`; distance is normalized by the
maximum board distance (12 cells). This adapts
[minimum-risk training](https://aclanthology.org/P16-1159/).

To start autonomous training when ready:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --seed 101 `
  --output runs\my-run --gpu 0 --gpu-profile rtx4070-laptop
```

Default progression is `cherry_bomb → cherry_bomb_obstacle → peashooter → easy → standard → shared`.
Each stage can also run independently. Explicit normal stages skip tile training:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage cherry_bomb --until-stage-complete `
  --seed 101 --output runs\cherry --gpu 0 --gpu-profile rtx4070-laptop
.\.venv\Scripts\python.exe -m pvz_rl train --stage cherry_bomb_obstacle --until-stage-complete `
  --seed 101 --output runs\cherry-obstacle --init-from runs\cherry\final.zip `
  --gpu 0 --gpu-profile rtx4070-laptop
```

Without `--until-stage-complete`, `--games` limits completed real episodes.
Peashooter uses easy waves, 50 starting sun, disabled mowers and configurable
1.5× sky-sun frequency. Cherry Bomb uses its exact attributed-hit oracle with no
exploration or search in either bomb stage. Obstacle boards contain 1–15 plants,
blocking competing blast cells so exactly one legal placement gives an attributed
hit; ordinary boards retain 1–8 plants. Each tile stage has its own episode budget and validation clock;
automatic progression preserves optimizer state, while separate `--init-from`
runs transfer weights into a fresh optimizer.
Cherry Bomb boards independently randomize each row's cart between ready and
disappeared; `training.tile_scenarios.bomb_mower_ready_probability` defaults to 0.5.
Resume keeps the saved probability; older runs without it keep all carts disappeared.
The live viewer follows new cohorts automatically. In Cherry Bomb, **Switch**
browses completed boards from the current cohort, including while fitting runs.
Both bomb stages default to 640 envs through `training.tile_scenarios.cherry_bomb_n_envs`
(`--cherry-bomb-envs`); `training.n_envs` (`--n-envs`) controls other stages.
Its preparation shares validated simulator setup, uses up to 32 persistent CPU
workers, and batches natural cooldown warmup and exact attributed-hit rollouts
on CUDA. Under
`training.performance`, `cherry_prepare_workers`, `cherry_oracle_lanes` and
`cherry_oracle_vram_mib` control execution (defaults: 32, 2048, 2048).
Workers respect the host-memory budget; scratch lanes shrink to retain GPU
headroom. Larger limits are allowed. `cherry_oracle = "cpu"` selects the exact CPU
reference; `cherry_cuda_warmup = false` keeps CPU cooldown advancement with the
CUDA oracle. One speculative CPU cohort overlaps fitting; unfinished preparation
is regenerated from unconsumed seeds on resume.
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

Each bomb stage independently checks validation every 10,000 completed episodes through
`curriculum.cherry_bomb_probe_interval_games`; Peashooter and normal stages check
every 2,000. Intervals round up to a fitted cohort boundary. A full cohort
must exceed 90% success to launch validation (normal 116/128; Cherry Bomb 577/640). Collection may be
search-assisted; mastery still requires 100/100 greedy network validation.
Each tile stage requires 100/100 on that skill alone, then completes or advances;
there is no joint probe or skill revisit in fresh runs. The combined `--stage pre`
is retired; old checkpoints retain their saved combined-stage workflow on resume.
Validation releases idle collection scratch storage and unused GPU caches before
allocating its oracle. Resume refreshes the configured Cherry interval, while an
already pending validation retries against the saved weights first.

Resume saved weights, optimizer, pending search, simulator and fitting progress:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --resume runs\my-run\latest.zip `
  --output runs\my-run --gpu 0 --gpu-profile rtx4070-laptop
```

Resume retains the saved bomb objective, stage order, search mode/settings and sky countdowns. Older currently
resumable checkpoints without search settings continue their original workflow.
Use `--init-from PATH` for a fresh optimizer and the current search configuration.
Annotation progress also supports compatible weight initialization; it is not an
autonomous recovery checkpoint.

After successful completion, autonomous training leaves only `final.zip`, containing
portable weights and the configuration needed for inference or `--init-from`.
Completed annotation/demo fitting likewise keeps only its `.pt` weight output;
annotation and demonstration inputs are preserved. Reports and recovery files are
kept while a run is interrupted, failed, time-limited or otherwise incomplete.
Completed weights cannot be resumed; start a new run with `--init-from`.

## Common checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\my-run --report-only
.\.venv\Scripts\python.exe tools\report_rewards.py runs\my-run 1
.\.venv\Scripts\python.exe tools\benchmark_cherry_collection.py `
  --envs 256 512 640 1024 --workers 8 16 24 32 --oracles cpu cuda `
  --warmup-backends cpu cuda `
  --output artifacts\cherry-collection.json
```

Default pytest runs two CPU checks. The explicitly selected learning check runs
two one-second CUDA games, one fitting pass and checkpoint reload, with search and
prerequisites disabled. These establish availability, not mastery. Bounded
diagnostic commands disable search; manual evaluation never uses it.
The Cherry benchmark performs collection without fitting and ranks combinations
by total collection seconds per completed env, including board preparation.

Reports separate Cherry Bomb, Peashooter and normal progress from oracle work, teacher losses and
greedy validation. `--report-only` loads no policy and starts no training. Reward
reports accept the one-based cohort number shown in logs.

- [Test ownership and code structure](AGENTS.md)
- [Architecture](docs/architecture.md)
- [Execution controls and diagnostics](docs/math/training-throughput.md)
- [Learning objective background](docs/math/training-objective.md)
- [Research references](docs/references.md)
- [Iteration history](docs/iteration.md)
