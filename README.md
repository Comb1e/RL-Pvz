# PVZ plant research

Train a stateless two-stage Transformer against a pinned Plants vs. Zombies
simulator. One network selects species/Wait/Dig; another scores all 45 tiles.
Both consume only public board facts. Fresh Peashooter and normal runs compare factual simulator forks to teach tile
preferences; main play and validation use the network alone.

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
fitting keeps its top-1 ranking hinge. Fresh Cherry Bomb stages use only
`-log(sum(successful-tile softmax probabilities))`: ordinary cross-entropy for the
obstacle stage's sole successful tile, and no prescribed probability split among
ordinary Cherry Bomb's successful tiles. All 45 outputs compete, including occupied
cells. Set `softmax_temperature` independently in `[training.cherry_bomb]` and
`[training.cherry_bomb_obstacle]`, using finite positive `zero_success` and
`full_success` endpoints; increasing, decreasing and constant schedules are supported.
Temperature changes linearly with the
completed training cohort's attributed-hit rate and stays fixed for its fitting
passes, including replay after interruption. A scalar retains a fixed temperature.
There is no hinge or distance penalty. Reports show cross-entropy and temperature;
selected-tile distance remains a diagnostic. Resume preserves saved legacy objectives
and settings; use `--init-from` for fresh fitting with the new loss. The set objective
is also studied in [partial-label learning](https://www.proceedings.com/content/079/079017-2841open.pdf).

To start autonomous training when ready:

Tune shared bomb settings in `[training.cherry_bomb]`, with an independent temperature
in `[training.cherry_bomb_obstacle]`, the mixed attacker exercise in
`[training.attacker_pre]`, Peashooter in `[training.peashooter]`, and normal play in `[training.normal]` (including its
`curriculum` tables). Shared fitting, rewards and resource limits retain their
common tables. The loader resolves stage tables into the existing runtime layout;
legacy configuration files and saved checkpoints remain supported.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --seed 101 `
  --output runs\my-run --gpu 0 --gpu-profile rtx4070-laptop
```

Default progression is `cherry_bomb → cherry_bomb_obstacle → attacker_pre → peashooter → easy → standard → shared`.
Each stage can also run independently. Explicit normal stages skip tile training:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage cherry_bomb --until-stage-complete `
  --seed 101 --output runs\cherry --gpu 0 --gpu-profile rtx4070-laptop
.\.venv\Scripts\python.exe -m pvz_rl train --stage cherry_bomb_obstacle --until-stage-complete `
  --seed 101 --output runs\cherry-obstacle --init-from runs\cherry\final.zip `
  --gpu 0 --gpu-profile rtx4070-laptop
.\.venv\Scripts\python.exe -m pvz_rl train --stage attacker_pre --until-stage-complete `
  --seed 101 --output runs\attacker-pre --init-from runs\cherry-obstacle\final.zip `
  --gpu 0 --gpu-profile rtx4070-laptop
```

Without `--until-stage-complete`, `--games` limits completed real episodes.
`attacker_pre` supplies a random Potato Mine, Peashooter, Repeater or Snow Pea.
Its boards contain 4–8 background plants across four occupied lanes, 1–2 Basic
zombies in the empty lane, and random ready/disappeared carts. Rules-derived
conservative firing/arming distances label placements without candidate rollouts.
Shooters must defeat both zombies; Potato Mine targets the zombie initially
closest to home. If no tile qualifies, that lane's first cell is the fallback.
Fitting uses successful-set cross-entropy, without search or exploration.
Validation counts a correct fallback as success and reports actual kills separately;
this exercise teaches lane selection, not optimal placement. Conservative bounds
can produce many fallbacks, particularly for two-zombie Peashooter/Snow Pea cases.
Tune its cohort size, mixture, generation, temperature and preparation settings
under `[training.attacker_pre]`; its scratch `oracle_lanes` and `oracle_vram_mib`
limits are independent of Cherry Bomb and apply only to cooldown warmup.
Measured RTX 4070 Laptop defaults are 640 envs, 16 workers, 640 scratch lanes and
a 1024 MiB scratch cap. Rebenchmark on other hardware.
Peashooter uses easy waves, 50 starting sun, disabled mowers and configurable
1.5× sky-sun frequency. Cherry Bomb uses its exact attributed-hit oracle with no
exploration or search in either bomb stage. Obstacle boards contain 1–15 plants,
blocking competing blast cells so exactly one legal placement gives an attributed
hit; ordinary boards retain 1–8 plants. Each tile stage has its own episode budget and validation clock;
automatic progression preserves optimizer state, while separate `--init-from`
runs transfer weights into a fresh optimizer.
Cherry Bomb boards independently randomize each row's cart between ready and
disappeared; `training.cherry_bomb.mower_ready_probability` defaults to 0.5.
Resume keeps the saved probability; older runs without it keep all carts disappeared.
The live viewer follows new cohorts automatically. In Cherry Bomb, **Switch**
browses completed boards from the current cohort, including while fitting runs.
Both bomb stages default to 640 envs through `training.cherry_bomb.n_envs`
(`--cherry-bomb-envs`); `training.normal.n_envs` (`--n-envs`) controls Peashooter and normal stages.
Its preparation shares validated simulator setup, uses up to 32 persistent CPU
workers, and batches natural cooldown warmup and exact attributed-hit rollouts
on CUDA. Under
`training.cherry_bomb`, `prepare_workers`, `oracle_lanes` and
`oracle_vram_mib` control execution (defaults: 32, 2048, 2048).
Workers respect the host-memory budget; scratch lanes shrink to retain GPU
headroom. Larger limits are allowed. `oracle = "cpu"` selects the exact CPU
reference; `cuda_warmup = false` keeps CPU cooldown advancement with the
CUDA oracle. One speculative CPU cohort overlaps fitting; unfinished preparation
is regenerated from unconsumed seeds on resume.
Normal fitting keeps three tile-only cohorts followed by one joint cohort, with
four passes each. Tile-only fitting freezes species parameters and species EMA.

Settings live in [train.toml](src/pvz_rl/data/train.toml). Peashooter uses
prioritized engine-labeled targets: spread Peashooters across lanes on a safe board,
defend a zombie lane without a Peashooter, then help an endangered defended lane.
The oracle tests only current visible zombies, without future spawns or further
planting. Main games stay greedy; labels never override actions. Tune
`column_decay` (0.5), `softmax_temperature` (1.0), `oracle_lanes` (256),
`oracle_vram_mib` (512) and `oracle_chunk_ticks` (256) under `[training.peashooter]`.
All 45 cells participate in cross-entropy. Existing recovery retains its saved
objective; use `--init-from` to transfer weights into the new fitting mode.

Normal stages expose `counterfactual_enabled` and `comparison_temperature` (default 1.0).
Every legal planting forks up to two same-species alternatives: next-best legal
tile and a distinct random legal tile. Followers replay accepted main commands,
without Transformer calls or recursive branching. Occupancy conflicts discard the
follower; resource shortages wait in order. Main actions stay greedy across all
45 tiles; normal species exploration is retained.

`[training.counterfactual]` bounds total resident environments, scratch VRAM and
chunk length. Pending roots share compact host/disk state under the trajectory RAM
budget. Collection includes final branch draining. Only main games advance budgets.
Defaults are 1,536 total lanes, a 1,536 MiB scratch budget and one decision per chunk,
selected using bounded collection benchmarks. The total includes inactive simulator
capacity retained by an automatic pipeline; the actual scratch allocation may shrink.
Planting tiles fit pairwise cross-entropy; species and accepted legal Dig retain
complete-return targets. Same-board duration shaping remains: a useful longer loss
can still rank below a shorter wasteful loss.

Each bomb stage independently checks validation every 10,000 completed episodes through
`training.cherry_bomb.probe_interval_games`; attacker_pre uses its own
`probe_interval_games`. Peashooter and normal stages check every 2,000.
Intervals round up to a fitted cohort boundary. A full cohort
must exceed 90% success to launch validation (normal 116/128; Cherry Bomb 577/640). Collection outcomes are the unassisted main games; mastery uses greedy network
validation without branches.
Cherry Bomb must strictly exceed `training.cherry_bomb.validation_success_threshold`
(default 0.95: 96/100 passes; 95/100 fails). Obstacle, attacker_pre, Peashooter and normal stages
still require 100/100. Each tile stage validates its own skill, then completes or advances;
there is no joint probe or skill revisit in fresh runs. The combined `--stage pre`
is retired; old checkpoints retain their saved combined-stage workflow on resume.
Validation releases idle collection scratch storage and unused GPU caches before
allocating its oracle. Resume refreshes the configured Cherry interval, while an
already pending validation retries against the saved weights first.

Resume saved weights, optimizer, pending factual forks, simulator and fitting progress:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --resume runs\my-run\latest.zip `
  --output runs\my-run --gpu 0 --gpu-profile rtx4070-laptop
```

Resume retains saved objectives, stage order, replay settings and sky countdowns.
Old non-search and placement-only recovery retains its saved workflow. Retired
slow search-teacher checkpoints require `--init-from PATH`; this transfers compatible
weights into a fresh optimizer and counters rather than resuming search.
Annotation progress also supports compatible weight initialization; it is not an
autonomous recovery checkpoint.

Validation checks create or refresh the report even when skipped; completed validation
refreshes it even when mastery passes. After successful completion,
autonomous training keeps `final.zip`, containing portable weights and the configuration
needed for inference or `--init-from`, together with generated `visualizations/` reports,
charts and recordings. Stages without a validated `best.zip` still generate a report.
Completed annotation/demo fitting likewise keeps only its `.pt` weight output;
annotation and demonstration inputs are preserved. Recovery files are kept while a
run is interrupted, failed, time-limited or otherwise incomplete.
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
.\.venv\Scripts\python.exe tools\benchmark_cherry_collection.py --stage attacker_pre `
  --envs 128 256 512 1024 --workers 4 8 16 32 --lanes 128 512 1024 `
  --vram-mib 128 512 --warmup-games 4096 --output artifacts\attacker-collection.json
.\.venv\Scripts\python.exe tools\benchmark_factual_collection.py --automatic-layout `
  --caps 768 1024 1536 2048 --vram 1280 1536 2048 --chunks 1 4 16 `
  --output artifacts\factual-collection.json
```

Default pytest runs two CPU checks. The explicitly selected learning check runs
two one-second CUDA games, one fitting pass and checkpoint reload, with counterfactuals and
prerequisites disabled. These establish availability, not mastery. Bounded
diagnostic commands disable counterfactuals; manual evaluation never creates branches.
The placement benchmark performs collection without fitting and ranks combinations
by total collection seconds per completed env, including board preparation and
policy inference. Worker startup and warmup samples are reported separately and
excluded from steady-state ranking. `--resume` continues an interrupted parameter sweep;
resource-infeasible combinations are recorded without stopping the sweep.
Attacker comparisons reset the distance-bound cache; `--warmup-games` sets the
minimum warmup episode count before fresh measurement panels.

The factual benchmark includes cloning, replay, spilling and final draining in
collection seconds per main env. It refuses concurrent training and runs no fitting;
its bounded workload does not predict full-game throughput.

Reports separate tile-stage and normal progress from oracle/branch work, pairwise
losses and greedy validation. `--report-only` loads no policy and starts no training. Reward
reports accept the one-based cohort number shown in logs.

- [Test ownership and code structure](AGENTS.md)
- [Architecture](docs/architecture.md)
- [Execution controls and diagnostics](docs/math/training-throughput.md)
- [Learning objective background](docs/math/training-objective.md)
- [Research references](docs/references.md)
- [Iteration history](docs/iteration.md)
