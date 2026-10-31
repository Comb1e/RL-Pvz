# Project instructions

Follow the global `AGENTS.md` for shared workflow, Git, documentation and testing
rules. This file adds project-specific constraints and ownership.

## Scope and invariants

- Use `.venv\Scripts\python.exe`. Keep one training/model framework and a separate,
  pinned game dependency. Smoke checks must not download dependencies.
- `src/pvz_rl/data/train.toml` is the single parameter source for training and demo
  overlays. Preserve the user's reward, exploration and cohort-cycle adjustments;
  read current values from configuration instead of duplicating them here. Group
  stage tuning under `training.cherry_bomb`, `training.peashooter` and `training.normal`;
  resolve these once at loading into the checkpoint-compatible runtime layout.
- Short integration checks are allowed. Do not launch formal training unless asked.
- Policy inputs contain public board facts only. Keep seeds, future schedules,
  snapshots, target identities, oracle labels and accounting metadata private.
- Keep independent, stateless species and conditional tile networks behind the
  shared policy runner. All 45 tiles remain proposal candidates; classify occupied
  cells after selection. Branch impossibility precedes tile invalidity.
- Manual annotations are positive tile sets only; there is no action annotation
  workflow or absolute-infeasible tile labeling. Annotation fitting keeps its top-1
  hinge. Both bomb stages use a configurable hit-ranking margin plus bounded
  expected distance over all 45 outputs; illegal tiles use maximum board distance.
  Legacy recovery retains its saved objective.
- Fresh autonomous training fits confirmed search values/differences and finite
  feasibility bounds through shared fitting. Legacy/diagnostic complete-return
  fitting retains its minimum-Q invalid targets and EMA species consistency.
  Tile-only fitting freezes species parameters and species EMA in both modes.
- Search uses public-board priors, one bounded root budget and fresh-world
  confirmation. Private world state never enters selection or network inputs.
  Only real episodes advance budgets; mastery validation never uses search.
- Preserve CPU/CUDA action cadence, legality, reward conversion and source credit
  agreement. Diagnostics and viewers must not change learning or add hot-loop CUDA
  synchronization.
- Recovery validates saved model, objective, schema and engine identities. Resume
  retains saved learning settings and committed updates; `--init-from` transfers
  compatible weights into fresh optimizer state. Do not silently migrate formats,
  replay committed passes or treat annotation progress as autonomous recovery.
- Cherry Bomb has its own configured cohort size; other stages retain `n_envs`.
  CPU preparation and the bounded CUDA oracle must preserve exact attributed-hit
  labels. Cherry carts start independently ready or spent, never moving; saved
  configurations without the setting retain all-spent carts. Prefetch is speculative:
  commit seeds/outcomes only on consumption, regenerate unfinished work on resume,
  and share the RAM/VRAM budgets. CUDA warmup reuses the same bounded oracle lanes.
  Completed autonomous training retains portable weights and generated visualizations;
  validation and completion create or refresh reports even when a check skips or
  mastery passes. Interrupted, failed and incomplete runs keep recovery artifacts;
  annotation inputs remain intact.

## Training stages and validation

- Default progression is `cherry_bomb → cherry_bomb_obstacle → peashooter → easy → standard → shared`.
  Each tile stage is separately selectable and validates only its own skill.
  Explicit fresh normal stages skip tile stages; legacy checkpoints retain their
  saved combined-stage workflow. Both bomb stages have no exploration or search.
  Obstacle boards use 1–15 plants and require exactly one engine-verified legal hit;
  scenario identity travels with worker/prefetch jobs and never enters policy input.
- Preserve configured Peashooter sky-sun scaling and saved countdowns without
  rescaling recovery state. Keep tile-stage episodes, budgets and validation clocks
  separate from one another and from normal games, exploration and cohort cycles.
  Validation and label-oracle rollouts do not count as training episodes.
- Autonomous validation checks occur at configured episode intervals after a fully
  fitted cohort. Use authoritative outcomes from the entire configured cohort,
  including attributed bomb hits; shortened cohorts and rolling statistics cannot
  qualify. A skipped check waits another interval.
- Both bomb stages independently use the Cherry Bomb probe interval; resume refreshes that interval
  but retries pending validation first. Validation releases idle collection scratch
  and graph storage without discarding CPU workers or speculative seed state.
- Keep cohort qualification separate from perfect validation mastery. Persist
  separate tile-stage/normal clocks and pending validation; retry interrupted validation
  against preserved weights before further fitting. Missing saved trigger settings
  preserve legacy behavior. Manual evaluation and annotation convergence are separate.

## Code structure

Paths below are relative to `src/pvz_rl/` unless stated otherwise.

| Area | Ownership |
| --- | --- |
| Repository `tools/bootstrap.*`, `tools/stage_game.py` | Verified, non-editable installation of the pinned engine from an existing checkout. |
| `config.py`, `data/train.toml`, `gpu.py` | Grouped stage tuning, shared settings, validation and device selection; configuration loading preserves the saved runtime layout. |
| `envs/` | CPU/CUDA simulator adapters, public encoding, actions, rewards and source credit; `tile_scenarios.py` and `sky_sun.py` own scenario generation and sun scaling. |
| `envs/scratch_game.py`, `bomb_preparation.py`, `cuda_bomb_oracle.py`, `cuda_snapshots.py` | Validated forks/cart setup, persistent workers/prefetch, CUDA cooldown warmup/attributed labels and batched snapshot transfers. |
| `policy/` | Independent two-stage Transformers, shared runner, all-tile selection, active batching and bounded collection/fitting CUDA graphs. |
| `learning/training.py`, `cuda_q.py`, `sequential_training.py` | Training orchestration, cohort lifecycle, collection, finalization and atomic recovery. |
| `learning/tile_curriculum.py`, `validation_trigger.py`, `budget.py`, `training_requirements.py` | Independent tile-stage clocks/progression, legacy prerequisite recovery, cohort validation gate and normal fitting schedule. |
| `learning/flat_q.py`, `objective.py`, `action_routing.py`, `bomb_objective.py` | Shared fitting/routing, legacy targets/consistency, annotation ranking and bomb hit/distance loss; `envs/tile_scenarios.py` shares illegal-aware distance labels with reporting. |
| `envs/search_worlds.py`, `learning/search_teacher.py`, `search_targets.py`, `search_settings.py` | Public world reconstruction, interruptible CUDA search/continuations, confirmed targets and versioned settings/recovery. |
| `learning/cuda_buffer.py`, `batch_transport.py`, `collection_scheduler.py`, `exploration.py` | Trajectories, bounded transport, decision tickets and persistent exploration commitments. |
| `learning/checkpoints.py`, `training_artifacts.py`, `performance.py` | Compatibility contracts, durable snapshots, completed-run weight/report retention and execution-only refresh settings. |
| `annotation/`, `learning/annotation_fit.py`, `demo_initialization.py` | Reachable-state sampling, atomic positive tile sets/conflicts, top-1 convergence and verified demo initialization. |
| `evaluation/`, `monitoring/`, repository diagnostic tools | Stateless evaluation, CPU verification of CUDA traces, bounded profiling, benchmarks and read-only reward reports; `tools/benchmark_cherry_collection.py` ranks env/worker/oracle combinations by total collection time per env. |
| `presentation/` | Shared renderers, annotation/live viewers and recordings; `live_capture.py`/`live_view.py` own cohort subscriptions and public result caching. `training_charts.py` owns panels; `visualization.py` reconciles resume history and builds policy-free reports. |

## Test ownership

`tests/test_smoke.py` is the entire permanent suite: exactly three tests, capped at
200 lines. Do not restore mathematical, boundary, backend or recovery suites. Use
temporary independent controls for targeted integration and boundary checks.

- CLI help: subprocess timeout of 15 seconds.
- CPU inference: a real public board, both policy stages, finite Q values with
  valid dimensions, and an action submitted to the simulator.
- Explicit `learning` check: two one-second CUDA games, two environments, one fitting
  pass, recovery/portable weight reload, completion cleanup and prediction. Disable
  compilation, presentation, prerequisites and search; use batch size 32, a two-minute allowance and a
  180-second subprocess timeout. Missing CUDA requirements fail.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
```

The default command runs the two CPU checks; the second selects CUDA learning.
All outputs use temporary directories. These checks establish workflow availability,
not mastery or detailed mathematical, backend, recovery or presentation correctness.
