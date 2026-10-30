# Project instructions

Follow the global `AGENTS.md` for shared workflow, Git, documentation and testing
rules. This file adds project-specific constraints and ownership.

## Scope and invariants

- Use `.venv\Scripts\python.exe`. Keep one training/model framework and a separate,
  pinned game dependency. Smoke checks must not download dependencies.
- `src/pvz_rl/data/train.toml` is the single parameter source for training and demo
  overlays. Preserve the user's reward, exploration and cohort-cycle adjustments;
  read current values from configuration instead of duplicating them here.
- Short integration checks are allowed. Do not launch formal training unless asked.
- Policy inputs contain public board facts only. Keep seeds, future schedules,
  snapshots, target identities, oracle labels and accounting metadata private.
- Keep independent, stateless species and conditional tile networks behind the
  shared policy runner. All 45 tiles remain proposal candidates; classify occupied
  cells after selection. Branch impossibility precedes tile invalidity.
- Fit complete returns through shared objective/routing code. Invalid targets use
  the detached responsible-head minimum plus the configured offset; accepted
  non-wait species targets use online selection and detached EMA tile evaluation.
  Tile-only fitting must freeze species parameters and species EMA.
- Preserve CPU/CUDA action cadence, legality, reward conversion and source credit
  agreement. Diagnostics and viewers must not change learning or add hot-loop CUDA
  synchronization.
- Recovery validates saved model, objective, schema and engine identities. Resume
  retains saved learning settings and committed updates; `--init-from` transfers
  compatible weights into fresh optimizer state. Do not silently migrate formats,
  replay committed passes or treat annotation progress as autonomous recovery.

## Training stages and validation

- Default progression is `pre → easy → standard → shared`. Explicit fresh normal
  stages skip prerequisites; legacy checkpoints retain their saved workflow.
- `pre` uses the Peashooter/Cherry Bomb state machine, joint validation and revisits
  of a failing skill. Preserve configured Peashooter sky-sun scaling and saved
  countdowns without rescaling recovery state.
- Keep pre episodes and normal games, budgets, exploration and cohort-cycle counters
  separate. Validation and label-oracle rollouts do not count as training episodes.
- Autonomous validation checks occur at configured episode intervals after a fully
  fitted cohort. Use authoritative outcomes from the entire configured cohort,
  including attributed bomb hits; shortened cohorts and rolling statistics cannot
  qualify. A skipped check waits another interval.
- Keep cohort qualification separate from perfect validation mastery. Persist
  separate pre/normal clocks and pending validation; retry interrupted validation
  against preserved weights before further fitting. Missing saved trigger settings
  preserve legacy behavior. Manual evaluation and annotation convergence are separate.

## Code structure

Paths below are relative to `src/pvz_rl/` unless stated otherwise.

| Area | Ownership |
| --- | --- |
| Repository `tools/bootstrap.*`, `tools/stage_game.py` | Verified, non-editable installation of the pinned engine from an existing checkout. |
| `config.py`, `data/train.toml`, `gpu.py` | Shared settings, validation and device selection. |
| `envs/` | CPU/CUDA simulator adapters, public encoding, actions, rewards and source credit; `tile_scenarios.py` and `sky_sun.py` own scenario generation and sun scaling. |
| `policy/` | Independent two-stage Transformers, shared runner, all-tile selection, active batching and bounded collection/fitting CUDA graphs. |
| `learning/training.py`, `cuda_q.py`, `sequential_training.py` | Training orchestration, cohort lifecycle, collection, finalization and atomic recovery. |
| `learning/tile_curriculum.py`, `validation_trigger.py`, `training_requirements.py` | Prerequisite state machine, cohort validation gate and normal fitting schedule. |
| `learning/flat_q.py`, `objective.py`, `action_routing.py` | Shared autonomous/demo fitting, selected-action targets, consistency, ranking and wait/invalid compaction. |
| `learning/cuda_buffer.py`, `batch_transport.py`, `collection_scheduler.py`, `exploration.py` | Trajectories, bounded transport, decision tickets and persistent exploration commitments. |
| `learning/checkpoints.py`, `performance.py` | Compatibility contracts, durable snapshots and execution-only refresh settings. |
| `annotation/`, `learning/annotation_fit.py`, `demo_initialization.py` | Reachable-state sampling, atomic labels/conflicts, head-specific convergence and verified demo initialization. |
| `evaluation/`, `monitoring/`, repository diagnostic tools | Stateless evaluation, CPU verification of CUDA traces, bounded profiling, benchmarks and read-only reward reports. |
| `presentation/` | Shared renderers, annotation/live viewers and recordings; `training_charts.py` owns panels, while `visualization.py` reconciles resume history and builds policy-free reports. |

## Test ownership

`tests/test_smoke.py` is the entire permanent suite: exactly three tests, capped at
200 lines. Do not restore mathematical, boundary, backend or recovery suites. Use
temporary independent controls for targeted integration and boundary checks.

- CLI help: subprocess timeout of 15 seconds.
- CPU inference: a real public board, both policy stages, finite Q values with
  valid dimensions, and an action submitted to the simulator.
- Explicit `learning` check: two one-second CUDA games, two environments, one fitting
  pass, checkpoint reload and prediction. Disable compilation, presentation and
  prerequisites; use batch size 32, a two-minute allowance and a 180-second subprocess
  timeout. Missing CUDA requirements fail.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
```

The default command runs the two CPU checks; the second selects CUDA learning.
All outputs use temporary directories. These checks establish workflow availability,
not mastery or detailed mathematical, backend, recovery or presentation correctness.
