# Project instructions

Follow the global `AGENTS.md` for shared workflow, Git, documentation and testing.
This file owns project-specific invariants and the code map.

## Scope and invariants

- Use `.venv\Scripts\python.exe`. Keep one training/model framework and the
  separate pinned game dependency; never edit the installed engine or its pin.
  No dependency downloads in smoke checks or formal training without a request.
- `src/pvz_rl/data/train.toml` is the single parameter source for training and demo
  overlays. Preserve user tuning. Group stage settings under `training.cherry_bomb`,
  `cherry_bomb_obstacle`, `attacker_pre`, `peashooter` and `normal`; resolve once
  into the runtime layout. Shared scenario sampling attempts live in `tile_scenarios`.
- Policy inputs contain public board facts only. Identities, schedules, seeds,
  snapshots, target sets, traces and accounting context remain private.
- Encode the board once within a decision/fitting call. Use a shared encoder,
  ten-way action Q readout and action-conditioned 45-query cross-attention decoder.
  Wait and unavailable branches skip decoding. All 45 tiles remain candidates;
  branch impossibility precedes tile invalidity. No across-decision feature cache.
- All autonomous stages train cell reach and home danger alongside their primary
  loss; normal cohorts jointly train action/tile Q returns and auxiliary forecasts.
  Preserve reward conversion, tick discounting, rejection compaction/weighting,
  species exploration, greedy tiles, four passes, BF16 runtime and FP32 masters.
  No factual followers or normal tile-max species consistency.
- Reach predicts independent future cell visitation and per-lane home entry,
  including current non-headless positions and later spawns. Count swept cells,
  including vaults; stop at defeat and capture actual home entry after mower
  resolution. Full-public pre-action facts precede encoding truncation. After
  the final spawn, cells right of the rightmost zombie and empty lanes are safe.
  Pending spawns never imply zero danger for empty lanes. Use masked FP32 BCE,
  equal lane/decision normalization and separate stage cell/home weights.
- Bomb/attacker reach labels use only root snapshot constraints, with no private
  tails. Keep placement oracles and validation. Peashooter privately Waits after
  cutoff to natural termination; normal games add no tail. Incomplete normal
  cutoffs preserve observed positives and provable zeros, leaving others unknown.
  Private work changes no primary outcomes/rewards, progress or policy calls.
- CUDA reach hooks are owned, guarded extensions. No extra environments, policy
  calls or hot-loop host synchronization. Charge private storage to shared budgets;
  tracker/storage failures retain recovery rather than fabricating labels.
- Preliminary stages keep their automatic targets. They train shared/tile/reach parameters
  while action-readout parameters and their EMA stay untouched; shared updates may
  change action predictions. Update EMA only for the phase's trained groups.
  Cell/home weights are independent stage settings. Demo initialization has no auxiliary loss.
- Bomb/attacker stages use successful-set cross-entropy over all 45 outputs,
  including occupied cells. Stage temperature follows completed-cohort accuracy,
  fixed across passes. Peashooter uses prioritized visible-lane engine labels and
  column-decaying cross-entropy: global coverage rule, then undefended lanes, then
  endangered defended lanes. Its labels never override main actions.
- Manual annotation commands, studio, datasets workflow and fitter are removed.
  Preserve existing user datasets/recordings. Retain human recording, replay
  verification and demo initialization with their demonstration objectives and
  no reach supervision.
- New shared-model/objective identities reject old initialization and recovery;
  no weight conversion. Compatible `--init-from` uses fresh optimizer/counters.
  Resume preserves saved settings, pending work and committed passes. Reach lifecycle:
  `collecting → traces_complete → labels_ready → fitting → committed`.
- Preserve CPU/CUDA cadence, legality, physical trajectories, reward conversion and
  attribution agreement. Viewers/diagnostics must not affect learning.
- Preparation/prefetch is bounded and speculative: commit seeds only on consumption,
  regenerate unfinished work on resume, share RAM/VRAM limits, and preserve exact
  attributed labels. Respect independent configured stage cohort/resource sizes.
- Completed autonomous runs retain portable weights and visualizations. Interrupted,
  failed and incomplete runs retain recovery. Validation passes/skips/completion
  create or refresh reports. Close cohort sidecar owners before the trajectory
  directory owner; recovery archives retain unfinished work independently.
  Leave `docs/` untouched unless explicitly requested.

## Stages and validation

- Fresh progression: `cherry_bomb → cherry_bomb_obstacle → attacker_pre → peashooter
  → easy → standard → shared`. Explicit normal stages start directly. Preserve
  normal stage indices, independent budgets/counters and saved stage orders.
- Both bomb stages have no exploration/search. Obstacle boards have exactly one
  engine-verified legal hit. Attacker exercises supply one of four attackers,
  use conservative distance labels without candidate rollouts, and report fallback
  correctness separately from attributed kills. Peashooter keeps easy waves,
  disabled mowers, configured sky-sun scaling and saved countdowns without rescaling.
- Only real completed episodes advance budgets and validation clocks. Check at
  configured intervals after fully fitted cohorts. Entire configured cohorts must
  strictly exceed the qualification threshold; shortened cohorts cannot qualify.
  Skips wait another interval. Cherry Bomb and Peashooter strictly exceed their
  configured mastery thresholds; other fresh stages require perfect validation.
- Every validation uses fresh random seeds outside reserved splits. Save panels
  before execution and retry interruptions against the same weights before fitting.
  Greedy validation has no exploration, scratch labeling or reach-loss updates.
  Release idle scratch/graphs without losing workers or speculative seed state.

## Code structure

Paths are relative to `src/pvz_rl/` unless stated otherwise.

| Area | Ownership |
| --- | --- |
| Repository `tools/bootstrap.*`, `tools/stage_game.py` | Verified installation of the pinned engine. |
| `config.py`, `data/train.toml`, `gpu.py` | Stage tuning, validation panels, model dimensions and device selection. |
| `policy/` | Shared public encoder, conditional tile decoder, action/tile/cell/home readouts, stateless runner and bounded graphs/batching. |
| `envs/` | CPU/CUDA adapters, public encoding, actions, rewards, source credit and scenarios. `reach_tracking.py` owns guarded movement hooks; compact visit timestamps also record home entry after mower resolution. |
| `envs/scratch_game.py`, `bomb_preparation.py`, `cuda_bomb_oracle.py`, `cuda_snapshots.py`, `attacker_scenarios.py` | Validated automatic exercises, persistent preparation, bounded oracles and conservative attacker labels. |
| `learning/training.py`, `cuda_q.py`, `sequential_training.py` | Cohort orchestration, collection/fitting, reporting and atomic recovery. |
| `learning/reach_supervision.py`, `reach_objective.py` | Private full-board root facts, future-cell/home labels, resumable Peashooter continuations, masked BCE and normalization. |
| `learning/flat_q.py`, `objective.py`, `action_routing.py`, `placement_labels.py`, `bomb_objective.py` | Shared Q fitting/economics, legality attribution and automatic placement losses. |
| `learning/peashooter_rules.py`, `peashooter_teacher.py`, `envs/lane_preparation.py`, `cuda_lane_oracle.py` | Prioritized lane targets and resumable bounded lane oracles. |
| `learning/cuda_buffer.py`, `batch_transport.py`, `collection_scheduler.py`, `exploration.py` | Bounded trajectories/transport, decision tickets and committed species exploration. |
| `learning/tile_curriculum.py`, `validation_trigger.py`, `budget.py`, `training_requirements.py` | Stage transitions, validation qualification, budgets and model/objective identities. |
| `learning/checkpoints.py`, `training_artifacts.py`, `performance.py`, `demo_initialization.py` | Recovery/weights, artifact retention, execution settings and verified human demo fitting. |
| `evaluation/`, `monitoring/`, repository diagnostic tools | Greedy evaluation, CPU verification, bounded profiling and benchmarks. |
| `presentation/` | Demo recording/replay, live cohort subscriptions and retained reports. `reach_view.py` distinguishes raw cell/home forecasts from masked snapshot/factual targets; both viewers place the latest timestamped prediction beside each board, using the episode-scoped journal cache without model calls. |

## Test ownership

`tests/test_smoke.py` is the entire permanent suite: exactly three tests, at most
200 lines. Use temporary independent mathematical, boundary, backend and recovery
controls; do not expand permanent suites or repeat full runs without a new cause.

- CLI help: 15-second subprocess timeout.
- CPU inference: real public board, action/tile/reach shapes, finite values and an
  action submitted to the simulator.
- Explicit CUDA learning: two one-second games, two envs, one pass, recovery and
  portable reload/prediction/cleanup. Disable compilation, presentation, preliminary
  targets and reach tracking; batch 32, two-minute allowance, 180-second subprocess
  timeout. Missing CUDA requirements fail.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
```

The default command selects the two CPU checks. Outputs use temporary directories;
these checks establish workflow availability, not mastery.
