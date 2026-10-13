# Validation scope

Tests protect the stateless two-stage Transformers, public simulator adapters,
separated learning targets, background collection and saved-run contracts.
Correctness checks are not evidence of improved win rate or curriculum mastery.

## Coverage ownership

Keep one owner per behavior; independent mathematics, backend parity and integration
at different interfaces are distinct controls. Retired inputs are tested only for
explicit rejection, never as supported models. Record executed results only in
[iteration history](iteration.md), preserving raw evidence.

| Behavior | Owning suites |
|---|---|
| Public records, phases, cap and private-state exclusion | `test_observations.py`, `test_cuda_learning.py` |
| Both-stage attention math, dimensions, conditioning, statelessness, padding/permutation | `test_two_stage_transformer.py` |
| Active buckets, canonical slot/RNG order, graph ownership/headroom/eviction | `test_two_stage_transformer.py`, `test_collection.py` |
| Selected-action attribution, detached consistency, probe/ranking math | `test_two_stage_q.py`, `test_q_math.py` |
| Flat batches, whole-pass updates, interrupted/restored fitting | `test_flat_training.py` |
| Background independence, source/lane isolation, FIFO backpressure, drain | `test_probe_rollouts.py` |
| Ragged storage, active rows, spill/prefetch, exact-once patches | `test_entity_storage.py` |
| Wait/dig/plant proposals, rejection and zero-time semantics | `test_q_selection_fallback.py`, `test_environment.py`, `test_action_timing.py` |
| Reward arithmetic, early Sunflower eligibility and CPU/CUDA caps | `test_net_value.py`, `test_reward_contract.py`, `test_kill_rewards.py` |
| Combined schedules and persistent species commitments | `test_exploration_schedule.py` |
| Replay verification, reward repricing and both-stage demo fitting | `test_demo_initialization.py`, `test_collision_follow_recording.py` |
| Fresh-weight preflight and complete current recovery contracts | `test_checkpoints.py`, `test_config.py`, `test_flat_training.py` |
| Curriculum, budgets, scheduling and lifecycle | `test_stage_training.py`, `test_game_budget.py`, `test_validation_schedule.py`, `test_training_lifecycle.py` |
| Evaluation, pinned engine, replay, statistics and reporting | `test_evaluation_refill.py`, `test_game_upgrade.py`, `test_baselines_statistics.py`, `test_reporting_config.py` |
| Read-only viewer, journal retention and output isolation | `test_live_view.py`, `test_action_journal.py`, `test_visualization.py` |
| Interval rates, cohort counts/returns, terminal cadence and snapshot isolation | `test_progress.py`, `test_training_progress_context.py` |

## Required independent and boundary controls

- Compare each stage's values/gradients with serial/dense attention, including exact
  fallback, empty/crowded inputs and required widths/layers/heads. Check permutation,
  padding, requested-branch conditioning and duplicate request reuse. Identical public
  current inputs must be independent of prior gameplay; position, health, resources
  and cooldown changes must have a controlled route to Q values.
- Use attribution counterexamples: poor actual tiles update only the tile network,
  waits update species wait, and consistency updates only species toward detached
  frozen-EMA geometry maxima. Cover correct species/wrong tile, wrong species/best
  tile, both wrong, ties and full-board fallback. Independent whole-cohort counts
  check 3x2 selected, 2x2 consistency and 10x2 probe strata, missing groups/outcomes,
  ranking anchors and flat-batch partition-invariant losses/gradients.
- Preserve ordinary/committed proposals, original-slot behavior draws, zero-time
  ordering, indefinite/full-board commitment waits and inactive-slot isolation.
  Wait skips tile inference; sun/cooldown do not hide ordinary branches.
- Compare independent CPU/CUDA forks for delayed shooting, mine arming and Sunflower
  income, including initial rejection, terminal/cutoff, horizon/cap and no fabricated
  waits/defeat. Accepted-only same-species alternatives must be distinct, exclude the
  actual tile and never recurse. Actual gameplay/accounting/RNG remains isolated.
- Exercise planting-owner/other-slot progress while probes remain unfinished,
  simultaneous triggers, repeated zero-time jobs and noncontiguous/shrinking live
  slots. Saturated capacity blocks only the retained eligible planting; FIFO and
  one-source/lane fallback must preserve every proposal/probe without redrawing.
- Storage owns out-of-order/multiple-job exact-once patches, RAM/spill, owned endpoint
  transfers and malformed/missing evidence rejection. Capacity gaps are neither
  decisions nor terminal boundaries. Flat transport excludes them and releases
  staging/index scratch after interruption. Endpoint dedup checks exact public inputs,
  EMA version and branch where relevant; ongoing simulations never merge.
- Recovery interrupts before execution, during auxiliary work and endpoint transfer,
  drains already-issued work, then compares restored execution with uninterrupted
  control. Missing evidence/old weights fail before simulator allocation. Rebuilt
  maps/caches are execution-only; committed optimizer/EMA updates are not repeated.
- Graph controls cover both encoders' overheads 1/47, owned replay/backward results,
  per-encoder cache eviction, zero entries, headroom and exact eager fallback.
  Quiescent collect/fit/collect changes must release outgoing resources without
  releasing pending consumers. Precision retries discard the whole uncommitted pass.

Original successful cases, known failures, boundaries and counterexamples belong
together; do not establish parity by weakening acceptance criteria.

## Running checks

Run from the repository root, with the GPU free of formal training:

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_two_stage_transformer.py tests/test_two_stage_q.py `
  tests/test_flat_training.py tests/test_entity_storage.py `
  tests/test_collection.py tests/test_probe_rollouts.py tests/test_q_math.py `
  tests/test_demo_initialization.py tests/test_checkpoints.py
```

Run owning targeted suites once and focused repairs as needed. A full suite is
reserved for relevant shared-core coverage, not repeated as an execution benchmark.
Some CUDA tests lack the `learning` marker; excluding that marker is not CPU-only.
Short tests/controls are not authorization for formal or unbounded training.

## Bounded performance controls

Measure both-stage inference/fitting time, species-consistency inference, selected
error, consistency error and conditional best-tile minus sampled-tile gap. Separate
setup/capture/warm-up, warmed trials, actual-window latency and probe drain time.
Report workspace/parameter/EMA/staging memory, logical/padded/requested rows,
auxiliary work/stops, queue age/backpressure and graph fallback activity.

Compare zero/one/simultaneous planting triggers, sparse/mixed/crowded boards and
128/64/32/16/4/1 noncontiguous live subsets in fixed 128-slot allocations.
Use a bounded collect → four-pass-fit → collect control and one bounded default
30-second-horizon probe; do not substitute a short horizon without labeling it.
Correctness and bounded fair scheduling do not imply unchanged throughput.

Historical `docs/evidence/` and `docs/figures/` remain unchanged. Their measurements
apply only to their recorded architectures/settings, not the current replacement.
Current owning controls, focused corrections and bounded measurements are indexed
in [verification evidence](evidence/two-stage-verification.json); results and
limitations belong in [iteration history](iteration.md). The planting fit control
explicitly shortens the horizon, while a separate one-probe control uses the default
30 seconds. Neither establishes general throughput or win-rate improvement. No
full-suite pass is claimed; existing net-value expectations for loss penalty 2.0
conflict with the unchanged shipped 1.0 setting.
