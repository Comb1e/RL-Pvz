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
- Fresh policy inputs are public board facts: entity records and 18 globals.
  Simulated field/probe outcomes are private auxiliary targets, never observations.
  Future spawns are suppressed inside forks; IDs, schedules, seeds and snapshots stay private.
- Encode once per decision/fitting call: shared encoder, ten-way action Q head,
  action-conditioned 45-query decoder. Wait/unavailable branches skip collection
  decoding; early rejected proposals still decode their own outcome token during
  fitting. Later stages use a pooled null branch for Wait labels. All tiles remain
  candidates. Cache memory within one forward only; branch impossibility precedes tile invalidity.
- All models are outcome-only. Factual prediction heads, horizon/snapshot labels,
  losses, summaries, storage and legacy resumes are removed. Early stages use
  `placement_outcomes`: exactly one post-proposal field/probe job per exercise,
  including rejections; no null/Wait target or alternatives. Keep placement teachers
  and stage weights. Train shared/tile/outcome parameters; freeze action-type/null
  parameters and EMA. Early tile scores receive no outcome-summary injection.
  Reserve no Wait boundary buffers for placement exercises; charge only their
  occupied proposal packets and admitted workers to auxiliary VRAM.
  Peashooter/normal use `action_outcomes` with explicit valid Wait references.
  Predict absolute field depth (11), breach bin (6), lane loss (2) and probe depth
  (11); preserve the 56-slot carrier, excluding reserved classes. Average known
  terms/lanes, valid tokens and roots equally; never invent unsimulated labels.
- Static forks own physical/RNG storage, suppress actions/future spawns/logs and
  resolve mowers before recording home entry. Continue other lanes after a loss;
  probes inherit damaged plants and spent mowers. The real game is unchanged.
  Fresh inference and validation compute no field/probe simulations. Learned
  outcome summaries retain detached tile-Q injection and its schedule.
- `absolute-wait-trajectory-v4` follows real Waits/rejections and records guarded per-tick
  field observations. One semantic probe belongs to each owned interval, anchored
  at its first field-clearing state (immediately when empty). Accepted planting/Dig,
  actual spawn-counter changes and reset/end close intervals; deaths do not.
  Capture pre-spawn full-tick states before physical/RNG changes, including spawns
  that die within one decision. Finish interrupted fields from those owned roots;
  never replay borrowed prefixes. Rebase breach times and exclude penetration
  before each decision, including within the preceding tick. A successful action's
  target belongs to its new post-action interval. Unexecuted alternatives need
  isolated forks. Cache entries remain exact, bounded and disposable; pending roots,
  choices, row references, anchors and partial jobs survive recovery. Queue closed
  intervals independently; publish exact labels before fitting. Resource failures
  retain work. Charge capture/roots/workers/cache against the shared fork VRAM cap
  and metadata against trajectory RAM; spill pending physical roots to owned storage.
- Public-model auxiliary execution uses one hybrid service for interval outcomes
  and the unchanged Peashooter teacher. Freeze occupied physical records before
  real mutation; asynchronous host buffers and device roots remain owned until
  their events finish. Main policy/environment work has GPU priority; admit only
  one bounded low-priority CUDA chunk and use measured CPU assistance. Never use
  GPU utilization percentages as admission control. Worker processes import no
  model/CUDA modules, own RNG/physical state, and persist across cohorts.
  Queue/dependency state, partial CPU/CUDA continuations and unpublished results
  survive recovery. Publish each stable job identity once. Checkpoints quiesce at
  chunk boundaries; events/process handles are execution resources, not state.
  Charge capture/packets/roots/workers/caches to the combined auxiliary VRAM cap
  and CPU worker reservations/metadata to shared trajectory RAM. Spill I/O and
  decoding belong to the coordinator. Resource failure preserves work; it never
  creates a label. Inference computes no private outcomes. Choose tuning by total
  time per environment and report main-decision latency and cohort drain separately.
- `interval_backend = "packed_async"` routes interval jobs independently of the
  teacher. Keep occupied roots device-resident from capture through private worker
  copies; ordinary collection must not decode, serialize, spill or re-upload them.
  Download roots only for recovery or bounded storage pressure. Event-polled
  pinned progress/results belong to one chunk until consumed. Refill persistent,
  memory-admitted panels; count trajectory/root storage incrementally and publish
  decision targets in interval batches. Use the configured synchronous or packed
  backend. Peashooter teacher rules, sampling and synchronous routing remain intact
  unless the shared hybrid setting explicitly selects both families.
- Teacher drain may pipeline incremental root preparation with its unchanged
  CUDA simulations. Poll double-buffered pinned progress only after its event;
  checkpoint by consuming completed progress before saving partial states.
  Validate owned roots once, batch slab uploads, charge shared cache allocations
  and in-flight refills, and commit queue ownership only after copy admission.
  Vectorized outcome gathers retain valid null/token/root loss weights.
  Collection decoder row padding never adds an action or removes a public entity.
- Legal non-Wait proposals always receive outcome supervision. Effective Waits
  sample alternatives with the configured probability. Two slots hold a legal
  branch runner-up (ties by action ID) and a uniform remaining legal action;
  keep five-slot storage with unused slots invalid. Fresh Peashooter uses zero
  alternatives, retaining its teacher and actual-action targets. Normal stages
  retain shared sampling.
- Normal cohorts jointly fit frozen-cohort n-step Double-Q targets and auxiliaries.
  Preserve economic conversion, tick discounting, rejection compaction/weighting,
  species exploration, greedy tiles, four passes, BF16 and FP32 masters. No
  counterfactual followers or tile-max species consistency. Forecast injection
  is detached; potential shaping starts disabled. There is no separate private
  forecast pretraining phase, sampler, graph, configuration or recovery state.
- Main-game guarded CUDA field traces record per-tick depth before defeat and
  actual home entries after mower resolution. They contain no factual horizons or
  bite-contact targets. Resource failures preserve exact work without creating labels.
- Preliminary stages retain automatic placement objectives. Peashooter additionally
  trains null parameters. Update EMA only for each phase's trained groups; shared
  updates may change action predictions. Demo initialization has no auxiliary loss.
- Bomb/attacker stages use successful-set cross-entropy over all 45 outputs,
  including occupied cells. Stage temperature follows completed-cohort accuracy,
  fixed across passes. Peashooter uses prioritized visible-lane engine labels and
  column-decaying cross-entropy: global coverage rule, then undefended lanes, then
  endangered defended lanes. Its labels never override main actions.
- Manual annotation commands, studio, datasets workflow and fitter are removed.
  Preserve existing user datasets/recordings. Retain human recording, replay
  verification and demo initialization with their demonstration objectives and
  no auxiliary supervision.
- Only `shared_public_outcomes_q_v5` checkpoints are supported. Reject every older
  checkpoint before environment/CUDA preparation for inference, resume and
  initialization; no conversions or legacy execution. New-version recovery retains
  optimizer/counters, owned roots/rows/choices, partial jobs, unpublished results
  and committed passes. Publish every target once before fitting.
  A setup failure before any issued decision may rebuild an empty scheduler from
  the saved boards. Missing tickets after collection starts remain invalid.
- Preserve CPU/CUDA cadence, legality, physical trajectories, reward conversion and
  attribution agreement. Viewers/diagnostics must not affect learning.
- Preparation/prefetch is bounded and speculative: commit seeds only on consumption,
  regenerate unfinished work on resume, share RAM/VRAM limits, and preserve exact
  attributed labels. Respect independent configured stage cohort/resource sizes.
  Release idle preparation scratch before allocating forks; live exercise viewing
  sizes staging from transferred live counts, never the full projectile capacity.
  Fresh cohorts draw independent seeds; save pending draws before prefetch so
  recovery reproduces unfinished boards. Seed-matched diagnostics disable this mode.
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
  use conservative distance placement labels, and report fallback
  correctness separately from attributed kills. Peashooter keeps easy waves,
  disabled mowers, configured sky-sun scaling and saved countdowns without rescaling.
- Only real completed episodes advance budgets and validation clocks. Check at
  configured intervals after fully fitted cohorts. Entire configured cohorts must
  strictly exceed the qualification threshold; shortened cohorts cannot qualify.
  Skips wait another interval. Cherry Bomb and Peashooter strictly exceed their
  configured mastery thresholds; other fresh stages require perfect validation.
- Every validation uses fresh random seeds outside reserved splits. Save panels
  before execution and retry interruptions against the same weights before fitting.
  Greedy validation uses public-only inputs and has no private simulations,
  exploration, auxiliary-label sampling or fitting updates.
  Release idle scratch/graphs without losing workers or speculative seed state.

## Code structure

Paths are relative to `src/pvz_rl/` unless stated otherwise.

| Area | Ownership |
| --- | --- |
| Repository `tools/bootstrap.*`, `tools/stage_game.py` | Verified installation of the pinned engine. |
| `config.py`, `data/train.toml`, `gpu.py` | Stage tuning, validation panels, model dimensions and device selection. |
| `policy/` | Shared public encoder, conditional tile decoder, action/tile/outcome readouts, stateless runner and bounded graphs/batching. |
| `envs/` | CPU/CUDA adapters, public encoding, actions, rewards, source credit and scenarios. `field_tracking.py` owns guarded per-tick depth and home-entry hooks. |
| `envs/scratch_game.py`, `bomb_preparation.py`, `cuda_bomb_oracle.py`, `cuda_snapshots.py`, `attacker_scenarios.py` | Validated automatic exercises, persistent preparation, bounded oracles and conservative attacker labels. |
| `learning/training.py`, `cuda_q.py`, `sequential_training.py` | Cohort orchestration, collection/fitting, reporting and atomic recovery. |
| `envs/field_tracking.py`, `wait_capture.py`, `static_fork.py`, `cuda_static_fork.py`, `fork_service.py`, `fork_execution.py/.cu` | Guarded field observations, exact real-tick capture, isolated field/probe phases, bounded refillable workers, verified suffix caching and queue recovery. |
| `learning/forecast_labels.py`, `forecast_objective.py`, `forecast_metrics.py` | Absolute field/probe targets, shared-forward categorical losses and calibration. |
| `learning/auxiliary_service.py`, `auxiliary_settings.py`, `async_teacher.py`, `envs/auxiliary_cpu.py`, `auxiliary_cuda.py`, `snapshot_packet.py`, `teacher_rules.py`, `outcome_math.py` | Shared execution with independent interval/teacher routing, immutable packed device roots, event-polled exact continuations, CPU workers and model-free simulation rules. |
| `learning/effect_collection.py`, `wait_outcomes.py`, `placement_outcomes.py`, `td_targets.py` | Saved candidate tickets, deferred interval and single-placement targets, and frozen-cohort n-step TD. |
| `learning/flat_q.py`, `objective.py`, `action_routing.py`, `placement_labels.py`, `bomb_objective.py` | Shared Q fitting/economics, legality attribution and automatic placement losses. |
| `learning/peashooter_rules.py`, `peashooter_teacher.py`, `envs/lane_preparation.py`, `cuda_lane_oracle.py` | Prioritized lane targets and resumable bounded lane oracles. |
| `learning/cuda_buffer.py`, `batch_transport.py`, `collection_scheduler.py`, `exploration.py` | Bounded trajectories/transport, decision tickets and committed species exploration. |
| `learning/tile_curriculum.py`, `validation_trigger.py`, `budget.py`, `training_requirements.py` | Stage transitions, validation qualification, budgets and model/objective identities. |
| `learning/checkpoints.py`, `training_artifacts.py`, `performance.py`, `demo_initialization.py` | Recovery/weights, artifact retention, execution settings and verified human demo fitting. |
| `evaluation/`, `monitoring/`, repository diagnostic tools | Greedy evaluation, CPU verification, bounded profiling and benchmarks. |
| Repository `tools/benchmark_peashooter.py`, `tools/draw_collection_process.py` | Seed-matched total-time/latency sweeps, optional real-collection host/CUDA event timing and two chronological CPU/GPU timelines. Collapse loops to start/end markers; identify waits and unavailable device timing explicitly. |
| Repository `tools/profile_training_workflow.py` | Bounded actual-CLI startup, collection and recovery audits, or one fully fitted production cohort with CPU/CUDA trace capture. Preserve production cutoff, initialization and presentation; do not equate shortened frozen-policy benchmarks with full training. |
| Repository `tools/profiling_metrics.py`, `tools/report_training_hotspots.py` | Per-thread exclusive host attribution, GC pauses, calibrated stage events, bounded CUPTI/cProfile windows, per-pass losses and costs per batch/board decision. Tracing must not retain released CUDA scratch; parameter sweeps release each model's graphs before creating the next model. |
| `presentation/` | Demo recording/replay, live cohort subscriptions and retained reports. `outcome_view.py` displays proposed-action field/probe predictions; disabled effects and unknown targets are explicit. Viewers reuse timestamped episode-scoped journal predictions without model calls. |

## Test ownership

`tests/test_smoke.py` is the entire permanent suite: exactly three tests, at most
200 lines. Use temporary independent mathematical, boundary, backend and recovery
controls; do not expand permanent suites or repeat full runs without a new cause.

- CLI help: 15-second subprocess timeout.
- CPU inference: real public board, action/tile/outcome shapes, finite values and an
  action submitted to the simulator.
- Explicit CUDA learning: two one-second games, two envs, one pass, recovery and
  portable reload/prediction/cleanup. Disable compilation, presentation, preliminary
  targets, forecasts and tracking; batch 32, two-minute allowance, 180-second subprocess
  timeout. Missing CUDA requirements fail.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
```

The default command selects the two CPU checks. Outputs use temporary directories;
these checks establish workflow availability, not mastery.
