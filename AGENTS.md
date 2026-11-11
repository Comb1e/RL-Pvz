# Project instructions

Follow the global `AGENTS.md` for shared workflow, Git, documentation and testing.
This file owns project-specific invariants and the code map.

If the user has not declared it in advance. Ignore the existing runs by default. New settings should need new runs to continue.

## Scope and invariants

- Use `.venv\Scripts\python.exe`. Keep one training/model framework and the
  separate pinned game dependency; never edit the installed engine or its pin.
  No dependency downloads in smoke checks or formal training without a request.
- `src/pvz_rl/data/train.toml` is the single parameter source for training and demo
  overlays. Preserve user tuning. Group stage settings under `training.cherry_bomb`,
  `cherry_bomb_obstacle`, `attacker_pre`, `prediction`, `attacker`, `peashooter` and `normal`; resolve once
  into the runtime layout. `prediction`, `cherry_bomb`, `cherry_bomb_obstacle`, `attacker_pre`,
  and `attacker` score each cohort once, then take one fitting pass. A saved file that still
  has `training.cohorts_per_epoch` keeps that file's one-pass block and memorized cohort.
  `peashooter`, `easy`, `standard`, and `shared` each own `n_epochs`. A live `3×` request
  trains the next cohort for one pass at `training.learning_rate_restart` times
  `training.learning_rate`. Teacher EMA uses `training.objective.ema_decay` on every stage.
  A stage advances after `training.stage_streak` consecutive full cohorts meet that stage's
  bar. Training does not run a held-out validation panel.
  Shared scenario sampling attempts live in `tile_scenarios`.
- Fresh policy inputs are public board facts: entity records and 18 globals.
  Simulated field/probe outcomes are private auxiliary targets, never observations.
  Future spawns are suppressed inside forks; IDs, schedules, seeds and snapshots stay private.
- Encode once per decision/fitting call. Prediction runs the shared encoder, the
  effect head, and the forecast-input embeddings, then submits Wait. Placement and
  Peashooter add the tile decoder and tile head for the supplied card. Easy,
  standard, and shared also run the ten-query action decoder and the action head.
  A Wait or excluded branch still skips the tile decoder.
  Encoder memory is the global token and the entities. The effect head cross-attends
  that memory with five learned lane queries and emits field depth, breach bin, and
  probe depth before the action is chosen. It does not see the action, the tile, or
  its own tokens. Detached class ids become five lane tokens placed after the global
  token on the memory the decoders read. Nothing from this prediction is written
  back into the encoder or kept for the next decision. Wait and an excluded
  branch skip tile decoding. Illegal cards and cells are
  excluded by setting that score to -inf on a selection copy. The choice is not
  rewritten into Wait. Preserve the proposal for Q and teacher fitting. Later
  stages retain valid Wait references. Every tile is still scored. A plant cell
  is a candidate only when it is empty, the card has enough sun, and its cooldown
  is finished. A dig cell is a candidate only when it holds a plant. Dig itself
  is excluded when no cell has a plant, and every plant card is excluded when
  the lawn is full. Cache memory within one forward only.
- All models are outcome-only. Factual prediction heads, horizon/snapshot labels,
  losses, summaries, storage and legacy resumes are removed. Early stages use
  `placement_outcomes`: exactly one pre-action field/probe job per exercise,
  including rejections. The fork is action 0 on the pre-mutation root. Keep placement teachers
  and stage weights. Train the tile decoder and tile head. Freeze the shared
  encoder, effect head, forecast-input embeddings, action decoder, action head,
  and their EMA. Early tile scores
  receive no outcome-summary injection.
  Reserve no Wait boundary buffers for placement exercises; charge only their
  occupied proposal packets and admitted workers to auxiliary VRAM.
  Peashooter/normal use `action_outcomes` with explicit valid Wait references.
  Predict absolute field depth (11), breach bin (6) and probe depth (11); preserve
  the 56-slot carrier, excluding reserved classes. Field class 10 is equivalent
  to a recorded breach. Static field and probe forecasts are the hard 0/1 depth
  step of the chosen class; home is 1 only for that class. Breach remains a class
  distribution. Average known
  terms/lanes, valid tokens and roots equally; never invent unsimulated labels.
  Each stage owns separate `static_field_prediction_weight` and
  `probe_prediction_weight`: scale their terms before the unchanged known-term
  average and shared warmup, including valid null targets. Zeroing a group never
  renormalizes the other group. Saved outcome-only single-weight runs retain
  that coefficient for both groups; model/labels/collection stay unchanged.
- Static forks own physical/RNG storage, suppress actions/future spawns/logs and
  resolve mowers before recording home entry. Continue other lanes after a loss;
  probes inherit damaged plants and mower state after all projectiles clear,
  queued shots finish, moving mowers stop and surviving Chompers are ready.
  The real game is unchanged.
  Inference and ordinary gameplay validation compute no field/probe simulations.
  Prediction validation compares the action-free head with one untouched fork of the
  current frame. The greedy proposal is not an input. There is no tile-Q injection.
- `absolute-wait-trajectory-v6` follows real Waits/rejections and records guarded per-tick
  field observations. One semantic probe belongs to each owned interval, anchored
  at the first fully settled tick after zombie clearance. Keep capturing the
  first zombie-clearance state and finish settling privately. Accepted
  planting/Dig, actual spawn-counter changes and reset/end close intervals; deaths do not.
  Capture pre-spawn full-tick states before physical/RNG changes, including spawns
  that die within one decision. Finish interrupted fields from those owned roots;
  never replay borrowed prefixes. Rebase breach times and exclude penetration
  before each decision, including within the preceding tick. Each decision's effect
  target is that rebased untouched interval. The post-action interval is still
  simulated so the next decision has a baseline, and it does not supervise this head.
  No alternative forks are submitted. Cache entries remain exact, bounded and disposable; pending roots,
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
- Every decision receives the untouched-board outcome, including legal non-Wait
  proposals and effective Waits. Five-slot tickets stay, and only slot 0 is valid.
  Fresh Peashooter keeps its teacher and that same untouched target. Normal stages
  use the same target. `extra_actions` stays 2 in configuration and is not sampled.
- Normal cohorts jointly fit frozen-cohort n-step Double-Q targets and auxiliaries.
  Preserve economic conversion, tick discounting, rejection compaction/weighting,
  species exploration, greedy tiles, that stage's `n_epochs`, BF16 and FP32 masters. No
  counterfactual followers or tile-max species consistency. Outcome injection into
  the tile head is removed. `injection_after_games` and `injected_n_step` stay in
  train.toml and are not read. `n_step` and the initial discount half-life stay
  fixed. Potential shaping starts disabled. There is no separate private
  forecast pretraining phase, sampler, graph, configuration or recovery state.
- Main-game guarded CUDA field traces record per-tick depth before defeat and
  actual home entries after mower resolution. They contain no factual horizons or
  bite-contact targets. Resource failures preserve exact work without creating labels.
- Preliminary stages retain automatic placement objectives. Prediction trains the
  shared encoder, the effect head, and the forecast-input embeddings. Later stages
  freeze those three. Placement and Peashooter train the tile decoder and tile
  head. Easy, standard, and shared train the action decoder, action head, tile
  decoder, and tile head. Update EMA only for each phase's trained groups and
  copy frozen weights onto the teacher. Demo initialization has no auxiliary loss.
- Bomb/attacker stages use successful-set cross-entropy over all 45 outputs,
  including occupied cells. Stage temperature follows completed-cohort accuracy,
  fixed across passes. Peashooter uses prioritized visible-lane engine labels and
  column-decaying cross-entropy: global coverage rule, then undefended lanes, then
  endangered defended lanes. Its labels never override main actions.
- Manual annotation commands, studio, datasets workflow and fitter are removed.
  Preserve existing user datasets/recordings. Retain human recording, replay
  verification and demo initialization with their demonstration objectives and
  no auxiliary supervision.
- Only `shared_public_outcomes_q_v11` checkpoints are supported. Reject every older
  checkpoint before environment/CUDA preparation for inference, resume and
  initialization; no conversions or legacy execution. New-version recovery retains
  optimizer/counters, owned roots/rows/choices, partial jobs, unpublished results
  and committed passes. Publish every target once before fitting.
  A setup failure before any issued decision may rebuild an empty scheduler from
  the saved boards. Missing tickets after collection starts remain invalid.
- Preserve CPU/CUDA cadence, legality, physical trajectories, reward conversion and
  attribution agreement. Viewers and diagnostics must not affect actions, rewards,
  or any other learning value. The live gradient scale is the exception: it multiplies
  the eligible master gradients before `max_grad_norm` and latches at the next cohort
  begin. The live `3×` restart latches at that same moment, forces one pass, and then
  returns to the configured rate. A checkpoint that omits the scale resumes at 1, and one
  that omits the restart flags stays idle. The cohort line prints each
  pass's unscaled gradient norm before that clip. A checkpoint that omits those norms
  resumes and records the passes that finish after recovery.
- Fitting encoders compile once per shape and keep that cache across cohorts and
  validation; releases free CUDA captures only. Recompile only for shape-changing
  pass fallbacks; per-cohort recompilation leaks compiler objects and slows fitting.
  Capture encoder graphs with `capture_begin(capture_error_mode="thread_local")`, not
  `torch.cuda.graph()`: auxiliary threads poll CUDA events during capture, and the
  context manager runs a full `gc.collect()` per capture.
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
- `jsonl_log.py` rolls each run JSONL log at `logging.jsonl_batch_bytes` into a
  folder named for that log, such as `hardware-metrics/`. An existing file larger
  than the limit is moved in whole, and later records append to a new file there.

## Stages and validation

- Fresh progression: `prediction → cherry_bomb → cherry_bomb_obstacle → attacker_pre → attacker → peashooter
→ easy → standard → shared`. Explicit normal stages start directly. Preserve
  normal stage indices, independent budgets/counters and saved stage orders.
- Both bomb stages have no exploration/search. Obstacle boards have exactly one
  engine-verified legal hit. Attacker exercises supply one of four attackers,
  use conservative distance placement labels, and report fallback
  correctness separately from attributed kills. Peashooter keeps easy waves,
  disabled mowers, configured sky-sun scaling and saved countdowns without rescaling.
- Attacker draws exact, shuffled Cherry Bomb/Obstacle/Attacker Pre quotas of
  64/256/320 for collection. Retain source exercise
  on each board and use its supplied branch, teacher, temperature and verification.
  One shared pool and one active CUDA oracle prepare the mixed cohort. The stage
  bar is the labeled-set check on that collected cohort. Weights-only stage
  transfers use `--init-from`; recovery retains the saved stage order.
- Prediction uses 640 seeded, engine-validated current frames per cohort and one
  exact untouched-board target per scene. Average all 15 known
  outcome terms per scene and weight scenes equally. Train the forecast path
  without warmup, Q, or teachers. Eligible parameters are the shared encoder, the
  effect head, and the forecast-input embeddings. That forward does not call the
  action decoder, action head, tile decoder, or tile head. Detached class ids give
  the decoders and forecast-input no gradient from that loss. This is the only stage
  that updates the shared encoder and the effect head. Easy, standard, and shared
  update the action decoder. The stage bar scores the cohort just collected:
  every scene passes only when every lane's field depth, breach bin, and probe
  depth match. Ten consecutive full cohorts at that rate advance the stage.
  Progress reports that fully-correct scene rate.
- Cohort fitting and the stage bar use the same collected cohort. Prediction, cherry bomb,
  cherry-bomb obstacle, attacker pre, and attacker take one fitting pass per cohort.
  A saved `cohorts_per_epoch` keeps its one-pass block and memorized cohort. Every cohort
  is scored once before its first pass. A one-pass cohort still takes that pass when the
  check is already perfect. Only a memorized cohort from a saved count is scored again
  after each pass. Prediction counts a scene when all five lanes match field depth, breach
  bin, and probe depth. The four placement stages count a board when its greedy legal tile
  is inside the labeled successful set; the attacker mix is one cohort, and a kill is not
  required. Cherry bomb's bar is strictly above 0.95. Obstacle, attacker pre, and attacker
  require 100%. Peashooter is strictly above 0.95 wins on the games just played. Easy,
  standard, and shared require every game of each required task to be won, and every
  required task to occur. A short cohort does not move the streak. Ten consecutive passes
  advance a full run, or mark a `--stage` run mastered. The report charts before-fit
  accuracy for the five stages, omits their exploration charts, and redraws every
  `visualization.report_interval_cohorts` cohorts, on a stage change, and at the end.
  A checkpoint that omits the before-fit pair or the streak still resumes.
- Only completed episodes or prediction scenes advance budgets. The passing cohort is
  fitted before the stage changes. Evaluation commands keep their panels. Release idle
  scratch/graphs without losing workers or speculative seed state.

## Code structure

Paths are relative to `src/pvz_rl/` unless stated otherwise.

| Area                                                                                                                                                                                    | Ownership                                                                                                                                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Repository `tools/bootstrap.*`, `tools/stage_game.py`                                                                                                                                   | Verified installation of the pinned engine.                                                                                                                                                                                                                                                |
| `config.py`, `data/train.toml`, `gpu.py`                                                                                                                                                | Stage tuning, validation panels, model dimensions and device selection.                                                                                                                                                                                                                    |
| `policy/`                                                                                                                                                                               | Shared public encoder, pre-action effect head, decoder-only forecast tokens, action decoder, conditional tile decoder, action/tile/outcome readouts, stateless runner and bounded graphs/batching.                                                                                          |
| `envs/`                                                                                                                                                                                 | CPU/CUDA adapters, public encoding, actions, rewards, source credit and scenarios. `field_tracking.py` owns guarded per-tick depth and home-entry hooks.                                                                                                                                   |
| `envs/scratch_game.py`, `bomb_preparation.py`, `cuda_bomb_oracle.py`, `cuda_snapshots.py`, `attacker_scenarios.py`, `prediction_scenarios.py`                                           | Validated automatic exercises, persistent preparation, bounded oracles, conservative attacker labels and seeded prediction frames.                                                                                                                                                         |
| `learning/training.py`, `cuda_q.py`, `sequential_training.py`                                                                                                                           | Cohort orchestration, collection/fitting, reporting and atomic recovery.                                                                                                                                                                                                                   |
| `envs/field_tracking.py`, `wait_capture.py`, `static_fork.py`, `cuda_static_fork.py`, `fork_service.py`, `fork_execution.py/.cu`                                                        | Guarded field observations, exact real-tick capture, isolated field/probe phases, bounded refillable workers, verified suffix caching and queue recovery.                                                                                                                                  |
| `learning/forecast_labels.py`, `forecast_objective.py`, `forecast_metrics.py`                                                                                                           | Absolute field/probe targets, shared-forward categorical losses, hard static/probe indicators, and calibration.                                                                                                            |
| `learning/auxiliary_service.py`, `auxiliary_settings.py`, `async_teacher.py`, `envs/auxiliary_cpu.py`, `auxiliary_cuda.py`, `snapshot_packet.py`, `teacher_rules.py`, `outcome_math.py` | Shared execution with independent interval/teacher routing, immutable packed device roots, event-polled exact continuations, CPU workers and model-free simulation rules.                                                                                                                  |
| `learning/effect_collection.py`, `wait_outcomes.py`, `placement_outcomes.py`, `td_targets.py`                                                                                           | Saved candidate tickets, deferred interval and single-placement targets, and frozen-cohort n-step TD.                                                                                                                                                                                      |
| `learning/flat_q.py`, `objective.py`, `action_routing.py`, `placement_labels.py`, `bomb_objective.py`, `cohort_accuracy.py`                                                            | Shared Q fitting/economics, legality attribution, automatic placement losses, and the collected-cohort accuracy stop.                                                                                                      |
| `learning/peashooter_rules.py`, `peashooter_teacher.py`, `envs/lane_preparation.py`, `cuda_lane_oracle.py`                                                                              | Prioritized lane targets and resumable bounded lane oracles.                                                                                                                                                                                                                               |
| `learning/cuda_buffer.py`, `batch_transport.py`, `collection_scheduler.py`, `exploration.py`                                                                                            | Bounded trajectories/transport, decision tickets and committed species exploration.                                                                                                                                                                                                        |
| `learning/tile_curriculum.py`, `validation_trigger.py`, `budget.py`, `training_requirements.py`                                                                                         | Stage transitions, validation qualification, budgets and model/objective identities.                                                                                                                                                                                                       |
| `learning/checkpoints.py`, `training_artifacts.py`, `performance.py`, `demo_initialization.py`                                                                                          | Recovery/weights, artifact retention, execution settings and verified human demo fitting.                                                                                                                                                                                                  |
| `evaluation/`, `monitoring/`, repository diagnostic tools                                                                                                                               | Greedy evaluation, CPU verification, bounded profiling and benchmarks.                                                                                                                                                                                                                     |
| Repository `tools/benchmark_peashooter.py`, `tools/draw_collection_process.py`                                                                                                          | Seed-matched total-time/latency sweeps, optional real-collection host/CUDA event timing and two chronological CPU/GPU timelines. Collapse loops to start/end markers; identify waits and unavailable device timing explicitly.                                                             |
| Repository `tools/profile_training_workflow.py`                                                                                                                                         | Bounded actual-CLI startup, collection and recovery audits, or one fully fitted production cohort with CPU/CUDA trace capture. Preserve production cutoff, initialization and presentation; do not equate shortened frozen-policy benchmarks with full training.                           |
| Repository `tools/profiling_metrics.py`, `tools/report_training_hotspots.py`                                                                                                            | Per-thread exclusive host attribution, GC pauses, calibrated stage events, bounded CUPTI/cProfile windows, per-pass losses and costs per batch/board decision. Tracing must not retain released CUDA scratch; parameter sweeps release each model's graphs before creating the next model. |
| `presentation/`                                                                                                                                                                         | Demo recording/replay, live cohort subscriptions and retained reports. The five overfit stages chart accuracy before fit and omit exploration. `outcome_view.py` shows this frame's pre-action field, breach bin, and probe when the journal recorded them; unknown targets stay blank. Viewers reuse timestamped episode-scoped journal predictions without model calls. |

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
