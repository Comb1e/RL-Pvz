# Iteration history

## 0.31.5 — 2026-09-29

- Problem/root cause: initialization compared historical reward prices against
  current TOML, then would have fitted archived totals. The human recording's
  decision 4930 rejected a recharging plant at the old penalty of 0.001; the
  current 0.003 penalty incorrectly triggered a reconstruction failure.
- Improvement: verify native replay hashes, observations, actions, outcomes,
  terminal states and configuration-independent reward facts. Check historical
  ledger arithmetic, finite values and penalty applicability, then recompute
  rewards through the shared reward function using current coefficients. Fitting
  consumes the verified in-memory transitions, so it cannot reread old targets.
  Keep original files unchanged and report reward changes/settings in verification
  metadata. No model, precision, dependency or user parameter changes.
- Verification: 26 focused checks passed for unchanged/changed rewards, independent
  returns, rejected plant/dig and terminal boundaries, corruption and CLI config.
  One additional tiny complete-pass CUDA initialization/reload check passed.
  The user's complete 15,372-decision recording verified with exactly one repriced
  reward; decision 4930 now costs -0.003 and next feedback remains `(0, 0, 1)`.
  SHA-256 checks confirmed archive, replay and manifest stayed unchanged. Ruff
  lint/format and Git whitespace checks passed. No full suite or formal training.
- Remaining limits: old archives store a configuration digest without original
  reward coefficients, so historical prices cannot be authenticated. Replay facts
  and accounting remain validated; those prices never train the model. Pinned
  engine and observation/action compatibility requirements remain in force.

## 0.31.4 — 2026-09-29

- Problem/root cause: demo and autonomous fitting read separate clipping limits.
  Fresh `--init-from` also copied the source configuration, so both easy runs
  inherited the initialization's invalid-plant penalty of 0.001 despite TOML
  having 0.003. The viewer correctly exposed the effective, unintended value.
- Improvement: keep only `training.max_grad_norm = 5`, preserving the user's
  clipping adjustment for both fitting paths. Remove the demo-only setting and
  override. Fresh initialization now resolves all settings from current TOML
  and transfers only compatible weights; reward changes do not block transfer.
  Resume keeps saved learning settings and rewards. Recording verification checks
  engine/schema and replay facts without binding optimizer settings to its digest.
- Verification: 22 focused controls passed, including the real tiny CLI
  initialization/training/evaluation/resume flow, shared clipping, archive
  corruption, architecture mismatch and retained viewer values. The existing
  human initialization transferred all 56 parameter tensors exactly and the
  user's command resolved penalty 0.003, clipping 5 and `features_bf16` without
  a refresh flag or starting simulation. Changed-file Ruff lint/format and Git
  whitespace checks passed. No full suite or formal training ran.
- Remaining limits: existing runs retain their saved penalty on resume; only a
  new run adopts changed learning settings. Old recordings still require their
  original reward configuration when verifying stored reward facts.

## 0.31.3 — 2026-09-29

- Problem/root cause: autonomous resume retains learning settings, but the live
  viewer only received display preferences, leaving active clipping and reward
  values invisible when the current TOML differed from a checkpoint.
- Improvement: send a one-time snapshot of the resolved run configuration to the
  viewer. A read-only strip shows learning rate, `max_grad_norm`, batch size,
  epochs, discount, sequence length, rewards and tile exploration. It wraps at
  narrow widths and **S** hides/shows it; small panels omit history footers when
  they would overlap board details. Demo-only clipping is not presented as an
  autonomous setting. No training parameters or checkpoint semantics changed.
- Verification: eight existing focused viewer controls passed. Offscreen review
  covered 1600×1050, 944×668 and 640×480, grid/focus layouts, hidden/visible settings,
  and the real S-key event loop before any board arrived. A deliberately different
  saved configuration retained its displayed values after performance refresh.
  The two layout controls passed again after the narrow-window adjustment.
  Ruff lint/format and Git whitespace checks passed. No formal training or full
  suite was run.
- Remaining limits: existing viewer processes receive the strip on their next
  launch; narrow windows still benefit from Focus or hiding settings for history.

## 0.31.2 — 2026-09-29

- Problem/root cause: the Windows CUDA-graphs encoder was replayed for successive
  entity microbatches while the recurrent loss still held the previous outputs
  for backward. CUDA-graphs reclaimed that storage, producing an overwritten
  tensor error and a pending-backward warning at fit startup.
- Improvement: capture AOT forward/backward separately and copy all outputs
  into independently owned storage, including saved activations and gradients.
  Preserve SDPA strides and refresh input weights on every invocation. Backward
  capture failure restarts the uncommitted pass eagerly at the same precision.
  CUDA-graphs compilation remains enabled; model shapes and precision are unchanged.
- Verification: five focused CUDA controls passed: compiled/eager FP32 and BF16
  output, gradient and update comparisons across empty/mixed/capped observations,
  partial microbatches and gradient accumulation, plus memory, nonfinite and
  backward-compilation retries without duplicated optimizer updates. A marker-only
  repair removed the warning but still produced about 11% gradient error during
  replay and was discarded. The owned-buffer implementation's BF16 gradient
  difference against eager measured 0.12–0.14% across four small updates.
- Device smoke check: three BF16 forward/backward/Adam steps per population on the
  RTX 4070 Laptop completed with `compiled / cudagraphs_owned` and no warnings.
  The 4,096-frame/40-entity case took 0.164–0.168 seconds per warmed step with
  1,730 MiB peak Torch allocation; 256 entities took 0.373–0.376 seconds and
  4,551 MiB. Initial tracing/capture steps took 1.078/1.194 seconds respectively.
  A truly empty 32-frame case also passed. These exclude simulation/storage and
  do not measure learning quality; owned copies add memory overhead. No formal
  training or full suite was run. Ruff lint, changed-file formatting, syntax and
  whitespace checks passed. The repository-wide format check flagged an existing
  assertion layout in `tests/test_demo_initialization.py`; that unrelated file
  was left unchanged.

## 0.31.1 — 2026-09-29

- Problem/root cause: enabling the fixed-shape encoder through the default
  Windows Inductor path traced PyTorch SDPA's pybind11 capability object, then
  fell back because the Windows environment has no Triton. The simulator also
  reserved only the proved future projectile bound, so a mid-game recovery
  snapshot containing active shots could exceed capacity.
- Improvement: keep SDPA capability probing out of Dynamo traces, use the
  available CUDA-graphs compiler on Windows, retain a one-time eager fallback,
  and allocate current-snapshot plus future projectile headroom. The shared
  reward table now has one invalid-plant penalty; the demo profile inherits it
  while preserving `training.demo.gradient_clip = 5`.
- Verification: direct CUDA forward/backward compilation completed with
  `compilation_status = compiled` and `compilation_backend = cudagraphs`; no
  pybind or max-autotune warning was emitted. Ruff, syntax and configuration
  checks were run without starting training.

## 0.31.0 — 2026-09-29

- Problem/root cause: recurrent fitting processed only four 256-step sequences
  per call, created a CUDA timing event for every chunk, and used a no-op
  compilation hook. Cohort telemetry showed 887–978 seconds of fitting with
  about 30–40% GPU utilization while transfer wait stayed below 0.1 seconds.
- Improvement: fuse four independent execution batches, increase the encoder
  execution budget to 128 observations/65,536 tokens, cache sequence indexes and
  ragged gathers, defer intermediate synchronization, and add bounded fixed-shape
  encoder compilation with an eager fallback. The learning batch, recurrent
  boundaries, complete-return weighting, precision boundaries and five demo
  passes remain unchanged.
- Verification: focused transport, recurrent-state, compilation-fallback and
  precision controls passed. The synthetic BF16 benchmark reached 11,318–11,356
  frames/s for mixed entity populations and 11,455–11,707 frames/s at the
  256-entity cap, with 838 MiB peak Torch allocation. The short CUDA collection
  check reached 2,939–3,249 decisions/s without the viewer and 2,938–3,121 with
  it; fitting preparation stayed below 0.05 seconds and transfer wait below
  0.005 seconds. The Windows compiler rejected the entity encoder because of a
  pinned-game pybind object, so the run recorded `compilation_status = fallback`
  and used the exact eager path. These checks are throughput evidence only, not
  a learning-quality claim.

## Documentation and test cleanup — 2026-09-28 (0.30.0 unchanged)

- Problem/root cause: guides accumulated retired event-memory, regional-input,
  two-coin exploration and minibatch-update descriptions alongside the current
  entity model. Validation and reference pages duplicated release narratives.
  Test suites retained obsolete training-mode exercises and repeated assertions.
- Improvement: delete eight redundant/retired guides; keep one current home for
  entity inputs, recurrent learning, rewards, rejection penalties and precision.
  Shorten the README to quick start, replace validation history with coverage
  ownership, and retain source attribution relevant to the current system.
  Historical narratives remain accessible through versioned Git links; raw
  evidence and recordings are preserved.
- Test cleanup: static inventory falls from 270 to 251 test functions and from
  36 to 35 test files. Remove two-coin/fixed/hybrid training exercises, duplicate
  report/reload runs, reward checks and viewer checks. Move unique deadline,
  profile, probe-size and rejected-dig assertions to their existing owners.
  Rename the retained SC2/compact/warmup suites for their current purpose.
  Keep independent math, CPU/CUDA parity, supported-reader compatibility,
  incompatible-checkpoint rejection and atomic interruption controls.
- Verification: Ruff lint and format checks, in-memory syntax compilation of
  104 Python files, PowerShell parsing and local Markdown link/anchor checks
  passed. No tests were collected or executed, and no benchmarks or simulation
  ran, as requested. These are static checks, not a new runtime validation claim.
- Isolation/limits: changes were made in a separate worktree. Runtime source,
  configuration, dependency pin, running training and run artifacts were untouched;
  the user's parameters remain intact. Behavioral validation is unexecuted.

## 0.30.1 — 2026-09-28

- Problem/root cause: `train --init-from` inherited stale encoder and precision
  settings from the demonstration checkpoint, so changing the current training
  profile required `--refresh-performance`.
- Improvement: new `initialization.pt` files omit execution-only performance
  fields. Loading a demonstration keeps structural metadata for validation and
  overlays the current train profile's execution settings automatically.
- Compatibility: demonstration weights and existing autonomous resume semantics
  remain unchanged; `--refresh-performance` remains an explicit autonomous-resume
  control.
- Verification: demonstration initialization, checkpoint hydration, CLI transfer,
  and performance-refresh controls passed in the focused test set; Ruff and
  whitespace checks passed.

## 0.30.0 — 2026-09-28

- Problem/root cause: small encoder batches, unconditional query splitting and
  nested activation checkpoints multiplied kernel launches and recomputation.
  Collection repeatedly extracted GPU scalars; fitting rebuilt dictionaries and
  synchronized each chunk. Sparse-game GPU utilization stayed near 35%.
- Improvement: full efficient SDPA, a 64-frame/32,768-token encoder budget,
  cached normalization constants, bulk ragged gathers, ordered double-buffer
  pinned prefetch, batched collection copies and deferred telemetry. Unsupported
  attention backends retain an exact query fallback.
- Precision: BF16 temporary features, embeddings and auxiliary projections;
  FP32 master weights, normalization/residual accumulation, recurrent core/state,
  Q heads, loss, accumulated gradients and optimizer state. No loss scaler.
  Failed BF16 passes restart in FP32 without losing an update. Allocation retries
  reduce only encoder microbatches and then enable one outer checkpoint.
- Compatibility: entity inputs, all parameter names/shapes, action semantics,
  rewards and the game pin remain unchanged. Ordinary resume preserves saved
  execution precision (older entity checkpoints default to FP32).
  `--refresh-performance` replaces only execution settings. Saved cohort fallback
  state survives SB3 policy reconstruction and uncommitted-pass recovery.
- Preserved user choices: `training.demo.passes=5`, four autonomous passes,
  128 environments, 1,024-decision batches, 256-step chunks, 256 entities, and
  train/demo invalid-plant penalties of 0.003/0.001. Demo fitting remains FP32.
- Test consolidation: eight overlapping test functions were removed; unique
  assertions moved into the existing owners. Batching, profile defaults,
  obsolete-checkpoint rejection, occupancy and collection recovery each have
  consolidated controls. New BF16, prefetch and retry coverage lives in existing
  model/storage/recurrent files. The final suite contains 536 cases, versus 537
  before this change; atomic signal interruption and independent math controls
  remain covered.
- Verification: 213 focused tests passed on the final implementation. An earlier
  full-suite run passed 545 cases before test consolidation and recovery refinements.
  The final 536-case full-suite rerun was stopped at the user's request; it is not
  reported as a completed run. Ruff lint, dependency checks, packaging and local
  documentation links passed. The existing five-pass entity demo checkpoint loads
  with FP32 parameters. No further tests were run after the stop request.
- Measured improvement on RTX 4070 Laptop: median short end-to-end throughput
  rose from 318 to 1,553 decisions/s without the viewer (4.89x), and 307 to 1,638
  with it (5.34x). Four-pass fitting fell from 35.97 to 6.27 seconds without the
  viewer (5.73x), and 37.55 to 5.84 seconds with it (6.43x). Synthetic fitting
  improved 7.37–16.05x across sparse, mixed and capped populations. Crowded
  synthetic Torch allocation peaked at 859.6 MiB; short real collection peaked
  at 143.0 MiB plus a separate 128.6 MiB CuPy pool.
- Evidence and limits: [raw repetitions and phase/memory measurements](evidence/training-throughput-v030.json).
  Collection used one-second cutoffs and three warmed repetitions per viewer
  setting, keeping four passes and the normal batch dimensions. Hardware sampling
  includes warmup. Baseline zero transport counters were uninstrumented.
  No formal training or learning-quality claim. Long-run throughput, disk-heavy
  cohorts and other GPUs require their own measurements; Windows FlashAttention
  and Triton are unavailable in the tested installation.

## 0.29.0 — 2026-09-28

- Problem/root cause: regional zombie aggregates detached health/armor and state
  from individual positions, fixed flat vectors omitted projectiles, and the
  alternative event-memory runtime duplicated storage and dispatch paths.
- Improvement: one `entity_v1` schema stores 11 integer public fields per physical
  entity and 18 globals, with shared 32-dimensional embeddings, a global plus 45
  tile readouts, two masked attention layers and the retained 256-unit LSTM.
  Public timers, headless zombies, projectiles and spent mowers are represented;
  IDs, RNGs, schedules and private movement state remain excluded.
- User-directed cap: default 256 based on RTX 4070 Laptop measurements, configurable
  in the single TOML. Retain mowers → plants → zombies → projectiles; omit
  projectiles first, then zombies, then plants, with nearest-house zombies first.
  Globals keep pre-cap counts and action journals/viewers expose omissions.
- Runtime/storage: GPU encoding stays on device, packs present slots and matches
  CPU public formulas. Ragged metadata/entity slabs share RAM and disk spill,
  recover with validated offsets/counts, and use versioned schemas. Encoder
  microbatches, query chunks and activation checkpointing bound attention memory.
  Refilled evaluation slots explicitly refresh their newly allocated observation.
- Removal/compatibility: event-memory model, temporal encoder, banks, caches,
  profile and dispatch are removed. Checkpoints, demos and trajectories require
  fresh entity-based initialization; existing recordings/run files are preserved.
  Greedy ten-way proposals, tile-only exploration, penalty targets and rejected
  `(previous_action=0, accepted=0, ticks=1)` feedback remain. The user's 0.003 train
  and 0.001 demo plant penalties are preserved; the game pin is unchanged.
- Verification: information counterexamples, all categories/public states,
  duplicates, caps, ID isolation, CPU/CUDA parity, padding/permutation invariance,
  independent float64 attention/gradient controls, causal recurrence, ragged spill,
  archive corruption, demo/viewer/replay, checkpoint and interruption recovery.
  All 537 collected tests verified: 489 in the full-suite run, then 48 after
  correcting two obsolete fixture assertions; production code was unchanged.
  The focused 92-case suite, Ruff, dependency checks, sdist/wheel builds and
  local documentation links pass. The wheel contains one bundled TOML and no
  retired model modules.
- Resource evidence: 256/512-entity fitting measured 686/247 frames/s; both fit
  in VRAM. A warmed 128-slot one-second collection/fitting check measured 1,080
  decisions/s, 95.4 MiB Torch peak allocation and 128.6 MiB CuPy pool reservation.
  A default 1,024-frame fitting-chunk control peaked at 46.3 MiB Torch allocation.
  [Full measurements](evidence/entity-inputs-v029.json) separate synthetic inputs,
  collection and memory counters. No formal training or learning-quality claim.
- Remaining limits: the cap loses individual facts above its limit; 32 dimensions
  and the cap require future learning evaluation. Quadratic attention arithmetic
  remains, while short-cutoff throughput is not a full-game speed estimate.

## 0.28.4 — 2026-09-27

- Problem/root cause: the same profile parameters were copied across root
  `configs/` files and bundled package data, so changing a setting could leave
  source and installed behavior out of sync.
- Improvement: keep one bundled `src/pvz_rl/data/train.toml` with sparse `demo`,
  `event-memory`, and `gpu_defaults` overlays. Loaders and the CLI resolve those
  overlays through one profile interface; `--profile event-memory` preserves the
  archived model choice.
- Verification: profile equivalence, custom-path loading, CLI training and the
  complete regression suite (546 tests) pass; no formal training was launched.
- Remaining limits: external scripts that refer to deleted `configs/*.toml`
  paths must use the bundled file and profile option.

## 0.28.3 — 2026-09-27

- Problem/root cause: collection still applied a species exploration coin after
  the ten-way comparison, so the submitted branch could differ from the
  highest-Q proposal. This made action exploration change the rejection and
  penalty distribution rather than only the requested tile target.
- Improvement: keep all ten branch choices greedy and apply the cohort budget
  only to the selected non-wait tile. Plant occupancy masks and unrestricted dig
  tiles remain unchanged; the branch coin field stays zero for journal/schema
  compatibility. The scheduler is recorded as `sequential_tile_epsilon_v1` so
  older two-coin checkpoints cannot be resumed under the new distribution.
- Verification: policy controls cover greedy branch selection at budgets 0, 10%
  and 100%, tile-only plant sampling, tile sampling for dig, and the existing
  rejection/environment parity cases. No formal training was launched.
- Remaining limits: the retained `per_head_epsilon` helper is a compatibility
  reader for archived schedule reports; current collection uses one tile coin.

## 0.28.2 — 2026-09-27

- Problem/root cause: the selector retained an illegal highest-Q proposal for
  penalty learning, but recurrent history could still describe that proposal as
  the previous action. Execution, metrics and diagnostics therefore disagreed
  about whether the simulator had waited.
- Improvement: keep proposal and execution channels separate across CPU, CUDA,
  recurrent collection, evaluation, event memory and journals. Rejected
  proposals remain targets and diagnostics; execution is wait (`0`) with
  `accepted=false`, the pinned reason, configured penalty and one-tick duration.
- Verification: focused policy, environment, recurrent, journal and CUDA parity
  controls cover unaffordable, cooling, full-board and empty-dig proposals plus
  legal and boundary actions. No formal training run was launched.
- Remaining limits: CUDA parity requires the pinned CUDA runtime; CPU fixed-tick
  compatibility retains its configured decision duration.

## 0.28.1 — 2026-09-27

- Previous problem/root cause: the ten-way selector was correct for the current
  occupancy masks, but the CPU/CUDA boundary did not make the proposal channel
  and CUDA rejection metadata explicit. Legacy executed-action history could
  therefore be mistaken for the recurrent proposal history when diagnosing an
  unavailable high-Q plant.
- Improvement: make the shared selector's all-ten-branch comparison explicit;
  keep plant tile masks independent of affordability and cooldown; retain the
  selected proposal through CUDA collection/evaluation; and publish CUDA
  acceptance, rejection reason and one-tick duration alongside the reward.
  Legacy event-memory keeps its executed-action convention, while recurrent
  Transformer–LSTM history uses the selected proposal and public outcome.
- Regression coverage: higher-Q peashooter at 50 sun, a cooling-down sunflower,
  occupied-tile geometry, full-board fallback, tie order, species/tile
  exploration, one-tick rejection, no plant creation, reason propagation and
  invalid-plant penalty. No formal training was launched.
- Verification: focused policy/environment/recurrent tests pass after the repair;
  full pytest, Ruff, dependency and doctor results are recorded with the final
  run. Remaining limitation: CUDA-specific execution still requires the pinned
  CUDA runtime and is skipped when unavailable.

## 0.28.0 — 2026-09-27

- Previous problems/root causes: demo weights could not enter autonomous training;
  training assumed an event-memory model and shuffled decisions. Initialization
  device/time CLI options mutated the recording verification configuration.
  The saving lesson was embedded in curriculum stages, CPU rules and CUDA hooks.
- Improvements: recurrent defaults throughout recording, initialization, training,
  evaluation and replay. Shared model/protocol inspection accepts demonstration
  weights or autonomous ZIPs, records source hashes and rejects incompatible
  transfers before creating environments. Resume restores optimizer and recovery
  state; weights-only initialization starts fresh. ZIPs embed run configuration.
- Recurrent lifecycle: fixed-weight cohorts retain previous proposals and public
  acceptance/duration, freeze completed slots and limit the final cohort to the
  remaining games. Whole-cohort equal-group regression recomputes episode states,
  carries detached state across 256-decision chunks and applies one clipped Adam
  update per full pass. Batch size 1,024 represents four sequences. Interrupted
  fitting restarts only the uncommitted pass; completed updates are preserved.
- User adjustments: remove the saving curriculum, rehearsal mixtures, custom sky
  rules, scenario generation, unused lesson defaults and obsolete feasibility
  tests/docs. Curriculum now starts at easy, then standard, then shared. Remove
  the max-minutes CLI flag; time allowances remain configuration settings. Keep
  recurrent 50%→1% tile exploration over 5,000 stage games, species-coin algebra,
  reward coefficients, engine pin and four autonomous passes. Event memory stays
  available through the event-memory overlay in the bundled train.toml profile.
- Verification: the existing human-1000 archive reconstructed all 16,364 decisions
  and the winning final state in 34.72 seconds, without fitting. Tiny complete-demo
  weights and same-device predictions transferred exactly, then autonomous fitting
  changed the weights. CLI initialization→training→evaluation→in-place resume
  passed. CPU/CUDA evaluation traces agree; collection/fitting interruption tests
  match uninterrupted weights and Adam state exactly. Independent returns, group
  means, padding, rejected/zero-tick actions, resets, gradient partitioning and
  atomic-save failure controls pass.
- Bounded GPU evidence: RTX 4070 Laptop GPU (8 GiB), 128 active slots, 384 collected
  decisions plus one synthetic 1,024-decision fitting chunk; peak PyTorch allocation
  924.68 MiB. That memory check completed zero games and performed zero optimizer
  updates. This allocation metric excludes CuPy/driver allocations and is not a
  full-game throughput or learning result.
- Final checks: full regression suite 536/536 passed; focused recurrent suite
  17/17 passed, including three added migration and CPU/CUDA sequence controls.
  The subsequent in-place elapsed-time control passed. Ruff lint/format, dependency
  checks and CLI help checks passed. In-place resume retains both elapsed status
  time and the prior best validation score.
- Remaining limits: no formal training or win-rate claim. Removed-lesson runtime
  checkpoints cannot resume those unavailable scenarios; compatible weights can
  initialize a fresh run. The exact archive digest migration permits only the
  known curriculum-only change; other recording modifications still fail checks.

## 0.27.0 — 2026-09-27

- Previous problems/root causes: early engine contact clamping, uncadenced bite
  transitions, inclusive pea tangency, missed mower crossings and low-HP defeat
  accounting. The human recorder conflated native/research settings and lacked
  its action codec. Shared demo defaults and unconditional cooldown encoding
  also changed the established collector's input contract.
- Improvements: pin merged game 1.7.0 / simulation 1.4.0 at
  `1424b4d802e783a36772091c49d96aa7c6036d7a` (engine PR #4).
  Share F/button follow-latest transitions, invalidate stale history responses,
  preserve horizontal scroll and isolate pending game replacements. Validate
  every recording artifact before opening the window; enforce one easy-stage
  attempt and keep interruptions incomplete. Give demo and collector commands
  separate bundled/configurable profiles; gate cooldowns to event_v8 on CPU/CUDA.
- Preserved work: integrate the unmerged Transformer–LSTM initialization and
  its complete archives; retain demo settings of 20 passes, 256-decision chunks,
  0.0003 learning rate, 0.5 gradient clipping and a 30-minute ceiling. Preserve
  the 50% to 1% tile-exploration schedule over 5,000 games in the recurrent
  profile and the existing collector's separate recipe.
- Cleanup: remove retired Q checkpoint inference, alternate CPU/CUDA masking,
  redundant mask/action helpers, unused transport flags and lesson-roster
  settings. Remove spatial/event-memory parameters from the recurrent demo
  profile. The active event-memory collector and historical recordings remain.
  Old model protocols fail before deserialization; current workflow instructions
  distinguish demo fitting from CUDA training.
- Regression correction: the unmerged demo tile-exploration adjustment had also
  changed the collector's coin probability. Restore its independently tested
  two-coin budget without changing the demo's 50% to 1% schedule or weakening
  the empirical distribution controls.
- Initialization correction: replace duplicated chunk-local group averaging
  with the shared complete-game balanced Q loss. Verify replay hashes, reward
  reconstruction and completion metadata before fitting; save each completed
  pass atomically and check recurrent checkpoint protocol on reload. A tiny
  complete replay independently checks group weights across unequal chunks.
  Recurrent selection now shares the public occupancy/branch selectors and
  applies full-board fallback per board, so one full board cannot unmask tiles
  on other boards. Remove unused record types and recurrent API aliases.
- Verification: independent collision and CPU/CUDA controls pass in the engine;
  headless native recorder startup, accepted/rejected actions, pause, disabled
  restarts/stage changes, interruption, natural loss, native replay and reusable
  archive verification pass. Grid/focused F and button event-loop checks cover
  repeated presses, stale responses, empty histories and unavailable panels.
  Exact collector regressions and final totals are recorded in validation.
  Final affected suites: 108 passes; engine rerun: 427 passes. The full research
  sweep and repaired-case rechecks are recorded explicitly in validation.
- Baseline compatibility: the unchanged research controller wins all nine
  versioned cases with new exact ticks. In particular, the previous hard/4 loss
  now wins at tick 43424. Independent failure/boundary controls remain strict;
  no controller or collision assertion was relaxed to recover wins.
- Limits: 1.6.0 recordings require their original engine. The recurrent demo
  checkpoint is a one-demonstration fit, not an event-memory collector resume.
  No formal training was launched.

Dates and results belong to their recorded version. Current behavior lives in
[architecture](architecture.md) and [training](training.md). Archived test counts,
artifact locations and learning outcomes remain in the historical validation snapshot.
The longer pre-consolidation notes remain in `git show b642c9c:docs/iteration.md`.

## 0.26.0 — 2026-09-26

- Problem: the previous controller used a compact 286-value spatial encoder and
  bounded event attention, with no recurrent state shared by waits, rejected
  proposals or duplicate digs. Human play could not be used as a verified
  initialization artifact.
- Cause: card countdowns were omitted from policy observations and the native
  replay was separate from learning transitions. No protocol linked a complete
  demonstration to recurrent Q fitting.
- Improvement: `event_v8` appends eight normalized cooldowns; the new entity
  Transformer–LSTM policy exposes shared single-step and sequence interfaces.
  `record-demo` preserves zero-tick action phases, pause queue order, native
  replay hashes and append-only transition records. `initialize-demo` verifies
  action order and observations, fits complete reward-to-go in 256-decision
  chunks, and publishes an initialization checkpoint, curves, coverage and a
  replay report.
- Verification: CPU encoding checks cover ready, initial recharge, immediate
  planting and cooldown normalization. Step/sequence recurrent parity, reset
  isolation, action/tile outputs, synthetic archive spill and native replay
  reconstruction pass. No formal autonomous training was launched.
- Remaining issues: CUDA kernel compilation and full 128-game recurrent
  sequence training need hardware runs; one demonstration gives no evidence of
  generalization or mastery. Legacy 0.25.0 checkpoints remain inference-only.

- Parameter adjustment: the later autonomous collector's tile exploration coin
  now starts at 50%, decays to 1% over 5,000 completed games, and stays at the
  floor. Deterministic validation clears both exploration budgets before any
  evaluation action.

## Full-game memory research — 2026-09-26

- Problem: a bounded event bank can lose early planting details, and rejected
  proposals become wait markers before entering policy memory.
- Cause: finite event/summary queues and executed-action-only memory inputs.
- Proposal: an append-only public event archive and joint causal sparse attention;
  record non-wait attempts, deduplicate consecutive digs at the same unchanged
  tile, and retain encoded plant/zombie/sun/mower-spent changes even during waits,
  using the existing observation schema without adding mower states.
  Compare reversible layers separately as an activation-storage optimization.
- Verification: inspected papers/project sources, ran synthetic existing-memory
  retention probes, independently enumerated sparse edge counts and same-tick
  causal counterexamples, and checked local runtime/GPU availability. Calculations
  are in [archived memory math](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/math/history-attention.md).
- Remaining issues: framework only; retrieval quality, actual sparse backend,
  gradients, throughput and learning quality are unverified. No policy change or
  training run was made. Current architecture documentation remains authoritative.

## 0.25.0 — 2026-09-26

- Request motivation: the user observed that earlier training often dug
  immediately after planting, even though the planting Q value was higher but
  masked. The previous controller removed unaffordable/cooling-down plant branches
  before comparison, allowing digging to win that restricted comparison.
  Historical viewer pages also truncated themselves when a row was selected,
  making the recorded reason difficult to inspect.
- Changes: all ten first-level Q values are now compared on every active state;
  plant tile masks describe occupancy only and digging covers all tiles.
  Rejected plant proposals remain in trajectories as automatic waits with a
  configurable penalty, while empty digs receive their own small penalty. CPU
  and CUDA use the same accounting. Journal paging keeps absolute offsets and
  immutable collection-time Q vectors.
- Verification: 544 maintained regressions passed, followed by 111 focused
  policy/environment/reward/journal/viewer controls after final adjustments.
  CPU/CUDA penalties, exact 128-game resume, old inference, viewer isolation,
  paging and reduced-size rendering pass; lint, doctor, dependencies and
  packaging pass. See validation for timings and artifact locations. No formal
  learning run or comparison was launched.
- Compatibility: the new identifiers are `event_sequential_q_v2`,
  `sequential_q_unmasked_penalty_v1`, and `sequential_q_mc_v2`. Previous
  0.24.0 checkpoints retain original inference and report reading but are rejected
  for new training, initialization, or resume.
- Remaining limitation: repeated invalid proposals can outweigh a victory reward;
  the earlier outcome-separation bounds cover only the economic/outcome component.
  This change does not establish that immediate digging or learning failure is fixed.

## 0.24.0 — 2026-09-26

- Problems: coarse zombie displacement/contact geometry omitted source gait and
  vault details. The viewer kept only a few selected-worker actions and discarded
  useful history on switching. Smaller cohorts limited measured device throughput.
- Causes and changes: source animation/ground-track and collision rules now drive
  independently implemented fixed-point CPU/CUDA mechanics in game 1.6.0. Speed,
  gait phase/remainders and portable RNG restore exactly. Mine trigger eligibility
  is separate from blast eligibility; airborne exclusions, chilling, vault landing,
  targeting and mower contact use inspected source rules. Matching mechanics remain.
- Research uses 128 complete games per unchanged sequential-Q update. Ten actual
  pre-action Q outputs piggyback on the existing collector readback. Every worker's
  entire current-game plant/dig journal uses bounded RAM and disk spill and streams
  into checkpoints. Focus/fullscreen, paged browsing and follow-latest expose raw
  historical returns without tile maps. Switching replaces board and history together.
- Verification: independent math/boundary controls preceded mechanics changes;
  CPU/CUDA states, events, RNG, snapshots/replays and all ten saving combinations
  pass. A 128-game short interruption control preserves policy, optimizer, RNG and
  history. Three warmed five-second cutoff viewer pairs measured **3.04% median
  overhead** and identical paired policy hashes, within 183.20 seconds total.
  See [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md) for regression checks, full measurements and scope.
- Compatibility: game simulation/snapshot identifiers and strict pin changed;
  fresh experiments are required. Q architecture and optimizer/exploration protocols
  are unchanged. Existing files and offline report readability are preserved.
- Remaining limits: reconstruction evidence and compact numerical mechanics are
  not original-binary equivalence. The frozen baseline now loses hard seed 4;
  this is recorded without tuning the lesson. Larger cohorts delay feedback and
  repeat the frozen controller longer. Five-second performance controls establish
  neither full-game throughput nor learning improvement. No formal training ran.

## 0.23.0 — 2026-09-26

- Problem: action kinds/digging used critic values while species/tiles used a
  separate PPO actor and alternating roles. The user requested one agent assembling
  commands through two Q-value judgments.
- Change: one shared spatial/history encoder, ten branch values and a shared
  nine-branch-conditioned 45-tile value head. Full-game Monte Carlo targets fit
  selected outputs with a balanced wait/plant/dig loss and one Adam optimizer.
  The actor, PPO/KL/entropy machinery and role schedule are removed. Compact
  simulator action transport, observations, memory, rewards and game pin remain.
- Exploration: only a winning planting branch explores. Independent species/tile
  coins use 1−sqrt(1−budget), preserving the requested 10% → 0.1% combined budget.
  The viewer shows raw Q values, branch comparisons, tile maps and overrides;
  terminal and report metrics expose pre-fit errors and missing action coverage.
- Evidence: derivation and independent controls preceded implementation. Bounded
  CUDA checks for seeds 101/102 finished in 8.61 seconds with real planting and
  16 Q optimizer steps each; complete throughput was 2,141/2,178 transitions/s.
  These one-second episodes establish operability only. See validation for tests,
  memory/timings and limitations; no formal learning was launched.
- Compatibility: fresh models required. Protocol manifests reject retired archives
  before class loading. New-protocol resume preserves one optimizer, RNGs, incomplete
  trajectories and fitting cursor. Historical runs, recordings and reports remain.
- Remaining limits: greedy selection cannot force exploration from waiting; untried
  values can still be wrong. Shared targets do not enforce exact branch/tile agreement.
  Neither the short controls nor the sequential-Q paper prove this agent will learn.

## 0.22.0 — 2026-09-25

Viewer follow-up: the Q row did not explain the comparison and could show a
planting value even when planting was unavailable. The controller and diagnostics
now share legal-value masking. Panels show the selected kind's lead over the next
legal kind, deterministic ties or a forced choice, with more precise Q values.
Independent arithmetic, legality/RNG controls and resized renders verify this
presentation change. Values can still be inaccurate; training and coverage are
unchanged. Existing checkpoints and runs remain compatible and preserved.

- Problem: the previous scheduler fitted both networks after every complete-game
  cohort, unlike the requested 256-game alternation. Long cohorts spent most time
  gathering full structured history and creating pinned buffers before critic work.
- Changes: an explicit saved role counter credits each optimized 32-game cohort,
  switches after 256 games, resets at stage changes, and records empty actor coverage.
  Inactive network/optimizer state stays fixed. Raw-only float32 gathering, bounded
  GPU token caching, reusable staging and one-batch-ahead transfer preparation reduce
  host work while preserving the authoritative CPU/disk trajectory and sample order.
- Viewer: critic return estimates and actual conditional planting probabilities,
  species-selectable tile heatmaps and decision timestamps. Manual and automatic
  replacement wait for unfinished undisplayed games; no viewer input resets a game.
- Verification: independent returns/token/gradient/Adam/RNG controls, role and
  interruption boundaries, viewer parity and matched bounded CUDA workloads.
  Measured median transport-control throughput improved 58.6%; see validation for
  conditions and limits. No learning-performance conclusion is drawn.
- Compatibility: new optimizer protocol rejects previous full resume; compatible
  0.21.0 inference/weight initialization and every existing artifact remain available.
  Greedy all-wait behavior and missing plant experience remain unresolved risks.

## 0.21.0 — 2026-09-25

The prior model sampled action kinds and fitted short-rollout values. The user
requested greedy top-level decisions with conditional planting PPO and actual
complete returns. The new controller has 47 critic outputs and an eight-species,
eight-tile-map actor, each with independent temporal features and Adam state.
Collection uses 32 complete games followed by separate critic and actor phases.
Short-rollout GAE, periodic overlap and initial actor warm-up are removed.

Game 1.5.0 corrects production/recharge/combat/headless mechanics and adds a
portable per-game RNG. Research adds five frontmost headless flags and compact
phase mapping, giving 286 observations and 289-value raw temporal tokens.
Old files remain preserved; changed signatures require fresh models. The old
five-flower/three-shooter saving control now loses; a revised public controller
uses a mine while saving for defense without changing the lesson configuration.

The architecture is rewritten around exact inputs and training responsibilities.
Derivations and source limits are in docs/math; verification evidence is recorded
in validation. Greedy untried values remain a coverage risk. No claim is made
that learning collapse is resolved, and no formal learning run was launched.

## 0.20.0 — 2026-09-25

- Problem: training exposes numerical curves but no live games; users cannot see
  sampled plant/dig behavior or follow individual games. The console also omitted
  cumulative net value and discounted return despite recording them in JSONL.
- Cause: full game state remains on CUDA and existing rendering serves offline
  recordings; per-tick synchronous rendering would interfere with the collector.
- Changes: one spawned four-panel renderer, private random subscriptions, manual
  per-panel switching, final-board retention and automatic replacement. Two pinned
  staging buffers sample selected present state before resets; bounded queues and
  generation/sequence checks discard stale frames. Closing/failing the UI leaves
  training active. The terminal adds explicitly scoped game reward/economy means.
- Organization: replace the flat implementation directory with `envs`, `policy`,
  `learning`, `evaluation`, `monitoring` and `presentation` packages. Update imports,
  resource ownership, tests and package data. Narrow deserialization aliases keep
  compatible 0.19.0 class references loadable without duplicate source files.
- Compatibility: output-only settings preserve 0.19.0 resume/weight transfer and the
  engine pin. Existing runs/artifacts remain intact. No formal training launched.
- Verification and overhead results are recorded in [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md).
  The final median viewer overhead was 0.55% (2,466 → 2,452 transitions/s), with
  matching final policy hashes in all three pairs. Individual comparisons ranged
  from −2.74% to +8.91%; performance remains hardware/load dependent.
  The viewer samples wall-clock frames, so it cannot display every intermediate
  state and is not a statistical replacement for completed-game diagnostics.

## 0.19.0 — 2026-09-25

- Problem: placement remained around 20% wins in the reported long run. Equal
  wait/plant-kind initialization did not give equal species probabilities, and
  injected branch exploration retained learned tile preferences. Ordinary training
  did not save hardware load samples, so stopping left no utilization history.
- Changes: remove placement and start with unrestricted, three-lane saving at 100
  sun. A checked no-income argument rules out winning without flower production;
  five public-state flower investments and reactive shooters win all ten lane
  combinations at 136.52 seconds on CPU and CUDA. These controls do not prove PPO
  will learn the lesson.
- Action initialization: wait weight 1.2, weight one per available species,
  exp(−12) total digging weight, and uniform legal tiles. A complete-action injected
  prior also explores uniform tiles. Joint likelihood, entropy and KL use that same
  mixture. Reward, architecture, engine pin, environment count and phase schedules
  are unchanged. New distribution signature requires fresh models.
- Hardware: one bounded background monitor for training/benchmarks; flushed raw
  JSONL, UUID-selected device metrics, session/activity labels, atomic hardware
  panels after evaluation attempts and graceful stopping, and model-free report
  regeneration. The terminal distinguishes recent task, window and run averages.
- Verification discovered unavailable compiler tracing interfering with same-seed
  reproducibility. The installed Torch compiler restores global CUDA RNG state;
  entering it beside collection is unsafe when tracing is unnecessary. Missing
  Triton is now detected before tracing. The strict output-enabled/disabled
  parameter-equality control then passed without relaxing its assertion.
- Current verification and telemetry overhead are recorded in
  [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md): 606 research regressions and 224 pinned game tests
  passed. Three paired telemetry checks retained identical policy hashes, with a
  2.16% median sampling cost. All existing runs and artifacts remain preserved.
  No formal training or learning comparison was launched. Lesson mastery and
  improved learned competence remain unmeasured.

## 0.18.0 — 2026-09-25

- Problem: the stopped placement run reached 3,781 games without mastery. Injected
  exploration fell from 10% during critic warm-up to about 0.016%, while recent wins
  declined to 8–10%. Actor learning also reduced the observed rate from about 4,068
  to 2,680 transitions/s.
- Exploration change: separate warm-up and formal schedules. Formal exploration starts
  at 5%, entropy starts at full strength, both decay over 3,000 actor-learning games,
  and floors of 0.1% injected noise and 10% entropy remain until mastery. Stage entry
  resets the clock. Validation explicitly disables injected exploration.
- Runtime change: CUDA sequence index templates avoid rebuilding full Python index
  lists; frozen evaluation uses inference mode; masked categorical hot paths use
  tensor-only operations; fixed-shape feature paths can use `torch.compile` without
  changing checkpoint keys. Host/device phase telemetry is recorded per rollout slot.
- Compatibility: retired-schedule checkpoints remain readable for inference but cannot
  resume or initialize new training. Same-protocol resume and stage handoff remain
  available. Existing `runs/` and artifacts are preserved.
- Verification: the final research regression passed 618 tests; CUDA doctor,
  dependency checks, packaging and Ruff passed. Three warmed eager benchmark
  repetitions reached a median 2,608 transitions/s, while the compiled path did not
  complete its bounded third repetition. The required 20% throughput gain was not
  demonstrated, so the result is inconclusive. No formal training comparison was
  launched by this change.

## 0.17.0 — 2026-09-24

- Problem: rare digging probabilities could rise abruptly despite sampled KL
  stopping; the 100 Hz discount weakened late outcomes enough for failed saving
  controls to outrank a successful control. Completed-game curves lagged policy
  changes and sparse action statistics could disappear during slot aggregation.
- Math first: documented and independently checked accounting, finite-game outcome
  bounds, GAE timing, exact hierarchical KL and a rare-action counterexample in
  [the math record](math/training-objective.md) before changing training behavior.
- Objective: gamma 1 preserves late outcomes; smaller development weight and mower
  expenditure prioritize natural wins under the documented default bounds. GAE
  retains a calibrated time-based trace. Values remain configurable, and no
  plant-retention rule or special digging reward is added.
- Updates: exact KL checks each available slot against frozen behavior. Excessive
  or non-finite movement restores actor weights and Adam state for the whole window;
  critic updates and collection continue. Sampled stopping persists across slots,
  advantages normalize once per rollout, and entropy decays with stage progress.
- Diagnostics: action-category sums/counts and pooled explained variance preserve
  sparse evidence; attempted/retained actor steps, rejection reasons, effective
  entropy and discounted reward components separate policy updates from past games.
- Compatibility: one architecture, one recipe, 128×128 collection and the same game
  pin. Existing runs are preserved. Earlier compatible weights support inference
  and initialization; full resume requires the new optimizer protocol.
- Verification: mathematical controls, CUDA integration, independent PPO/Adam,
  checkpoint/replay checks and complete regressions are recorded in validation.
  No learning comparison or formal training was launched; collapse resolution and
  throughput effects remain unmeasured. The user will run learning experiments.

## 0.16.0 — 2026-09-24

- Problem: the synchronous scheduler left the CUDA simulator idle during PPO updates
  and the learner idle during rollout collection; live traces showed roughly 1.7 s
  collection followed by 2.7–2.8 s updates.
- Change: periodic two-slot on-policy collection is now the only training scheduler.
  A frozen behavior snapshot fills both slots while the learner consumes the first;
  policy-version hashes, CUDA-stream events, queue waits and overlap are recorded.
- Compatibility: checkpoints require the periodic pipeline signature and fresh models
  are required. Files under `runs/` are preserved for the next session; historical
  generated artifacts outside `runs/` are cleanup candidates.
- Verification: compact-policy regression, CUDA smoke, checkpoint reload and periodic
  resume passed. Three short warmed measurements improved median throughput by
  5.8% (2,570 vs 2,430 transitions/s) at 46% more allocated VRAM. No games completed;
  learning effects remain unmeasured. Final review also replaced estimated overlap
  with interval measurements, made slots reusable, and deferred Ctrl+C until both
  updates and the episode ledger are committed.

## 0.15.0 — 2026-09-24

- Problem: plant species competed directly with waiting and digging, and stage runs
  could exhaust their time/game allowance before meeting mastery.
- Changes: a three-way wait/dig/plant head, conditional species head and nine tile
  maps use one forward pass with consistent masked mixture probabilities and PPO
  ratios. The initial dig bias stays -12. Conditional species entropy is explicit.
- Stage execution: optional `--until-stage-complete` disables both training ceilings,
  keeps completed-game probes and episode cutoffs, and stops only after the selected
  stage passes. Interruption/resume retains both optimizers, counters and pending
  validation; ordinary bounded runs remain the default.
- Representation: centralized command dimensions preserve compact simulator IDs.
  The factorized history trial was 4–5% slower and its bounded competence checks
  were inconclusive, so the existing history embedding remains. Temporary comparison
  code is removed; measurements are in [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md).
- Compatibility: event_transformer_v2 requires fresh weights; one policy/recipe,
  281 observations, 128 environments, game checkout and dependency pin remain unchanged.
- Verification: independent probability/gradient/PPO controls, stage state-machine
  tests, current regression suites and bounded CUDA checks are recorded in validation.
  Neither the hierarchy nor unlimited stopping establishes improved playing strength.

## 0.14.0 — 2026-09-24

- Problem: placement frequently dug its single shooter during actor warm-up; training
  throughput was dominated by PPO updates rather than simulation.
- Cause/evidence: repeated stochastic choices accumulated a substantial dig hazard at
  100 Hz. Small categorical embedding backward sorted many repeated history indices.
  These findings do not establish the sole cause of poor lesson success.
- Changes: event_v6 has 281 public inputs, omitting projectiles, spawned counts and all
  mower details except spent. Small trainable tables use equivalent matrix products
  during backpropagation. Distribution construction applies masking once. Initial dig
  bias is -12; the user reduced parallelism to 128, with 128 transitions each.
- Compatibility: fresh models; one architecture/recipe, existing runs and reports preserved,
  game checkout and pin unchanged. Warm-up, rewards, learning rates and PPO remain unchanged.
- Verification and limitations: measured checks and bounded comparison results are recorded
  in [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md). No formal training or claim of solved lesson learning.

## 0.13.0 — 2026-09-24

- Problem: each region duplicated zombie state/pole counts alongside threat composition.
- Change: nine regional values and 342 total inputs; nearest unused-pole distance
  replaces the pole count and five state counts. Negative distance sentinels preserve
  presence changes without retaining every movement tick. CPU/CUDA share layout offsets.
- Compatibility: fresh event_v5 models; game 1.4.0, simulation 1.1.0 and source pin unchanged.
- Maintenance: remove duplicate historical lesson/observation tests and consolidate
  the standalone dig-bias test into policy controls. Current lesson and math controls remain.
- Verification/results: see validation. Learning benefit and sufficiency of the
  reduced behavior information remain unmeasured; no formal training or comparisons.

## 0.12.0 — 2026-09-24

- Problem: explicit countdowns bypass temporal inference, while dense tick history
  would grow excessively at 100 Hz.
- Change: 417 timer-free inputs, independent actor/critic event Transformers,
  deduplicated raw history, contiguous chunks with public-prefix reconstruction,
  actor-only evaluation and 25 FPS presentation independent of simulation.
- Engine: game 1.4.0 / simulation 1.1.0 at 100 Hz. Fresh models are required.
  Local 20 Hz recordings were removed by metadata; the other game checkout is untouched.
- Checks and measurements: see validation. Retaining gamma/lambda 0.999 per tick
  shortens physical credit; per-tick exploration also occurs more often per second.
  Both are explicit confounds. Correctness is not a learning success claim.
- The timer/history controls passed, but the bounded paired learning check did not:
  the candidate reached about 1.6k transitions/s versus 8.7–9.9k for the archived
  feed-forward source and completed no training games in either 120-second window.
  Saving validation remained 0/3 for both seeds. Keep this method experimental.

## 0.11.1 — 2026-09-23

- Problem: saving collapsed into waiting; the transferred critic was optimistic
  about saving and the final critic undervalued rarely visited planting states.
- Change: configurable wait/plant exploration starts at 10%, holds during 1,024-game
  critic-only adaptation, then exponentially reaches 0.1% at stage game 3,000 and
  continues toward zero. Ordinary masked PPO uses the actual mixed probabilities.
  Digging remains legal and trainable. No reward or network changes.
- Compatibility: 0.11.0 weights still load. Explicit current configuration enables
  the new settings on weights-only transfer; resume retains the saved experiment.
  Warm-up and decay reuse saved stage residency and preserve optimizer identities.
- Verification and bounded learning findings: see [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md).
  The initial 256-game adaptation was insufficient; 1,024 games retained placement
  in short checks, but saving still failed and critic errors remained mixed.
  Only three actor-updating rollouts fit each final check. Remains experimental.

## 0.11.0 — 2026-09-23

- Problem: saving-stage destructive digging remained despite tighter lessons.
  Prior logs showed zero saving wins and a late window of 200 digs per 200 plants.
  Digging itself was negatively rewarded, but potential shaping telescoped and
  zero-time actions advanced decision discounting. These are mechanisms to test,
  not a proven sole explanation of the learned behavior.
- Change: one `net_value_v1` account values actual production, effective plant
  damage, health-weighted assets and mower expenditure. Purchases conserve value;
  losses and damage are charged/credited once. Historical maximum/drawdown is
  diagnostic. Gamma and lambda now advance with simulation ticks.
- Implementation: shared CPU/CUDA metric schema, research-local integer event
  counters, duration tensors, terminal-observation timeout targets and physical-time
  GAE. Actor/critic, input, curriculum, PPO settings and game pin remain unchanged.
- Compatibility: fresh 0.11 models required; retired reward and checkpoint
  conversion removed. New-method stage initialization and resume remain supported.
  Old generated outputs are retired; one recipe remains.
- Verification: independent ledger/time controls, complete regressions and the
  bounded two-seed fresh-saving comparison are recorded in [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md).
  No formal training or final-test evaluation is launched. Learning remains experimental.

## 0.10.4 — 2026-09-23

- Problem/hypothesis: early digging recurred. Free lesson income and spare time
  may make wasted purchases too recoverable; this is a curriculum trial, not a
  proven explanation of the easy-stage collapse.
- Change: the current recipe disables sky income in placement/saving. Placement
  has 100 sun and ticks 1/21/41; saving has 150 sun, two lanes and repeated waves
  at ticks 860/1100/1340. All quantities remain configurable. No legacy lesson
  mode, behavioral restriction, extra reward or alternative policy is added.
- Implementation: detached CPU episode rules and a narrow checked CUDA adapter
  supply per-game sky amounts, preserving mixed batches, reset boundaries,
  recording hashes and the installed game pin. No game checkout changes.
- Verification: income/rate calculations, all lane cases, exact delay boundaries,
  plant/dig failures, mixed income/cap events, CPU/CUDA agreement, recordings
  and regressions are recorded in [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md).
- Limit: verified feasibility does not demonstrate improved learning. Tight
  lessons can hurt exploration and transfer; normal wins and planting must be
  measured alongside digs. No formal training launched.

## 0.10.3 — 2026-09-23

- Problem/cause: rollout wording suggested a 128-action game cap, and the requested
  default parallelism is now 256. The actual collector continues unfinished games
  across updates; waiting was included in the ambiguous decision count.
- Change: 256 environments, 32,768 transitions per rollout; explicit CLI/startup
  wording and separate accepted plant/dig diagnostics in episode records, progress
  and reports. Waits/rejections do not increment that diagnostic. Existing saved
  configurations retain their own parallelism on resume.
- Investigation: archived comparisons contain about 99.2% waiting transitions,
  often with alternatives still legal. Record weighted waiting-sample optimization
  and temporal-abstraction tradeoffs in the existing research/reference documents.
  Neither proposal is enabled; rewards, per-tick choices and PPO losses are unchanged.
- Verification: see [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md) for regression and continuation
  controls. No formal training or stronger-learning claim.

## 0.10.2 — 2026-09-23

- Problem/cause: normal easy/standard/hard evaluation ran periodically and at
  completion even while the current curriculum stage had not passed; 500-game
  probes added frequent evaluation work.
- Change: mastery probes every 2,000 completed games; normal validation only
  after each individual stage succeeds. Keep 100/100 mastery and all evaluation
  cases. Save pending stage-success evaluations so resume can finish them before
  further learning; avoid duplicate final evaluation. Reports/checkpoints remain
  available without a validated model, and suites label missing evaluations.
- Compatibility: saved periodic recipes keep their schedules. Explicit current
  config with weights-only initialization adopts the change. Game pin, policy,
  rewards and optimizer settings are unchanged.
- Verification: 388 research, 207 CPU game and 222 CUDA game tests passed;
  scheduling, deadlines, checkpoint/resume, verified demos, reports, packaging
  and documentation checks are recorded in [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md).
- Limit: fewer normal validation points delay detection of within-stage
  regressions. No formal training or learned-performance claim.

## 0.10.1 — 2026-09-23

- Problem: free sky sun let the single-lane saving lesson pass without any
  sunflower. Mastery mainly repeated placement; living-resource shaping was
  stronger than requested, and hardware comparisons stopped at 128 games.
- Change: three simultaneous basic zombies in three distinct lanes at tick 1000,
  still 50 starting sun and ordinary rules. The 275-sun pre-breach budget cannot
  buy three shooters; a two-sunflower control supplies a feasible win in every
  lane combination. Lane count is configurable; omitted counts retain old cases.
- Reward: reduce economy weight from 0.5 to 0.1 without adding reward terms,
  planting restrictions or a direct purchase bonus. Death/digging shaping also
  becomes weaker; no win-rate improvement is claimed from this change alone.
- Runtime: configurable CUDA benchmark sizes through 1024 games, three repetitions,
  failed/incomplete-profile exclusion, warmup separation and load/memory records.
  New default is 1024×128 after a 41.87% median throughput gain and a longer
  completion/reset confirmation of 42.40%; peak sampled GPU memory is 1427 MiB.
  Verification: 376 research, 207 CPU game and 222 CUDA game tests pass, plus
  focused controls and a 1024-game batch smoke with three verified demos.
  Full measurements and limits are recorded in [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md).
- Compatibility: unchanged network and game pin. Explicit current-config
  weights-only initialization adopts new settings; resume retains saved settings.
- Remaining limits: task feasibility is not learned mastery; the old saving
  scores are not comparable to the new task. Larger rollouts may change learning
  behavior. No formal training or new learning-performance comparison is launched.

## 0.10.0 — 2026-09-22–23

- Problem: easy-stage training rapidly increased early digs and lost lesson skill.
  Mixed completed-game logs initially showed 100% wins from short rehearsals alone.
  Shared learned features let value-loss updates directly alter action probabilities;
  their causal contribution to collapse requires comparison evidence.
- Change: independent actor/value encoders (169,467 total parameters), separate Adam
  states and clipping, critic epochs continuing after actor KL stop. Frozen critic
  inference is batched after action collection. Rewards and exploration are unchanged.
- Diagnostics: separate task windows, plant usage, episode/transition composition,
  legal-dig probabilities, gradient norms, optimizer steps and post-update drift.
- Compatibility: tensor-only 0.9.0 weights conversion duplicates the shared encoder;
  current checkpoints resume both optimizers. Preserve runs and the game pin.
- Sources: inspected DeepSeek report, What Matters architecture findings and SB3
  interfaces; adopted gradient isolation and phase-specific work, not LLM algorithms.
- Verification/results: independent controls, full regressions and paired five-minute
  transfer comparisons are recorded in [archived validation](https://github.com/Comb1e/RL-Pvz/blob/40529687133cf897ad401034991f755bada34954/docs/validation.md). The first 200
  games are examined separately. No formal training is launched.
- Remaining problem: easy validation improved for both transfer seeds, but saving
  retention failed and seed 102 developed destructive digging later. Throughput
  was lower. The method remains experimental; separation alone is not a fix.

## 0.9.0 — 2026-09-22

- Problems: too many recipes and reward terms; destructive digging, unstable saving
  mastery and large PPO divergence. Choice-only reduction amplified a small subset
  of transitions; the old representation duplicated several signals.
- Changes: one 500-value categorical observation, one 118,611-parameter spatial
  grouped policy, one TOML recipe, three reward components and standard PPO means
  with configurable KL stopping. Remove old policies, recipes, comparison commands,
  event rewards and the unused early-dig penalty proposal. Keep digging legal.
- Checkpoint behavior: structural signatures gate weights-only initialization;
  new optimizer/counters/schedules allow explicit reward/PPO/curriculum changes.
  Resume preserves the saved experiment. Retired weights cannot load; reports and
  recordings remain independent. Obsolete local checkpoints were deleted at the
  user's request, preserving their recorded evidence.
- Verification: complete research/game regression, CPU/CUDA math and gameplay
  controls, reload/transfer/stage/report/replay tests, packaging and bounded fresh
  comparisons. Exact counts and results are in validation.md.
- Results/limits: both compact pilots scored 0/20 deterministic placement wins,
  versus 20/20 for both references; the improvement criterion failed. The method
  remains experimental. Regional compression is partial observation, resume resets
  active episodes, and the short comparison cannot isolate individual changes.

## 0.8.1 — 2026-09-22

- Follow-up: The user requested perfect 100-game mastery and less frequent validation.
  Replace the permissive 20-case gates with 100/100 per required task, including
  final shared-stage mastery. Probe every 500 training games; validate checkpoints
  every 2,000 with the original 50 cases/difficulty. A separate 100-case curriculum
  seed range avoids enlarging normal validation or touching held-out seeds.
  Resume preserves saved gates; explicit stage handoff can select the new protocol.
  README now lists every stage's pass criteria and the two schedules.
- Problem/cause: teaching stages advanced within one run; strict resume restored
  the same budget and offered no explicit checkpoint handoff to another stage.
- Change: select one stage, retain its rehearsal mix, stop on mastery or budget,
  and save probe/final checkpoints. Initialize a chosen stage from compatible
  weights and optimizer state with fresh local schedules and parent provenance.
  Existing automatic curriculum and same-stage resume remain available.
- Verification: stage/state boundaries, CUDA handoffs through all five stages,
  exact restored weights/Adam moments, interrupted resume, mismatch rejection,
  normal-game checkpoint selection, CLI/report smoke and regression suites;
  recorded results are in validation.md.
- Remaining limits: inherited learning settings must match; stage budgets do not
  certify mastery. Short checks establish integration, not improved win rates.

## 0.8.0 — 2026-09-22

- Problem/cause: CPU collectors, hybrid jobs, and old recipes remained reachable;
  installer HEAD requirements and README examples did not match the current folders.
- Change: CUDA-only training preflight; remove CPU collectors/transport, hybrid
  training, four standalone recipes, and legacy pilot/benchmark commands. Keep
  four direct conditions, CUDA SC2 comparisons, and CPU verification/inference.
  Archive the pinned game commit without switching its source checkout. README
  now maps the actual folders and uses consistent fresh output names.
- Compatibility: CUDA weights, optimizer state, learning settings and engine pin
  remain compatible. Ignore dormant conditions for individual resume; replace
  retired buffer metadata for historical inference. CPU/hybrid resume is rejected.
- Verification: full research and both game suites, CUDA smoke/report/replay,
  checkpoint action/hash controls, pinned source staging, documentation examples,
  lint, dependency checks and packaging; exact results are in validation.md.
- Remaining limits: CUDA hardware is required for training. The plant-reward
  candidate remains experimental; this cleanup establishes no learning improvement.
  The CPU simulator remains necessary for correctness checks and replay verification.

## Documentation consolidation — 2026-09-22 (0.7.1 unchanged)

- Problem/cause: release notes accumulated into repeated installation, reward,
  architecture and experiment guides, leaving obsolete advice beside current behavior.
- Change: replace the 1,000-line README with a daily-use guide; combine paper/SC2/
  reward notes into `research.md`, GPU measurements into `validation.md`, and game/
  GPU integration into `architecture.md`. Deduplicate references and version history;
  remove the seven superseded documents after moving their useful content.
- Verification: local links/anchors, CLI examples and source/config references are
  checked; committed evidence files and executable settings remain unchanged.
  No learning run or new performance claim is part of this documentation change.
- Limit: older external bookmarks to removed documents need the new README index;
  detailed historical prose remains in Git and local artifact paths may be absent
  from fresh clones.

## 0.7.1 — 2026-09-21

- Problem/cause: native viewers rejected zero-time research recordings; SC2 normal
  validation remained 42%/0%/0%; frequent early digging and forced-action samples
  motivated exploration/credit hypotheses. Mower-free lesson logs lacked context.
- Change: compatible source readers in both game branches; optional choice-state
  PPO, GAE 0.999 and minibatches 1,024; refilled GPU validation; lesson-aware logs;
  offensive potential, small non-nut eaten penalty and trainable initial dig bias.
  Installed game stays pinned at 1.3.0; legacy profiles retain their semantics.
- Verification: 374 research, 207 CPU-game and 222 CUDA-game tests; exact replay
  hashes, gradients/optimizer, reward boundaries, reload and shared-demo checks.
- Limit: final candidate improved early lesson learning but normal validation was
  2/15 and 0/15 in two three-minute checks. Experimental; no two-hour advantage proved.

## 0.7.0 — 2026-09-21

- Problem/cause: absent sustained attackers, joint entropy favoring large action
  groups, frequent validation and no cumulative runtime bound.
- Change: balanced type/tile exploration, spatial grouped policy, stage-start
  residency, 100-game probes, 1,000-game validation, result reuse, cumulative time
  allowance and explicit A–F comparisons. Game remains 1.3.0.
- Verification: 338 research/214 game tests, later focused deadline checks and
  two five-minute diagnostics; single environment/workspace consolidated.
- Limit: both diagnostics failed placement; normal validation incomplete at the
  deadline. No candidate recommended or A–F/formal matrix launched.

## 0.6.0 — 2026-09-21

- Problem/cause: CPU simulation and transfers dominated small-policy GPU learning.
- Change: game 1.3.0 CuPy/NVRTC backend, ordered batched combat, tensor PPO/GAE,
  CPU-verified demos, capacity checks and 128 decisions per parallel game.
  Promote 128 GPU games after correctness/speed gates; retain explicit CPU hybrid.
- Verification: 312 research/214 game tests; three-repeat benchmark 10,733 vs
  2,835 decisions/s (3.79×); installed pin `8861824`, simulation 1.0.0.
- Limit: gain combines backend and larger rollouts; eight-game CUDA is slower.
  Compact host synchronization remains. No thermal-soak or convergence claim.
  Temporary development environment was moved to Recycle Bin; one research `.venv` remains.

## 0.5.0 — 2026-09-20

- Problem/cause: ten-tick decisions limited immediate actions; reward lacked defense
  events; decision counts did not reflect the requested game-based progress.
- Change: zero-time plant/dig adapter and versioned replays; one-tick wait/rejection;
  loss −2, activation sun cost, wall-nut absorption, empty-blast cost; completed-game
  budgets/curriculum/validation and checkpoint counters. Game stays 1.2.1.
- Verification: 283 research/199 game tests, all-lane lesson controls, source/tick
  rewards, game overshoot/resume, Windows/CUDA and native video checks.
- Limit: discount remains per decision; final rollouts can exceed targets; action
  adapter uses pinned private hooks. No new learning pilot validated these changes.

## 0.4.1 — 2026-09-20

- Problem/cause: kill-only feedback ignored partial damage; capped economy omitted
  valuable living plants and wealth above the scale.
- Change: mower kill −2/N, nonlethal HP reward, empty activation −0.2; uncapped
  plant-plus-sun potential; separate logs and legacy compatibility. Game stays 1.2.1.
- Verification: 207 research/199 game tests, 40 new reward controls and native outputs.
- Limit: gradual damage can outscore instant kills; empty mower activations normally
  do not occur; previous pilot results used a different objective.

## 0.4.0 — 2026-09-20

- Problem/cause: 10,229-game archive lacked sustained attackers; flat sampling gave
  wait 1/136, weak economic scaling and no mastery gates.
- Change: tactical 1,140-value encoder, grouped type/tile policy, teaching state
  machine, capped paired pilot, +1/N plant and −1/N mower kill reward. Game stays 1.2.1.
- Verification: 166 research/199 game tests; original pilot interrupted for reward
  change, revised matched 65,536-decision results inconclusive; both diagnostics failed.
- Limit: immediate plant/dig cycles persisted; joint entropy still preferred large
  groups. Separate kill objectives and incomplete studies cannot be pooled.

## 0.3.2 — 2026-09-20

- Problem/cause: repeated mask IPC/legality queries and rollout copies; game had
  advanced to 1.2.1 while research pinned 1.2.0.
- Change: cached authoritative legality, masks sent with observations, reusable
  CUDA rollout, phase timings/paired benchmark and non-editable 1.2.1 installation.
- Verification: 120 research/199 game tests; exact learned tensors/Adam state in
  reference comparisons; final three-pair speed gain 22.4% (2,176→2,664 decisions/s).
- Limit: CPU simulation and small inference batches remained; startup/validation
  excluded from timing; no learned-skill claim. New game pin required fresh training.

## 0.3.1 — 2026-09-20

- Problem/cause: configuration selected CPU despite available CUDA.
- Change: default policy device CUDA, explicit CPU override and unavailable-device
  errors; simulation still on CPU. Game stays 1.2.0.
- Verification: 87 research/199 game tests; CUDA-tagged policy/Adam tensors and reload.
- Limit: availability did not establish speed; existing CPU runs retain saved settings.

## 0.3.0 — 2026-09-20

- Problem/cause: duplicate rendering, bulky outputs and conflated package/simulation
  identity while game 1.2.0 introduced native presentation APIs.
- Change: pin `a47056d`, native renderer, metadata, compressed demos and seekable
  viewer; MP4 optional, separate versions and strict old-checkpoint rejection.
- Verification: 87 research/199 game tests, no-FFmpeg default demos, archive reads,
  native/browser playback, source staging, render/hash purity and packaging.
- Limit: old weights need their original pin; metadata needs whole-recording hashes.

## 0.2.0 — 2026-09-20

- Problem/cause: little operational output and manual visualization; shared model
  identity and presentation-vs-research compatibility were unclear.
- Change: throttled logs, post-update metrics/TensorBoard, offline curves, shared
  checkpoint demos/video and export recovery with separate compatibility settings.
- Verification: 154 combined research/game checks; shared hashes, unchanged weights
  with output toggled, old/empty reports, interruption, corruption and browser video.
- Limit: rendering/FFmpeg dependencies, extra export time and missing historical metrics.

## 0.1.0 — 2026-09-20

- Problem/cause: empty research directory; game lacked an RL wrapper and protocol.
- Change: Gym adapter, fixed 406 actions, spatial encoder, masked PPO, baseline/hybrid
  controls, seed separation, reward shaping, checkpoint recovery, benchmark/reporting
  tools and pin `b3cfbd8` from the independent game.
- Verification: action/mask/terminal controls, known heuristic wins/losses, short
  CPU/CUDA/Windows learning, full miniature protocol and package checks.
- Limit: no trained-skill/generalization claim; partial observation and inexact
  rollout/RNG resume. Formal runs remain user initiated; no remote was configured.
