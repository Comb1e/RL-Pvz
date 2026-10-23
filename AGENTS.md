# Project instructions

- Search relevant research and projects when establishing solutions;
  record the sources actually used in `docs/references.md`.
- Use Git, feature/fix branches, and Conventional Commits. Never push directly to
  main. Use PRs, rebase before merging, and prefer squash merges.
- Keep shared constants in configuration and repeated behavior behind shared interfaces.
- Use explicit state machines for complex transitions.
- Keep current architecture and workflow diagrams in `docs/architecture.md`.
- Record date, previous problems, root causes, improvements, verification, and
  remaining issues for each version in `docs/iteration.md`.
- Keep only the three core smoke tests described below, per the user's selected scope.
  Do not restore mathematical, boundary, backend or recovery regression suites.
- Keep the game dependency separate and pinned. Never expose seeds, future schedules,
  or snapshots to policy inputs.
- User's current scope: complete the code and training README; availability and short
  integration tests are allowed. Do not launch formal training unless requested.
- In this project, always keep only one training and model framework.
- Python: .venv\Scripts\python.exe

## Code structure

- `tools/bootstrap.ps1` and `tools/bootstrap.sh` install the CUDA environment from
  an existing game checkout. Both reuse `tools/stage_game.py` for verified,
  non-editable installation of the engine lock, without changing the game checkout.
  Bash accepts `--game-repo` / `--python`, handles POSIX and Windows virtual
  environments, and cleans its temporary archive on exit. Do not download
  dependencies in smoke tests.
- `tools/collect_fit_diagnostics.py` wraps the existing autonomous train command
  with temporary host-timing wrappers and an optional bounded CPU/CUDA profiler.
  It reuses `monitoring/hardware.py` with an optional independent detail sampler
  for thread CPU, process I/O and Linux cgroup throttling. Streaming chunk/worker
  events retain exact batch shapes, compiler paths, graph state and fallbacks.
  The lightweight collector adds no CUDA waits/readbacks or learning changes;
  detailed profiler overhead is explicit.
  Collection counters include the small-active tail; return finalization, spool
  preparation and checkpoint serialization have separate spans. `pipeline_benchmark.py`
  calls production lifecycle phases on fresh scratch cohorts and stops after
  four commits and the first decision of the next cohort. `tools/tune_gpu.py`
  owns synthetic/trajectory/full-cycle selection and the paired promotion gate.
- `src/pvz_rl/data/train.toml` is the single parameter source for train/demo overlays.
  Both use `entity_v1`, `two_stage_transformer_q_v1` and
  `complete_return_action_global_v3`. Old-objective resumes are rejected; structurally compatible
  BF16-master weights support inference/fresh initialization; verified recordings are reconstructed without changes. Preserve
  user reward adjustments. `training.demo.passes` is 20, independently of autonomous
  `training.n_epochs` (4). Tile (50% to 1%) and accepted-plant species (10% to 1%)
  probabilities decay over 10,000 stage games and freeze per cohort.
  `reward.early_sun_spawn_fraction=[1,3]` and `early_sun_extra_multiplier=2` triple
  early actual Sunflower income value. Shared clipping uses `training.max_grad_norm=5`.
  `training.objective.species_consistency_weight=1.0` controls online-selected,
  detached EMA-evaluated tile targets. `species_consistency_huber_delta=1.0` under
  the same objective table is finite, positive and non-boolean; consistency uses
  twice unreduced Huber to preserve small-error MSE scale. No gates or sample
  exclusion apply to accepted non-wait consistency. `training.performance.fit_batch_groups=4` groups flat batches.
- `envs/encoding.py` defines the shared 11-field entity schema, 18 globals,
  normalization, truncation diagnostics and `EntityBatch`. Order is mowers, plants,
  zombies, projectiles; nearest zombies come first. Cap 256 drops from the end;
  globals count all entities. CUDA encodes the same public facts, not reserved slots.
- `policy/entity_attention.py` owns shared embedding/attention interfaces, pre-LN
  blocks, padding masks, SDPA, exact query fallback and outer checkpointing.
  `policy/two_stage_transformer.py` owns `TwoStageTransformerPolicy`: independent
  species encoder (128 width, 4 layers, 8 heads, FF 512) and conditional tile encoder
  (192 width, 6 layers, 12 heads, FF 768), with zero dropout and 16 features/head.
  Species uses one global token; tiles use global/branch tokens plus 45 spatial queries.
  There is no learned history, scalar bypass, fusion or feature-sharing between stages.
  Species Q values are direct predictions; only requested board/branch tile pairs
  are evaluated, with duplicate requests reused. `selected_values` supervises wait
  through species and non-wait through tiles. `two_stage_policy.py` owns `TwoStageQPolicy`.
- `policy/runner.py` is the shared stateless runner for collection/evaluation.
  `policy/sequential_q.py` owns all-45 tile proposals, greedy ten-way selection
  and balanced selected-action loss. Occupancy is checked after a tile is
  selected, so full boards and occupied plots remain representable as invalid
  proposals; sun/cooldown do not mask ordinary branches. `active_batch.py` packs active canonical slots into public-width
  and power-of-two buckets with stage-specific token overheads 1/47. Padding has
  no actions, RNG or supervision. Behavior draws retain original-slot shape/order.
- `policy/cudagraph_backend.py` owns separate collection/fitting captures and owned
  outputs/activations/gradients. Every online/EMA encoder has a bounded collection
  LRU: 8 entries and 512 MiB VRAM headroom by default. Eviction fences copies and
  releases retirees through existing handoffs; pressure/failure uses exact eager
  execution. Fitting compilation owns a separate code object per encoder/phase,
  admits up to six fitting shapes and pads partial microbatches before trimming readouts.
  Quiescent phase changes release outgoing captures and renew only the owner's
  fitting compiler cache; never raise global Dynamo limits or reset unrelated caches.
  `compile_collection` and `compile_fitting` are independent central settings;
  `compile_kernels` is rejected except as ignored historical weight metadata.
- `envs/` owns simulator adapters, action outcomes and reward ledgers;
  `action_credit.py` maps ordered plant/projectile events back to source actions
  without entering observations, while `cuda_accounting.py` supplies read-only
  aggregate reward counters. Rejection retains its
  proposal/reason and zero penalty audit fields, executes wait and advances one tick; accepted plant/dig
  advances zero ticks. Accounting is not a model input. `rewards.py` owns the
  six-component global terminal/asset and action damage/blocking/sun/mower
  objective plus diagnostic proximity/time fields. `CompleteGameBuffer` can
  apply validated source credits before chronological return finalization.
  `rewards.py` owns early
  Sunflower eligibility, physical conservation, proximity penalties and cohort-median
  victory finalization. Recording fact checks remain configuration-independent.
- `learning/exploration.py` owns persistent accepted-plant species commitments.
  Choose another species uniformly; wait ordinarily for sun/cooldown, with no
  timeout. All 45 tiles remain proposal candidates, so occupancy is classified
  after selection. Success clears commitment without recursively drawing a new one.
  Provenance and original policy proposals remain diagnostics. Exploration is
  disabled for evaluation, demonstrations and auxiliary games.
- `learning/collection_scheduler.py` retains actual decision tickets, original-slot
  draws and per-slot counters. Accepted planting returns immediately to normal
  collection; there is no counterfactual source, probe lane or rollout state.
- `learning/sequential_training.py` owns `CudaSequentialQ`: canonical collection,
  background drains, complete returns and four whole-cohort fitting passes.
  `learning/flat_q.py` owns shared autonomous/demo fitting: selected regression,
  double-estimator species consistency and demo ranking.
  `learning/objective.py` owns the shared detached online argmax/EMA evaluation
  target, selected tile IDs, estimator gap and MSE-matched Huber. Reuse existing
  online tile tables and EMA forward; add no Transformer forwards. All 45 tile
  candidates and lowest-index ties remain; occupancy and branch availability are
  routed after selection. Targets
  follow actual proposals, including rejection/commitment, never diagnostic
  proposals. Selected counts are 3x2 wait/plant/dig by acceptance; consistency
  counts are 2x2 plant/dig with rejected columns zero. Denominators cover the whole cohort. No non-wait sampled
  return directly trains species. One clipped Adam update commits per pass, then
  both EMA networks update with decay 0.95. No detach boundaries or recurrent fitting.
- `learning/cuda_buffer.py` stores executed active metadata and ragged entity
  slabs under one RAM/disk budget. `source_actions.py` defines private
  `(environment, episode, ordinal)` references and append receipts. Per-episode
  integer lookup slabs rebuild and validate at recovery. Inactive-gap carry is
  persisted before padding is dropped. Active flat batches exclude capacity gaps/padding;
  complete returns remain chronological per episode. `batch_transport.py` owns
  shared prepared sources, cached CPU plans, bounded slim scans and counting
  partitions, plus ordered double-buffer `BatchPrefetch`. Pinned entity staging
  grows to the current bucket only after its copy fence. `action_routing.py`
  owns segmented invalid compaction and chunk carry state; first-row
  representatives per 100-row window affect fitting only. `step_result.py`
  supplies array-backed step results and lazy compatibility infos. `host_transfer.py` owns reusable pinned
  `HostHandoff`; views are consumed before reuse.
  `PreparedDeviceBatches` in `batch_transport.py` owns immutable resident raw
  inputs/labels with the original source batch ranges and padding; annotation
  admission covers train/validation together and retains streaming fallback.
- `learning/checkpoints.py` validates model/objective/schema and demo v10,
  action-global v4 recovery/trajectory contracts before simulator allocation;
  the objective protocol is v3 because fitting representative weighting changed.
  Model-state `stateless-two-stage-bf16-master-state-v2` stores FP32 online
  masters, FP32 Adam state and BF16 EMA runtime weights. No migration or
  partial transfer exists. Fresh compatible `--init-from` transfers full weights
  only; resume retains saved learning settings. `cuda_q.py` owns cohort lifecycle and
  atomic recovery. Saving stops selection and finishes issued tickets before
  serializing simulator, accounting, controller,
  scheduler, RNG, trajectory, optimizer and both networks. Pools/maps/caches rebuild.
  Fitting/finalization/EMA changes require complete actual rows. Uncommitted passes
  restart; committed steps are not repeated. `performance.py` owns execution refresh.
  Annotation progress v4 is self-contained and accepted for weight-only fresh
  initialization, including zero-pass/incomplete snapshots. Shared immutable-byte
  loading hashes the loaded version and retries Windows sharing conflicts;
  destination provenance records source pass/kind/convergence. Annotation progress
  is never autonomous recovery. Progress v3 first needs annotation resume to v4.
  `SnapshotStager` groups borrowed authoritative master/Adam tensors by CUDA
  device/dtype into owned CPU snapshot storage. Cached layouts rebuild on Adam
  state changes; bounded allocation failure uses direct copies. Serialization
  waits for D2H, and each annotation pass remains durably atomic before advancing.
- `presentation/demo_recording.py` records reusable v2 archives/native replays.
  `learning/demo_initialization.py` verifies replay facts/outcomes/timing and reprices
  rewards without rewriting recordings, then fits 20 passes using both online/EMA
  stages. Journals/viewers are read-only actual Q/proposal/execution diagnostics,
  not learned history.
  Demo ranking reuses batch-scoped online forwards from `flat_q.py`; annotation
  fitting prepares label masks once and preserves indexed extrema tie gradients.
  `live_layout.py` owns paging/follow-latest; `live_view.py`
  snapshots effective settings without reloading TOML/device values.
- `annotation/dataset.py` owns versioned rich snapshots, canonical entity rows,
  atomic annotation storage and durable pending/labeled/discarded records.
  `annotation/sampler.py` samples reachable simulator states. Tile records
  select only currently usable plant cards; action records accept any running
  endpoint and expose complete branch availability without a sampler-selected
  action. Live and batch generation share uniform
  0–20,000-step endpoints, bounded terminal/unusable retries, 500–1,500-step
  planting intervals and an eight-plant cap from `training.annotation`. Omitted
  annotation seeds use system entropy; explicit seeds remain reproducible.
  Each interval draws equally from wait and the eight configured plant types;
  unavailable choices become wait without redrawing. Usable choices select random
  legal plots; only engine actions change sun, health, cooldowns or mowers.
  `presentation/web_assets.py` owns the shared
  Canvas renderer; `presentation/annotation_viewer.py` serves the one-map-at-a-time
  CPU annotator with difficulty selection, pointer-drag painting, synchronized
  Wait/Dig controls, restart recovery, and relative APIs. Sampler and validation
  share automatic action-label derivation: impossible plants stay locked; Dig is
  an editable default, including on empty boards. Generation failures retain the
  saved state and retry through relative POST without restarting the server.
  `dataset.py` also groups contradictory labeled records by
  canonical plant/entity/global inputs; the annotator shows that conflict queue
  before generation, applies one revision atomically to every group member, and
  supports Save and exit. Browser-local history navigation revisits completed
  maps; revision-checked edits persist in place without consuming the pending map
  or invoking the sampler. Pending resolution alone generates the next map.
  `learning/annotation_fit.py` freezes the opposite stage for tile/action fitting
  and publishes a compatible final initialization only after exact train-and-validation
  feasible top-1 and absolute-bottom convergence over all outputs.
  Its top-1/bottom-k hinge ignores ordering inside label sets, defaults to unlimited
  passes, and enters an explicit L2 phase only after training first converges.
  It streams metrics, records the regularization phase, and saves separate atomic
  annotation weights/Adam/RNG/pass progress. Matching dataset/split/settings and
  L2 identity gate resume; omitted options inherit saved settings and pass limits
  count additional commits per invocation. Progress v3 resumes into v4 with its
  obsolete cap ignored. History tails reconcile against durable pass metrics;
  partial gradients are discarded and commits defer Ctrl+C until durable.
  Annotation training reuses resident raw batches when admitted under the central
  cache cap and GPU headroom; learned activations always recompute. Train/validation
  evaluation preserves separate shapes, order and totals with one combined metric
  readback. Test-cache admission follows convergence and fresh BF16 verification.
  All sources/caches close on exit and rebuild after resume. `performance.py`
  resolves optional stage fitting limits; annotation compilation stays disabled.
  Cache/execution settings are reported but remain outside fitting identity.
  Fresh reloaded BF16 verification permits zero-pass publication. Long histories use the standalone
  `tools/plot_annotation_loss.py` min/max-envelope canvas viewer.
  It never creates an autonomous resume state.
- `learning/action_routing.py` classifies pre-action public legality and actual
  outcomes, with branch impossibility preceding bad tile attribution. Trajectories
  persist route/branch usability/reason. `flat_q.py` skips irrelevant stages via
  CPU-planned indices: wait/species-invalid use species only, tile-invalid uses
  tiles only, accepted non-wait adds both online stages and EMA tile inference.
  Invalid regression uses `training.objective.invalid_action_target=-1.0`, never
  episode rewards; rejection penalty settings are removed and audit fields are zero.
  Consecutive invalid runs are compacted by this shared module, while complete
  returns retain every row. Consistency balances accepted plant/dig counts only.
  Verified demos reprice historical rewards and share these routes without
  rewriting recordings.
- `evaluation/` uses the stateless runner and CPU-verifies CUDA action traces.
  `monitoring/` owns availability, both-stage entity/throughput benchmarks, bounded
  collection/four-pass-fit/collection controls and objective diagnostics.
  `DeviceProfiler` bounds pending events with central `telemetry_span_limit`,
  reuses completed events and reports drops without synchronization. Progress
  uses host records for cohort wins/returns and interval active rates, distinct from
  lifetime totals. CUDA timings cross existing handoffs; telemetry adds no hot-loop
  device synchronization. Pass-boundary aggregates report `species_consistency_loss`
  (balanced scaled Huber), `species_consistency_mae`,
  `species_consistency_saturation_fraction` and `species_selection_evaluation_gap`;
  absent non-wait rows are nullable/`n/a`. Gap is EMA candidate max minus EMA at the
  online-selected tile: disagreement, not proven overestimation. Selected error and
  conditional best-tile minus sampled-tile placement gap stay separate.
- `docs/architecture.md` traces current ownership/workflows/failures.
  `README.md` owns quick-start commands; this file maps test ownership.
  `docs/math/entity-inputs.md` owns encoding/attention,
  `sequential-training.md` learning/attribution,
  `training-objective.md` reward economics, `invalid-action-penalties.md` rejection,
  and `training-throughput.md` execution;
  results/history belong in `docs/iteration.md`, raw measurements in `docs/evidence/`.
- `gpu.py` resolves a visible index or CUDA UUID and applies execution-only native-BF16
  profiles; `monitoring/fit_benchmark.py` measures the complete fitting path and
  `tools/tune_gpu.py` runs isolated bounded calibration trials. `presentation/web_viewer.py`
  serves the existing bounded live-view mailboxes from a CPU process for JupyterLab
  Server Proxy; it does not own model or simulator state.

- Annotation protocol v2 supports `annotate --kind tile|action`, separate JSONL stores, feasible and locked absolute-infeasible labels, shared conflict detection, and head-specific fitters. The action annotator exposes plant cards as Canvas hit targets; the duplicated per-action HTML button strip is removed. `initialize-demo --init-from` creates fresh demo optimizer/EMA state from either annotation artifact.

## Test ownership

`tests/test_smoke.py` is the entire test suite, capped at 200 lines with three tests:

- CLI help starts successfully within a 15-second subprocess timeout.
- CPU inference consumes a real public board, runs both policy stages, returns finite
  Q values with valid dimensions, and submits an action to the simulator.
- The explicitly selected `learning` test collects two one-second CUDA games with two
  environments and one fitting pass, then reloads a checkpoint and predicts an action.
  Compilation and presentation are disabled; batch size is 32, training allowance is
  two minutes, and the subprocess timeout is 180 seconds. Missing CUDA requirements fail.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m learning
```

The default command runs only the two CPU checks. All outputs use temporary
directories. These smoke checks establish core workflow availability; they do not
establish mathematical correctness, backend parity, detailed recovery behavior or
annotation/demo/viewer/reporting correctness. Previous regressions remain in Git
history; there is no archived test suite in the working tree.
