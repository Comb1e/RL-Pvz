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
- Verify successful, failing, boundary, and independent mathematical controls together.
- Reduce duplicate tests: keep each behavior in its owning suite and preserve distinct
  mathematical, boundary, backend and recovery controls.
- Keep the game dependency separate and pinned. Never expose seeds, future schedules,
  or snapshots to policy inputs.
- User's current scope: complete the code and training README; availability and short
  integration tests are allowed. Do not launch formal training unless requested.
- In this project, always keep only one training and model framework.

## Code structure

- `src/pvz_rl/data/train.toml` is the single parameter source for train/demo overlays.
  Both use `entity_v1`, `two_stage_transformer_q_v1` and
  `complete_return_sequential_tile_probe_v1`. Old weights/resumes require fresh
  initialization; verified recordings are reconstructed without changes. Preserve
  user reward adjustments. `training.demo.passes` is 20, independently of autonomous
  `training.n_epochs` (4). Tile (50% to 1%) and accepted-plant species (10% to 1%)
  probabilities decay over 10,000 stage games and freeze per cohort.
  `reward.early_sun_spawn_fraction=[1,3]` and `early_sun_extra_multiplier=2` triple
  early actual Sunflower income value. Shared clipping uses `training.max_grad_norm=5`.
  `training.objective.species_consistency_weight=1.0` controls detached EMA tile-max
  targets. `training.performance.fit_batch_groups=4` groups flat batches for execution.
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
- `policy/runner.py` is the shared stateless runner for collection/evaluation/probes.
  `policy/sequential_q.py` owns candidate geometry, greedy ten-way selection and
  balanced selected-action loss. Empty plant tiles/all dig tiles are candidates;
  full-board plants use tile-zero rejection fallback. Sun/cooldown do not mask
  ordinary branches. `active_batch.py` packs active canonical slots into public-width
  and power-of-two buckets with stage-specific token overheads 1/47. Padding has
  no actions, RNG or supervision. Behavior draws retain original-slot shape/order.
- `policy/cudagraph_backend.py` owns separate collection/fitting captures and owned
  outputs/activations/gradients. Every online/EMA encoder has a bounded collection
  LRU: 8 entries and 512 MiB VRAM headroom by default. Eviction fences copies and
  releases retirees through existing handoffs; pressure/failure uses exact eager
  execution. Quiescent phase changes release outgoing captures before new allocation.
- `envs/` owns simulator adapters, action outcomes and reward ledgers;
  `cuda_accounting.py` supplies read-only reward counters. Rejection retains its
  proposal/reason/penalty, executes wait and advances one tick; accepted plant/dig
  advances zero ticks. Accounting is not a model input. `rewards.py` owns early
  Sunflower eligibility, physical conservation, proximity penalties and cohort-median
  victory finalization. Recording fact checks remain configuration-independent.
- `learning/exploration.py` owns persistent accepted-plant species commitments.
  Choose another species uniformly; wait ordinarily for sun/cooldown/geometry, with
  no timeout. Success clears commitment without recursively drawing a new one.
  Provenance and original policy proposals remain diagnostics. Exploration is
  disabled for evaluation, demonstrations and auxiliary games.
- `learning/probe_layout.py` owns 1–44 tile-only probe layouts. Every accepted
  normal/exploratory planting queues up to four distinct other empty tiles of its
  species from the pre-action board. `collection_scheduler.py` retains tickets,
  original-slot draws and per-slot counters under local FIFO source backpressure.
  `envs/probes.py` owns immutable pre-action simulator/accounting sources, `ProbeJob`
  and `ProbePool`. Real games continue alongside independent occupied lanes; forks
  follow frozen greedy EMA without recursive probes. Defaults are 256 lanes,
  128 pending jobs, 1 auxiliary transition/round, 30 seconds, 4096 transitions and
  64-round control reports. Headroom fallback reaches one source/lane without dropping
  probes. Exact public endpoint/EMA-version dedup precedes species-tail bootstrap;
  requested branch participates when conditional tile requests are deduplicated.
- `learning/sequential_training.py` owns `CudaSequentialQ`: canonical collection,
  background drains, complete returns and four whole-cohort fitting passes.
  `learning/flat_q.py` owns selected regression, detached frozen-EMA candidate-max
  consistency, probe Huber and demo ranking. Selected counts are 3x2 wait/plant/dig
  by acceptance; consistency counts are 2x2 plant/dig; probe counts are 10x2 branch
  by initial acceptance. Denominators cover the whole cohort. No non-wait sampled
  return directly trains species. One clipped Adam update commits per pass, then
  both EMA networks update with decay 0.95. No detach boundaries or recurrent fitting.
- `learning/cuda_buffer.py` stores canonical slot-shaped metadata and ragged entity
  slabs under one RAM/disk budget. Active flat batches exclude capacity gaps/padding;
  complete returns remain chronological per episode. `batch_transport.py` owns
  ordered double-buffer `BatchPrefetch`. Pending endpoint patches are exact-once in
  RAM/spilled storage. `host_transfer.py` owns reusable pinned `HostHandoff`; views
  are consumed before reuse and endpoints own their transferred evidence.
- `learning/checkpoints.py` validates model/objective/schema and advanced demo,
  trajectory and recovery contracts before simulator allocation. No migration or
  partial transfer exists. Fresh current-contract `--init-from` transfers full weights
  only; resume retains saved learning settings. `cuda_q.py` owns cohort lifecycle and
  atomic recovery. Saving stops selection, finishes issued tickets and drains every
  probe/bootstrap/patch before serializing simulator, accounting, controller,
  scheduler, RNG, trajectory, optimizer and both networks. Pools/maps/caches rebuild.
  Fitting/finalization/EMA changes require zero pending evidence. Uncommitted passes
  restart; committed steps are not repeated. `performance.py` owns execution refresh.
- `presentation/demo_recording.py` records reusable v2 archives/native replays.
  `learning/demo_initialization.py` verifies replay facts/outcomes/timing and reprices
  rewards without rewriting recordings, then fits 20 passes using both online/EMA
  stages. Journals/viewers are read-only actual Q/proposal/execution diagnostics,
  not learned history. `live_layout.py` owns paging/follow-latest; `live_view.py`
  snapshots effective settings without reloading TOML/device values.
- `evaluation/` uses the stateless runner and CPU-verifies CUDA action traces.
  `monitoring/` owns availability, both-stage entity/throughput benchmarks, bounded
  collection/four-pass-fit/collection controls and objective diagnostics. Progress
  uses host records for cohort wins/returns and interval active rates, distinct from
  lifetime totals. CUDA timings cross existing handoffs; telemetry adds no hot-loop
  device synchronization. Diagnostics report selected error, species consistency
  and conditional best-tile minus sampled-tile placement gap, never memory metrics.
- Owning suites: `test_two_stage_transformer.py` covers attention/dimensions,
  statelessness, active packing, precision and graph ownership; `test_two_stage_q.py`
  and `test_q_math.py` cover attribution, detached consistency, probe/ranking weights
  and partition invariance; `test_flat_training.py` covers fitting/recovery.
  `test_probe_rollouts.py` owns isolation/backpressure/draining; `test_entity_storage.py`
  owns flat active rows, spill/prefetch and exact-once patches. Observation, reward,
  exploration, demonstration, evaluation and presentation suites retain their owners.
- `docs/architecture.md` traces current ownership/workflows/failures.
  `docs/training.md` owns commands and recovery; `docs/math/entity-inputs.md` owns
  encoding/attention, `sequential-training.md` learning/attribution,
  `training-objective.md` reward economics, `invalid-action-penalties.md` rejection,
  and `training-throughput.md` execution. `docs/validation.md` maps verification;
  results/history belong in `docs/iteration.md`, raw measurements in `docs/evidence/`.
