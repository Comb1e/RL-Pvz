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
  `complete_return_sequential_no_probe_v1`. Old weights/resumes require fresh
  initialization; verified recordings are reconstructed without changes. Preserve
  user reward adjustments. `training.demo.passes` is 20, independently of autonomous
  `training.n_epochs` (4). Tile (50% to 1%) and accepted-plant species (10% to 1%)
  probabilities decay over 10,000 stage games and freeze per cohort.
  `reward.early_sun_spawn_fraction=[1,3]` and `early_sun_extra_multiplier=2` triple
  early actual Sunflower income value. Shared clipping uses `training.max_grad_norm=5`.
  `training.objective.species_consistency_weight=1.0` controls online-selected,
  detached EMA-evaluated tile targets. `species_consistency_huber_delta=1.0` under
  the same objective table is finite, positive and non-boolean; consistency uses
  twice unreduced Huber to preserve small-error MSE scale. No gates or sample
  exclusion apply. `training.performance.fit_batch_groups=4` groups flat batches.
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
  execution. Fitting compilation owns a separate code object per encoder/phase,
  admits up to four shapes and pads partial microbatches before trimming readouts.
  Quiescent phase changes release outgoing captures and renew only the owner's
  fitting compiler cache; never raise global Dynamo limits or reset unrelated caches.
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
- `learning/collection_scheduler.py` retains actual decision tickets, original-slot
  draws and per-slot counters. Accepted planting returns immediately to normal
  collection; there is no counterfactual source, probe lane or rollout state.
- `learning/sequential_training.py` owns `CudaSequentialQ`: canonical collection,
  background drains, complete returns and four whole-cohort fitting passes.
  `learning/flat_q.py` owns shared autonomous/demo fitting: selected regression,
  double-estimator species consistency and demo ranking.
  `learning/objective.py` owns the shared detached online argmax/EMA evaluation
  target, selected tile IDs, estimator gap and MSE-matched Huber. Reuse existing
  online tile tables and EMA forward; add no Transformer forwards. Geometry-only
  candidates, tile-zero full-board fallback and lowest-index ties remain. Targets
  follow actual proposals, including rejection/commitment, never diagnostic
  proposals. Selected counts are 3x2 wait/plant/dig by acceptance; consistency
  counts are 2x2 plant/dig. Denominators cover the whole cohort. No non-wait sampled
  return directly trains species. One clipped Adam update commits per pass, then
  both EMA networks update with decay 0.95. No detach boundaries or recurrent fitting.
- `learning/cuda_buffer.py` stores canonical slot-shaped metadata and ragged entity
  slabs under one RAM/disk budget. Active flat batches exclude capacity gaps/padding;
  complete returns remain chronological per episode. `batch_transport.py` owns
  ordered double-buffer `BatchPrefetch`. `host_transfer.py` owns reusable pinned
  `HostHandoff`; views are consumed before reuse.
- `learning/checkpoints.py` validates model/objective/schema and demo v9,
  no-probe recovery v1 and trajectory v1 contracts before simulator allocation.
  Model-state `stateless-two-stage-bf16-master-state-v2` stores FP32 online
  masters, FP32 Adam state and BF16 EMA runtime weights. No migration or
  partial transfer exists. Fresh current-contract `--init-from` transfers full weights
  only; resume retains saved learning settings. `cuda_q.py` owns cohort lifecycle and
  atomic recovery. Saving stops selection and finishes issued tickets before
  serializing simulator, accounting, controller,
  scheduler, RNG, trajectory, optimizer and both networks. Pools/maps/caches rebuild.
  Fitting/finalization/EMA changes require complete actual rows. Uncommitted passes
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
  device synchronization. Pass-boundary aggregates report `species_consistency_loss`
  (balanced scaled Huber), `species_consistency_mae`,
  `species_consistency_saturation_fraction` and `species_selection_evaluation_gap`;
  absent non-wait rows are nullable/`n/a`. Gap is EMA candidate max minus EMA at the
  online-selected tile: disagreement, not proven overestimation. Selected error and
  conditional best-tile minus sampled-tile placement gap stay separate.
- Owning suites: `test_two_stage_transformer.py` covers attention/dimensions,
  statelessness, active packing, precision and graph ownership; `test_two_stage_q.py`
  and `test_q_math.py` cover attribution, detached double-estimator targets,
  piecewise Huber values/gradients and ranking weights
  and partition invariance; `test_flat_training.py` covers fitting/recovery.
  `test_entity_storage.py` owns flat active rows and spill/prefetch. Observation, reward,
  exploration, demonstration, evaluation and presentation suites retain their owners.
- `docs/architecture.md` traces current ownership/workflows/failures.
  `docs/training.md` owns commands and recovery; `docs/math/entity-inputs.md` owns
  encoding/attention, `sequential-training.md` learning/attribution,
  `training-objective.md` reward economics, `invalid-action-penalties.md` rejection,
  and `training-throughput.md` execution. `docs/validation.md` maps verification;
  results/history belong in `docs/iteration.md`, raw measurements in `docs/evidence/`.
