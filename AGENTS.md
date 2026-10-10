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
  Both use `entity_v1`, `transformer_lstm_q_v4`, `public_event_history_v1` and
  `complete_return_joint_tile_probe_v1`; old weights/resumes
  require fresh initialization; verified recordings are reconstructed without changes. Keep user reward adjustments when updating defaults.
  `training.demo.passes` controls demonstration initialization (20 by default),
  independently of autonomous `training.n_epochs` (4 by default).
  Tile (50% to 1%) and accepted-plant species (10% to 1%) probabilities decay over
  10,000 stage games and freeze per cohort. `reward.early_sun_spawn_fraction=[1,3]`
  and `early_sun_extra_multiplier=2` triple early actual Sunflower income value.
  `training.max_grad_norm` is the sole clipping limit for demo and autonomous
  fitting; historical demo clipping metadata has no effect.
- `src/pvz_rl/envs/encoding.py` defines the shared 11-field entity schema,
  normalization, truncation diagnostics and `EntityBatch` collator. Canonical order
  is mowers, plants, zombies, projectiles; overflow drops from the end, with nearest
  zombies first. CUDA features encode the same public facts on device. Reserved
  simulator slots never become policy tokens. Globals count all present entities.
- `policy/active_batch.py` owns host `ActiveBatchPlan` construction from
  `enabled_envs` and published `summary_host` counts, public entity-width and
  power-of-two batch buckets, canonical gathers/scatters and the active-only tile
  wrapper. Original slot identities and inactive memory are preserved; selection
  RNG draws keep their original slot-shaped dimensions and order.
- `envs/` owns simulator adapters, geometry masks and rewards; `cuda_accounting.py`
  supplies read-only counters. The shared execution contract retains the proposal,
  reason and penalty, exposes `executed_action=0` for rejection and advances one tick.
  `rewards.py` also tracks one-time zombie entries into the two house-side columns;
  complete-game fitting finalizes victory-time shaping against the current cohort median.
  `envs/probes.py` owns isolated CPU/CUDA counterfactual execution and round-robin
  schedules. Shallow batch/feature prefix views reuse existing scratch allocation;
  source-ID remapping copies original live slots into compact lane prefixes and
  scatters public features/metadata back to canonical game/probe rows.
  `learning/objective.py` owns six-component targets, unified complete-action probe Huber loss
  and accepted-demonstration ranking. EMA history follows actual behavior, while
  forked probe histories never reach actual gameplay or journals.
- `policy/entity_attention.py` owns shared embeddings, readout tokens, padding masks,
  full efficient attention, exact query fallback and optional outer checkpointing.
  Fixed-shape CUDA execution uses direct no-grad collection graphs and an owned
  AOT fitting backend on all platforms, with an explicit eager fallback.
  `policy/cudagraph_backend.py` owns separate collection/fitting captures and
  per-encoder collection LRU caches: eight live entries and 512 MiB VRAM headroom
  by default. Replay/output-copy fences protect retired captures; the existing
  handoff releases completed retirees without a new wait. Retirement is separately
  bounded by the entry limit;
  capacity/headroom/capture failures retain exact eager execution. Phase boundaries
  release the outgoing captures before allocating the next phase's workspaces.
  It captures AOT forward/backward separately and copies their outputs, including
  saved activations and gradients,
  out of graph storage with SDPA strides intact. Mutable capture storage is allocated
  outside inference mode so evaluation/collection mode switches remain safe.
  A backward capture failure restarts
  the uncommitted pass eagerly. The compiled path stays tensor-only so simulator pybind
  objects are never traced. Fitting uses BF16 temporary features with FP32
  master weights, LSTM and Q heads. `transformer_lstm.py`
  owns event LSTM, one wait head and conditional joint-tile values; `recurrent_policy.py` adapts the training lifecycle.
  `sequential_q.py` owns shared `ActionQValues` and derives species/dig values
  from candidate-tile maxima; all ten branches remain greedy, independent of sun/cooldown;
  occupancy-only plant tiles and unrestricted dig tiles constrain tile exploration.
  `learning/exploration.py` owns the separate committed-plant state machine:
  accepted normal plants can select another species uniformly; actual waits persist
  until resources/cooldown/geometry permit planting or the episode ends, without a timeout.
  Exploration provenance and original policy proposals are diagnostics, never LSTM inputs.
  A full-board plant still proposes tile zero. `runner.py` owns recurrent lifecycle;
  `transformer_lstm.py` defines `EventMemoryState`: hidden/cell, pending gross public
  event facts and elapsed
  ticks. `envs/history.py` defines `HistoryEvent`, extraction and normalization.
  Sunlight gains/spending, zombie spawn/defeat/removal and plant addition/removal
  write memory once using the resulting board. Quiet decisions preserve memory
  exactly and read current board features through the shared Q fusion layer.
- `learning/checkpoints.py` inspects saved protocols/schema before simulation.
  `cli.py` resolves fresh `--init-from` runs from current configuration, transferring
  only weights; demonstration weights require only the pinned input/output model
  interface; autonomous weight transfer also checks timing/returns, while recovery
  requires the current committed exploration/trajectory protocols. `--resume`
  and evaluation retain saved settings. Weight-only reuse never loads a retired
  optimizer or collector.
  `cuda_q.py` owns cohort lifecycle and atomic recovery; `recurrent_q.py` collects
  chronological transitions and accumulates whole-pass gradients. Sparse fitting
  packs event rows, then deterministically gathers memory onto all decisions, with
  existing 256-decision detach boundaries. Selected complete-action losses (no duplicated branch-return regression) balance nonempty groups,
  then accepted/rejected strata 50/50 using full-cohort counts. `cuda_buffer.py`
  stores fixed metadata plus ragged entity slabs under one RAM/disk budget and
  validates offsets/counts and categories on recovery. `sequence_transport.py` owns
  ordered double-buffer prefetch; `performance.py` defines the execution-only
  refresh allowlist, including fused `fit_sequence_groups`. Recurrent fitting
  groups independent slot batches without sharing hidden state, records one
  device timing event per pass, and uses cached intermediate telemetry.
  Demonstration initialization checkpoints omit those fields
  and hydrate them from the current train profile when loaded. `host_transfer.py`
  owns the sole reusable pinned `HostHandoff`. Behavior evidence, auxiliary
  control and terminal headers/totals share each stream wait. Owned endpoint
  transfers precede complete-state CPU deduplication and compact nonterminal
  bootstrap inference; the queued copy is consumed at the next handoff.
  Actual rows append immediately, then `cuda_buffer.py` patches complete endpoint
  evidence exactly once before finalization/fitting; saving drains every patch.
  `PackedEntityBatch` carries validated slab offsets directly into storage.
  Behavior globals transfer active rows only and reconstruct zero-filled canonical
  host rows; probe globals follow the same compact/scatter contract. Fixed metadata
  may retain original slot-shaped staging.
  `learning/probe_layout.py` owns 1–44 tile-only probe shapes. Every accepted
  actual normal/exploratory planting queues up to four distinct other empty tiles
  of that species; branches are not probed. `collection_scheduler.py` retains
  canonical decision tickets, per-slot actual counters/reset markers and behavior
  draws under local source-capacity backpressure. `envs/probes.py` owns `ProbeJob`
  and `ProbePool`: immutable pre-action sources, FIFO work, persistent independent
  lanes, immediate retirement and exact-once delayed endpoint patches.
  Real slots, including a planting owner, continue alongside auxiliary games.
  Defaults allow 256 auxiliary lanes, 128 pending source jobs and one auxiliary
  transition per scheduler round. Allocation fallback halves capacities to one
  source/lane; it never drops a probe. Frozen greedy EMA stops at true
  terminal/cutoff, 30 simulated seconds or 4096 transitions; no recursive probes
  or exploratory commitments run in forks. Heavy reporting stays at 64 rounds.
  Complete-state endpoint dedup precedes tail inference; proposal/reward evidence
  is never merged. Sources, retained tickets, states, endpoints and handoff buffers
  participate in memory accounting. Source IDs/schedules never enter policy inputs.
  Cohort reward finalization and fitting require all expected patches. Saving
  stops new selection, finishes issued tickets and drains auxiliary work before
  serializing canonical recovery state; pools/maps/caches rebuild on restore.
  New demo/objective/trajectory/recovery protocols reject previous checkpoints
  before allocation. No migration or legacy collector is supported.
  Sequence fitting gathers actual decisions per original slot, ignores capacity
  gaps and keeps 256-actual-decision detach boundaries. Its chronological maps
  share staging budgets, spill to temporary disk maps and release on early close.
  Current execution/logging
  defaults refresh on compatible recovery.
  Fitting retries whole uncommitted passes on BF16/memory failure.
  Curriculum stages are easy, standard, shared.
- `presentation/demo_recording.py` records structured v2 archives and compact viewer
  history; `learning/demo_initialization.py` reconstructs replay facts, checks the
  archived ledger's facts/arithmetic, and recomputes rewards with current settings.
  Its verified in-memory transitions are the sole source of demonstration returns;
  historical reward prices never supervise fitting or rewrite the user's archive.
  `envs/rewards.py` declares which ledger fields are configuration-independent facts.
  `action_journal.py` retains actual Q values, proposal/execution and entity omissions;
  `live_layout.py` owns paging and F follow-latest. Presentation never feeds the model.
  `live_view.py` snapshots effective learning settings once from the environment's
  resolved run config; the read-only viewer strip never reloads TOML or device values.
- `evaluation/` uses the same runner and checks CUDA traces against CPU replay.
  `monitoring/entity_benchmark.py` measures cap-dependent inference/fitting cost;
  `throughput_benchmark.py` measures fixed 1,024-frame/four-pass execution.
  `collection_benchmark.py` owns cold/warmed sparse/mixed/crowded snapshots at
  128/64/32/16/4/1 live games within fixed 128 original simulator slots, with
  noncontiguous original IDs, bounded auxiliary-capacity controls and an opt-in bounded
  collect/four-pass-fit/collect cycle. Cutoff controls allow bounded extra ordinary
  decisions for zero-time placements/digs; snapshot windows stay unchanged.
  Its default performs no fitting;
  `throughput_benchmark.py` owns synthetic four-pass fitting, not snapshot collection.
  `monitoring/progress.py` owns compact terminal/train.log events, independent
  terminal/JSON cadences, duplicate suppression and redirected/interactive modes.
  Its host-only interval monitor distinguishes active games, decisions/game/s and
  active transitions/s from lifetime totals, excluding fitting/recovery downtime.
  `cuda_diagnostics.py` pools timing events and reads them after the existing handoff.
  `monitoring/objective_diagnostic.py` reports one 16-game/four-pass, 30-second-cutoff
  event-memory control and fixed-history sensitivity; measurements are not mastery evidence.
- `tests/test_observations.py`, `test_transformer_lstm.py`, `test_entity_storage.py`
  and `test_recurrent_training.py` cover information preservation, CPU/CUDA parity,
  independent attention math, order/padding invariance, ragged recovery and training.
- `test_q_math.py` owns selected-Q gradient and atomic interruption controls;
  `test_collection.py` owns serial/batched equivalence, allocation fallback,
  deduplication counterexamples and compiled inference buckets. The
  active-bucket/slot/RNG math and graph LRU/headroom/phase ownership controls belong
  in `test_transformer_lstm.py`; storage owns exact-once endpoint patching/spill.
  Interval-rate, shrinking-cohort and phase/reset controls belong in
  `test_progress.py`, with callback snapshot isolation in its context suite.
  Independent constant-bootstrap CPU/CUDA controls remain in
  `test_q_selection_fallback.py`.
  `test_progress.py` owns terminal cadence/modes; callback telemetry and active
  accumulator isolation belong in `test_training_progress_context.py`.
  `test_training_lifecycle.py` owns validation/finalization integration.
  `test_reward_contract.py` owns asset conservation and reward-mode boundaries;
  `test_exploration_schedule.py` checks both schedules and persistent species commitments.
  Reward controls include exact event-time early Sunflower eligibility and CPU/CUDA caps.
- `docs/architecture.md` traces ownership and failures; `docs/training.md` documents
  recording, fresh initialization, training and recovery. `docs/math/entity-inputs.md`
  and `recurrent-training.md` define encoding, attention and the complete-return objective.
  `training-objective.md` owns reward economics, `invalid-action-penalties.md` owns
  rejection costs, and `training-throughput.md` owns precision/transfer controls.
  `docs/validation.md` maps verification ownership; results/history belong only in
  `docs/iteration.md`, with raw measurements in `docs/evidence/`.
