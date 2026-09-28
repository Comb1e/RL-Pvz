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
- Keep the game dependency separate and pinned. Never expose seeds, future schedules,
  or snapshots to policy inputs.
- User's current scope: complete the code and training README; availability and short
  integration tests are allowed. Do not launch formal training unless requested.

## Code structure

- `src/pvz_rl/data/train.toml` is the single parameter source for train/demo overlays.
  Both use `entity_v1` observations and `transformer_lstm_q_v2`; old inputs/weights
  require fresh initialization. Keep user reward adjustments when updating defaults.
- `src/pvz_rl/envs/encoding.py` defines the shared 11-field entity schema,
  normalization, truncation diagnostics and `EntityBatch` collator. Canonical order
  is mowers, plants, zombies, projectiles; overflow drops from the end, with nearest
  zombies first. CUDA features encode the same public facts on device. Reserved
  simulator slots never become policy tokens. Globals count all present entities.
- `envs/` owns simulator adapters, geometry masks and rewards; `cuda_accounting.py`
  supplies read-only counters. The shared execution contract retains the proposal,
  reason and penalty, exposes `executed_action=0` for rejection and advances one tick.
- `policy/entity_attention.py` owns shared embeddings, readout tokens, padding masks,
  chunked attention and checkpointed encoder microbatches. `transformer_lstm.py`
  owns LSTM and Q heads; `recurrent_policy.py` adapts the training lifecycle.
  `sequential_q.py` keeps all ten branches greedy, independent of sun/cooldown;
  occupancy-only plant tiles and unrestricted dig tiles allow tile-only exploration.
  A full-board plant still proposes tile zero. `runner.py` owns recurrent state and
  feeds rejection back as `(previous_action=0, accepted=0, ticks=1)`.
- `learning/checkpoints.py` inspects saved protocols/schema before simulation.
  `cuda_q.py` owns cohort lifecycle and atomic recovery; `recurrent_q.py` collects
  chronological transitions and accumulates whole-pass gradients. `cuda_buffer.py`
  stores fixed metadata plus ragged entity slabs under one RAM/disk budget and
  validates offsets/counts on recovery. Curriculum stages are easy, standard, shared.
- `presentation/demo_recording.py` records structured v2 archives and compact viewer
  history; `learning/demo_initialization.py` verifies and fits only fresh archives.
  `action_journal.py` retains actual Q values, proposal/execution and entity omissions;
  `live_layout.py` owns paging and F follow-latest. Presentation never feeds the model.
- `evaluation/` uses the same runner and checks CUDA traces against CPU replay.
  `monitoring/entity_benchmark.py` measures cap-dependent inference/fitting cost.
- `tests/test_observations.py`, `test_transformer_lstm.py`, `test_entity_storage.py`
  and `test_recurrent_training.py` cover information preservation, CPU/CUDA parity,
  independent attention math, order/padding invariance, ragged recovery and training.
- `docs/architecture.md` traces ownership and failures; `docs/training.md` documents
  recording, fresh initialization, training and recovery. `docs/math/entity-inputs.md`
  and `recurrent-training.md` define encoding, attention and the complete-return objective.
