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

- `configs/train.toml`, `configs/demo.toml` and bundled profiles use the recurrent
  Transformer–LSTM by default; `configs/event-memory.toml` is the explicit alternative.
- `src/pvz_rl/envs/` adapts the pinned game, public observations and rewards;
  `cuda_accounting.py` adds read-only accounting counters to the pinned CUDA kernel.
- `src/pvz_rl/policy/transformer_lstm.py` owns the entity Transformer and recurrent
  Q heads. `recurrent_policy.py` adapts the same weights to the training lifecycle;
  `runner.py` owns public action/outcome history for collection, evaluation and replay.
  `sequential_q.py` shares selection and group-weighted loss across both model families.
  `event_memory.py`, `temporal.py` and `spatial_policy.py` retain the explicit alternative.
- `src/pvz_rl/learning/checkpoints.py` inspects protocols, saved configurations and
  model dispatch. `demo_initialization.py` verifies and fits a completed archive.
  `cuda_q.py` owns cohort lifecycle/atomic recovery; `recurrent_q.py` adds chronological
  recurrent collection and whole-pass gradient accumulation. `cuda_buffer.py` bounds
  host trajectory memory with disk overflow. Curriculum stages are easy, standard, shared.
- `src/pvz_rl/presentation/demo_recording.py` owns the one-attempt human recorder,
  append-only transition archive, compact history, replay and output validation.
  `live_layout.py` owns viewer layout, paging, generation checks and F follow-latest.
  `action_journal.py` retains Q values captured during actual collection decisions.
- `src/pvz_rl/evaluation/` runs deterministic held-out games through the shared runner
  and verifies recorded CUDA traces against CPU replay. `tests/test_recurrent_training.py`
  covers recurrent handoff, fitting, state boundaries, recovery and mathematical controls;
  legacy model tests select the explicit event-memory configuration.
- `docs/training.md` documents recording, initialization, CUDA training, resume and
  evaluation; `docs/architecture.md` describes current ownership and failure paths;
  `docs/math/recurrent-training.md` defines the sequence objective and independent controls.
