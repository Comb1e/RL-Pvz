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

- `configs/demo.toml` and bundled `data/demo.toml` define the recurrent human
  workflow; `configs/train.toml` and bundled `data/research.toml` retain the
  separate CUDA collector. CLI commands select their matching profile.
- `src/pvz_rl/envs/` adapts the pinned game, encodes the versioned public
  observation (including card cooldowns), and computes transition/accounting
  rewards.
- `src/pvz_rl/policy/transformer_lstm.py` contains the 45-tile/15-region entity
  Transformer, recurrent state interface, and branch/tile Q heads for demo fitting.
  `policy/event_memory.py`, `temporal.py`, and `spatial_policy.py` implement the
  active CUDA collector. `sequential_q.py` shares action algebra and selection.
  Retired checkpoint inference and its alternative masks have been removed.
- `src/pvz_rl/presentation/demo_recording.py` owns the interactive easy-game
  recorder, append-only transition archive, compact history index, native
  action-phase replay, output-path validation, and one-session human controls.
- `src/pvz_rl/presentation/live_layout.py` owns the multi-panel viewer layout,
  history paging, generation-safe responses, and F follow-latest state changes;
  horizontal table scrolling is independent from selection/follow state.
- `src/pvz_rl/learning/demo_initialization.py` verifies a complete archive and
  replay, computes complete returns, performs chunked initialization, and writes
  the versioned checkpoint and diagnostics. It does not start formal 128-game
  training.
- `src/pvz_rl/learning/` contains the later CUDA complete-game collector and
  optimizer; `src/pvz_rl/evaluation/` runs held-out games; `tests/` mirrors these
  boundaries with CPU, CUDA, integration, and independent math controls.
- `docs/training.md` documents recording, initialization, explicit CUDA training,
  resume and evaluation; `docs/live-view.md` describes viewer controls.
