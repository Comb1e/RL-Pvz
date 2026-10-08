# Validation scope

Tests protect the current entity Transformer–LSTM, public simulator adapters,
complete-return learning and saved-run interfaces. A passing implementation
check is not evidence of improved win rate or curriculum mastery.

## Coverage ownership

Keep one owner for each behavior. CPU/CUDA parity, independent mathematical
controls and integration at a different interface are distinct checks. Keep
legacy cases only when they protect a supported reader or explicit rejection
of incompatible input; do not exercise retired models as supported methods.

| Behavior | Maintained test files |
|---|---|
| Public records, categories, caps and private-state exclusion | `test_observations.py`, `test_cuda_learning.py` |
| Attention math, permutation/padding, recurrence, precision | `test_transformer_lstm.py` |
| Ragged storage, offsets, disk spill, ordered prefetch | `test_entity_storage.py` |
| Proposals, execution, timing and event-only memory | `test_q_selection_fallback.py`, `test_environment.py`, `test_action_timing.py`, `test_recurrent_training.py` |
| Reward arithmetic and attribution | `test_net_value.py`, `test_reward_contract.py`, `test_kill_rewards.py` |
| Complete returns, balanced loss, whole-pass updates, recovery | `test_q_math.py`, `test_recurrent_training.py` |
| Demonstration verification and initialization | `test_demo_initialization.py`, `test_collision_follow_recording.py` |
| Curriculum, budgets, evaluation scheduling and integration | `test_stage_training.py`, `test_game_budget.py`, `test_validation_schedule.py`, `test_training_lifecycle.py`, `test_training_integration.py` |
| Evaluation refill, pinned engine, replay, statistics and reports | `test_evaluation_refill.py`, `test_game_upgrade.py`, `test_baselines_statistics.py`, `test_reporting_config.py` |
| Viewer state, journal retention and output isolation | `test_live_view.py`, `test_action_journal.py`, `test_visualization.py` |

Configuration, exploration, runtime-cache and progress controls remain alongside
these suites. Shared fixtures suppress native training windows except in tests
that explicitly request the viewer. Different corruption fields and boundary
values are intentional cases, not duplicate functionality.

## Running checks

Collection ownership is `tests/test_collection.py`: serial/batched state and RNG
isolation, gross CPU/CUDA public-event controls, unchanged EMA history,
one-lane allocation fallback, exact device
deduplication counterexamples and compiled inference buckets. Independent CPU/CUDA
danger-zone/bootstrap controls live in `test_q_selection_fallback.py`; packed
offset/spill/recovery controls live in `test_entity_storage.py`. Recurrent lifecycle
tests cover interrupted collection and reconstruction without repeating probe math.
Event-only identity/pending recovery, controlled current-state sensitivity and
independent serial sparse gradients belong in `test_transformer_lstm.py`.
Hierarchical outcome weights and missing strata belong in `test_q_math.py`;
corrupt event/write-mask/stratum metadata belongs in `test_entity_storage.py`.
Native event reconstruction belongs in `test_demo_initialization.py`.
`test_progress.py` covers terminal modes, cadence, duplicate suppression and failure
events; `test_training_progress_context.py` covers detailed snapshot preservation
and prohibition on reading active fitting accumulators.

Run from the repository root. Static inspection is suitable while a training
process is active; use a separate checkout for edits.

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
```

When test execution is authorized and the GPU is available:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The full suite includes short fitting, CUDA simulation, replay and viewer checks.
Some CUDA tests do not carry the `learning` marker: excluding that marker does
not make a run CPU-only. Do not run it alongside a training job whose GPU or
process state must remain untouched. A test run is never a formal training run.

## Evidence

[Iteration history](iteration.md) is the sole chronological record of releases
and their verification. The committed [entity measurements](evidence/entity-inputs-v029.json)
and [throughput measurements](evidence/training-throughput-v030.json) retain raw
trials and workload limits. Their results apply to the recorded implementation,
not to an unexecuted cleanup. Earlier raw files remain in `docs/evidence/` and
`docs/figures/`; historical validation narratives are recoverable with
`git show 4052968:docs/validation.md`. Ignored local artifacts are not required by
a fresh clone.
