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
| Active buckets, canonical slot/RNG controls, owned graph cache/phase lifetime | `test_transformer_lstm.py` |
| Scratch isolation, valid-only probes, exact deduplication and bootstrap | `test_collection.py` |
| Ragged storage, offsets, disk spill, pending bootstrap patches, ordered prefetch | `test_entity_storage.py` |
| Proposals, execution, timing and event-only memory | `test_q_selection_fallback.py`, `test_environment.py`, `test_action_timing.py`, `test_recurrent_training.py` |
| Reward arithmetic and attribution | `test_net_value.py`, `test_reward_contract.py`, `test_kill_rewards.py` |
| Combined schedules, accepted triggers and persistent plant commitments | `test_exploration_schedule.py` |
| Complete returns, balanced loss, whole-pass updates, recovery | `test_q_math.py`, `test_recurrent_training.py` |
| Demonstration verification and initialization | `test_demo_initialization.py`, `test_collision_follow_recording.py` |
| Curriculum, budgets, evaluation scheduling and integration | `test_stage_training.py`, `test_game_budget.py`, `test_validation_schedule.py`, `test_training_lifecycle.py`, `test_training_integration.py` |
| Evaluation refill, pinned engine, replay, statistics and reports | `test_evaluation_refill.py`, `test_game_upgrade.py`, `test_baselines_statistics.py`, `test_reporting_config.py` |
| Viewer state, journal retention and output isolation | `test_live_view.py`, `test_action_journal.py`, `test_visualization.py` |
| Active interval rates, terminal cadence and callback snapshot isolation | `test_progress.py`, `test_training_progress_context.py` |

Configuration, exploration, runtime-cache and progress controls remain alongside
these suites. Shared fixtures suppress native training windows except in tests
that explicitly request the viewer. Different corruption fields and boundary
values are intentional cases, not duplicate functionality.

## Running checks

Collection ownership is `tests/test_collection.py`: serial/batched state and RNG
isolation, gross CPU/CUDA public-event controls, unchanged EMA history,
one-lane allocation fallback, exact device
deduplication counterexamples, valid nonterminal representative inference and
compiled inference buckets. Independent CPU/CUDA
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
and prohibition on reading active fitting accumulators. Early Sunflower CPU/CUDA
threshold, ordered production and cap controls belong in `test_reward_contract.py`;
demonstration repricing remains in its initialization suite. Committed recovery and
old-framework weight-only reuse belong in recurrent integration; storage owns
controller provenance corruption/spill. Terminal ownership includes current-cohort
reward means, finalization, reset and narrow-width retention.

### Active-only execution controls

Keep original successful paths, known failures, boundaries and independent controls
together; do not establish parity by weakening constraints:

- The model suite owns public-width and power-of-two planning for
  128/64/32/16/4/1 live games, interleaved canonical IDs and zero live rows. Compare
  outputs/event memory against independent serial calls, freeze inactive memory,
  and check original slot-shaped RNG draws. Include empty/capped entity lists,
  non-power-of-two batches and active-only tile-head behavior.
- The collection suite owns unchanged proposals/RNG/probe schedules, one-lane
  allocation fallback, compact scratch prefix views and valid-only dedup before
  bootstrap. Source-ID controls must copy the correct original RNG/accounting/
  proximity state and scatter public features/metadata to canonical rows for
  noncontiguous live slots, without reducing private capacity or probe coverage.
  Active-only behavior/probe global transfers reconstruct zero-filled canonical
  host rows without copying inactive globals just to preserve shape. Equal boards with
  different globals, gross events or elapsed timing remain distinct. Disabled,
  inactive and terminal probes must not cause next-state network work; terminal
  bootstrap remains zero. GPU representatives become host-visible at the existing
  handoff, not through a new ordinary count wait.
- The model suite owns per-encoder LRU hits/eviction, owned replay outputs, retired
  capture release at the existing handoff and separate fitting/collection owners.
  Cover zero entries, cache limits, headroom before/after admission, allocation
  failure and capture failure without stale-buffer reuse or changed eager math.
  Collection/fitting/collection boundaries must release the outgoing captures;
  no-grad direct graphs and fitting AOT graphs use the same all-platform contract.
- Storage owns canonical delayed bootstrap patching in RAM and spilled blocks,
  invalid/nonfinite updates and rejection after target finalization. Recurrent
  integration owns last-copy checkpoint/finalization drains, resumed state with
  rebuilt execution maps/caches and saved compilation-status failure handling,
  including unavailable status. Those controls must not change recovery protocols.
- Progress owns independent host-clock controls for shrinking games, interval
  versus lifetime counts, phase/cohort changes, restored counters and zero-duration
  `n/a`. Compare transitions/collection seconds and transitions/active-game seconds
  against hand-computed denominators. Callback context owns unchanged JSON/terminal
  cadence and prohibition on device reads or active fitting accumulator access.

Run from the repository root. Static inspection is suitable while a training
process is active; use a separate checkout for edits.

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
```

When test execution is authorized and the GPU is available, select owning suites:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_collection.py tests/test_transformer_lstm.py tests/test_entity_storage.py `
  tests/test_progress.py tests/test_training_progress_context.py tests/test_recurrent_training.py
```

Do not run a full suite for an execution-only change; expand to full testing only
when a shared core-package change requires it. These suites include short fitting,
CUDA simulation and recovery checks.
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

The collection benchmark owns cold/warmed sparse/mixed/crowded controls at
128/64/32/16/4/1 live games in fixed 128 original simulator slots, plus opt-in
bounded four-pass post-fit collection. Noncontiguous original IDs distinguish
active-only execution from simply allocating fewer environments.
Default snapshot checks do no fitting. Separate setup, cold capture, warmed trials,
post-fit trials, active/padded/unique work counts and phase memory; never infer
improvement from configured capacity or an unexecuted benchmark matrix. Publish
new verification and raw evidence only after the parent implementation's results.
