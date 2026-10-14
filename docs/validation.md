# Validation ownership

Focused suites are run once per change; full-suite repetition is avoided.

| Area | Owner |
|---|---|
| Transformer dimensions, statelessness, BF16 runtime and master lifecycle | `test_two_stage_transformer.py` |
| Action attribution, detached consistency, Huber and weighting math | `test_two_stage_q.py`, `test_q_math.py` |
| Flat fitting, spool/padding parity and recovery | `test_flat_training.py`, `test_entity_storage.py` |
| Observation schema, reward accounting and exploration | Their existing owning suites |
| Demonstration reconstruction and checkpoint contracts | `test_demo_initialization.py`, checkpoint suites |
| Active-only collection and throughput telemetry | Collection benchmark controls |

## Required controls

- The species and tile stages independently reproduce serial values, gradients,
  padding invariance and current-board responsiveness.
- Compiler controls cover both stages, all four public width buckets, partial
  microbatches and changed fitting microbatch limits across phase releases. CUDA
  values/gradients match eager BF16, capture cleanup is owned, excess shapes stay
  eager and compiler failure remains an explicit fallback without global resets.
- Runtime parameters/activations are BF16; FP32 masters and Adam moments receive
  updates; checkpoints restore BF16 runtime state from FP32 masters.
- Poor sampled-tile return does not directly update species values. Detached
  online tile selection and EMA evaluation route consistency gradients only to
  the species network.
- Complete-return losses preserve 3x2 selected and 2x2 consistency group/outcome
  denominators under missing strata, width reordering, padding, RAM spill and
  batch partitioning.
- The collector records planting immediately. No simulator copy, auxiliary lane,
  30-second horizon, endpoint patch or probe loss occurs.
- Collection controls compare live counts 128, 64, 32, 16, 4 and 1, including
  noncontiguous completion and simultaneous terminal slots. Bounded
  collect → four-pass-fit → collect controls report BF16 timing, memory,
  optimizer completion, selected error and consistency diagnostics.

These controls establish contract and numerical correctness. They do not promise
training speed, win-rate improvement or mastery.
