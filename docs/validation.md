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
| Pinned game staging and Bash installation/cleanup | `test_cuda_only.py` |
| Optional fit collector transparency, bounded traces and host resource deltas | `test_fit_diagnostics.py`, hardware cases in `test_reporting_config.py` |
| Reachable tile snapshots, annotation persistence, drag protocol and tile-only warm-start | `test_tile_annotations.py`, annotation fitting controls |

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


## Optimization and portability controls

- `test_entity_storage.py` and `test_flat_training.py` verify bucket boundaries, row ordering, partial batches, RAM/disk spill, CPU selection parity and recovery.
- `test_two_stage_transformer.py` verifies collection cache limits separately from fitting graph ownership and masked partial-batch padding.
- `test_gpu_configuration.py` verifies profile overlays, resume learning retention and device-independent execution settings.
- `test_live_view.py` remains the shared mailbox/reconnect/history owner; `web_viewer.py` is CPU-only and uses the same command protocol.
- `test_tile_annotations.py` covers complete rich observations, plant-availability filtering, legal tile bounds, grouped annotation storage, durable pending/labeled/discarded records, automatic one-map progression, relative browser APIs, and the Canvas drag/discard controls. History controls cover navigation boundaries, persistent in-place edits, discard restoration, unchanged pending maps and sampler counts, invalid/stale revisions, independent HTTP clients, and browser draft/stale-response behavior through Node.js when available. Sampling controls verify inclusive random horizons, 500–1,500-step planting intervals, sparse caps, equal wait/eight-species proposals without tile-count bias, unavailable choices waiting without redraw, terminal/unusable retries, signed zombie coordinates, hash-order independence and independent simulator replay. The annotation fit additionally checks order-invariant set loss, frozen species parameters and weight-only checkpoint transfer.
- `tools/tune_gpu.py` records three repetitions, cold setup, transfer/preparation/graph metrics and memory headroom across the original and larger grouped microbatch/thread cases. Learning recommendations require paired seeds and completed cohorts before promotion.


## Annotation convergence and selective attribution

- `test_tile_annotations.py`: independent hinge values/gradients, duplicate/order
  invariance, partial-batch record weighting, legal masks, deterministic ties,
  all-positive states, empty/conflicting inputs, unlimited default, incomplete
  caps, exact weights/Adam recovery after interruption, changed dataset/settings
  rejection, species freezing, zero-pass verification and artifact reload.
- `test_flat_training.py`, `test_two_stage_q.py`, `test_q_math.py`: every routed
  head and exact forward counts, detached accepted consistency, branch precedence,
  legal boundaries, fixed-target isolation, global balancing, storage/prefetch,
  graph/allocation recovery and interrupted pass restart.
- `test_reward_contract.py`, existing environment/net-value/rejection controls:
  rejected-vs-wait CPU/CUDA resulting state, one tick, terminal/development rewards,
  zero audit charges, retained proposals/reasons. Demo suite verifies historical
  facts and current repricing without rewriting recordings.
- Checkpoint/config suites verify changed-objective resume rejection while
  compatible BF16-master weights remain loadable for inference/fresh starts.
  `test_tune_gpu.py` verifies coherent routed cases at all existing widths;
  the shared benchmark supplies local CUDA execution evidence separately from
  annotation train/held-out accuracy or game performance.
