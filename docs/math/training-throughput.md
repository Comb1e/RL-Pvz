# Precision and execution controls

The collector and fitter share the stateless two-stage policy. Active original
slots are packed by public entity width and results scatter back to canonical
slots. Finished slots are inactive, never refilled and never inferred. Padding
has no action, RNG, loss or storage evidence. Species inference uses one extra
token; conditional tile inference uses 47.

## Runtime precision

Every collection, evaluation, demonstration and fitting forward uses BF16
parameters and activations. The online model keeps separate FP32 master tensors;
Adam and its first/second moments reference those masters only. Runtime BF16
gradients accumulate into FP32 master gradients, clip once per pass, then Adam
updates the masters. Updated masters are copied to the BF16 runtime model and
the BF16 EMA teacher updates with decay 0.95.

`training.performance.fit_precision = "bf16_master"` is the only supported
precision contract. BF16 availability is checked before simulator allocation.
Nonfinite loss or gradients fail explicitly. Memory pressure may reduce fitting
microbatches or enable activation checkpointing, but never falls back to FP32.
INT8 and INT4 execution are not implemented.

## Flat fitting

Complete returns remain chronological per original episode. Genuine active rows
are materialized in a reusable fitting spool, optionally spilled through the
trajectory RAM/disk budget, and bucketed by capped public entity width. Fixed
padded batches are prefetched through the ordered double buffer. Padding and
reordering do not affect whole-cohort denominators or gradients. The spool is
discarded after fitting and is never simulator recovery state.

Base batches contain 1,024 rows and `fit_batch_groups=4` groups transport work;
there is one optimizer commit per autonomous pass (four) or demonstration pass
(twenty). Collection geometry remains `encoder_microbatch=128`; fitting defaults
use `fit_encoder_microbatch=192` and `fit_token_budget=65536`.

## Collection

`ActiveBatchPlan` uses published live flags and public entity counts only. It
never inspects future schedules or private simulator state. Behavior draws keep
original-slot order. The actual environment executes immediately after every
proposal, including planting. There are no alternative-action copies, rollout
lanes, probe endpoints, rollout caps or post-plant drain.

Graph caches are bounded to eight collection shapes per encoder with 512 MiB
headroom. Collection and fitting captures have separate ownership; quiescent
phase changes release outgoing captures. Capture failure uses eager BF16
execution. Recovery rebuilds captures, prefetch workers, maps and fitting spools.

## Rates and evidence

Report active decisions/game/s and active transitions/s from host-side interval
records, separately from lifetime totals. Completed-game rewards and wins use
current-cohort records; no device synchronization is added for terminal output.
Diagnostics include both-stage inference/fit time, fitting rows/s, memory,
optimizer completion, selected-action error, species-consistency diagnostics and
conditional best-tile minus sampled-tile placement gap.

Independent controls cover 128/64/32/16/4/1 live slots, sparse/mixed/crowded
boards, zero/one planting and recovery. Bounded collect → four-pass-fit → collect
checks establish implementation correctness only, not a speed or win-rate
guarantee.
