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

Fitting compiler caches are private to each encoder and fitting phase: separate
Python code objects prevent species/tile stages or different model instances from
sharing Dynamo's recompilation allowance. Up to four observed fitting shapes are
compiled per phase; additional shapes use eager BF16. Partial compiled microbatches
are padded and trimmed before supervision. Quiescent fitting release renews that
encoder's compiler cache as well as releasing captures, so changed microbatch
limits do not accumulate stale specializations across cohorts. Global compiler
limits are not increased and unrelated compiler caches are not reset.

After updating compiler code, restart the training process from its coherent
current-contract recovery checkpoint. Existing weights and recordings remain
compatible; fresh demonstration initialization is not needed for this execution fix.

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

## Diagnose cloud fitting stalls

Wrap the ordinary autonomous training command with the collector. Keep the same
training arguments, checkpoint, configuration and output directory; choose an
unused diagnostic directory:

```bash
.venv/bin/python tools/collect_fit_diagnostics.py --output artifacts/fit-profile -- \
  train --output runs/easy --resume runs/easy/latest.zip --no-live-view
```

The lightweight run records all fitting chunks. Let at least one whole fitting
pass finish so both startup and repeated batch work are visible. Training keeps
its normal budget and interruption/recovery behavior. The collector neither
ends training after collecting data nor changes execution or learning settings.
On Windows, use `.\.venv\Scripts\python.exe` with the same script arguments.

For a second diagnostic run, add `--trace-batches 6` before `--`. The profiler
skips 20 fitting data batches, warms up for one batch, then traces six batches
once. Adjust `--trace-skip-batches` if the interesting pause occurs elsewhere.
Pass-boundary work between those batches may also appear. Early interruption
can yield a partial trace, reported in the summary. Profiling startup/export
can synchronize CUDA and alter timing; compare hardware utilization against
the lightweight run. The lightweight wrappers add host timing/logging overhead
but no CUDA waits, device tensor readbacks or new CUDA events. Shapes, stacks and
allocation tracing are disabled to bound profiler overhead. CUDA trace support
depends on the cloud build/driver; unavailable CUDA profiling is reported and
the CPU trace remains useful. CUDA Graph kernel visibility is build dependent.

The collector reuses the hardware sampler and temporarily wraps the existing
learner; it does not introduce another training framework. Host labels distinguish
spool construction, prefetch startup/next-batch waits, encoder forwards,
compilation backend work, new graph setup, backward, master-gradient accumulation,
pass metrics and weight/EMA updates. Each chunk records actual rows, non-wait
fraction, exact transport width, compilation paths/shapes, graph cache counters,
effective settings/fallbacks and PyTorch allocated/reserved memory. Worker
preparation events include slot-reuse waits; their separately recorded preparation
time excludes those waits. CPU thread rates, process I/O/fault counters, available
RAM/swap, host pressure/CPU steal counters and cgroup v2 throttling deltas help
detect CPU/container limits and disk-backed storage. Unsupported platform facts
remain absent/null. GPU snapshots record device UUID, driver, limits, clock
reasons and competing process allocations; these snapshots are startup facts,
not continuous process-attributed GPU utilization.

Send `summary.json`, `metadata.json`, `hardware-metrics.jsonl` and
`fit-events.jsonl`, plus `fit-trace.json.gz` / `operators-*.txt` when tracing.
The diagnostic directory also contains GPU startup snapshots. Package it with:

```bash
tar -czf artifacts/fit-profile.tar.gz -C artifacts fit-profile
```

Interpret measurements using these controls:

| Observation | Candidate bottleneck / next evidence |
|---|---|
| Long `next_batch` and busy `pvz-batches` thread | Row-by-row validation/padding in CPU preparation; compare worker preparation events |
| Long `spool_build`, major faults or increasing read bytes | First-pass materialization or RAM-budget spill; inspect RAM and process I/O together |
| Long compile/backend or graph-setup spans at new widths | Lazy compilation/capture; compare subsequent chunks of the same shape |
| Frequent `extra_shape_eager` microbatches | Exact transport maxima exceed the four fitting shapes admitted per encoder; inspect kernel-launch gaps |
| CUDA runtime waits around `aten::nonzero` / indexing | Device-dependent output sizes synchronize the host; locate waits in the trace |
| Many short kernels and gaps with small effective microbatches | Host launch/control overhead; inspect forward/backward/master-gradient spans |
| Increasing cgroup `throttled_usec`, CPU pressure or steal time | Cloud CPU quota or contention; total CPU percentage can hide a saturated thread |
| High GPU utilization with long kernels/copies | GPU compute, memory or transfer work; distinguish kernels from memory copies in the trace |

These are hypotheses, not automatic tuning recommendations. In particular,
4,096 transported rows still execute in microbatches capped at 192 by default.
The spool is sorted by entity count, while each transport batch uses its exact
maximum entity count; sorting does not by itself guarantee four fixed widths.
Teacher tile inference takes the no-grad collection-graph path even during fit.
Increasing transport groups alone does not enlarge encoder microbatches.

Host spans are inclusive and nested; producer work overlaps the main thread.
Do not sum those spans or treat them as GPU kernel durations. The existing
`device_compute_seconds` brackets a whole fitting pass with CUDA events, so
idle gaps between enqueued operations contribute too. It cannot establish
GPU busy time. NVIDIA utilization measures whether any kernel ran during a
device sample window, not SM occupancy, achieved FLOPs or training throughput.
A one-second poll can miss shorter bursts and is device-wide.


## Selective fitting workloads

The fitting spool groups by entity bucket and persisted route. CPU selections
are transferred with the existing pinned batches: species-invalid performs no
tile/EMA forward, tile-invalid performs no species/EMA forward, and accepted
non-wait reuses online tile tables for detached EMA consistency. No CUDA boolean
compaction or deduplication is needed for this validated path. Rejection-heavy
cohorts can therefore have much less EMA work without altering the simulator or
ordinary controller selection.

The existing `tools/tune_gpu.py` calls the shared complete-fit benchmark. Every
sparse/medium/dense/bucket-boundary workload now includes wait-only, mixed,
accepted tile-heavy, species-invalid-only, tile-invalid-only and rejection-heavy
cases with coherent public availability and outcomes. JSON reports routed rows,
online species/tile rows, EMA tile rows and skipped EMA rows alongside transport,
cold four-pass setup, warmed repetitions, graph caches and peak memory. These
are execution measurements, not game-quality evidence. Compare warmed complete
work and cold setup separately; GPU utilization alone cannot establish speedup.
