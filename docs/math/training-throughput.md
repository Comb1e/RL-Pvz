# Precision and execution controls

For the RTX 4070 investigation using `runs/compact-stages-101`, see the
[training pipeline audit](../evidence/training-performance-audit-20261021.md).
It distinguishes real run telemetry, a CPU/GPU fitting replay of saved rows,
and synthetic component controls. Candidate settings are not promoted defaults.
The [pipeline implementation evidence](../evidence/training-pipeline-optimization-20261021.md)
records current complete-cycle trials, including incomplete comparisons.

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

Complete returns remain chronological per original episode. Storage contains
executed active decisions only; private `(slot, episode, ordinal)` references
resolve source credits through RAM/disk integer slabs. Explicit inactive-gap
boundaries survive compaction and recovery. The spool scans slim metadata with
segmented NumPy operations, counts width/route partitions, and gathers selected
rows/entities in bounded chunks. It never copies the complete trajectory dtype
just to compute representatives. Return counts stay raw; fitting denominators
use representatives without frequency weights. Genuine waits and identical
invalid actions retain the first of each configured 100-row window; accepted
non-wait rows remain intact. Wait and invalid reductions are reported separately.

Prepared sources share ordered double-buffer transport across autonomous, demo
and annotation fitting. Each pinned slot grows to its required entity bucket
only after its previous copy event completes. Source indices, spool arrays,
cached CPU routing plans and staging share the storage budget. Preparation and
copy overlap GPU fitting through one producer and a separate copy stream;
consumer stream ownership is recorded before pinned slots are reused. A staging
budget failure restarts the uncommitted pass with eager transport. The disposable
spool is never serialized as simulator recovery state.

Base batches contain 1,024 rows and `fit_batch_groups=4` groups transport work;
there is one optimizer commit per autonomous pass (four) or demonstration pass
(twenty). Collection geometry remains `encoder_microbatch=128`; fitting defaults
use `fit_encoder_microbatch=192` and `fit_token_budget=65536` before GPU-profile
overlays. The 4070 keeps 64/64 and 16,384; 128/128 and 32,768 are candidate overrides.
Transport grouping does not change encoder microbatch limits.

Whole-cohort balancing coefficients are transferred once. Accepted consistency
uses direct all-45 reductions without rebuilding proposal masks. Demo ranking
shares online species/tile tensors with regression until the batch backward.
Annotation inputs and dense label masks are prepared once; indexed max/min
preserve tie gradients and pass-boundary totals avoid redundant train metrics.
FP32 gradient additions, BF16 runtime copies and EMA updates use grouped foreach
operations; clipping and the one-commit-per-pass boundary remain unchanged.

## Annotation fitting

Both annotation commands use packed checkpoint staging and automatically admit
prepared train/validation inputs to GPU memory. On the measured 4070 Laptop
datasets, complete warm passes improved by 13–22%, including fresh evaluation
and durable saving. See the [paired measurements and limits](../evidence/annotation-fitting-performance-20261023.md).
This measures execution throughput; convergence time and grokking were not tested.

The central `training.performance` settings are:

| Setting | Default | Meaning |
|---|---|---|
| `annotation_input_cache` | `"auto"` | Cache both training and validation if admitted; `"off"` streams inputs. |
| `annotation_cache_mib` | `256` | Combined resident input cap; zero disables admission for nonempty inputs. |
| `annotation_use_fit_limits` | `false` | Opt into the existing species/tile fitting microbatch and token limits. |

For example, place these values under the existing table in a custom TOML
configuration and pass it with the usual `--config` option:

```toml
[training.performance]
annotation_input_cache = "auto"
annotation_cache_mib = 256
annotation_use_fit_limits = false
```

Admission includes padded entities, globals, padding masks, label masks, branch
IDs and row indices. It preserves the exact existing batch boundaries, widths,
record order and normalization. Training and validation are admitted together
only if their combined estimate fits the cap and leaves at least the larger of
1 GiB or 20% of total VRAM free. CPU execution, oversized inputs, insufficient
headroom or allocation failure use the existing streaming/prefetch transport.
Partially allocated caches are released before fallback. Admission checks input
memory at setup; later learned activations still need their usual memory budget.

Only immutable raw inputs reside in the cache. All learned forwards and
backwards run again with current weights. Fresh post-update BF16 train/validation
evaluation retains separate accumulation order and metrics, with a combined
final host readback. Validation stays gradient-free and outside the training
source. The test cache is prepared only after exact train/validation convergence
and fresh saved-weight verification, after releasing train/validation caches.
Sources and transports close on completion, interruption or failure.

Checkpoint staging groups authoritative model and Adam tensors by CUDA device
and dtype, copies packed buffers to owned CPU storage, and preserves the existing
checkpoint dictionary's names, shapes, dtypes and values. Layouts are reused
until tensor structure changes, including Adam moment creation. Scratch is
bounded by the configured storage RAM allowance; allocation pressure uses direct
per-tensor transfers. D2H completes before serialization. Saving remains serial,
with a temporary file, flush, `fsync`, Windows-sharing retries and atomic replace
after every committed pass. No next pass starts before that durable save.

Reports and progress record cache request/admission, bytes/reason, staging mode
and effective encoder limits. These execution settings can change on resume;
they do not change objective compatibility. Caches rebuild after resume and are
never serialized. Progress v3/v4 and unfinished-weight initialization retain
their existing rules; no checkpoint protocol or model layout changed.

Default annotation forwards retain policy encoder limits (128 rows, 65,536
tokens under the measured configuration). To experiment with stage fitting
limits, set `annotation_use_fit_limits=true`, then use `fit_species_microbatch`,
`fit_tile_microbatch` and `fit_token_budget` in the same performance table.
The 4070 profile's existing 64-row/16,384-token limits are a control, not the
annotation default. Tile 192/256 controls improved pass timing but changed BF16
values and updates, so larger limits remain opt-in. Compilation stays inactive
for annotation fitting. A grouped explicit FP32 L2 candidate matched arithmetic
but failed repeated complete-pass timing acceptance; the existing L2 reduction,
activation timing, Adam and clipping remain in use.

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

`compile_collection` and `compile_fitting` are independent central execution
settings. The old `compile_kernels` key is rejected in user configuration.
`telemetry_span_limit` bounds pending diagnostic events and their reusable pool.
Ready events are resolved without waits; if the bound remains full, skipped spans
are counted. Hardware samples and host counters remain separate from device spans.
Both compilation settings are disabled in the initial 4070 profile. Eager,
collection-graph and fitting-compiler controls must be measured independently.

Fitting compiler caches are private to each encoder and fitting phase: separate
Python code objects prevent species/tile stages or different model instances from
sharing Dynamo's recompilation allowance. Up to six observed fitting shapes are
compiled per phase; additional shapes use eager BF16. Partial compiled microbatches
are padded and trimmed before supervision. Quiescent fitting release renews that
encoder's compiler cache as well as releasing captures, so changed microbatch
limits do not accumulate stale specializations across cohorts. Global compiler
limits are not increased and unrelated compiler caches are not reset.

Trajectory/recovery v5 stores active identities and boundary carry state and
identifies the changed wait-representative and quadratic reward semantics.
Earlier recovery files cannot resume; compatible weights and recordings remain
valid for fresh initialization. Prepared caches rebuild and are never model data.

## Rates and evidence

Report active decisions/game/s and active transitions/s from host-side interval
records, separately from lifetime totals. Completed-game rewards and wins use
current-cohort records; no device synchronization is added for terminal output.
Diagnostics include both-stage inference/fit time, fitting rows/s, memory,
optimizer completion, selected-action error, species-consistency diagnostics and
conditional best-tile minus sampled-tile placement gap.

The current [smoke suite](../../AGENTS.md#test-ownership) checks one tiny CUDA
learning/checkpoint workflow with compilation disabled. It does not verify
execution equivalence, compiler behavior, recovery or throughput. Historical
controls and measurements remain in iteration history and retained evidence;
manual benchmarks supply new performance evidence.

The [2026-10-27 wait-compaction control](../evidence/wait-compaction-20261027.md)
records raw/retained rows, spool preparation and four-pass fitting on the 4070.
It is a bounded synthetic fitting control, not a complete training-cycle or
convergence measurement.

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
Use `--trace-phase collection` for a bounded collection window instead of the
default fitting window. Lightweight collection counters include the tail with
eight or fewer active games. Finalization, complete returns, spool build and
checkpoint serialization have distinct host spans.

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
| Frequent `extra_shape_eager` microbatches | Shapes exceed the six fitting specializations admitted per encoder; inspect kernel-launch gaps |
| CUDA runtime waits around `aten::nonzero` / indexing | Device-dependent output sizes synchronize the host; locate waits in the trace |
| Many short kernels and gaps with small effective microbatches | Host launch/control overhead; inspect forward/backward/master-gradient spans |
| Increasing cgroup `throttled_usec`, CPU pressure or steal time | Cloud CPU quota or contention; total CPU percentage can hide a saturated thread |
| High GPU utilization with long kernels/copies | GPU compute, memory or transfer work; distinguish kernels from memory copies in the trace |

These are hypotheses, not automatic tuning recommendations. In particular,
4,096 transported rows still execute in microbatches capped at 192 by default.
The spool partitions by configured public-width bucket and route. Teacher tile
inference uses its own no-grad fitting graph owner when fitting compilation is enabled.
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
