# Annotation fitting execution — 2026-10-23

Packed checkpoint transfers, resident prepared inputs and combined evaluation
readback passed exact-equivalence controls and repeated complete-pass timing
acceptance on the RTX 4070 Laptop. They are enabled by default. Encoder limits,
L2 arithmetic, Adam, clipping, the objective and model architecture are unchanged.

| Head / phase | Baseline warm pass | Candidate warm pass | Reduction |
|---|---:|---:|---:|
| Tile / initial | 129.75 ms | 111.66 ms | 13.9% |
| Tile / L2 active | 147.15 ms | 128.04 ms | 13.0% |
| Action / initial | 93.69 ms | 72.84 ms | 22.3% |
| Action / L2 active | 99.55 ms | 84.65 ms | 15.0% |

These measure execution throughput with fixed inputs. They do not measure time
to convergence, validation generalization or grokking.

## Workload and timing

The baseline fitter is revision `5c801d0`; the measured candidate is `144b69e`
with the packed staging implementation from `856bc39`. The final follow-up
corrects staging-mode reporting and formatting without changing numerical work.
Both use shared production components. The baseline fitter was loaded from Git
into a temporary module; no alternate training implementation remains installed.

The device is an 8 GiB RTX 4070 Laptop, driver 616.56, PyTorch 2.8.0+cu128,
Python 3.12, Windows. Trials use the 4070 profile, two CPU threads, seed 101 and
fresh identical random weights/Adam/RNG state. Annotation batch size remains
256; both default encoder limits remain 128 rows / 65,536 tokens. There is no
compilation or profiler. Datasets are copied to scratch storage on the project
volume. Tile has 199 labeled records (160 train, 22 validation, 17 test); action
has 50 (40/5/5). Dataset/split digests, initial-weight hash, exact configuration,
hardware details and trial summaries are retained in the
[measurement JSON](annotation-fitting-performance-20261023.json).

Each head/phase uses ABBA ordering, with 48 optimizer passes per invocation.
The first five passes are excluded from warm medians. Each reported cell is
the median of its two trial medians. Adjacent A/B pairs improved by 8.9–30.2%;
none regressed. Every complete pass includes training backward, optional L2,
clipping, Adam, runtime synchronization, fresh train/validation evaluation,
snapshot staging, durable atomic checkpoint saving, history and flushed console
output. Console output uses an identical scratch file for both implementations;
interactive-terminal rendering is not measured. L2 trials use a controlled
convergence predicate to activate the real regularized path from pass one;
these random initial weights did not actually satisfy the training gate.

First optimizer-pass times are retained separately: tile baseline 149–289 ms
and candidate 130–159 ms; action baseline 97–101 ms and candidate 80–105 ms.
The first baseline also bears process/device startup. Invocation elapsed and
time outside passes 2–48 are retained, but the latter includes setup, first
pass and teardown, so it is not an isolated cold-preparation measurement.
Candidate input-cache admission took about 1–2.3 ms in these controls.

Host spans are diagnostic and inclusive: baseline serialization includes its
implicit D2H transfers; candidate staging and serialization are separate.
Baseline evaluation spans cover one split, candidate evaluation spans cover
both. L2 schedule spans exclude its later backward. CUDA execution is
asynchronous, so these spans must not be summed or interpreted as isolated
kernel times. Headline elapsed times use one wall-clock interval per pass.
Separate four-pass controls with explicit phase-boundary CUDA synchronization
record CPU preparation, ranking backward, L2 forward/backward, optimizer,
runtime synchronization, evaluation, snapshot staging and serialization in the
JSON. They are excluded from headline comparisons. Per-split CPU preparation
medians were about 0.94–1.32 ms. L2 backward measured about 8.20 ms for tiles and
4.10 ms for actions in these short synchronized controls; this diagnostic does
not replace the ordinary complete-pass measurements.

## Memory and durable saving

The combined resident train/validation inputs occupy 294,476 bytes for tiles
and 65,700 bytes for actions. Admission uses a 256 MiB cap and at least the
larger of 1 GiB or 20% VRAM headroom. Candidate peak PyTorch allocations were
about 591 MiB (tile) and 72 MiB (action), versus 780–794 / 274 MiB baseline.
Separate complete-pass microbatch controls observed at least 6,244 MiB free at
pass boundaries, including non-PyTorch/device allocations. These boundary
samples are not continuous device-memory traces.

Maximum sampled process RSS ranged from 1,622–1,865 MiB baseline and 1,845–2,223 MiB
candidate in the shared benchmark process. CPU allocator retention and trial
ordering affect these numbers; no RSS reduction is claimed. Progress artifacts
remain complete (about 34.2 MiB tile and 19.8 MiB action). Packing changes storage
grouping, not tensor names, dtypes, values or checkpoint protocols. Per-pass
flush, `fsync`, replacement and Windows-sharing retries remain synchronous.

## Optional controls

| Tile limits | Backward control | Complete warm pass | Numerical difference from 128 |
|---|---:|---:|---|
| 128 / 65,536 tokens | 33.84 ms | 104.9–106.5 ms in paired controls | Reference |
| 64 / 16,384 tokens | 39.34 ms | 110.98 ms | Same values/loss; max gradient delta 0.0007324 |
| 192 / 65,536 tokens | 28.65 ms | 97.58 ms | Max value/gradient delta 0.00390625 |
| 256 / 65,536 tokens | 28.89 ms | 97.35 ms | Max value/gradient delta 0.00390625 |

Complete-pass controls use ABBA, 24 passes and five warmups; each candidate has
its own 128-row control. Tile loss delta for 192/256 is 0.00009751 and maximum
updated-parameter delta is about 0.0006000. These numerical controls use one
initial-state optimizer update; long-run divergence is not measured. Larger
limits remain opt-in through `annotation_use_fit_limits`. The 40-row action
training split fits in one microbatch at either 64 or 128, so its noisy isolated
timings do not establish a batching improvement.

Grouped elementwise FP32 powers with the original per-parameter sum order gave
exact L2 penalties, gradients and final model/Adam tensors in the controls.
However, tile regularized ABBA medians were 138.15, 127.30, 135.05, 128.96 ms:
one pair regressed by more than 2% and the aggregate improvement was below 5%.
Action controls regressed in aggregate. The candidate is rejected; the original
explicit L2 implementation remains the only implementation.

## Correctness and limits

Temporary controls for both heads compared the Git baseline, streaming and
resident execution, including partial batches and empty validation. Losses,
metrics, FP32 gradients, model/frozen weights, Adam state, RNG state and
pass/phase counters matched exactly. ABBA production runs matched their final
state exactly as well. Snapshot controls covered noncontiguous/scalar/empty
tensors, owned CPU storage, roundtrip serialization, cap fallback and injected
allocation failure. Cache controls covered cap, headroom and allocation rejection,
including a failure after the first split was already resident. Interruption and
injected fitter failures released all prepared sources and device caches.

CPU interrupted-backward and SIGINT-during-commit controls resumed to the same
three-pass state as uninterrupted fitting, including delayed L2 state. They
checked additional-pass limits, fixed training membership, execution-setting
changes on resume, dataset mismatch without overwriting progress, v3-to-v4
resume, zero-pass/incomplete initialization and concurrent Windows save/load.
Controlled publication checks for both heads verified zero-pass publication,
test evaluation/cache admission only after the convergence gate, direct staging
fallback, and resource release. CPU smoke checks passed (2 passed, 1 deselected);
focused Ruff lint/format and `git diff --check` passed. No permanent tests were
added. Bounded performance controls finished within the 30-minute allowance.

No formal training, push or PR was performed. User datasets, existing runs,
architecture documentation and `tools/pre_train.ps1` are untouched. Temporary
datasets/checkpoints/scripts are removed after retaining this evidence. Larger
datasets and other GPUs still need workload-specific performance measurements;
the streaming path remains available.
