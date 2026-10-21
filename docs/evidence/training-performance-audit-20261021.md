# RTX 4070 training pipeline audit — 2026-10-21

The strongest opportunities are CPU launch/control overhead during collection,
larger fitting microbatches for the observed sparse workload, and eliminating
repeated preparation/forwards. Increasing transfer bandwidth or removing EMA is
not supported as the first optimization by the available evidence.

This is an inspection and bounded measurement, not an implementation change.
Architecture, learning settings, rewards and user checkpoints were not changed.
Raw measurements and methodology are in
[the audit JSON](training-performance-audit-20261021.json).

## Evidence from the user's training run

Primary evidence is `runs/compact-stages-101/easy`: logs, 568 hardware samples,
and public trajectory arrays read directly from `interrupted.zip`. That run
uses reward/recovery v2, before the current unmasked-tile/early-rejection change.
It is useful workload evidence, not a current-contract resume or a before/after
comparison. No simulator or optimizer recovery state was resumed.

| Observed fact | Implication |
|---|---|
| 128 games, 2,320,769 decisions collected in 552.8 s: 4,199 decisions/s | Collection is a substantial measured cost. |
| Collection median process CPU 92.4%, GPU utilization 39%, GPU power 57.3 W | Host scheduling/launch work deserves investigation; these aggregates alone cannot prove the GPU's stall cause. Process 100% means approximately one logical core, not the whole CPU. |
| 1,959,182 species-invalid, 358,235 waits, 3,352 accepted non-wait, zero tile-invalid rows | Accepted EMA inference is a small fraction of this historical workload. Current early branch rejection already avoids much tile work that this older run performed. |
| All observed boards have 5–43 encoded entities; 96.36% have at most 32 | Dense synthetic boards are boundary controls, not the typical easy-stage tuning workload. |
| 3,398,784 stored slot rows, 2,320,769 active rows | 31.72% of stored metadata rows are inactive slot gaps; model inference already packs active rows. |
| At least the final 73 s of collection had seven or fewer live games | Small-active-batch latency matters, as well as full-batch throughput. |
| Fitting was interrupted in pass 1; optimizer step remained zero | There is no measured completed fitting pass or full collect/fit/collect cycle in this run. |
| The 17 fit-phase samples show median CPU 100%, GPU 0%, RAM rising to 6,182 MiB | This phase includes spool setup, asynchronous launch work and interruption/save handling. A trace is needed to separate them; this is not proof of a transfer bottleneck. |

Applying the shared compactor to the entire saved chronological metadata retains
414,842 of 1,959,182 invalid rows, a 78.83% reduction. Including waits and accepted
actions leaves 776,429 fitting representatives. There are 414,841 invalid runs:
nearly all retained invalid rows represent genuinely different short runs, so
the existing 100-row rule cannot reduce them further without changing supervision.
Compaction on slim metadata took 2.26 s, excluding archive reads and spool copies.

## Bounded fitting replay with CPU/GPU overlap

The replay uses actual saved public observations, actions, routes and targets.
Every 64th representative is selected separately within each entity bucket and
route: 12,139 records comprising 5,601 waits, 6,484 species-invalid proposals and
54 accepted non-wait actions. Already-compacted representatives are not compacted
a second time across sampled gaps.

Both cases use the shared RAM buffer, fitting spool, CPU producer, two pinned
slots, transfer stream, 2,048-row transport, selective online/EMA inference,
backward, FP32 master accumulation, clipping, Adam, runtime sync and EMA update.
Each repetition completes four passes on temporary models initialized from the
saved model weights. Optimizer/EMA state is fresh and no weights are published.

| Execution setting | First four-pass replay | Warm four-pass replay | Peak PyTorch allocation |
|---|---:|---:|---:|
| Microbatch 64, token budget 16,384; current 4070 profile | 8.44 s | 8.12 s | 1,387.6 MiB |
| Microbatch 128, token budget 32,768 | 4.57 s | 4.57 s | 1,396.8 MiB |

The candidate reduces this replay's warmed time by 43.7% (1.78× throughput).
CPU preparation totals were 0.13–0.14 s and copy-stream totals about 0.056 s
across eight passes per case; main-thread waits for prepared batches were below
1 ms in total. These spans overlap and must not be added as disjoint wall costs.
For this in-RAM sample, transfer is not the dominant measured fitting cost.

This is not a promised training speedup: the sample excludes full-cohort
materialization, simulator allocations, the live viewer and collection. Candidate
order was fixed, with only one warm repetition per setting. Full production
memory headroom and sustained clocks remain unverified.

Synthetic controls show why workload matching matters. With accepted actions
and 64 entities, the same candidate improves median four-pass time from 0.258
to 0.240 s. With 256 entities, it worsens 0.743 to 0.846 s. Enabling the existing
AOT CUDA-graph backend on the 64-entity case worsens warmed time to 0.308 s,
increases peak allocation from 876 to 2,087 MiB, and has a 4.36 s cold cycle.
Do not enable compilation globally or raise every microbatch from these results.

## Improvements, in priority order

1. **Validate a larger sparse-workload fitting microbatch.** The replay supports
   testing 128 with a 32,768 token budget on this 4070. Retain bounded memory
   fallback and dense/bucket-boundary controls. Keep the optimizer's one commit
   per pass and whole-cohort denominators. Transport groups do not enlarge encoder
   microbatches; increasing groups can retain more autograd activations until
   the batch backward and increase memory pressure.

2. **Reduce collection launch/control overhead, especially at the cohort tail.**
   The current collector's short initial-state probe reaches about 11,729
   decisions/s, but differs from the trained policy and mature boards, so it is
   not comparable to the historical 4,199/s. Its trace still has 978 CUDA launches,
   four `nonzero` calls and 22 stream synchronizations across two decisions.
   Inspect the real trained-state timeline before attributing these waits.
   Cache immutable branch bitmasks/cost tensors and reduce repeated small tensor
   assembly. The ordinary one-request-per-board path can avoid the general
   `unique` deduplication in `conditional_tiles`; retain the generic interface
   for callers that actually request duplicates. Separate collection-graph and
   fitting-compile switches so collection acceleration can be evaluated without
   imposing the slower measured fitting backend. Preserve required branch
   skipping, RNG order, graph ownership and fallback behavior.

3. **Reuse online forwards during demo fitting.** `batch_loss` evaluates online
   species/tile values for accepted non-wait rows. The subsequent
   `demonstration_rank_loss` is called without its optional cached species/tile
   arguments, so it computes them again. A shared forward-result interface can
   serve both losses with correct row alignment and gradients. Zero dropout
   makes reuse particularly suitable here. This is a source-confirmed duplicate;
   its end-to-end benefit has not been measured in this audit.

4. **Vectorize and prepare annotation/demo data once.** Annotation ranking builds
   label masks and converts CUDA predicates to Python booleans inside a per-record
   loop. Each batch also synchronizes for finite checks, scalar loss and CPU
   metrics, and reconstructs encoded observations. Prevalidate/precompute boolean
   label sets, batch the same max/min hinge, bucket by public width, and aggregate
   metrics on device until the existing pass boundary. Demo fitting similarly
   repeats CPU collation and blocking metadata transfers each pass instead of
   using the autonomous prefetch path. Preserve exact ties, empty/all-positive
   behavior, required full-training evaluations and atomic pass saves.

5. **Bound full-cohort CPU materialization and storage.** `FittingSpool` creates
   complete temporary record arrays for selection, sorting and gathering in
   addition to its accounted persistent arrays. The compactor uses Python sets
   and repeatedly scans interleaved environments. Use slim metadata, vectorized
   run boundaries and bounded gathers through the shared interfaces; retain the
   exact first-of-100 rule and episode boundaries. A future compact active-row
   storage path could remove inactive slot gaps, but needs an explicit mapping
   for source-action identity and recovery. This is more invasive than tuning.

6. **Reduce recurring collection payloads and Python reporting work.** Every step
   currently copies full episode totals, headers and the all-45 policy mask, then
   constructs one nested info object per slot. In the 128-slot probe those buffers
   alone reserve 112,640, 21,504 and 51,968 bytes respectively. Keep fields needed
   for public counts/accounting, transfer full totals for completed lanes, and
   reconstruct the unchanged proposal mask from active flags where valid.
   Offer packed internal results while preserving the external adapter/viewer.
   Measure fewer allocations and transfers together; bandwidth alone is not
   established as the limiting resource.

7. **Reduce small fitting kernels before optimizing infrequent optimizer work.**
   One warmed 128-row accepted-action batch has 2,572 `cudaLaunchKernel` calls and
   726 `aten::_to_copy` calls. Efficient SDPA is already active. FP32 norms,
   runtime/master gradient casts, pointwise losses and tiny coefficient transfers
   are fusion/batching candidates. The current custom compiler captures AOT
   operations; it does not provide Inductor pointwise fusion. Preserve FP32
   reductions and BF16 runtime/master numerics. Reuse cohort coefficient tensors
   and group compatible gradient/EMA operations. Fused Adam alone is lower
   priority because Adam commits only four times per cohort.

## Verification needed before adopting settings

Measure an isolated, current-contract run with the same initial weights,
128-game easy cohort, exploration, viewer state and storage budget. Compare the
complete collect → finalize → returns → four-pass fit → collect path, including
cold setup, cohort-tail latency, RAM/disk spill, host/GPU timing and actual row
counts. Repeat settings in alternating order and compare total elapsed time,
not utilization or a sum of overlapping profiler spans. Do not refill completed
lanes or overlap a new cohort across weight updates merely to improve utilization:
that would change the current cohort/return/behavior semantics.

The existing `tools/collect_fit_diagnostics.py` can profile the user's next normal
training invocation with the same arguments. The archived v2 checkpoint must
not be resumed under current v3 semantics; compatible weights can initialize a
fresh run. No formal run was launched for this audit.
