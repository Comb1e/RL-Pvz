# Precision and execution controls

## Collection execution contract

Fitting packs genuine event rows into the LSTM and gathers each latest memory
output onto all decisions. Quiet rows still supervise current-state predictions.
The memory-read backward uses a deterministic matrix reduction rather than repeated
index atomic accumulation, preserving exact interrupted-pass recovery. Event metadata
and acceptance strata use the existing pinned transport and shared storage budget.
Detach boundaries remain 256 decisions, not 256 events.

Collection leaves the model, reward prices, probe schedule, recurrent feedback,
loss denominators and fitting settings unchanged. Two independent scratch lanes
execute the two branch probes and two tile probes in two passes. If scratch
allocation fails, one lane executes all scheduled slots; no probe is omitted.
Every pass restores its own simulator RNG, accounting and proximity ledger.
One EMA evaluation advances actual history; stacked fork inputs use that history
without ever replacing it with a probe result.

For public retained count `C`, pending-zombie count `Z` and present plant count `P`,
`min(cap, C + Z + 2P + 1)` bounds one decision's retained next entities. Rejection
and wait advance at most one tick; the pinned engine emits at most two shots per
existing plant in that tick. Accepted planting/digging advances no tick and adds
at most one plant. When already capped, the bound remains the cap. This count-only
bound exposes neither future spawn ticks nor RNG. Buckets 32/64/128/256 hold it;
CPU/CUDA canonical ordering, multiplicity and omissions remain unchanged.

Probe equivalence is scoped to the same actual game/decision and requires exact
kept entity records, globals and gross public event/timing inputs. All forks
in that scope clone the complete actual EMA event-memory state. Thus both observation and
recurrent inputs match before an entity offset can alias. Proposals, reasons,
components and bootstrap targets are never deduplicated. Gross-event, memory-timing,
global or entity counterexamples keep independent offsets.

Persistent feature stores and device packing replace per-probe width reads.
Behavior transfer uses its already-known bucket; probe transfer copies a bounded
packed bucket slab. Its unused tail can still cross PCIe: exact variable-length
DMA would need an extra host count boundary. Deduplication reduces stored records
and host comparison work, but does not guarantee proportional transfer savings.
Pinned buffers are reusable and charged to trajectory RAM. All copies precede
one collection-stream wait; headers/totals for terminal episodes use that handoff.
Storage checks counts/offsets before ingestion and recovery validates saved slabs.

Encoder graphs use the existing tensor-only, owned-output wrapper for collection
and fitting. Shape caches remain bounded; unavailable compilation records one
eager fallback. CUDA timings are read only after the existing handoff. Device
spans and host storage/presentation/synchronization timings are reported separately
and need not sum to wall time. Terminal formatting only consumes cached host
metrics; its cadence does not control the 15-second detailed snapshot cadence.

## Fitting execution contract

The optimization changes execution, not the entity representation or complete-return
objective. For each cohort, the same active decisions, group denominators and
targets contribute to each of four whole-pass gradients. Chronological 256-step
chunks retain the original detached recurrent boundaries. The execution path
may concatenate four independent 1,024-decision groups, but this does not define
a new optimizer minibatch or alter the loss denominator. The current encoder
microbatch is 128 observations with a 65,536-token budget.

Let `theta` be FP32 master weights and `Q_b(theta, x)` the network with eligible
feature operations evaluated using BF16 computation copies. Fitting accumulates

\[
g = \sum_{c \in \text{chronological chunks}}\nabla_\theta L_c(Q_b(\theta,x),G),
\qquad \theta' = \operatorname{Adam}(\theta,\operatorname{clip}(g)).
\]

The gradient is accumulated in FP32; no loss scale is applied. BF16's exponent
range matches FP32, but its shorter significand changes rounding. This changes
numerical results and can change subsequent learned decisions. It does not
establish equivalent long-term training trajectories or learning quality.
Keeping master weights in FP32 preserves small updates that would disappear if
rounded immediately to BF16 storage.

Entity embedding outputs and eligible feature matrix operations use BF16.
Embedding sums, residual accumulation, normalization reductions, explicit fallback
softmax, recurrent computation/state, Q heads, objective and optimizer remain
FP32. Encoder outputs are converted to FP32 before the recurrent and tile-head
boundaries. FP64 independent controls retain their precision.

Independent tests compare BF16 branch/tile outputs against FP32 with absolute
tolerance 0.002 and relative tolerance 0.01. The aggregate gradient relative L2
error must be at most 2% on seeded sparse/crowded controls. Internal cell states
can accumulate input quantization differences; a separate control evaluates the
FP32 LSTM on the actual BF16-derived feature inputs and checks its state exactly.
FP32 attention, sequence and gradient controls remain independently tested.
For Adam, the independent first-step formula is `delta = lr*g/(abs(g)+epsilon)`.
Softmax makes a shared key bias mathematically irrelevant, so its tiny FP32
gradients can be rounding noise. Adam can amplify differences in those near-zero
gradients; controls bound update differences by this formula rather than assuming
bitwise-identical parameters across different reduction orders.

An uncommitted pass contributes no durable optimizer update. Nonfinite BF16 loss
or gradients discard all accumulated gradients and restart that pass in FP32.
The remainder of the cohort stays FP32. A nonfinite FP32 retry fails. Allocation
failure uses successively smaller encoder microbatches, then one outer encoder
checkpoint; it never changes observations, targets or recurrent chunk boundaries.
Restarted passes are not added to completed-pass or optimizer-step counters.

Two reusable pinned slots are charged to the trajectory RAM allowance. A single
producer preserves sequence order; each slot's transfer event completes before
it is overwritten. The compute stream waits for transfer completion, and copied
device tensors retain stream ownership until their consumers finish. Checkpoints
drain queued copies and retain durable arrays plus effective execution settings;
prefetched chunks and partial gradients are reconstructed after interruption.

If independent sequence groups are (G_1,…,G_k), with losses (L_i), the fused
call computes

\[
L_{\mathrm{fused}} = \sum_i L_i,
\qquad
\nabla L_{\mathrm{fused}} = \sum_i \nabla L_i,
\]

because each group has its own hidden and cell state and attention is scoped to
one observation row. The optimizer still clips and updates once after the full
cohort pass. A reset is inserted between slot groups, and a partial final group
is padded with inactive rows.

Per-chunk CUDA timing and metric reads are intentionally absent from the hot
loop. One timing event is recorded for each pass; intermediate logs reuse the
last completed metric snapshot. This avoids host synchronization while keeping
pass-end finite checks, optimizer timing, checkpointing and recovery explicit.

Performance evidence must separate setup/warmup, collection, host preparation,
transfer wait, transfer device time, fitting, optimizer time and total wall time.
Preparation and transfer can overlap computation, so their durations are not
additive. Memory reports distinguish Torch allocation, CuPy pools and host RAM.
Short throughput checks are not formal training or learning-quality estimates.
