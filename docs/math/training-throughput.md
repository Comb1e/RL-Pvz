# Precision and execution controls

## Collection execution contract

Fitting packs genuine event rows into the LSTM and gathers each latest memory
output onto all decisions. Quiet rows still supervise current-state predictions.
The memory-read backward uses a deterministic matrix reduction rather than repeated
index atomic accumulation, preserving exact interrupted-pass recovery. Event metadata
and acceptance strata use the existing pinned transport and shared storage budget.
Detach boundaries remain 256 decisions, not 256 events.

Collection leaves the model, reward prices, probe schedule, recurrent feedback,
loss denominators and fitting settings unchanged. A host `ActiveBatchPlan` uses
the canonical live-slot mask (`enabled_envs`) and already-published public
retained-entity counts (`summary_host`). For each public entity-width bucket,
it gathers live observations and complete event memory into power-of-two batches,
bounded by encoder microbatch/token budgets. Masked execution padding creates
neither transitions nor recurrent writes. Outputs scatter to original slot IDs;
inactive hidden/cell, pending facts and elapsed ticks stay unchanged. No live rows
means no network call. Both online and actual-history EMA use the plan. The tile
head processes active rows through a wrapper, while selection/exploration RNG
keeps the original slot-shaped draw dimensions and order.

For a width bucket with `m > 0` logical rows and batch capacity `K`, each chunk
uses `B = 2^ceil(log2(min(m_remaining, K)))` rows, with `K` bounded by the
microbatch and token allowance for entities plus 46 readout tokens. Thus chunk
padding stays below twice its logical row count; it is not cohort capacity.
Compaction does not promise bitwise-equal floating-point reductions across batch
shapes. Independent serial controls still check outputs, memory and random draws
without weakening the representation or selected-action contract.

Two independent scratch lanes execute the two branch probes and two tile probes
in two passes. Shallow batch/feature prefix views reuse the existing allocation;
source-ID remapping copies original live slots into compact lane prefixes before
simulation/encoding. Scheduled-role masks preserve eligibility. Public feature
records and metadata scatter back to canonical game/probe rows before dedup and
bootstrap. Scratch execution width follows live games, but allocation capacity
and private simulator storage bounds remain unchanged. Neither private bounds
nor source IDs determine policy tokens. If scratch
allocation fails, one lane executes all scheduled slots; no probe is omitted.
Every pass restores its own simulator RNG, accounting and proximity ledger.
One compact EMA evaluation advances actual history; independently gathered fork
inputs use that history without ever replacing it with a probe result.

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
components and stored bootstrap targets remain separate. Only valid nonterminal
representatives enter compact EMA next-state inference; values scatter to every
original probe record and terminal bootstrap is zero. Gross-event, memory-timing,
global or entity counterexamples keep independent offsets.

Persistent canonical feature stores and device packing replace per-probe width
reads. Behavior globals transfer active rows only; the host scatters them into
zero-filled original-slot rows. Probe globals follow the same compact/canonical
scatter contract; fixed metadata may retain original slot-shaped staging.
Entity transfer uses already-published retained counts
for behavior and summed public next-entity bounds for active probe games, not
original slot capacity or private simulator bounds. The packed probe slab's unused
tail can still cross PCIe: exact variable-length
DMA would need an extra host count boundary. Deduplication reduces stored records
and host comparison work, but does not guarantee proportional transfer savings.
Pinned buffers are reusable and charged to trajectory RAM. Behavior/probe evidence,
dedup representatives and terminal headers/totals precede one existing
collection-stream wait. Only then are GPU dedup results known to the host planner.
The current canonical record appends first; compact bootstrap inference queues
its CPU copy after that wait. The next existing handoff consumes the copy and
`buffer.update_probe_bootstrap` patches the already-appended rows, including
spilled blocks. Finalization, fitting entry, checkpoint, interruption and shutdown
drains flush the last pending copy before any target or saved trajectory uses it.
No ordinary second count/bootstrap wait is added. Pending copies cannot outlive
their row identities or be overwritten before consumption.
Storage checks counts/offsets before ingestion and recovery validates saved slabs.

All CUDA platforms use direct no-grad `CUDAGraph` collection and the custom AOT
owned forward/backward fitting backend, without a Windows/Inductor split.
Each encoder separately owns a collection LRU cache and fitting captures. Defaults
are eight live collection entries and 512 MiB headroom, configured through
`training.performance.collection_graph_cache_entries` and
`collection_vram_headroom_mib`. Capture keys include shape, stride, dtype/device
and computation mode; online and EMA do not share captures. Eviction retires the
least-recently-used capture and fences its last replay/output copies. The existing
handoff releases completed retirees without a new wait; cache misses may also poll
completed fences. Retirement is separately bounded by the entry limit, so resident
counts include live entries plus not-yet-released retirees. Headroom is checked
before and after admission.
Capacity/headroom pressure or capture allocation failure selects the exact eager
encoder; other capture failures record an explicit fallback. Zero entries disables
collection capture only.

At quiescent phase boundaries, collection captures/probe workspaces release before
fitting, and fitting captures release after all delayed backwards/prefetch work
and before collection allocation. Fitting's four-shape bound is independent of the
collection cache. Active maps, captures and caches are execution-only, rebuilt on
resume rather than serialized. Recovery/trajectory protocols and durable rows
remain unchanged. Cache counters and phase-memory samples are diagnostics, not
speed or learning-quality evidence.

CUDA timings are read only after the existing handoff. Device
spans and host storage/presentation/synchronization timings are reported separately
and need not sum to wall time. Terminal formatting only consumes cached host
metrics; its cadence does not control the 15-second detailed snapshot cadence.

## Active collection rate controls

Let `D` count committed actual live transitions in the reporting interval, `T`
count collection seconds and `A = sum_i(a_i * delta_t_i)` count active-game
seconds, using active games before each callback interval. The monitor reports

\[
\text{active transitions/s} = D/T,\qquad
\text{decisions/game/s} = D/A.
\]

Padding, finished games and counterfactual simulation do not increment `D`.
Using `A`, rather than initial cohort capacity times `T`, preserves the per-live-
game interpretation as slots finish. Phase/cohort changes or restored counters
restart the interval; fitting, validation and recovery downtime are excluded.
Zero/unmeasured denominators show `n/a`. Interval rates and lifetime transition,
collection-time and active-game-time totals remain distinct. Host counters and
clocks are the sole inputs; sampling adds no CUDA read or wait.

Benchmark controls hold 128 original simulator slots and select noncontiguous
128/64/32/16/4/1 live-game subsets, not six different allocation sizes. They
distinguish sparse/mixed/crowded boards, cold capture, warmed repeats and an opt-in
bounded four-pass collect/fit/collect cycle. Evidence must report logical/padded rows, unique valid
probe representatives, cache hit/miss/eviction/fallback counts and phase memory,
not just nominal slot capacity. Results and verification status belong only in
[iteration history](../iteration.md); execution controls alone establish no speedup.

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
