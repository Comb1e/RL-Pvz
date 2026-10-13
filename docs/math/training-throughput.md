# Precision and execution controls

Execution optimizations preserve decisions, reward prices, exploration, tile
probe coverage, complete-return targets and whole-cohort loss denominators.
Two current-board encoders replace all learned history; this is not a promise
of lower cost or improved learning.

## Active-only two-stage inference

`ActiveBatchPlan` uses canonical enabled-slot masks and already-published public
entity counts. Each public-width bucket gathers live boards into power-of-two
batches bounded by encoder microbatch/token budgets, then scatters values to
original slots. No live rows means no inference. Padding creates no transitions,
RNG or supervision. Finished slots are never refilled or moved.

Species inference has one extra token; conditional tile inference has 47. For
m logical rows and capacity K, a chunk uses
\(B=2^{\lceil\log_2(\min(m_{remaining},K))\rceil}\), where K respects each stage's
overhead, width and execution budget. Tile calls evaluate requested board/branch
pairs only, reusing duplicate requests. Waits require no tile call. Commitments
can request their forced species alongside original-policy diagnostic requests.
Behavior selection retains original-slot draw shapes/order even for unused draws;
auxiliaries consume no behavior RNG.

Compaction changes floating-point reduction order, not representation. Serial
controls compare both stages within established tolerances. Private simulator
allocation bounds never become entity tokens or future-state features.

## Independent background execution

An immutable pre-action source bank retains eligible actual plantings. Actual
acceptance authorizes same-species alternative tiles. FIFO jobs advance through
reusable occupied auxiliary lanes while ready real slots continue, including the
planting owner. Defaults are 256 lanes, 128 sources and one auxiliary transition
per scheduler round; unused allocation rows execute nothing.

Source pressure holds only another eligible planting's retained proposal/draws,
without artificial waits or time advancement. Older waiting tickets have priority.
Unfinished, ready, capacity-blocked and terminal masks remain distinct. Allocation
fallback halves effective capacities to one source/lane without dropping probes;
failure at the minimum is explicit. GPU contention remains despite independence.

Public count-only growth bounds support auxiliary entity-width buckets without
examining future schedules. Sources own simulator/accounting evidence; forks
follow frozen greedy EMA. No recurrent state or event-history tensors are allocated.
Owned endpoint boards/components survive lane reuse. Exact public-input and EMA
version comparisons deduplicate endpoint bootstrap; conditional requests also
compare branch. Ongoing simulations and proposal/reward evidence never merge.
Terminal/cutoff bootstrap is zero; only unique nonterminal endpoints need species
EMA tail inference.

Behavior globals transfer issued rows and reconstruct zero-filled canonical host
rows; fixed metadata remains slot-shaped. Actual rows append with pending counts.
Later reusable pinned handoffs attach owned endpoints/bootstrap values exactly
once, including spilled storage, without an extra ordinary bootstrap wait. Heavy
auxiliary control reports remain at 64 continuation rounds. Consume reusable views
before overwriting them.

Saving stops selection, finishes issued tickets and drains every queued/running
job, endpoint/bootstrap copy and patch. Finalization/fitting/EMA changes require
zero pending evidence. Transient forks are not serialized; canonical simulator,
controller, scheduler, accounting and RNG state recover independently of the model.

## Capture ownership and headroom

Both online and EMA stages have separate collection graph caches. Per-encoder
defaults are eight entries and 512 MiB VRAM headroom, controlled by
`collection_graph_cache_entries` and `collection_vram_headroom_mib` under
`training.performance`. Keys include shape/stride/dtype/device and computation
mode. Eviction fences replay/output copies; existing handoffs release completed
retirees, whose separate bound prevents indefinite accumulation.
Headroom/capacity/allocation/capture failure reports exact eager fallback.
Zero cache entries disables collection capture, not fitting.

All CUDA platforms use direct no-grad collection graphs and owned AOT fitting
captures. Saved activations, outputs and gradients are copied out of mutable
graph storage with SDPA strides preserved. Simulator pybind objects are never
traced. Collection and fitting have separate phase ownership; quiescent changes
drain consumers and release outgoing captures/workspaces before next-phase
allocation. Recovery rebuilds maps/caches instead of serializing captures.

## Flat whole-pass fitting

Complete returns are calculated chronologically per original episode. Fitting
then flattens genuine decisions and excludes capacity gaps/padding. No temporal
detach boundaries, hidden/cell states or episode-sequence reconstruction exist.
Base batch size is 1,024. `training.performance.fit_batch_groups=4` groups flat
batches for execution, not four optimizer minibatches. The full cohort supplies
selected 3x2, consistency 2x2 and probe 10x2 outcome denominators.

With batch contributions \(L_j\) and EMA held fixed for the pass,

\[
g=\sum_j\nabla_{\theta,\phi}L_j,\qquad
(\theta',\phi')=\operatorname{Adam}((\theta,\phi),\operatorname{clip}_5(g)).
\]

Partitioning/grouping preserves additive loss/gradients within numerical tolerance.
Clip once and commit one Adam step per complete pass, then update both EMA stages
with decay 0.95. Autonomous fitting makes four passes; demonstrations make twenty.

`fit_precision=features_bf16` uses BF16 temporary encoder feature operations;
`fp32` supplies a reference. Master weights, optimizer state, Q heads, objective
and accumulated gradients remain FP32; there is no FP16 loss scaler.
Collection/evaluation and default demo initialization remain FP32. Quantization
can change learned trajectories; selective precision is not numerical identity.
FP64 independent mathematical controls retain their precision.

Nonfinite BF16 loss/gradients discard the uncommitted pass and retry in FP32.
Allocation retries shrink encoder microbatches, then enable one outer checkpoint,
never change entity cap, targets, coverage or denominators. Failed backward capture
restarts the uncommitted pass eagerly. Exhaustion stops without an optimizer/EMA
commit; retries do not increment completed-pass counters.

Ordered double-buffer `BatchPrefetch` charges both pinned slots and flat index
maps to trajectory RAM. Slabs/index scratch may spill. A slot's transfer fence
completes before reuse; the compute stream waits on copies and consumers retain
ownership. Early close drains copies and releases staging/index scratch.
Interrupted passes reconstruct batches from durable evidence, not prefetched
tensors or partial gradients.

## Rates and evidence

For actual committed interval decisions D, collection time T and active-game
seconds \(A=\sum_i a_i\Delta t_i\), report active transitions/s = D/T and
decisions/game/s = D/A. Padding, auxiliary transitions and finished slots do not
increment D. Phase/cohort/reset boundaries restart intervals; fitting/recovery
downtime is excluded. Zero/unmeasured denominators show `n/a`. Lifetime totals
remain separate. Host clocks/records add no CUDA read/wait.

Device timings cross existing handoffs/pass boundaries, not per-batch reads.
Reports separate species/tile inference, EMA consistency, auxiliary/tail work,
preparation, transfer wait/device time, fitting, optimizer and total wall time.
Overlapped spans are not additive. Memory distinguishes parameters/EMA/gradients,
Torch allocation, CuPy pools, host RAM, staging and auxiliary/source workspaces.
Report selected-action error, species-consistency error and conditional placement
gap, not a derived species-versus-sampled-return loss or memory saturation.

Bounded controls compare cold/warmed sparse/mixed/crowded boards at
128/64/32/16/4/1 live slots in fixed 128-slot allocations. Distinguish real-window
latency and probe drain time for zero/one/simultaneous triggers; include bounded
collect → four-pass-fit → collect and a default-horizon probe. Record logical/
padded/requested work, auxiliary queue/backpressure/age, stop reasons, graph
fallbacks, workspace and both-stage costs. `rollout_enqueue_seconds` is host enqueue
overhead, not GPU wall time. Historical one-network benchmarks do not measure the
new architecture. Results belong in [iteration history](../iteration.md) only
after execution; short checks establish neither mastery nor performance guarantees.
