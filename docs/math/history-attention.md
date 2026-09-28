# Full-game history: cost and causal controls

Historical calculations for the [retired memory proposal](../full-game-memory.md).
Current observation attention is specified in [entity inputs](entity-inputs.md).
They are not measured sparse-transformer performance or a training result.

## Dense scores and retained state

Let E be retained records, S tokens per record, N = ES total tokens, H heads,
B batch size, L layers, d embedding width, and b bytes per element. Full-frame
tokenization uses S = 45 + 15 + 1 = 61. Delta tokenization can reduce N, but the
following formulas still apply to the actual token count.

Materialized dense scores take BHN²b bytes per layer. Attention arithmetic scales
as O(BN²d). FlashAttention avoids the materialized scores, not the dense pairwise
arithmetic. Conventional layer K/V storage takes 2BLNdb bytes, excluding inputs,
activations, workspaces, gradients, parameters, optimizer states and masks.

For B = H = 1, b = 2 (hypothetical fp16), L = 2 and d = 128:

| Records | Tokens | Dense scores, one head, GiB | Two-layer K/V, GiB |
|---|---|---|---|
| 48 | 2,928 | 0.01597 | 0.00279 |
| 1,000 | 61,000 | 6.93090 | 0.05817 |
| 120,000 | 7,320,000 | 99,805.0 | 6.98090 |

The last row models one rejected proposal per tick at 100 Hz for 1,200 seconds;
it is not a bound on all decisions because successful zero-tick actions can add
decisions. Four heads multiply scores by four; concurrent environments multiply
K/V residency. Current policy precision is float32, so fp16 numbers are optimistic
illustrations, not a precision change. Waiting with public movement can also
generate an event each tick under the user's retention requirement.

## Sparse edges and an all-prefix counterexample

At most k keys per token yields at most Nk attention edges and O(BNkd) attention
arithmetic. Projections and feed-forward work remain, and block padding adds
edges. This is not a GPU timing prediction.

For causal local width w including self and g terminal queries each reading the
whole prefix, the exact edge count is:

    sum(i=1..N, min(i,w)) + gN <= N(w+g).

This is linear in one prefix evaluation for fixed w and g. A full-history query
after every event requires total historical edges:

    sum(i=1..N, i) = N(N+1)/2.

Thus making every historical event a global readout restores quadratic total
work, even if inference evaluates one query at a time. Reversible layers do not
change this. Exact storage of arbitrary events grows with their count; fixed
GPU residency can use CPU/disk overflow, but fixed storage cannot promise exact
unbounded history without lossless redundancy.

## Independent controls performed

A Python arithmetic probe evaluated the table directly from the byte formulas.
Independent edge-set enumeration for N in {1, 7, 19}, w = 4 and g = 3 agreed with
the sparse formula/bound and triangular all-prefix counterexample.

A separate causality counterexample used decisions at ticks [0, 0, 1]. Tick-only
masking allowed exactly one future edge, from the first decision to the second.
Decision-order masking allowed none. Same-observation spatial tokens may still
attend bidirectionally. Outcomes must be available before a query can read them.

A CPU probe of the actual EventMemory used 2,000 synthetic observations, the
shipped 8/32/8 capacities and summary stride 100:

| Pattern | Valid entries | Earliest raw tick | Represented count |
|---|---|---|---|
| Unchanged state and wait marker | 17 | 0 | 800 |
| Non-wait action every tick | 48 | 1,299 | 800 |
| Public plant-health change every tick | 48 | 1,299 | 800 |

The initial quiet-state event survives because quiet frames do not push its event
queue. Continuous admission loses its original raw state. Summary start/count
metadata can describe an earlier interval but cannot reconstruct all its frames.
These synthetic memory controls are not simulated games or evidence of learned
timing. Collector/environment source separately confirms that rejected planting
proposals become executed wait markers before reaching policy memory.
