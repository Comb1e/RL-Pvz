# Individual entity inputs and attention

The `entity_v1` schema stores an unpadded int32 matrix `[n,11]` and float32
globals `[18]`. Here n is the number **retained after the configured cap**.
Before capping, there is one row for every present physical entity, including
headless zombies, spent mowers and duplicate records. The cap is a deliberate
information loss, not an attention approximation.

| Index | Field | Normalization at model input |
|---|---|---|
| 0 | type | Learned categorical lookup; plants 1–8, zombies 9–13, pea/icy projectile 14/15, mower 16 |
| 1 | state | Learned namespaced public-state lookup; 0 for projectiles |
| 2 | row | Lane / maximum lane index (4) |
| 3 | x | `(x - house_x) / (spawn_x - house_x)` |
| 4 | health | Health / pinned species maximum health |
| 5 | armor | Armor / maximum pinned armor (1,100) |
| 6 | phase_ticks | Public behavior countdown / tick rate (100) |
| 7 | slow_ticks | Public remaining slowing duration / tick rate |
| 8 | has_pole | 0 or 1 |
| 9 | headless | 0 or 1 |
| 10 | damage | Projectile damage / normal-pea damage (20) |

Plant x is its tile center (`col * 1000 + 500` under the pin). Occupancy recovers
the column using integer division by the pinned units per tile. Inapplicable
raw fields are zero; type identifies their meaning. Values are not clipped,
rounded or regionally aggregated. Unknown categories and invalid type/state
combinations fail explicitly. Fields remain integers on disk; normalization
occurs in the shared model input path.

Globals retain sun / largest card cost, elapsed / cutoff, current/total waves /
wave scale, initial/defeated zombies / count scale, and the eight card cooldowns
in plant order as `cooldown_ticks / (recharge_ticks + 1)`. Four present-list
counts (plants, zombies, projectiles, mowers) follow, each / count scale (75).
Counts include entities omitted by the cap and preserve multiplicity.
For a card, ready is exactly zero and an instantaneous placement sets the
normalized cooldown to one. After one wait it is
`recharge_ticks / (recharge_ticks + 1)`, preserving the pinned countdown boundary.

## Selection and information boundary

Canonical ordering is **mowers → plants → zombies → projectiles**. Mowers and
plants sort by lane then horizontal position; zombies and projectiles sort by
increasing x, nearest the house first. The remaining raw public fields break
ties. Taking the first `max_entities` rows therefore omits projectiles first,
then zombies, then plants. Five mower records are always retained. Equivalent
duplicate records are interchangeable but remain separate rows below the cap.
No ID, private allocation slot, seed, spawn schedule, RNG or movement phase is
used for sorting or encoding.

A counterexample to the former aggregation is two same-species zombies at x=1000
and x=2000 with health 100/200 versus 200/100. Count, total health and nearest x
are identical, but the new records differ. The encoder preserves the association
of health with position whenever those zombies are retained. Above the cap,
full counts cannot recover omitted individual attributes.

## Shared embedding and readouts

For the nine normalized numerical fields u and categorical type t and state s:

\[
e_i=\operatorname{LayerNorm}(E_{type}[t_i]+E_{state}[s_i]+Wu_i+b),
\qquad W\in\mathbb{R}^{32\times9}.
\]

This yields `[n,32]`. Width 32 is a configurable compact starting capacity,
not a proven minimum. All parameters learn through complete-return Q regression.
No language model or text tokenizer participates.

A projection of the globals initializes one board readout token. Forty-five
tile queries use normalized lane/tile-center coordinates, the shared numerical
projection and a learned tile marker. Concatenation with physical embeddings
gives `[batch,n_max+46,32]`. The extra 46 tokens are readouts, not entities.
Two bidirectional attention layers (4 heads, head width 8, feed-forward 64,
dropout 0) produce the board summary and tile features.

For each attention head, padded keys receive negative infinity before softmax:

\[
A(Q,K,V)=\operatorname{softmax}(QK^T/\sqrt{8}+M)V.
\]

Every layer masks keys and zeroes padded outputs. All readout tokens remain
valid, so even n=0 has finite outputs. There are no entity-list position
embeddings or causal attention masks. For a permutation matrix P on entity
rows, attention permutes entity outputs while leaving readout outputs unchanged;
the shared pointwise projections, normalization and feed-forward operations
preserve that property. Thus branch Q, tile Q and next LSTM state are invariant
up to floating-point reduction differences. The LSTM enforces time causality.

## Computation and storage bounds

Let L=n+46. Dense attention arithmetic is O(batch × L² × width). Query chunking
splits Q into at most 64 rows, but every chunk attends to all K,V; concatenation
is exactly the full result. The implementation uses PyTorch SDPA, or explicit
scaled scores, mask, softmax and multiplication for its independent fallback.
One fallback query chunk uses O(microbatch × heads × 64 × L) score storage.
Autograd may retain multiple chunks for backward; the allocation fallback and
outer checkpoint bound practical usage. This is not a linear total-memory claim.

Encoder microbatches contain at most 128 frames and at most the configured 65,536
tokens when possible. Optional outer activation checkpointing recomputes these frames
without nested attention checkpoints. Frames are reassembled in original batch/time order before
the LSTM; encoder scheduling cannot change the recurrent objective or the single
optimizer update per whole-cohort fitting pass. Unrecoverable allocation errors
are surfaced; capacity is never silently reduced in response.

Trajectories keep fixed transition metadata/globals plus entity offset/count and
append-only `[total_retained_rows,11]` slabs. Each raw entity costs 44 bytes.
Metadata and entity slabs share the configured host RAM allowance and spill to
disk. Archive recovery checks protocol, dtype, block shape, contiguous offsets,
counts, cap and final entity extent before exposing data to training.

## Independent controls

Tests compare attention values and gradients to an independently constructed
float64 dense formula, including the chunked fallback. They check permutation,
padding, individual/batch and step/sequence agreement, future-input independence,
reset isolation, empty and crowded input finiteness, and CPU/CUDA raw facts.
FP32 invariance controls disable cuDNN TF32 to avoid batch-dependent precision;
float64 optimizer controls avoid Adam amplifying numerical noise in gradients
whose true value is zero (for example softmax key bias).

Schema tests cover all public categories and phases, health-position aliasing,
armor/slowing/pole/headless/projectile facts, duplicate retention, coordinate and
cap boundaries, and excluded private metadata. Ragged storage controls exercise
empty rows, shared budgets, disk spill, archive round trips and corrupt offsets.
These are implementation controls; cap benchmarks do not establish learning quality.

The default cap is 256; its detail/throughput tradeoff and recorded device
measurements are in [iteration history](../iteration.md#0290--2026-09-28).
Execution precision and retry controls are in [training throughput](training-throughput.md).
