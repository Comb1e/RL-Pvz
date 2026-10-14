# Individual entity inputs and spatial attention

The unchanged `entity_v1` schema stores an unpadded int32 `[n,11]` entity matrix
and float32 `[18]` globals. Here n is the retained count after the configured
256-entity cap. Present physical records include headless zombies, spent mowers
and duplicates. The cap loses information; it is not an attention approximation.

| Index | Field | Model-input normalization |
|---|---|---|
| 0 | type | Learned lookup: plants 1–8, zombies 9–13, pea/icy projectile 14/15, mower 16 |
| 1 | state | Learned namespaced public-state lookup; projectiles use 0 |
| 2 | row | Lane / maximum lane index (4) |
| 3 | x | `(x - house_x) / (spawn_x - house_x)` |
| 4 | health | Health / pinned species maximum health |
| 5 | armor | Armor / maximum pinned armor (1,100) |
| 6 | phase_ticks | Public behavior countdown / tick rate (100) |
| 7 | slow_ticks | Public remaining slowing duration / tick rate |
| 8 | has_pole | 0 or 1 |
| 9 | headless | 0 or 1 |
| 10 | damage | Projectile damage / normal-pea damage (20) |

Plant x is its tile center (`col * 1000 + 500` under the pin). Occupancy uses
integer division by pinned units per tile. Inapplicable raw fields are zero.
Numerical values are not clipped or regionally aggregated. Unknown categories
and invalid type/state combinations fail. Integers stay raw on disk; shared
normalization occurs at model input.

Globals contain sun / largest card cost, elapsed / cutoff, current/total waves /
wave scale, initial/defeated zombies / count scale, eight plant cooldowns as
`cooldown_ticks / (recharge_ticks + 1)`, and four present-list counts
(plants/zombies/projectiles/mowers) / 75. Counts include omitted entities.
A ready cooldown is zero; instantaneous placement sets it to one, and one wait
yields `recharge_ticks / (recharge_ticks + 1)`.

## Information boundary

Canonical order is mowers → plants → zombies → projectiles. Mowers/plants sort
by lane/position; zombies/projectiles sort by x, nearest the house first. Raw
public fields break ties. Overflow drops projectiles first, then zombies, then
plants; five mower records remain. Duplicates retain multiplicity below the cap.
No private ID, allocation slot, seed, schedule, RNG or movement phase enters
sorting/encoding. Two zombies with swapped health at distinct positions remain
distinguishable below the cap even if aggregate counts/health match.

Both networks receive current elapsed time, cooldowns and entity countdowns.
There are no frame stacks, historical events, temporal attention or learned
memory. Identical current inputs and requested branches give identical outputs,
independent of earlier gameplay, within fixed execution/precision tolerances.

## Independent embeddings and readouts

Each encoder uses its own categorical embeddings and numerical projection:

\[
e_i=\operatorname{LayerNorm}
(E_{type}[t_i]+E_{state}[s_i]+Wu_i+b).
\]

The nine numerical fields and shared normalization semantics are the same, but
learned parameters are independent. Width is 128 in species and 192 in tiles.

| Stage | Tokens in addition to entities | Layers / heads / feed-forward | Q head |
|---|---|---|---|
| Species | One projected global readout | 4 / 8 / 512 | Global readout, 128 → 128 → 10 |
| Conditional tile | Global readout, selected branch token, 45 coordinate queries | 6 / 12 / 768 | Shared per tile, 192 → 128 → 1 |

Tile queries encode normalized lane/tile-center coordinates, not entity-list
positions. The tile network receives the selected plant/dig branch, not species
features or scores. Wait skips this stage. Neither network has a scalar bypass
or post-attention history fusion. Both use pre-normalized blocks, dropout zero
and 16 features per attention head.

Each head computes
\(A(Q,K,V)=\operatorname{softmax}(QK^T/\sqrt{16}+M)V\).
Padded keys receive negative infinity; padded outputs are zeroed each layer.
Readout/query tokens remain valid even for empty boards. Bidirectional spatial
attention has no causal mask or entity-list position embeddings. Permuting
entities permutes their outputs without changing readouts, up to floating-point
reduction differences. Current position/health/resources/cooldowns can influence
both stages without any history update.

## Work and storage bounds

Token length is \(L=n+1\) for species and \(L=n+47\) for tiles. Dense attention
arithmetic is \(O(batch\,L^2\,width)\). SDPA uses its supported efficient backend;
the exact fallback chunks queries but attends to every valid key/value.
One 64-query chunk uses \(O(microbatch\,heads\,64\,L)\) score storage; backward
may retain multiple chunks, so this is not a linear total-memory claim.

Encoder microbatches have at most 128 boards and, when possible, 65,536 tokens.
Active packing uses each stage's overhead, not one shared hardcoded token count.
Optional outer checkpointing recomputes activations. Allocation retries reduce
work size, never entity detail, action coverage or learning denominators.
Exhausted retries fail explicitly.

Trajectories keep fixed metadata/globals, offsets/counts and append-only raw
entity slabs. Each raw entity costs 44 bytes. Slabs/staging share RAM allowance
and spill to disk. Recovery checks protocol, dtype, shapes, offsets, counts,
cap and final slab extent before exposing data to training.

## Independent controls

Owning model controls compare both stages' values/gradients with independent
serial/dense attention, including fallback, permutation, padding and batched
requests. They verify required dimensions, branch conditioning, duplicate-request
reuse, statelessness, current-state sensitivity and empty/crowded finiteness.
FP64 mathematical controls avoid precision artifacts; FP32 comparisons account
for reduction/TF32 differences rather than weakening schema/attribution checks.
Observation/storage suites own raw CPU/CUDA facts, all public phases, multiplicity,
cap/private-state boundaries, ragged spill and corruption.

Historical cap measurements remain in [iteration history](../iteration.md);
they do not measure this two-network architecture. See
[execution controls](training-throughput.md) for precision, transport and retries.
