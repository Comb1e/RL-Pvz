# Complete-game learning

Current returns, sequential action values, balanced regression and plant-only
exploration are derived and independently controlled in
[sequential-q-control.md](sequential-q-control.md).

The CPU trajectory buffer remains authoritative. A raw token is 289 float32
values, or 1,156 bytes. The 48-entry memory bank references prior raw tokens;
indices are stored as uint32, bit-preserved during collection readback rather
than numerically converted to float32. A 2 GiB GPU cache can hold at most
floor(2×2^30 / 1156) raw tokens before block rounding and the free-VRAM reserve.
It stores no activations and changes neither returns nor sample order.

Full returns accumulate backward separately for each environment, including
instantaneous actions. Paused environments add no transitions. Checkpoints retain
partial games and fitting cursors; a partial game is not converted into a terminal
loss by interruption. The per-game cutoff is a labelled failure, not a bootstrap.

Independent maintained controls cover CPU/disk spill, causal reconstruction,
cache misses, partial batches, transfers and exact continuation after interruption.
Historical actor/critic schedules and their evidence live in iteration records.
