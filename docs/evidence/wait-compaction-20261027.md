# Wait-compaction fitting control — 2026-10-27

One bounded synthetic control exercised production `CompleteGameBuffer`,
`FittingSpool`, `BatchPrefetch` and `flat_q.batch_loss` on the NVIDIA GeForce
RTX 4070 Laptop GPU with PyTorch 2.8.0+cu128. No gameplay or formal training ran.

Both controls used 2,048 original rows with eight public entities per board,
257-row storage blocks, eager fitting, current configured transport and encoder
limits, fresh seed-101 weights and Adam state, and four complete fitting commits
with runtime synchronization and EMA updates. A one-row window supplies the
uncompacted comparison; a 100-row window exercises the delivered behavior.
All original stored targets were checked unchanged after spool preparation and
after fitting. Returns and delayed credits remain outside fitting compaction.

| Workload | Window | Raw / fitting rows | Spool preparation | Four fitting passes | Peak PyTorch allocation |
|---|---:|---:|---:|---:|---:|
| All wait | 1 | 2,048 / 2,048 | 35.97 ms | 880.47 ms | 424.47 MiB |
| All wait | 100 | 2,048 / 21 | 1.83 ms | 67.00 ms | 69.25 MiB |
| Mixed routes | 1 | 2,048 / 2,048 | 30.41 ms | 926.65 ms | 1,431.32 MiB |
| Mixed routes | 100 | 2,048 / 530 | 8.79 ms | 563.78 ms | 1,432.55 MiB |

The all-wait input is one uninterrupted run: 21 windows retain the first row
of each window, including the final partial window. Mixed input has four
512-row runs: genuine wait, species-invalid, tile-invalid and accepted non-wait.
Compaction retains six waits, six species-invalid rows, six tile-invalid rows
and all 512 accepted rows. Its online species/tile/EMA row counts are 524/518/512.
Invalid routes retain their separate head-specific targets.

These are single, ordered, non-warmed controls; the uncompacted case ran first
for each workload. Timing includes transport, forward/backward, clipping, Adam,
runtime synchronization and EMA, with CUDA synchronization at the outer timing
boundaries only. It excludes collection, return calculation, checkpoint saving
and viewer overhead. The allocation metric excludes simulator/CuPy memory.
The control establishes bounded fitting availability and records an observed
reduction in work; it does not establish a full-cycle speedup, a 100-fold speedup
or improved convergence. Representative-count weighting intentionally changes
the contribution of long wait runs. Temporary cohorts were deleted on exit.
