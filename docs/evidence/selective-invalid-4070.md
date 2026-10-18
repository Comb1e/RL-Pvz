# Selective fitting local rehearsal — 2026-10-18

Device: NVIDIA GeForce RTX 4070 Laptop GPU. PyTorch: 2.8.0+cu128.
The full architecture was retained. This is a small execution/correctness rehearsal, not a throughput preset or learning-quality comparison. Each width has one row; fitting uses 16-row species/tile microbatches, token budget 4096, grouping 1, two CPU threads and eager kernels. Each repetition commits four complete passes; three warmed repetitions follow one cold repetition.

The existing tuner shared workload completed 24 cases (four width families and six action mixtures). Raw evidence is retained locally at `artifacts/selective-invalid-4070/benchmark.json`.

Cold total: 12.433 s. Three warmed repetitions across all cases: 34.737 s. Maximum allocated CUDA memory: 274.73 MiB.

| Width family | Action mix | Rows | Online species | Online tile | EMA tile | Skipped EMA | Median four-pass seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| sparse | wait | 17 | 17 | 0 | 0 | 0 | 0.2533 |
| sparse | mixed | 17 | 17 | 2 | 2 | 0 | 0.5470 |
| sparse | tile | 17 | 17 | 17 | 17 | 0 | 0.5600 |
| sparse | species_invalid | 17 | 17 | 0 | 0 | 17 | 0.2147 |
| sparse | tile_invalid | 17 | 0 | 17 | 0 | 17 | 0.2818 |
| sparse | rejected_heavy | 17 | 10 | 10 | 3 | 14 | 0.9887 |
| medium | wait | 12 | 12 | 0 | 0 | 0 | 0.3072 |
| medium | mixed | 12 | 12 | 1 | 1 | 0 | 0.6701 |
| medium | tile | 12 | 12 | 12 | 12 | 0 | 1.1115 |
| medium | species_invalid | 12 | 12 | 0 | 0 | 12 | 0.3686 |
| medium | tile_invalid | 12 | 0 | 12 | 0 | 12 | 0.3836 |
| medium | rejected_heavy | 12 | 7 | 7 | 2 | 10 | 1.0082 |
| dense | wait | 10 | 10 | 0 | 0 | 0 | 0.1178 |
| dense | mixed | 10 | 10 | 1 | 1 | 0 | 0.3288 |
| dense | tile | 10 | 10 | 10 | 10 | 0 | 0.2163 |
| dense | species_invalid | 10 | 10 | 0 | 0 | 10 | 0.1224 |
| dense | tile_invalid | 10 | 0 | 10 | 0 | 10 | 0.1378 |
| dense | rejected_heavy | 10 | 6 | 6 | 2 | 8 | 0.3864 |
| boundary | wait | 11 | 11 | 0 | 0 | 0 | 0.2670 |
| boundary | mixed | 11 | 11 | 1 | 1 | 0 | 0.4984 |
| boundary | tile | 11 | 11 | 11 | 11 | 0 | 0.8189 |
| boundary | species_invalid | 11 | 11 | 0 | 0 | 11 | 0.3219 |
| boundary | tile_invalid | 11 | 0 | 11 | 0 | 11 | 0.4454 |
| boundary | rejected_heavy | 11 | 7 | 6 | 2 | 9 | 1.2077 |

Counts are logical rows per pass; encoder cache statistics retain padding/capture details. The benchmark validates exact skipping in rejected-only workloads. Times do not establish a production speedup; route fragmentation and tiny batches can dominate this rehearsal.

A separate generated reachable annotation record started at 0/1 training hits and converged at 1/1 after seven committed passes. The saved BF16 runtime was reloaded and verified, then all 151 FP32 weight tensors loaded on CPU through the fresh initialization path. Report/history remain at `artifacts/selective-invalid-4070/generated-init.report.json` and `.history.jsonl`. No held-out records or game-performance study were present, so those results are undefined.

A read-only check of the user dataset found three contradictory wall-nut pairs, at JSONL lines 29/51, 34/56 and 36/58. Identical encoded model inputs have disjoint acceptable sets; 100% deterministic training accuracy is impossible until those labels are reconciled or a record is explicitly discarded. The fitter reports all six record IDs before fitting. Details remain in `artifacts/selective-invalid-4070/annotation-conflicts.json`; original `data/` records were not modified.

The cloud GPU gate and equal-game learning comparisons were not run.
