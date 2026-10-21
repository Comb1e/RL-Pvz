# RTX 4070 pipeline implementation and bounded measurements

The structural optimization is implemented. **No execution profile is promoted**:
the baseline timed out in pass four, there are no two completed paired
comparisons, and whole-device memory samples fail the required headroom.
The earlier 8.12 → 4.57 s replay remains a component result.

## Conditions and provenance

- RTX 4070 Laptop GPU, 8,187.5 MiB, driver 616.56, Windows, PyTorch 2.8.0+cu128.
- Both main trials start fresh from the weights in
  `runs/compact-stages-101/easy/interrupted.zip`, SHA256
  `c964fa8e7f4773a98c780b0e4b398140c3bbe28304bafb915d67b487da74a8ac`.
  Historical optimizer/simulator state is never resumed.
- Seed 101, 128-game initial/easy cohort, production exploration, four fitting
  passes, viewer disabled, two threads, transport grouping two. Both compilation
  switches are disabled. The baseline uses the corresponding retired switch.
- Baseline: `c21fe01` plus its benchmark device-literal correction. Candidate:
  `8235da0` plus historical execution-metadata loading and stage-dispatch fixes.
  Later identity fast paths, bounded telemetry and diagnostic refinements are
  covered by short controls and owning tests, not another full cycle.
- Final implementation revision: `235cead`; documentation/evidence are committed
  separately. Focused owning checks, Ruff lint/format and whitespace checks pass.
- Performance-trial wall window: **57.75 minutes**. Detailed profiler traces are
  excluded. Brief correctness checks overlapped portions of the trials; these
  measurements are insufficient for a rigorous paired speedup claim.

[Retained JSON](training-pipeline-optimization-20261021.json) includes exact
effective configurations, source notes, initial-weight identity, phase durations,
raw workload counts and periodic hardware summaries. Temporary cohorts and the
baseline worktree were removed. Automatic approval review rejected scratch-log
deletion as "blocked by policy"; those external JSON/JSONL files remain.

## Main observations

| Measurement | Baseline 64/64, 16,384 tokens | Intermediate candidate 128/128, 32,768 |
|---|---:|---:|
| Collected active decisions | 2,290,461 | 2,290,461 |
| Public entity rows | 39,500,521 | 39,500,521 |
| Physical metadata rows | 4,058,368 | 2,290,461 |
| Collection | 486.01 s | 561.34 s |
| Tail, ≤8 active games | 107.53 s | 130.49 s |
| Fitting pass 1 | 222.77 s | 115.45 s |
| Fitting pass 2 | 214.53 s | 110.73 s |
| Fitting pass 3 | 255.43 s | 121.13 s |
| Fitting pass 4 | incomplete | 110.02 s |
| Complete collect → fit → next decision | unavailable | 1,022.94 s |
| Sampled peak process RSS | 7,453 MiB | 6,402 MiB |
| Sampled peak whole-device memory | 7,938 MiB | 7,923 MiB |

The candidate completes four optimizer commits and the next collection decision.
It stores 43.6% fewer metadata rows while retaining identical decision/entity
counts. Its four passes take 457.32 s together, but collection is 15.5% slower
in this intermediate implementation. Subsequent review removed repeated
per-slot NumPy identity searches and bounded the cohort-long CUDA event queue;
their full-cycle effect has not been measured.

Candidate fitting retains 1,783,532 representatives, including 1,395,574 of
1,902,503 invalid rows (26.65% invalid-row reduction). All 2,290,461 executed rows
still contribute to returns. Four-pass host preparation totals 6.71 s and copy
stream time 0.80 s; those spans overlap fitting and must not be summed into a
cycle estimate. Whole-cycle time is measured directly.

The required free VRAM is 1,637.5 MiB. Sampled remaining memory is only about
250–265 MiB in the main trials, so the promotion gate fails independently of
timing. Windows allocator peaks can exceed physical VRAM through paging; the
dense controls demonstrate pressure, not an acceptable throughput profile.
Future benchmark reports use sampled whole-device memory together with the
allocator peak reserve instead of checking free memory only after cleanup.

## Final controls and correctness

- A 128-game crowded-board cutoff control completed four commits. An
  accepted-action control completed four commits and returned to collection.
  Accepted-action and rejection-heavy controls also ran with four active games.
- A **32 MiB storage-budget override** forced spill and eager transport fallback
  through the same production components; four commits completed. This is a
  pressure/correctness control, not a speed comparison. The override is recorded
  separately from ordinary production configuration.
- A final 40-second production-collection rehearsal exercised bounded telemetry
  and actual execution flags; its intentional timeout is marked incomplete.
- The first short control exposed a stale timeout diagnostic after different
  slots finished on different steps. The corrected control reads retained
  terminal records and passes without weakening the cutoff-loss assertions.
- Owning checks cover active identities and exact compaction boundaries,
  RAM/disk restore and stale credits, prepared alignment, loss/gradient controls,
  graph ownership/fallbacks, selective forward counts, interrupted/OOM pass
  restart, demo shared forwards, annotation convergence/recovery, and model
  serialization/replay. No training-parameter-constant tests were added.
- Existing reward controls had hard-coded the previous scale. They now calculate
  the independent physical conversion from configuration. The old CUDA parity
  failure also reproduces on the baseline: initial plants have no CPU placement
  origin, whereas CUDA reports aggregate event economics. The aggregate parity
  control now compares the same accounting level; causal ownership retains its
  separate tests. Reward implementation and objective identities are unchanged.

ABBA repetition, final full-cycle timings, independent thread/group tuning and
collection-graph calibration remain unverified within this measurement budget.
The 4070 keeps 64/64, 16,384-token, grouping-two defaults. Model architecture,
observation encoding, reward economics and weight layout remain unchanged;
only trajectory/autonomous recovery advance to v4, with no old-state migration.
