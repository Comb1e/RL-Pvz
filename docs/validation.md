# Availability and verification — 0.3.1

## CUDA default — 2026-09-20

PyTorch `2.8.0+cu128` detects the NVIDIA GeForce RTX 4070 Laptop GPU. Research
package 0.3.1 is installed; the pinned game package remains 1.2.0.

| Check | Result |
|---|---|
| Complete research regression/integration suite | **87 passed in 59.47 seconds**, including all five CPU learning conditions and Windows/CUDA workers |
| Complete upstream game suite | **199 passed in 9.43 seconds**; bytecode and pytest cache writes disabled for the game checkout |
| CUDA default smoke | CLI run without `--device`: 64 diagnostic decisions, two spawned workers, four optimization epochs, complete status, checkpoint, report, and compact demo |
| Actual GPU execution | Saved policy tensors and Adam moment tensors loaded without device remapping retain CUDA device tags; checkpoint reload places all policy parameters on CUDA |
| Missing CUDA | Simulated unavailable GPU raises the existing `--device cpu` guidance before creating an output directory |
| Tooling | Ruff lint/format, pip dependency check, editable package installation, and Git whitespace checks passed |

Evidence: `artifacts/cuda-research-tests.txt`, `artifacts/cuda-engine-tests.txt`,
`artifacts/cuda-default-smoke-console.txt`, and `artifacts/cuda-verification.json`.
The saved run is `artifacts/cuda-default-smoke`; its offline report is
`visualizations/index.html`. No formal research training was launched, and this
diagnostic run does not establish full-game competence or a GPU speed advantage.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -B -m pytest -q E:/Projects/pvz/tests `
  -p no:cacheprovider --basetemp artifacts/cuda-engine-test-temp
.\.venv\Scripts\python.exe -m pvz_rl train `
  --condition masked --family diagnostic --seed 101 --steps 64 --n-envs 2 `
  --rollout-size 64 --batch-size 32 --eval-interval 64 --validation-count 1 `
  --output artifacts/cuda-default-smoke
```

The game checkout was clean after verification. Existing CPU checkpoints retain
their original device setting and require `--device cpu` when resuming.

## Previous 0.3.0 verification

Date: 2026-09-20. Windows, Python 3.12, PyTorch 2.8.0+cu128,
Gymnasium 1.2.3, SB3/SB3-Contrib 2.7.1. Game package 1.2.0, simulation 1.0.0,
commit `a47056d8141ec635d3ff3f4d5561d6a75cfca2cc`.

## Current migration checks

The research package and the installed game were tested together, including the
complete upstream game suite. Availability/smoke tests use short learning budgets;
no formal training or final-test performance study was launched.

| Check | Evidence |
|---|---|
| Research regression/integration suite | **87 passed in 66.19 seconds** in the final complete pass |
| New game regression suite | **199 passed in 9.61 seconds** in the final complete pass, using the installed 1.2.0 package and read-only access to upstream tests |
| Native game installation | Non-editable installation from a verified Git archive staged in research `build/game-1.2.0`; engine source manifest verified |
| Package/simulation identity | Separate checks for package 1.2.0 and simulation 1.0.0, source hashes, and unchanged rules hash |
| Known gameplay controls | Existing easy/standard/hard success and failure cases retain expected outcomes/ticks |
| Learning | All five conditions, CPU and two-worker Windows/CUDA, checkpoint serialization, and same-pin resume |
| Default recording | Fresh subprocess blocks pygame imports and uses an absent FFmpeg path; training still generates all three compact demos and the HTML report |
| Shared model | Identical checkpoint hash in all easy/standard/hard demos; output-enabled and disabled short runs retain identical learned tensors |
| Compatibility boundaries | Old source-pin checkpoints rejected before deserialization; archived reports/recordings processed without model loading |
| Recordings | `.pvzdemo`, `.json.gz`, and `.json` equivalence; embedded provenance; metadata-independent observations/masks/hashes; legacy sidecar fallback |
| Native viewer | Mid-entry and backward seeks, completion/rewind, pause, single-step, speed, restart, and inspection exercised by game tests; smoke viewer rendered and inspected |
| Optional videos | Native packed RGB bytes, configurable even dimensions, H.264/yuv420p MP4; natural outcomes and truncation preserved |
| Errors | Missing encoder, failed encoder process, replay corruption, altered metadata checksums, and rendering/export recovery |
| Browser playback | Microsoft Edge played all three locally exported 1280×820 videos at 2×, sought to the middle, and reported no media errors |
| Build/install tooling | Ruff lint/format, pip dependency check, PowerShell installer syntax, wheel/source distribution build, and packaged manifest checks passed |

The final complete pass contains **286 passing tests**, with no failures. Logs:
`artifacts/pvz12-final-research-tests.txt`, `artifacts/pvz12-final-engine-tests.txt`,
and `artifacts/pvz12-final-build.txt`. Doctor output is
`artifacts/pvz12-availability.json`; browser playback evidence is
`artifacts/pvz12-browser-playback.json`. The rendered report/video and native viewer
were visually inspected. `git -C E:/Projects/pvz status --porcelain=v1` stayed empty.

Commands used for the final complete test pass:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -B -m pytest -q E:/Projects/pvz/tests `
  -p no:cacheprovider --basetemp artifacts/pvz12-final-engine-test-temp
.\.venv\Scripts\ruff.exe check src tests tools
.\.venv\Scripts\ruff.exe format src tests tools --check
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build --no-isolation
```

## Saved fresh-run evidence

`artifacts/pvz-1.2-shared-smoke` contains a fresh 128-decision CPU run with one worker
and one validation seed per difficulty. Its shared selected checkpoint SHA-256 is
`7f5e83e60f5ec1da3da45f9e7142cd2ed89327f9618582f92bde03addb3e4cd4`.
Default training created the report and three compact demos, with no video directory.
Optional video export was requested separately after confirming this default behavior.

| Difficulty / seed | Observed outcome | Compact bytes | Final simulation hash |
|---|---|---:|---|
| Easy / 100000 | Won; all five mowers used | 6616 | `571a1d13a203ec1af454f4dc3105acf3b559995e3503dfee09b268c6f19c1970` |
| Standard / 100000 | Lost | 7205 | `f3ea524b87f2a2fac241cf3e10087f7b906d1eac49e0a236496a8ea27e9feb3e` |
| Hard / 100000 | Lost | 7492 | `232526650f507859a242a24a243724a2db15258ce37529852acc58a804934aa2` |

These predetermined demonstrations are compatibility checks, not estimates of
win rate. Their final simulation hashes reproduce the previous version's smoke
games. The native seekable viewer screenshot is `artifacts/pvz-1.2-native-viewer.png`.

## Historical verification before the upgrade

The following 0.2.0 evidence used the previous game source pin and is retained as
historical context. It must not be pooled with current-pin experiment results.

## Availability and verification — 0.2.0

Date: 2026-09-20. Verified locally on Windows with Python 3.12, an Intel
Core i9-14900HX, approximately 32 GB RAM, and an RTX 4070 Laptop GPU (8 GB).
Installed learning libraries: PyTorch 2.8.0+cu128, Gymnasium 1.2.3,
Stable-Baselines3 2.7.1, and SB3-Contrib 2.7.1.

## Executed checks

| Check | Result |
|---|---|
| Automated research suite | **67 passed in 50.92 s** in the final full run, including the original 56 regression checks and new visualization/logging/recovery checks |
| Upstream game regression suite | **87 passed in 2.84 s**; run with bytecode and pytest cache writes disabled, and temporary output in the research project |
| Gymnasium API checker | All five conditions passed |
| CUDA availability | GPU identified and a CUDA tensor computation passed |
| Short real learning | All five conditions completed 128 decisions in integration tests |
| Windows subprocess execution | Two-worker 64-decision CUDA training run completed |
| Checkpoint serialization | Predictions agree after save/load; parameters are finite |
| Interruption and recovery | Synthetic interruption after first PPO update resumes to the budget and preserves the earlier tied best checkpoint |
| End-to-end orchestration | Five-condition miniature protocol completed training, baseline/OOD evaluation, verified replays, statistics, and report; completed-suite resume is idempotent |
| Benchmark code availability | Separate 64-decision CPU and CUDA trials completed; these trials are too short to select formal-study hardware |
| Engine identity | Installed source manifest and rules digest match the pinned game |
| Dependencies | `pip check`: no broken requirements |
| Lint | `ruff check src tests tools`: passed |
| Packaging | Wheel and source distribution built successfully, including configuration and engine manifest |
| Installer | PowerShell parser accepted `tools/bootstrap.ps1`; the already installed environment was used for the checks |
| Report and replay CLI | Development evaluation, report generation, and replay verification completed |
| Rendering/report layout | Off-screen frame shape and simulation purity checked; generated multi-panel report figure visually inspected |
| Shared policy | One policy/optimizer survives curriculum changes; one load of `best.zip` serves all three demos with identical hashes |
| Learning unaffected by output | Short runs with visualization disabled/enabled produce identical learned tensors and deterministic evaluation actions |
| Training metrics | Post-update metrics captured at steps 64 and 128, including the final update; TensorBoard retained |
| Operational logs | Timestamped console/file output agrees; fake-clock test checks the 15-second throttle without episode spam |
| Report compatibility | Empty/old runs render; resume boundaries exclude overlapping ancestor points; presentation-only settings can change on resume |
| Failure recovery | Interrupted training report, completion-only recovery, absent FFmpeg, encoder process failure, and corrupt replay covered |
| Full-game video availability | One 128-decision smoke checkpoint exported easy/standard/hard games at 1000×600, 20 fps, with verified final hashes |
| Browser playback | Headless Microsoft Edge loaded all three local MP4s, played each at 2×, sought successfully, and reported no media errors |
| Regeneration | `visualize --run` rebuilt the report and reused verified videos without additional training |
| Replay export CLI | `replay --video` verified and encoded a controlled winning replay; ffprobe confirmed H.264, yuv420p, 1000×600, 20 fps |

The final complete test pass contained **154 passing tests**, with no failures.
Raw outputs are retained locally in `artifacts/final-research-tests.txt` and
`artifacts/final-engine-tests.txt`; the final package build log is
`artifacts/final-build.txt`. Lint, formatter checks, dependency checks, and Git
whitespace checks passed. `git -C E:/Projects/pvz status --short` remained empty.

Commands used for the final full automated pass:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -B -m pytest -q E:/Projects/pvz/tests `
  -p no:cacheprovider --basetemp artifacts/engine-test-temp
.\.venv\Scripts\ruff.exe check src tests tools
.\.venv\Scripts\ruff.exe format src tests --check
.\.venv\Scripts\python.exe -m build --no-isolation
.\.venv\Scripts\python.exe -m pip check
```

## Saved visualization smoke artifacts

The local run is `artifacts/shared-policy-visual-smoke`: 128 decisions, learner seed
101, one CPU worker, and one validation seed per difficulty. It is an availability
check, not a trained-policy benchmark. Its first 128 decisions did not complete a
training episode, so episode-statistic panels correctly show that no completed
training episodes are available. Optimizer and validation panels contain measured
data; no synthetic curves are inserted.

All three full-game demos use validation seed 100000 and checkpoint SHA-256
`b418f18023bdec8218ba5c9abc77d6ebab0dc5ec3418e89b35280fa1706a0eae`:

| Difficulty | Observed demo outcome | MP4 duration, including final hold |
|---|---|---:|
| Easy | Won; used all five mowers | 193.85 s |
| Standard | Lost | 227.55 s |
| Hard | Lost | 241.55 s |

One winning validation case after a tiny update budget is not evidence of learned
competence. These outcomes are reported to verify that successful and unsuccessful
replays display their real results using the same selected weights.

Local artifacts (ignored by Git) include the offline report under the run's
`visualizations/index.html`, browser playback checks in
`artifacts/browser-playback.json`, a report screenshot in
`artifacts/report-preview.png`, and a rendered replay frame in
`artifacts/replay-preview.png`. `artifacts/availability-v0.2.0.json` records the
engine, CUDA, renderer, and installed FFmpeg 8.1 checks.

The infinite Gymnasium `Box` bounds generate advisory checker warnings. They are
intentional: custom crowds can exceed the normalization scales, and the encoder
preserves counts rather than silently clipping them. Encoded observations in tests
are finite and satisfy their declared space.

## Independent controls and counterexamples

- Every one of the 406 indices is checked against an independently constructed
  action formula, including all board corners.
- Masks are checked against engine validation with exact cost, one sun short,
  occupied tiles, dig legality, cooldown completion, and unavailable hybrid strategies.
- A 10-tick action is compared with one single-tick action and nine waits using
  complete engine hashes.
- Win, loss, external truncation, final partial tick batches, reset, and vector
  wrapper terminal-observation handling are checked separately.
- Two games with different hidden future schedules, names, and seeds have identical
  encoded initial observations when their public states match.
- Entity permutation and changed IDs preserve encoding. A 120-zombie crowd retains
  its full count despite exceeding the usual 75-zombie scale.
- Discounted potential terms are independently summed and compared with the
  telescoping formula. Plant/dig cycles do not produce free cumulative reward.
- A concurrent spawn and defeat is counted using the explicit defeat counter,
  not the change in living population.
- Changed-wave families preserve their specified opening waves, rosters, sizes,
  or spawn timing; curriculum boundary decisions are tested exactly.
- Bootstrap tests use independently known means, identical-policy zero differences,
  and failures for duplicate, missing, unpaired, or incompatible data.

## Frozen heuristic regression cases

These are known development cases, not held-out research estimates.

| Level | Seed | Expected and reproduced outcome | Final tick |
|---|---:|---|---:|
| Easy | 0 | Won | 3146 |
| Standard | 0 | Won | 6398 |
| Standard | 1 | Lost | 5395 |
| Hard | 0 | Lost | 6219 |
| Hard | 4 | Won | 9401 |
| Hard | 6 | Lost | 4463 |
| Easy | 42 | Won | 3221 |
| Standard | 42 | Won | 6105 |
| Hard | 42 | Won | 8526 |

The separate diagnostic task also has a tested hand-chosen winning peashooter
placement with no mowers. This confirms task solvability without relying on learned
behavior as the correctness reference.

The new video tests independently construct an empty winning scenario, an immediate
house-threat losing scenario, and a future-spawn scenario with an external cutoff.
Corrupting a final replay hash or forcing an encoder process failure must preserve
any previous successful output and remove temporary video files. Missing video
dependencies must leave completed training checkpoints and the report intact.

## What has not been run

The user requested code and training instructions, allowing availability tests.
Consequently, no full 25-run experiment, meaningful performance benchmark, or
formal trained-agent test study was launched. Tiny integration runs use deliberately
short cutoffs and reduced seed sets; their statistics do not measure gameplay skill.
No running background training job is needed to use the delivered code.

The README provides explicit commands for pilots, benchmarking, the formal suite,
held-out evaluation, and report generation. A fresh CPU-only installation and
cross-platform operation have not been exercised; CPU execution was tested using
the installed CUDA-capable PyTorch distribution.
