# Availability and verification — 0.7.1

## Replay, optimization and plant-role rewards — 2026-09-21

The research simulator remains **1.3.0**, pinned at `8861824`, with unchanged rules.
Native reader patches are CPU-game **1.2.2** / `a95524e` and CUDA-game **1.3.1** /
`314528a`. They do not replace the training installation. Existing run checkpoints
and recordings are preserved.

| Check | Result |
|---|---|
| Final complete research regression | **374 passed, 2 warnings in 271.86 s** |
| Complete CPU-game regression | **207 passed in 11.83 s** |
| Complete CUDA-game regression | **222 passed in 41.48 s** |
| Reward/kill/progress focused checks | **18 + 18 passed** in separate, overlapping runs |
| CUDA evaluation/learning focused checks | **24 passed, 1 warning in 34.49 s** |
| Final report-only change | **6 passed, 5 deselected in 44.99 s** |
| Lint, formatting, dependency check | Passed |
| Wheel/sdist and editable research install | Built and installed **0.7.1** |
| Native demos | Three original recordings verify in both source readers; exact final hashes and ticks, including rewind |

Focused checks overlap the complete suites; their counts are not added together.
An earlier complete run passed 371 tests; the final run includes the additional
digging-initialization and CPU custom-optimizer reload controls.
Warnings are the existing SB3 reference-policy GPU advisories. Game caches and test
outputs were kept outside their checkouts. No gameplay constraints or tolerances
were weakened. The source launcher was verified without changing the installed pin.

The new reward controls independently confirm +0.0999 offensive shaping on first
placement, −0.1 on removal and zero discounted gain for plant/dig cycles; −0.02
for a non-wall-nut eaten event; no eaten penalty for nuts, digging or detonation;
and the unchanged −0.2 empty-blast penalty. CPU and GPU agree to 1e-10 on double
reward components, with existing float32 rollout tolerances unchanged. Terminal
potential is zero and time-limit potential is retained.

Actor controls verify independent clipped losses, gradients and Adam updates;
forced-only and singleton choice batches remain finite. Slot-refill tests cover
out-of-order completion, slot reuse, wins, losses, truncations, exact action traces,
CPU hashes and plant/mower kill totals. Legacy replay and native viewer tests pass.

Logs and images are under `artifacts/`, including `research-final-v071.txt`,
`native-game-tests-v071.txt`, `cuda-game-tests-v071.txt`, `plant-rewards-tests-v071.txt`,
`report-focused-v071.txt` and `native-sc2-demo-v071.png`. The native winning final
frame was inspected visually; its outcome and 15/15 defeated counter are visible.

### Bounded learning evidence

Seed 101 optimization-only control, five-minute allowances, five common normal
validation cases per difficulty: baseline collected 1,024 games at 5,093 training
decisions/s; efficient collected 1,792 at 10,007 decisions/s. Both scored 0% normal
validation and remained in placement. The second-seed comparison was interrupted
when the scope changed to plant-role rewards; it is incomplete, not paired evidence.
This is a preliminary throughput observation, not a two-hour win-rate gain.

Reward-only v1: two five-minute allowances, both normal validations 0/15 and no
sampled training wins. Final v2 adds a trainable initial dig bias of −6, verified
to retain nonzero legal probability, finite gradients, learnable preference reversal
and exact checkpoint reload. Two three-minute allowances completed:

| Seed | Games | Seconds | Training wins (all / last 100) | Early digs/game (last 100) | Normal validation |
|---|---:|---:|---|---:|---|
| 101 | 821 | 173.50 | 144 / 64 | 0.01 | 2/15, both easy |
| 102 | 1,030 | 174.98 | 318 / 84 | 0.04 | 0/15 |

Both reached a 20/20 placement probe. Seed 101 advanced to saving and achieved two
20/20 saving probes; minimum stage residency prevented premature promotion. Seed
102 lacked the second consecutive placement pass before the deadline. Both remain
curriculum-incomplete. These diagnose improved early learning, **not** normal-game
improvement on both seeds or a two-hour result. V2 remains experimental.

Six completed check allowances total 26 minutes; actual elapsed was about 24.5
minutes, plus a short interrupted check. Only development seeds were used.
The final evidence summary is linked from `sc2-learning-fix.md` and stored in
`docs/evidence/sc2-v071/diagnostics.json`. Existing completed-run evidence, controls,
and new learning diagnostics must not be pooled. No formal training or final-test
evaluation was launched.

A loaded copy of the user's selected checkpoint also produced identical actions,
episode metrics and final hashes with fixed versus refilled evaluation batches
on easy development seeds 0, 1 and 2, including both losses and a win. See
`artifacts/refill-real-policy-v071.json`; its single timing pair is not a benchmark.

After training, the final v2 seed-101 checkpoint generated three more compact
demos at validation seed 100000. All use SHA-256
`33fe56f22763f56e6a6a5490f467e9d26bbd485dc4267eed7165e7a7037f8288`,
and each recording's CPU replay matches its CUDA evaluation state hash.

| Difficulty | Outcome / final tick | Plant / mower kills |
|---|---|---|
| Easy | Won / 3,419 | 6 / 9 |
| Standard | Lost / 3,652 | 2 / 11 |
| Hard | Lost / 3,652 | 2 / 11 |

The report and recordings are in
`artifacts/sc2-retention-check/sc2-plant-rewards-101/visualizations/`.
The training curves and easy final frame were visually inspected; previews are
`artifacts/retention-curves-preview-v071.png` and `artifacts/retention-demo-v071.png`.
Export took 23.32 seconds after the diagnostic run, without additional learning
or MP4 encoding. These predetermined demonstrations are not an additional
win-rate estimate.

## SC2-inspired learning — 2026-09-21

Research **0.7.0** retains game package **1.3.0**, simulation **1.0.0**, engine
commit `8861824df6893a34c2cd4df7f9b68613376d7964`, and the existing combat/reward
coefficients. Windows/Python 3.12, RTX 4070 Laptop GPU, Torch 2.8.0+cu128 and
CuPy 13.6.0 were used. Both game checkouts remained clean.

| Check | Result |
|---|---|
| Complete research regression | **338 passed, 2 warnings in 265.26 s** |
| Complete upstream game regression | **214 passed in 44.74 s**; all cache, bytecode and temporary output outside the game checkout |
| Final focused regression | **41 passed in 84.43 s**, covering all new SC2 tests, reports, and CPU/CUDA GAE at both gamma values after the final curve-marker changes |
| Export budget recovery | **3 passed in 41.36 s** after an independent simulated-time interruption exposed missing export time; final status now includes presentation and cleanup on every exit path |
| Probability and optimizer controls | Independent entropy arithmetic and gradients, zero illegal-action probabilities, wait-only/single-tile boundaries, true joint PPO clipping, losses, gradients and Adam updates |
| Spatial policy | Tactical layout, board dependence, finite gradients, complete optimizer parameter coverage, CPU collection with CPU/CUDA learning, CUDA simulation and checkpoint reload |
| Curriculum and scheduling | Episode-start stage residency, failed/passing probes, retained policy/optimizer identity, coalesced validation thresholds, same-weight result reuse, earlier-checkpoint ties and cumulative-time resume |
| Rewards and game controls | Existing successful and failing lesson controls, plant/dig cycles, cutoff/terminal precedence, reward parity and timeout bootstrap; both gamma values telescope correctly |
| Demos and reports | Three CPU-verified compact replays from the same spatial checkpoint; offline relative assets, truncated outcomes, report regeneration and existing export-failure checks pass |
| Availability | NVRTC compilation, DLPack/shared stream, simulation hash, 1000×600 RGB rendering, seekable compact replays and FFmpeg/libx264 all available |

The two complete-suite warnings are the existing SB3 small-MLP GPU warnings in
stock reference implementations used for comparison. `doctor` also retains the
Gymnasium advisory about unbounded observations. No gameplay assertions or
floating-point tolerances were weakened. The final focused run added the second
gamma to existing CPU/CUDA GAE controls; its results are reported separately,
not as a new complete-suite count.

The CUDA spatial smoke completed **2 games / 64 decisions** with two parallel
games and a deliberately one-second cutoff. All easy/standard/hard demos are
correctly **truncated**, share one checkpoint SHA-256, and verify against the
CPU engine. This establishes integration, not full-game competence. The teaching
stage remains placement and `curriculum_incomplete` is true. Single-point behavior
curves, optimizer curves and a native replay frame were visually inspected.
The rendering inspection preserved state hash
`7ff2c79ce9f35cf4689ecc9932543245cf79b088b7b3bea49f8416ed06d25554`.

Implementation used a separate code worktree and the existing environment, with
no second virtual environment. Full suites and GPU learning checks ran without
the original job. Its results are not used to judge the new candidate.

Evidence is retained in `artifacts/research-tests-v070.txt`,
`artifacts/game-tests-v070.txt`, `artifacts/focused-tests-v070.txt`,
`artifacts/availability-v070.json`, `artifacts/export-budget-final-v070.txt`, and
`artifacts/sc2-replay-frame-v070.png`.
The retained short smoke run is `artifacts/sc2-smoke-v070`.
The A–F comparison and formal research training were not launched; final-test
seeds remain untouched. Candidate E/F remain experimental.

### Bounded learning diagnostics and delivery

Two profile-E runs used learner seeds 101 and 102, each capped at five minutes.
Both completed 768 placement games; they collected 851,968 and 835,584 decisions
respectively and stopped at completed-update boundaries. Neither won a training
game or passed any of six 20-case placement probes. Mean early digs/game were
3.000 and 2.909, with three peashooter purchases/game. Saving was not reached.
This is an unsuccessful short diagnostic; it does not establish full-game
improvement or compare the proposed method with baseline.

Both final normal-game evaluations completed 128/150 cases before the deadline,
then correctly recorded `validation_pending=true`. Partial results did not
produce `best.zip`; final checkpoints and available records survived. The measured
300.03-second duration per run includes cooperative shutdown overhead. Their
reports were subsequently regenerated without training. Detailed limitations and
the committed summary are in [SC2 adaptation](sc2-adaptation.md).

Final Ruff checks, formatting, `git diff --check`, and dependency checks pass.
The 0.7.0 wheel and sdist built successfully. Editable installation in the existing
canonical `.venv` imports research 0.7.0 from this checkout and game 1.3.0 from its
installed dependency; `pip check` passes. All 106 retained artifact files were
checksum-verified on consolidation. The smoke and both diagnostic reports then
regenerated with valid offline assets and unchanged model files; all three smoke
recordings verified again. Evidence: `artifacts/build-v070.txt`,
`artifacts/install-v070.txt`, and `artifacts/regeneration-v070.json`.

The temporary implementation worktree and absent GPU-worktree registration were
removed. The single working research checkout is
`E:/Projects/Tower-Defence-AI/PVZ-plant`, using its existing `.venv`, on
`feature/sc2-learning`. Neither repository has a remote, so no PR was created.

## CUDA simulation and installed environment — 2026-09-21

Research **0.6.0**, game package **1.3.0**, simulation **1.0.0**, pinned game commit
`8861824df6893a34c2cd4df7f9b68613376d7964`. Combat rules retain hash
`a7486688fa43a9b50193f82715fa519bd27cfebc7ed054bebe4f7990f9c268de`.
Windows, Python 3.12, RTX 4070 Laptop GPU, Torch 2.8.0+cu128, CuPy 13.6.0.

| Check | Final result |
|---|---|
| Complete research suite in the main `.venv` | **312 passed in 218.54 s** |
| Complete new game suite in the main `.venv` | **214 passed in 49.23 s**; all bytecode, pytest/Hypothesis caches and temporary outputs outside the game checkout |
| CUDA checks after final inference-timing adjustment | **19 passed in 23.36 s** |
| Game equivalence | Exact integer snapshots, acceptance, masks, ordered events and canonical hashes; all plant/zombie mechanics, crowded mower sweeps, immediate actions, rejected requests, resets, storage limits and archived easy/standard/hard replay outcomes |
| Numeric parity | Both encoders, lesson/changed-scenario states, unclipped crowds, order and private-schedule invariance; reward components, GAE, SB3 minibatch ordering, losses, gradients and optimizer updates |
| Learning integration | Masked, unmasked, sparse, mixed CUDA paths; CPU hybrid and Windows workers; grouped policy reload, unchanged policy/optimizer identity through curriculum changes, completed-game stopping and interrupted resume |
| Selected checkpoint and demos | One `best.zip` produces three CPU-verified compact demos with identical checkpoint identity; no FFmpeg call for default output; old-pin models remain rejected and archived recordings remain readable |
| Rendering and reports | Offline assets, native renderer purity, visible truncation outcome, optional MP4/replay controls and failure recovery pass existing regressions; curves and representative replay frame visually inspected |
| Availability | NVRTC compilation, DLPack pointer/shared-stream writes, oracle hash, free memory, 1000×600 Gym RGB, native compact playback and H.264 export available |
| Packaging and lint | Research/game wheel and sdist build; CUDA kernels and bundled config included; Python Ruff and CUDA clang-format checks, `git diff --check`, `pip check` pass |
| Installation and cleanup | Game installed non-editably from a verified archive staged inside research `build/`; research installed from the main checkout; main `.venv` retained, temporary CUDA `.venv` moved to Recycle Bin |

The two warnings in the complete research run are SB3's existing small-MLP GPU
warning on **stock reference** PPO paths used for numerical comparison. The
device collector does not issue that CPU-observation warning. Earlier development
runs exposed stale expected engine commit/version strings; those expectations
were updated to the actual committed 1.3.0 pin before the successful full run.
No gameplay assertions or numerical tolerances were weakened.

Integer state/masks/hashes are exact. Encoder comparisons use `atol=1e-7,
rtol=1e-6`; float32 reward/GAE comparisons use up to `2e-6` absolute tolerance,
with double reward components checked to `1e-10`. Fixed-rollout parameter and
gradient comparisons use `atol=2e-7, rtol=2e-6`. Independently trained policies
are not required to remain bitwise equal.

The CUDA smoke collected one 256-decision rollout (two parallel games × 128),
completed four PPO epochs, saved/reloaded checkpoints and generated an offline
report with easy/standard/hard demos. The configured two-game target was exceeded
to finish the update; ten one-second-cutoff games were recorded. All demos
correctly say **truncated**. This validates integration, not learned competence.
The separate interrupted-resume tests perform further updates and retain global
game/curriculum counters. No final-test seeds or formal training suite were used.

The native replay frame, training and optimizer PNGs, and the static benchmark
plot were visually inspected. Full-game wins/losses are covered by the archived
reference replays and engine differential tests; the short training artifact
does not claim those outcomes.

An empty task-local CuPy cache measured **1.658 s** for Torch context startup,
**2.324 s** for simulation/encoder compilation, allocation and 128-game initial
reset, and **0.043 s** for GAE compilation/execution. These costs exclude Python
imports and are separate from steady-state throughput. CUDA evaluation now uses
device events for amortized batch inference time instead of host enqueue time.

The reviewed three-repeat benchmark selected **128 parallel games × 128 decisions**,
**10,733 decisions/s**, **3.79×** the current CPU-simulation baseline. CUDA at
eight games is slightly slower than equivalent CPU; validation/export and long
thermal behavior are outside that speed claim. See [measurements and limitations](gpu-performance.md).

Local evidence is retained in `artifacts/research-tests-v060.txt`,
`artifacts/game-tests-v130.txt`, `artifacts/cuda-tests-final-v060.txt`,
`artifacts/availability-v060.json`, `artifacts/cold-compile-v060.json`,
`artifacts/visual-check-v060/`, and `.test-tmp/research-v060/`. Benchmark evidence
and the reviewed recommendation are committed under `docs/evidence/gpu-v060/`.
Both repositories have feature-branch Conventional Commits. Neither has a remote,
so no PR was created. The original `E:/Projects/pvz` remains unchanged.

The second `.venv` isolated CUDA installation/testing from the existing setup.
After consolidation, automatic approval review blocked permanent recursive
deletion. Reversible Recycle Bin cleanup succeeded; the main environment imports
the installed game from its own `site-packages` and research from this checkout.

## Earlier verification records


## Per-tick actions, defense rewards and completed-game schedules — 2026-09-20

Pinned game package **1.2.1**, simulation **1.0.0**, commit
`6fd1f54706369915013a49eab5c1790f8c55ab0a`. The game checkout remains clean;
installed source hash `8d948f608d9b8a3c6e789eef35f2580bb850519d7b197524160cc3918d50db45`
and rules hash `a7486688fa43a9b50193f82715fa519bd27cfebc7ed054bebe4f7990f9c268de`
pass verification. All upstream bytecode/cache/temp output stays outside the checkout.

| Check | Result |
|---|---|
| Complete research regression | **283 passed in 248.80 s**; one existing SB3 small-MLP GPU utilization warning |
| Complete upstream regression | **199 passed in 10.75 s** |
| Same-tick actions | Multiple card placements/digs at one tick, no artificial cap (45 digs), exact costs/cooldowns, invalid/restricted requests advance, reset and win/loss/cutoff boundaries |
| Independent timing controls | Action plus wait matches native full state hash; all-lane placement controls win at tick 875 and saving at 2074; waiting loses at 1000/2199. Legacy placement remains tick 883 under its original cadence. |
| Rewards | 29 new arithmetic/engine defense checks, plus preserved kill/damage/economy controls: sun/spend/income order, multiple kills per activation, partial/full/custom-health wall-nut bites, armor-only and source/tick explosions, two-bomb contention, untriggered mines and reset |
| Reward algebra | Configurable loss −2, legacy loss −1; immediate-action shaping telescopes; plant/dig cycles lose value; natural terminals zero potential, truncations preserve it |
| Completed-game schedules | All five conditions; count timeouts, exclude validation, finish optimization at game target and record excess; global count across Windows workers/CUDA; checkpoint reload and interrupted resume; fixed/mastery gates ignore decisions and keep optimizer identity |
| Replays/presentation | Compact/plain equivalence, embedded hashes, corrupt recordings, trailing/empty action phases, mid-game cache-boundary seeking, completion/rewind, unchanged render hashes, optional MP4 decodes with one frame per tick |
| Shared policy | All three automatic demos reference one selected checkpoint; default compact export does not call FFmpeg; old configs/reports/replays remain supported and changed-protocol resume is rejected |
| Tooling | Ruff, formatting and pip checks passed; editable installation and wheel/sdist build **0.5.0** passed; no remote configured |

Evidence: `artifacts/v050-release-research-tests.txt`,
`artifacts/v050-final-engine-tests.txt`, `artifacts/v050-games-tests.txt`,
`artifacts/v050-game-mastery.txt`, `artifacts/v050-doctor.json`, and
`artifacts/v050-build.txt`. The doctor verifies instant placement, compact seeking,
all five Gym interfaces, CUDA, 1000×600 Gym RGB output and FFmpeg 8.1/libx264.
The existing infinite observation-bound advisories and SB3 small-MLP GPU utilization
warning remain; numerical and API checks pass.

The inspected game-count smoke run is
`artifacts/pytest-v050-final/test_global_games_across_windo0/game-smoke`.
Two CUDA workers target two games with a deliberately one-second cutoff and one
128-decision rollout. They complete **four games**, record **two extra games**,
and perform exactly one optimization update. All demos are correctly labeled
truncated and reference shared checkpoint SHA-256
`4ae4a9744bbf5d544d8521b2108e983e2201a892a3dbaf8665e44782163585a4`.
The new game-axis/defense plots were inspected in
`artifacts/v050-games-curves-preview.png` and `artifacts/v050-defense-curves-preview.png`.
They have one point and zero combat events because of the short cutoff, not a
learning-performance curve. The full offline report retains all optimizer panels.

The inspected final native frame is `artifacts/v050-demo-final.png`, 1280×820,
with visible truncated outcome. Rendering preserves simulation SHA-256
`aafa91cc7825efc9adeb1d84ffebd3c554dc03a97ca2641090699b39e3fc61a4`.
Tests also decode explicit H.264 exports and verify natural winning/losing frames.
No new timed pilot, held-out study, or formal training suite was launched.

The former version records below are historical evidence under their original
reward, timing and budget protocols; do not pool their results with this release.

## Damage, mower and economic-value rewards — 2026-09-20

Game package **1.2.1**, simulation **1.0.0**, source pin
`6fd1f54706369915013a49eab5c1790f8c55ab0a` remain unchanged. Verification found no
tracked or untracked changes in the game checkout. Tests used the installed game;
upstream bytecode writes were disabled and all caches/temp output stayed here.

| Check | Result |
|---|---|
| Complete research suite | **207 passed in 104.49 s**; one existing SB3 small-MLP CUDA utilization warning |
| Complete upstream game suite | **199 passed in 9.09 s** |
| New independent reward controls | 40 tests: starting versus remaining HP, 200/400-HP types, armor-only/mixed hits, lethal exclusion, earlier damage and mower finishes, entirely new spawn/death batches, same-tick/lane activation matching |
| Potential and lifecycle | All eight plant purchases preserve value; digging/death remove full cost above the scale; plant damage leaves value intact; multiple plants/order invariance; custom rule costs/HP; terminal zero and truncation bootstrap; reset and telescoping/cycle controls |
| Engine batch equivalence | One-tick and ten-tick shooting episodes have identical final hashes and combat totals; real mower activation has zero empty-activation penalty |
| Learning availability | CPU/CUDA, Windows spawned workers, grouped/shared policies, all five conditions, checkpoint reload, unchanged optimizer through curriculum stages, same-version resume |
| Compatibility | Legacy potential/kill configs reproduce old formulas and reload; resuming into revised reward settings is rejected; malformed new settings rejected |
| Reports and demos | New metrics reach episode files and post-update summaries; offline assets resolve; compact demos and optional H.264 MP4 decode; identical checkpoint identity across easy/standard/hard |
| Tooling/package | Ruff, format checks, pip dependency check, 0.4.1 editable install and wheel/source-distribution build passed |

Evidence files: `artifacts/v041-research-tests.txt`, `artifacts/v041-engine-tests.txt`,
`artifacts/v041-doctor.json`, and `artifacts/v041-build.txt`. The doctor confirms
CUDA, all five Gymnasium conditions, native rendering and compact playback, and
FFmpeg 8.1/libx264. Its pre-existing infinite observation-bound warnings preserve
crowd totals and are unchanged.

The report/MP4 smoke run is at
`artifacts/pytest-v041-rewards/test_shared_checkpoint_reports0/shared`. It collects
128 decisions with a two-second test cutoff. All three demos are correctly labeled
truncated and identify checkpoint SHA-256
`407e040a50e93b12383ab4bdbe89067abf73c63ab3068403c9b624cd0ccbab48`.
Its ten-panel training chart was visually inspected, including the new damage
reward and empty-activation curves. A native 1280×820 final frame was inspected at
`artifacts/v041-demo-final.png`; rendering preserved final simulation hash
`69c2aeda7bc19c17b7a88723ab089ce499915b2931bfeb1c3ea3f8498408c44e`.

These runs verify integration, not gameplay skill. No new timed comparison pilot
or formal training was launched. All previous pilot results below used the old
reward and must not be presented as evidence for version 0.4.1.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/pytest-v041-rewards
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HYPOTHESIS_STORAGE_DIRECTORY = 'E:/Projects/Tower-Defence-AI/PVZ-plant/artifacts/hypothesis-v041-engine'
.\.venv\Scripts\python.exe -B -m pytest E:/Projects/pvz/tests -q `
  -o cache_dir=E:/Projects/Tower-Defence-AI/PVZ-plant/artifacts/engine-pytest-cache `
  --basetemp E:/Projects/Tower-Defence-AI/PVZ-plant/artifacts/engine-v041-tests
```

## Pure-RL profiles and plant/mower rewards — 2026-09-20

Game package **1.2.1**, simulation **1.0.0**, source pin
`6fd1f54706369915013a49eab5c1790f8c55ab0a` remain unchanged. The game checkout stays
clean. PyTorch 2.8.0+cu128 detects the RTX 4070 Laptop GPU.

| Check | Result |
|---|---|
| Complete research suite with revised kill rewards and final reporting/diagnostic guards | **166 passed in 100.40 seconds**; one expected upstream warning about small MLP GPU utilization |
| Complete upstream game suite | **199 passed in 9.82 seconds**; bytecode disabled, caches/temp under research artifacts |
| Tactical observations | 1,140 values, exact region boundaries, scaling, crowd totals, entity-order/ID invariance, no task/seed/future-schedule leakage |
| Grouped distribution | Independent NumPy joint probabilities, entropy/log-probabilities, illegal zero mass, CPU/CUDA finite gradients, forced wait and deterministic type/tile counterexample |
| Placement/saving controls | Every lane: shooting control wins at ticks 883/2074; wait loses at 1000/2199; restrictions, legal digs, exact costs, cooldowns, cutoff and reset covered |
| Curriculum and recovery | Two passes, failed streak reset, minimum residency, rehearsal, next-reset transition, same policy/optimizer identity, persisted resume, incomplete-budget state |
| Kill attribution | Six attacking plants, damaging-projectile kills, bombs/mines/chomper, multiple mower kills, mixed batch, plant-wounded/mower-finished zombie, activation-only and nonlethal events, duplicate/unattributable event rejection |
| Reward boundaries | Signed normalized/unnormalized contributions, configurable weights, legacy zero defaults, terminal mower victory, reset and known unsuccessful close-threat controls |
| Integration | All original five conditions, Windows spawned workers, CPU/CUDA, serialization, final post-update entropy capture, completed-update deadline |
| Shared demos | Three verified normal games from identical checkpoint weights; native final frames preserve simulation hashes and display real outcomes |
| Packaging/tooling | Ruff and formatting, dependency check, editable 0.4.0 install, wheel/source distribution build, Git whitespace checks |

Evidence: `artifacts/v040-release-tests.txt`, `artifacts/v040-engine-tests.txt`,
`artifacts/v040-doctor.json`, and `artifacts/v040-build.txt`. The new regression cases
are in `tests/test_pure_rl.py` and `tests/test_kill_rewards.py`. The source-attribution
controls rely only on public events; no game source or private state was changed.

The installed engine's complete tests ran with:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -B -m pytest E:/Projects/pvz/tests -q `
  -o cache_dir=E:/Projects/Tower-Defence-AI/PVZ-plant/artifacts/engine-pytest-cache `
  --basetemp E:/Projects/Tower-Defence-AI/PVZ-plant/artifacts/engine-v040-tests
```

The first timed pilot at `artifacts/paper-pilot-v040` was interrupted when the user
changed the reward objective. Its validated checkpoints and episode files remain.
Placement reached 114,688 decisions with 0/20 best validation wins; saving reached
73,728 with 1/20. No matched comparison across both seeds and all profiles exists
for that objective. Its status/report explicitly mark this as inconclusive.
Conservative recorded elapsed time was 583.81 seconds, leaving 20 minutes for a
fresh revised-reward pilot. Earlier and revised results are not pooled.

### Revised-reward pilot

`artifacts/paper-pilot-kill-reward-v040` completed all eight comparison jobs and
both diagnostics. Its 20-minute budget took **1,210.92 seconds including final
checkpoint/report cleanup**. Adding the conservative 583.81 seconds of the stopped
pilot gives **1,794.73 seconds**, within the original 30-minute allowance. The
revised run collected 1,355,776 decisions including diagnostics. No formal research
training or final-test performance evaluation was launched.

The largest validation budget completed by every profile and both learner seeds
was **65,536 decisions**. Every cell below uses the same 15 cases: validation seeds
100000–100004 on easy, standard and hard, equal weighted by difficulty.

| Profile | Learner 101 macro win rate | Learner 102 macro win rate |
|---|---:|---:|
| Baseline | 0.0% | 0.0% |
| Tactical observation | 0.0% | 0.0% |
| Tactical + grouped policy | 0.0% | 0.0% |
| Full teaching method | 0.0% | 6.7% |

Placement diagnostics collected 131,072 decisions with 0/20 best validation wins.
Saving collected 114,688 with 1/20. Neither passed. Full-method seeds 101/102
collected 86,016/81,920 decisions, both still in placement, with zero training
episode wins or plant kills. The selected comparison result is therefore
**experimental/inconclusive**: the method does not improve both seeds and fails the
diagnostic rule. The single normal-game win is not evidence of reliable defense.

`comparison.json`, `comparison.png`, `report.md`, all run configurations, logs,
checkpoints, curriculum probes, and episode records are preserved under that root.
The report independently checks complete scenario sets, profile identities, engine/
reward protocol hashes and agreement between curve scores and episode-level wins.
Only common completed budgets enter its curves; higher-budget results remain in
individual run reports. Curves were visually inspected. No different objectives
or engine pins are pooled, and no best-seed selection is used.

Predetermined diagnostic replays and action traces show immediate plant/dig cycles;
the entropy counterexample and proposed follow-ups are in `paper-adaptation.md`.
Those follow-ups were documented, not silently applied to the timed comparison.

### Saved integration presentation

The 64-decision CUDA/two-worker grouped-policy integration run is preserved at
`artifacts/pytest-v040-kill-full/test_grouped_spawn_and_same_ch0/grouped-spawn`.
It produces an offline report and three native demos from one checkpoint:
`d908b7f90c4153379433458dc95b13d38dc560194e633a9bf6826e524bf0563f`.
These are availability checks, not claims of learned competence.

| Difficulty | Outcome | Plant / mower kills | Plant reward / mower penalty |
|---|---|---:|---:|
| Easy | Won | 3 / 12 | +0.20 / -0.80 |
| Standard | Lost | 4 / 12 | +0.10 / -0.30 |
| Hard | Lost | 3 / 15 | +0.04 / -0.20 |

All three complete simulation hashes match the corresponding pre-reward smoke
games, demonstrating that accounting does not mutate gameplay. Native final frames
and hash/reward verification are saved under `artifacts/pure-rl-presentation-final`.
Relative HTML assets resolve locally with no network URLs. Final frames and the
diagnostic economy/entropy curves were visually inspected for readable labels,
actual outcomes, and missing-data handling. An archived
`artifacts/speed-shared-smoke/best.zip` also loads with its original 2,719-value
encoder and absent (zero) kill weights.

## Runtime performance and PVZ 1.2.1 — 2026-09-20

Research package 0.3.2 uses game package 1.2.1 at
`6fd1f54706369915013a49eab5c1790f8c55ab0a`; simulation remains 1.0.0.
Diff inspection confirms only rendering and the package version changed in game
source. Engine, combat rules, replay, and scenario code have identical hashes.
The new manifest was exported with bytecode disabled into the research project;
the game was built non-editably from `build/game-1.2.1`.

| Check | Result |
|---|---|
| Full research regression/integration suite | **120 passed in 76.22 seconds** in the final run; expected upstream warning that small MLP PPO may underutilize CUDA |
| Full game regression suite | **199 passed in 10.01 seconds**, with bytecode/cache writes disabled and temporary output in research artifacts |
| All five learning conditions on CPU and CUDA | Optimized/reference learned policy tensors and Adam state match exactly after multiple updates |
| Device buffer | Same dtype, shapes, values, minibatch order, next RNG draw, multiple epochs, partial final batch, and reset as stock SB3 |
| Legality and worker caches | Independent engine validation, cooldown/cost boundaries, bomb disappearance, reset, natural win/loss, external truncation, and Windows spawned-worker trajectories |
| Full gameplay controls | Identical observations/actions/rewards/masks and complete hashes at every decision for known seed-42 wins across all difficulties; existing known failing seeds also pass |
| Same-pin checkpoint recovery | Stock buffer checkpoint resumes with current runtime defaults, preserving optimizer and earliest best checkpoint; different source pins remain rejected |
| Native 1.2.1 renderer | Defeated/total labels verified for 0/15, 2/15, 15/15, 0/0, 200/200; Gym dimensions and rendering purity retained |
| Timing | Validation gaps excluded from collection/update totals, final update captured, paired benchmark accounts for warmup separately |
| Tooling | Ruff checks, formatting, pip dependency check, non-editable game install, editable research install, wheel/source build, Git whitespace checks |

Evidence: `artifacts/speed-final-research-tests.txt`,
`artifacts/speed-full-engine-tests.txt`, `artifacts/speed-doctor.json`, and
`artifacts/speed-final-packaging.txt`.
The performance benchmark and smoke results below are short engineering checks,
not evidence of learned skill or research win-rate improvements.

Final paired CUDA benchmark on the RTX 4070 Laptop GPU, eight Windows workers,
with three seeds and **32,768 measured decisions plus 4,096 warmup decisions per
run**. The reference disables all three runtime flags; every research setting is
identical. Runs were sequential with alternating order and no overlapping tests
or training. Callback reference cycles and unused allocator memory are reclaimed
before each run so peak memory belongs to that measurement.

| Learner seed | Reference decisions/s | Optimized decisions/s | Throughput increase | Final weights |
|---|---:|---:|---:|---|
| 800 | 2,187 | 2,645 | 20.9% | Exactly equal |
| 801 | 2,151 | 2,729 | 26.9% | Exactly equal |
| 802 | 2,176 | 2,664 | 22.4% | Exactly equal |
| Median | **2,176** | **2,664** | **22.4%** | 3/3 pairs equal |

Median collection time decreased from 12.26 to 10.13 seconds; median update time
decreased from 2.80 to 2.17 seconds. Peak allocated PyTorch CUDA memory was stable
at 51.85 MiB reference versus 100.99 MiB optimized. The approximately 49 MiB extra
is the cached rollout; these figures exclude CUDA driver/context overhead counted
by tools such as Task Manager. Low overall VRAM use remains expected.

Raw measurements, full policy hashes, and pair ratios are in
`artifacts/speed-pvz121-final/measurements.json` and `runtime-comparison.json`.
This measures early easy-curriculum collection/updates on one laptop, not a
confidence interval or a full-run wall-time guarantee. Startup, validation, reports,
and demos are excluded and reported separately where applicable. An earlier
three-pair run measured 25.3% median improvement; the final memory-isolated results
above are the reported result. Preliminary overlapping profiler/test runs are not
used for the speed claim.

The saved CUDA smoke run `artifacts/speed-shared-smoke` collected 8,192 decisions
using eight workers, the default 256–256 networks, 4,096-decision rollouts,
256-sample minibatches, and four epochs. Only the short budget and one-case-per-
difficulty validation were overridden. Both post-update metric rows were captured.
The offline report and all relative assets resolve; a CUDA checkpoint reload passed.
All three compact demos use checkpoint
`594a2a9babc62b359e73c09db1947e2915708b8b399e21c5ab4e763f59f3dfd9`
and validation seed 100000. Their final hashes verify. Outcomes were losses:
easy tick 2678, standard/hard tick 3652. These are integration cases, not failures
of the proposed research targets after meaningful training.

Visually inspected the optimization/training curves and mid-game/final native
replay frames. The 1.2.1 HUD clearly shows `2/75` mid-game and `13/75` at the hard
loss, with the actual outcome banner. No MP4 encoder was invoked by default.
Evidence is in `artifacts/speed-smoke-verification.json` and the smoke run's
`visualizations` directory. Full-suite tests separately exercise optional MP4
export, all three outcomes, diagnostic runs, interrupted runs, and report recovery.

All three compact recordings from the user's existing game-1.2.0 run also verify
under 1.2.1 with identical final hashes; the archived run was read only. See
`artifacts/speed-archived-replay-verification.json`. The separate game checkout
remains clean at the new pin. No formal research training was launched, and the
user's running training job was allowed to complete without interruption.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -B -m pytest -q E:/Projects/pvz/tests `
  -p no:cacheprovider --basetemp artifacts/speed-engine-test-temp
.\.venv\Scripts\python.exe -m pvz_rl benchmark `
  --workers 8 --devices cuda --steps 32768 --repeats 3 --compare-runtime `
  --output artifacts/speed-pvz121-final
.\.venv\Scripts\python.exe -m pvz_rl train `
  --condition masked --seed 101 --steps 8192 --eval-interval 4096 `
  --validation-count 1 --output artifacts/speed-shared-smoke
```

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
