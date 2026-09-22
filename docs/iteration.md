# Iteration history

Dates and results belong to their recorded version. Current behavior lives in
[research design](research.md) and [architecture](architecture.md); test counts,
artifact locations and learning outcomes live in [validation](validation.md).
The longer pre-consolidation notes remain in `git show b642c9c:docs/iteration.md`.

## 0.8.1 — 2026-09-22

- Follow-up: Leafy requested perfect 100-game mastery and less frequent validation.
  Replace the permissive 20-case gates with 100/100 per required task, including
  final shared-stage mastery. Probe every 500 training games; validate checkpoints
  every 2,000 with the original 50 cases/difficulty. A separate 100-case curriculum
  seed range avoids enlarging normal validation or touching held-out seeds.
  Resume preserves saved gates; explicit stage handoff can select the new protocol.
  README now lists every stage's pass criteria and the two schedules.
- Problem/cause: teaching stages advanced within one run; strict resume restored
  the same budget and offered no explicit checkpoint handoff to another stage.
- Change: select one stage, retain its rehearsal mix, stop on mastery or budget,
  and save probe/final checkpoints. Initialize a chosen stage from compatible
  weights and optimizer state with fresh local schedules and parent provenance.
  Existing automatic curriculum and same-stage resume remain available.
- Verification: stage/state boundaries, CUDA handoffs through all five stages,
  exact restored weights/Adam moments, interrupted resume, mismatch rejection,
  normal-game checkpoint selection, CLI/report smoke and regression suites;
  recorded results are in validation.md.
- Remaining limits: inherited learning settings must match; stage budgets do not
  certify mastery. Short checks establish integration, not improved win rates.

## 0.8.0 — 2026-09-22

- Problem/cause: CPU collectors, hybrid jobs, and old recipes remained reachable;
  installer HEAD requirements and README examples did not match the current folders.
- Change: CUDA-only training preflight; remove CPU collectors/transport, hybrid
  training, four standalone recipes, and legacy pilot/benchmark commands. Keep
  four direct conditions, CUDA SC2 comparisons, and CPU verification/inference.
  Archive the pinned game commit without switching its source checkout. README
  now maps the actual folders and uses consistent fresh output names.
- Compatibility: CUDA weights, optimizer state, learning settings and engine pin
  remain compatible. Ignore dormant conditions for individual resume; replace
  retired buffer metadata for historical inference. CPU/hybrid resume is rejected.
- Verification: full research and both game suites, CUDA smoke/report/replay,
  checkpoint action/hash controls, pinned source staging, documentation examples,
  lint, dependency checks and packaging; exact results are in validation.md.
- Remaining limits: CUDA hardware is required for training. The plant-reward
  candidate remains experimental; this cleanup establishes no learning improvement.
  The CPU simulator remains necessary for correctness checks and replay verification.

## Documentation consolidation — 2026-09-22 (0.7.1 unchanged)

- Problem/cause: release notes accumulated into repeated installation, reward,
  architecture and experiment guides, leaving obsolete advice beside current behavior.
- Change: replace the 1,000-line README with a daily-use guide; combine paper/SC2/
  reward notes into `research.md`, GPU measurements into `validation.md`, and game/
  GPU integration into `architecture.md`. Deduplicate references and version history;
  remove the seven superseded documents after moving their useful content.
- Verification: local links/anchors, CLI examples and source/config references are
  checked; committed evidence files and executable settings remain unchanged.
  No learning run or new performance claim is part of this documentation change.
- Limit: older external bookmarks to removed documents need the new README index;
  detailed historical prose remains in Git and local artifact paths may be absent
  from fresh clones.

## 0.7.1 — 2026-09-21

- Problem/cause: native viewers rejected zero-time research recordings; SC2 normal
  validation remained 42%/0%/0%; frequent early digging and forced-action samples
  motivated exploration/credit hypotheses. Mower-free lesson logs lacked context.
- Change: compatible source readers in both game branches; optional choice-state
  PPO, GAE 0.999 and minibatches 1,024; refilled GPU validation; lesson-aware logs;
  offensive potential, small non-nut eaten penalty and trainable initial dig bias.
  Installed game stays pinned at 1.3.0; legacy profiles retain their semantics.
- Verification: 374 research, 207 CPU-game and 222 CUDA-game tests; exact replay
  hashes, gradients/optimizer, reward boundaries, reload and shared-demo checks.
- Limit: final candidate improved early lesson learning but normal validation was
  2/15 and 0/15 in two three-minute checks. Experimental; no two-hour advantage proved.

## 0.7.0 — 2026-09-21

- Problem/cause: absent sustained attackers, joint entropy favoring large action
  groups, frequent validation and no cumulative runtime bound.
- Change: balanced type/tile exploration, spatial grouped policy, stage-start
  residency, 100-game probes, 1,000-game validation, result reuse, cumulative time
  allowance and explicit A–F comparisons. Game remains 1.3.0.
- Verification: 338 research/214 game tests, later focused deadline checks and
  two five-minute diagnostics; single environment/workspace consolidated.
- Limit: both diagnostics failed placement; normal validation incomplete at the
  deadline. No candidate recommended or A–F/formal matrix launched.

## 0.6.0 — 2026-09-21

- Problem/cause: CPU simulation and transfers dominated small-policy GPU learning.
- Change: game 1.3.0 CuPy/NVRTC backend, ordered batched combat, tensor PPO/GAE,
  CPU-verified demos, capacity checks and 128 decisions per parallel game.
  Promote 128 GPU games after correctness/speed gates; retain explicit CPU hybrid.
- Verification: 312 research/214 game tests; three-repeat benchmark 10,733 vs
  2,835 decisions/s (3.79×); installed pin `8861824`, simulation 1.0.0.
- Limit: gain combines backend and larger rollouts; eight-game CUDA is slower.
  Compact host synchronization remains. No thermal-soak or convergence claim.
  Temporary development environment was moved to Recycle Bin; one research `.venv` remains.

## 0.5.0 — 2026-09-20

- Problem/cause: ten-tick decisions limited immediate actions; reward lacked defense
  events; decision counts did not reflect the requested game-based progress.
- Change: zero-time plant/dig adapter and versioned replays; one-tick wait/rejection;
  loss −2, activation sun cost, wall-nut absorption, empty-blast cost; completed-game
  budgets/curriculum/validation and checkpoint counters. Game stays 1.2.1.
- Verification: 283 research/199 game tests, all-lane lesson controls, source/tick
  rewards, game overshoot/resume, Windows/CUDA and native video checks.
- Limit: discount remains per decision; final rollouts can exceed targets; action
  adapter uses pinned private hooks. No new learning pilot validated these changes.

## 0.4.1 — 2026-09-20

- Problem/cause: kill-only feedback ignored partial damage; capped economy omitted
  valuable living plants and wealth above the scale.
- Change: mower kill −2/N, nonlethal HP reward, empty activation −0.2; uncapped
  plant-plus-sun potential; separate logs and legacy compatibility. Game stays 1.2.1.
- Verification: 207 research/199 game tests, 40 new reward controls and native outputs.
- Limit: gradual damage can outscore instant kills; empty mower activations normally
  do not occur; previous pilot results used a different objective.

## 0.4.0 — 2026-09-20

- Problem/cause: 10,229-game archive lacked sustained attackers; flat sampling gave
  wait 1/136, weak economic scaling and no mastery gates.
- Change: tactical 1,140-value encoder, grouped type/tile policy, teaching state
  machine, capped paired pilot, +1/N plant and −1/N mower kill reward. Game stays 1.2.1.
- Verification: 166 research/199 game tests; original pilot interrupted for reward
  change, revised matched 65,536-decision results inconclusive; both diagnostics failed.
- Limit: immediate plant/dig cycles persisted; joint entropy still preferred large
  groups. Separate kill objectives and incomplete studies cannot be pooled.

## 0.3.2 — 2026-09-20

- Problem/cause: repeated mask IPC/legality queries and rollout copies; game had
  advanced to 1.2.1 while research pinned 1.2.0.
- Change: cached authoritative legality, masks sent with observations, reusable
  CUDA rollout, phase timings/paired benchmark and non-editable 1.2.1 installation.
- Verification: 120 research/199 game tests; exact learned tensors/Adam state in
  reference comparisons; final three-pair speed gain 22.4% (2,176→2,664 decisions/s).
- Limit: CPU simulation and small inference batches remained; startup/validation
  excluded from timing; no learned-skill claim. New game pin required fresh training.

## 0.3.1 — 2026-09-20

- Problem/cause: configuration selected CPU despite available CUDA.
- Change: default policy device CUDA, explicit CPU override and unavailable-device
  errors; simulation still on CPU. Game stays 1.2.0.
- Verification: 87 research/199 game tests; CUDA-tagged policy/Adam tensors and reload.
- Limit: availability did not establish speed; existing CPU runs retain saved settings.

## 0.3.0 — 2026-09-20

- Problem/cause: duplicate rendering, bulky outputs and conflated package/simulation
  identity while game 1.2.0 introduced native presentation APIs.
- Change: pin `a47056d`, native renderer, metadata, compressed demos and seekable
  viewer; MP4 optional, separate versions and strict old-checkpoint rejection.
- Verification: 87 research/199 game tests, no-FFmpeg default demos, archive reads,
  native/browser playback, source staging, render/hash purity and packaging.
- Limit: old weights need their original pin; metadata needs whole-recording hashes.

## 0.2.0 — 2026-09-20

- Problem/cause: little operational output and manual visualization; shared model
  identity and presentation-vs-research compatibility were unclear.
- Change: throttled logs, post-update metrics/TensorBoard, offline curves, shared
  checkpoint demos/video and export recovery with separate compatibility settings.
- Verification: 154 combined research/game checks; shared hashes, unchanged weights
  with output toggled, old/empty reports, interruption, corruption and browser video.
- Limit: rendering/FFmpeg dependencies, extra export time and missing historical metrics.

## 0.1.0 — 2026-09-20

- Problem/cause: empty research directory; game lacked an RL wrapper and protocol.
- Change: Gym adapter, fixed 406 actions, spatial encoder, masked PPO, baseline/hybrid
  controls, seed separation, reward shaping, checkpoint recovery, benchmark/reporting
  tools and pin `b3cfbd8` from the independent game.
- Verification: action/mask/terminal controls, known heuristic wins/losses, short
  CPU/CUDA/Windows learning, full miniature protocol and package checks.
- Limit: no trained-skill/generalization claim; partial observation and inexact
  rollout/RNG resume. Formal runs remain user initiated; no remote was configured.
