# Iteration history

## 0.4.1 — 2026-09-20

### Previous issues and root causes

- Kill-only combat rewards supplied no credit for useful partial damage. The
  user requested −2/N mower kills, nonlethal HP reward, and −1/5 empty activations.
- Separate capped sun and sunflower potentials did not value other living plants
  and could hide loss of economic value above their caps.

### Improvements

- One public-event accounting interface distinguishes lethal hits, earlier
  nonlethal HP damage, plant/mower kills, and same-tick/lane empty activations.
  Normalize damage by active-rule starting HP and initial zombie count.
- Keep plant kills +1/N; change mower kills to −2/N. Add the requested damage and
  empty-activation terms with independent configurable weights.
- Replace separate economy terms with `0.5*(sun + living plant costs)/300`,
  uncapped, retaining the 0.5 zombie-progress term and discounted difference.
- Expose reward components in episode/evaluation records and rolling logs/curves.
  Version all supplied profiles together. Preserve old saved reward configurations
  and reject resume across different reward settings.
- Document assumptions, examples, and mathematical controls in `reward-design.md`.
  PVZ 1.2.1, shared policy, action space, curriculum, and PPO settings are unchanged.

### Verification and remaining issues

- Complete research suite: **207 passed in 104.49 s**. Complete upstream game
  suite: **199 passed in 9.09 s**, with caches and temporary output outside the game.
- Forty new numerical/engine controls plus a legacy-checkpoint compatibility test;
  CPU/CUDA, Windows workers, all five conditions, rendering, replays, and reports
  pass. New curves and a native final frame were visually inspected.
- Ruff, formatting, dependency checks, editable installation and wheel/sdist build
  pass. The game checkout remains clean. Details and artifacts: `validation.md`.
- Learning efficacy remains unmeasured for this objective. The 0.4.0 pilot cannot
  validate it; no additional timed pilot or formal study was run. Nonlethal damage
  intentionally gives gradual kills more cumulative credit than instant kills.
  Empty mower activations normally cannot occur in the pinned engine.

## 0.4.0 — 2026-09-20

### Previous issues and root causes

- An archived 10,229-episode run bought no sustained attackers. Flat legal-action
  sampling gave waiting only 1/136 initial probability; initial sun was scaled by
  9990. This motivated controlled representation/exploration changes, not a claim
  that one factor alone caused poor learning.
- Difficulty increased on a fixed schedule even when elementary placement/saving
  behavior was not demonstrated. There was no persisted mastery-gate state.
- The reward did not distinguish kills earned by plants from mower rescues. The
  user explicitly requested positive plant-kill credit and negative mower credit.

### Improvements

- Add tactical v2 (1,140 values), three distance regions per lane, economic card
  signals and lane summaries. Preserve spatial v1 and every original action index.
- Add a masked type/tile policy with true joint PPO probability/entropy and greedy
  type-then-tile evaluation. Equal logits give initial waiting probability 1/4.
- Add placement/saving lessons and a five-stage mastery state machine. Stage changes
  apply at episode resets, retaining one policy and optimizer. Checkpoints persist
  gates; normal easy/standard/hard validation alone chooses the shared checkpoint.
- Add versioned profile configurations and a capped pilot with two learner seeds,
  alternating profile order, matched completed validation budgets, and explicit
  missing comparisons. Keep unsuccessful candidates experimental.
- Add configurable +1/N plant-kill rewards and -1/N mower-kill penalties using public
  killing-damage events. Projectiles, mines, bombs and chomper receive plant credit.
  Record kill sources and contributions separately, and keep legacy missing weights
  equivalent to zero. Changed objective/representation profiles require fresh runs.
- Extend progress, episode metrics, and offline charts for economy, attackers,
  grouped entropy, curriculum state and plant/mower kills. Keep CUDA transport,
  native compact replays, and a single model for every normal difficulty.
- Document paper evidence, local diagnosis, chosen/rejected methods, adjustments,
  and unsuccessful diagnostics in `paper-adaptation.md`. PVZ 1.2.1 is unchanged.

### Verification and limitations

The complete research suite passes **166 tests**; the upstream game suite passes
**199 tests**. Independent distribution algebra, all five lesson lanes, meaningful
plant/mower kill controls, failed close-threat cases, Windows workers, CPU/CUDA,
curriculum resume, shared checkpoint demos, rendering/replay hashes, package build,
dependency and lint checks pass. Exact records and pilot results are in
`validation.md`. No formal training suite or final-test evaluation was launched.

The first pilot was interrupted to adopt the requested kill-source reward; its
results remain separate. Failed diagnostics do not establish that the method
improves play, and mastery gates may retain a lesson for the whole run. Regional
aggregation loses some precise enemy positions. Event rewards intentionally change
the objective; only the potential term has the policy-invariance guarantee.

## 0.3.2 — 2026-09-20

### Previous issues and root causes

- The small CUDA policy used little VRAM; collection spent time in worker mask
  requests, repeated engine legality queries, and small GPU operations. Low memory
  allocation alone did not measure GPU utilization or training throughput.
- Each PPO minibatch copied rollout arrays from CPU again, even when subsequent
  epochs reused exactly the same rollout. Logs did not separate collection/update
  time, and short hardware benchmarks included one-time initialization costs.
- The separate game checkout had advanced to 1.2.1, with defeated/total HUD text,
  while research still pinned 1.2.0.

### Improvements

- Cache authoritative engine legality while its public inputs remain equivalent.
  Requery on occupancy/status/card-availability changes or reset. Hybrid strategy
  proposals still refresh each decision; direct masks remain unchanged.
- Deliver masks alongside observations through stock SB3 workers, preserving reset
  and terminal-observation semantics. Cache one completed rollout on CUDA for all
  PPO epochs, retaining CPU GAE and exact NumPy sample order.
- Keep networks, optimizer, precision, workers, rollout/minibatch sizes, losses,
  seed splits, reward, curriculum, and shared checkpoint selection unchanged.
  Optional runtime flags and resolved metadata support comparisons and same-pin
  resume without changing research compatibility.
- Add phase timings to logs/metrics and paired benchmarks with warmup, alternating
  run order, separate startup time, policy hashes, and CUDA memory measurements.
- Pin and stage game 1.2.1 without writing into its checkout. Reuse the native
  defeated/total renderer; preserve strict rejection of older source-pin models.

### Verification

The complete research suite passes **120 tests** and the upstream game suite
passes **199 tests**. CPU and CUDA runs across all five conditions retain exactly
equal learned weights and optimizer state between reference and optimized paths.
Complete seed-42 games retain identical actions, observations, masks, rewards, and
state hashes on all difficulties. Natural terminals, truncation/reset masks,
cooldowns, exact sun thresholds, plant disappearance, minibatch dimensions/order,
cache reset, same-pin resume, and 1.2.1 HUD boundary counts are covered.
Measured speed, smoke artifacts, packaging, and final checks are in `validation.md`.
The final paired CUDA benchmark measured 22.4% higher median throughput
(2,176 to 2,664 decisions/s), with equal final policy weights in all three pairs.
A saved 8,192-decision shared-policy smoke generated verified demos and inspected
curves using the 1.2.1 native renderer.

### Remaining limits and adjustments

Small GPU inference batches and CPU simulation still limit throughput. The rollout
cache intentionally adds only about 49 MiB at the default size; filling VRAM is
not an objective. Measurements on a laptop depend on other work, power, and thermal
state. No formal research suite was launched or running user job interrupted.
Source-pin isolation requires fresh training after moving from game 1.2.0 to 1.2.1;
old recordings/reports remain readable. Optional upstream performance suggestions
are recorded in `engine-notes.md`; the game folder remains unchanged.

## 0.3.1 — 2026-09-20

### Previous issue and cause

The bundled training configuration explicitly selected CPU even when the installed
CUDA-enabled PyTorch detected the laptop's NVIDIA GPU.

### Improvements

- Select CUDA by default for training and suite runs through the existing device
  interface. Keep explicit CPU overrides and the unavailable-CUDA error.
- Document GPU setup, CPU fallback, hardware override precedence, and the matching
  device requirement for resume. Keep one shared policy across all difficulties.
- Pin general learning regression fixtures to CPU so they remain portable; retain
  the dedicated Windows-worker integration test that uses CUDA when available.

### Verification

All 87 research tests and 199 upstream game tests passed. A fresh 64-decision,
two-worker diagnostic run without a device override completed on the RTX 4070
Laptop GPU. Saved policy tensors and Adam moment tensors retain CUDA device tags;
checkpoint reload on CUDA, missing-CUDA handling, lint, formatting, and dependency
checks passed. See `validation.md` for commands and artifacts.

### Remaining issues and limits

GPU execution does not imply faster throughput for this small MLP and Python
simulation. Simulation workers remain on CPU. No formal research training was run.
Existing compatible CPU runs must still resume with their original CPU setting.
The game folder remains unchanged; no Git remote is configured for a PR.

## 0.3.0 — 2026-09-20

### Previous issues and causes

- The research environment was pinned to game package 1.0.0; the new 1.2.0 game
  source would fail its manifest check despite unchanged simulation semantics.
- The research package duplicated the game's board drawing and required sidecars
  for external replay outcomes. Game 1.2.0 now supplies both native interfaces.
- Demo creation was coupled to MP4 encoding, adding an encoder dependency and
  export delay even when compact timed-operation recordings were sufficient.
- Package version and simulation compatibility version were treated as one value.

### Improvements

- Adopted game commit `a47056d8141ec635d3ff3f4d5561d6a75cfca2cc`, package 1.2.0,
  retaining simulation 1.0.0. Verify source, package, simulation, and rules separately.
- Stage a hash-checked Git archive inside the research project before non-editable
  installation, so packaging does not write into the game checkout.
- Record compressed `.pvzdemo` files with native checkpoint identity, experiment
  provenance, outcome, and termination reason. Hash whole recordings in manifests.
- Generate compact demos by default, independently of optional MP4 encoding;
  retain one shared checkpoint for easy, standard, and hard.
- Use BoardRenderer/RGBFrame for rendering and Playback's outcome/operation/seek
  APIs for presentation. Preserve Gym's RGB array size and expose native video size.
- Add `--videos` and native replay `--speed`, preserve legacy file/sidecar reads,
  and rebuild archived reports/export existing recordings without loading old models.
- Require fresh training for old source-pin checkpoints; same-pin resume still
  restores policy and optimizer. Update README, architecture, references, and engine notes.

### Verification

Full research and game suites, short CPU/CUDA integration runs, a 128-decision
fresh shared-policy smoke run, native replay controls, optional MP4 playback,
availability, dependency, and package checks are recorded in `validation.md`.
Known win/loss ticks and the rules hash remain unchanged. Tests cover compact
recording without pygame/FFmpeg, metadata isolation/precedence, strict version
rejection, legacy archives, seeking, and tampered recording checksums.

### Remaining issues and limits

- The smoke run is not evidence of learned competence; formal training was not run.
- Old model checkpoints require their original environment; no migration is offered.
- Metadata is caller-supplied. Simulation hashes verify dynamics, while whole-file
  hashes detect changes to stored experiment annotations.
- Native viewing/rendering requires pygame; optional MP4 encoding requires FFmpeg.
- Static reports require browser reload after refresh. Resume is not exact RNG or
  interrupted-rollout restoration. Font pixels may differ across platforms.
- The game folder is unchanged. There is no configured remote for opening a PR.

## 0.2.0 — 2026-09-20

### Previous issues and causes

- Training used SB3 `verbose=0` with no operational progress reporter, leaving
  collection, validation, and checkpoint activity unclear in the terminal.
- Reports required a separate evaluation command; game replays required the
  interactive UI and lacked exported videos with research context.
- Shared-policy training already existed, but the output did not make it obvious
  that one selected checkpoint was used across all difficulties.
- Exact whole-configuration comparisons would treat presentation changes as
  experiment changes, preventing otherwise compatible resume operations.

### Improvements

- Added throttled timestamped console/file progress, explicit phases, rolling
  aggregates, post-update PPO metrics, and preserved TensorBoard logging.
- Added automatic offline HTML reports, validation/training/optimizer PNG curves,
  labeled resume segments, and support for missing older metrics.
- Added fixed validation demos of one shared `best.zip`, public-state HUD rendering,
  verified replay sidecars, and streamed H.264 MP4 export with playback controls.
- Separated export outcomes from learning success; added regeneration commands,
  encoder availability checks, output configuration defaults, and compatibility
  comparisons that exclude presentation settings.
- Updated training instructions, architecture diagrams, implementation references,
  and optional game-engine suggestions in `engine-notes.md`. The game folder was
  not edited.

### Verification

See `validation.md` for executed checks and saved smoke artifacts. Verification
covers shared checkpoint identity, unchanged learned weights with visualization,
post-update/final metrics, log cadence, old configurations, resume boundaries,
natural wins/losses and external cutoffs, corrupt replays, missing FFmpeg, and an
encoder process failure. Only short learning runs and availability checks were run.

### Remaining issues

- Learning performance and generalization still require the formal experiment.
- Video export requires pygame-ce and an FFmpeg build with libx264. It adds time
  after training; progress and export duration are recorded separately.
- Older runs lack newly added aggregate metrics; their missing panels remain blank.
- Reports are static files: reload the page after a validation refresh.
- Raw older replays without wrapper metadata cannot identify external cutoffs;
  they retain their actual engine status.
- Resume still starts fresh game episodes and is not exact rollout/RNG restoration.
- No Git remote is configured, so the feature commit is prepared locally for a PR.

## 0.1.0 — 2026-09-20

### Previous state and issues

The research directory was empty. The separate game exposed a deterministic API
but had no learning wrapper, reward, experiment protocol, or training documentation.
One winning seed-42 replay did not establish generalization.

### Root causes addressed

- A placement controller needs a stable action schema and legality masks for a
  combinatorial plant/tile action space.
- Win/loss feedback is delayed across a long economic horizon.
- A time cutoff is different from an engine defeat; confusing them breaks value
  bootstrapping and terminal reward shaping.
- Evaluating during collection can save pre-update weights under post-collection
  step labels and miss the final optimized policy.
- Comparing different scenario sets or pooling different training configurations
  can produce invalid apparent improvements.

### Improvements

- Added an external Gymnasium package with 406 direct actions or five hybrid
  strategies, 2,719 public-state features, a fixed 0.5-second decision interval,
  and explicit episode states.
- Added sparse/potential rewards with independent mathematical tests, a fixed
  curriculum, five PPO conditions, and four non-learning baselines.
- Preserved the original heuristic and verified both known wins and known losses.
- Added separated seed sets, three changed-wave families, paired evaluation,
  predetermined verified replays, bootstrap statistics, and scientific figures.
- Added validation after completed PPO updates, best/latest/final checkpoints,
  recovery without overwriting attempts, and a restartable experiment journal.
- Added pinned game-source verification, dependency locks, Windows installation,
  CPU/CUDA availability checks, complete-loop hardware benchmarking, and training
  instructions in the README.
- Recorded the literature actually used and documented the current architecture.

### Verification

See `validation.md` for the final executed checks. Availability testing includes
short actual PPO updates and Windows subprocess/CUDA execution. Full training and
held-out research evaluation were not launched, following the user's scope change.

### Remaining issues and limits

- No trained-policy win-rate or generalization claim has been established.
- The representation aggregates nearby entities; it is partially observed.
- Resume restores policy/optimizer but starts fresh episodes; interrupted rollouts
  and random-generator histories are not restored exactly.
- Large simultaneous crowds remain limited by the existing Python engine.
- Recurrent policies, DQN, human imitation, screenshots, and commercial-game transfer
  are outside the current study.
- Bootstrap intervals can degenerate for all-win/all-loss samples; five independent
  runs and transparent sample counts remain essential.
- Windows/Python 3.12 is the verified development environment. No remote is configured,
  so the implementation is kept on a local feature branch for later PR review.
