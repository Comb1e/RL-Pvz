# Iteration history

Dates and results belong to their recorded version. Current behavior lives in
[research design](research.md) and [architecture](architecture.md); test counts,
artifact locations and learning outcomes live in [validation](validation.md).
The longer pre-consolidation notes remain in `git show b642c9c:docs/iteration.md`.

## 0.24.0 — 2026-09-26

- Problems: coarse zombie displacement/contact geometry omitted source gait and
  vault details. The viewer kept only a few selected-worker actions and discarded
  useful history on switching. Smaller cohorts limited measured device throughput.
- Causes and changes: source animation/ground-track and collision rules now drive
  independently implemented fixed-point CPU/CUDA mechanics in game 1.6.0. Speed,
  gait phase/remainders and portable RNG restore exactly. Mine trigger eligibility
  is separate from blast eligibility; airborne exclusions, chilling, vault landing,
  targeting and mower contact use inspected source rules. Matching mechanics remain.
- Research uses 128 complete games per unchanged sequential-Q update. Ten actual
  pre-action Q outputs piggyback on the existing collector readback. Every worker's
  entire current-game plant/dig journal uses bounded RAM and disk spill and streams
  into checkpoints. Focus/fullscreen, paged browsing and follow-latest expose raw
  historical returns without tile maps. Switching replaces board and history together.
- Verification: independent math/boundary controls preceded mechanics changes;
  CPU/CUDA states, events, RNG, snapshots/replays and all ten saving combinations
  pass. A 128-game short interruption control preserves policy, optimizer, RNG and
  history. Three warmed five-second cutoff viewer pairs measured **3.04% median
  overhead** and identical paired policy hashes, within 183.20 seconds total.
  See [validation](validation.md) for regression checks, full measurements and scope.
- Compatibility: game simulation/snapshot identifiers and strict pin changed;
  fresh experiments are required. Q architecture and optimizer/exploration protocols
  are unchanged. Existing files and offline report readability are preserved.
- Remaining limits: reconstruction evidence and compact numerical mechanics are
  not original-binary equivalence. The frozen baseline now loses hard seed 4;
  this is recorded without tuning the lesson. Larger cohorts delay feedback and
  repeat the frozen controller longer. Five-second performance controls establish
  neither full-game throughput nor learning improvement. No formal training ran.

## 0.23.0 — 2026-09-26

- Problem: action kinds/digging used critic values while species/tiles used a
  separate PPO actor and alternating roles. The user requested one agent assembling
  commands through two Q-value judgments.
- Change: one shared spatial/history encoder, ten branch values and a shared
  nine-branch-conditioned 45-tile value head. Full-game Monte Carlo targets fit
  selected outputs with a balanced wait/plant/dig loss and one Adam optimizer.
  The actor, PPO/KL/entropy machinery and role schedule are removed. Compact
  simulator action transport, observations, memory, rewards and game pin remain.
- Exploration: only a winning planting branch explores. Independent species/tile
  coins use 1−sqrt(1−budget), preserving the requested 10% → 0.1% combined budget.
  The viewer shows raw Q values, branch comparisons, tile maps and overrides;
  terminal and report metrics expose pre-fit errors and missing action coverage.
- Evidence: derivation and independent controls preceded implementation. Bounded
  CUDA checks for seeds 101/102 finished in 8.61 seconds with real planting and
  16 Q optimizer steps each; complete throughput was 2,141/2,178 transitions/s.
  These one-second episodes establish operability only. See validation for tests,
  memory/timings and limitations; no formal learning was launched.
- Compatibility: fresh models required. Protocol manifests reject retired archives
  before class loading. New-protocol resume preserves one optimizer, RNGs, incomplete
  trajectories and fitting cursor. Historical runs, recordings and reports remain.
- Remaining limits: greedy selection cannot force exploration from waiting; untried
  values can still be wrong. Shared targets do not enforce exact branch/tile agreement.
  Neither the short controls nor the sequential-Q paper prove this agent will learn.

## 0.22.0 — 2026-09-25

Viewer follow-up: the Q row did not explain the comparison and could show a
planting value even when planting was unavailable. The controller and diagnostics
now share legal-value masking. Panels show the selected kind's lead over the next
legal kind, deterministic ties or a forced choice, with more precise Q values.
Independent arithmetic, legality/RNG controls and resized renders verify this
presentation change. Values can still be inaccurate; training and coverage are
unchanged. Existing checkpoints and runs remain compatible and preserved.

- Problem: the previous scheduler fitted both networks after every complete-game
  cohort, unlike the requested 256-game alternation. Long cohorts spent most time
  gathering full structured history and creating pinned buffers before critic work.
- Changes: an explicit saved role counter credits each optimized 32-game cohort,
  switches after 256 games, resets at stage changes, and records empty actor coverage.
  Inactive network/optimizer state stays fixed. Raw-only float32 gathering, bounded
  GPU token caching, reusable staging and one-batch-ahead transfer preparation reduce
  host work while preserving the authoritative CPU/disk trajectory and sample order.
- Viewer: critic return estimates and actual conditional planting probabilities,
  species-selectable tile heatmaps and decision timestamps. Manual and automatic
  replacement wait for unfinished undisplayed games; no viewer input resets a game.
- Verification: independent returns/token/gradient/Adam/RNG controls, role and
  interruption boundaries, viewer parity and matched bounded CUDA workloads.
  Measured median transport-control throughput improved 58.6%; see validation for
  conditions and limits. No learning-performance conclusion is drawn.
- Compatibility: new optimizer protocol rejects previous full resume; compatible
  0.21.0 inference/weight initialization and every existing artifact remain available.
  Greedy all-wait behavior and missing plant experience remain unresolved risks.

## 0.21.0 — 2026-09-25

The prior model sampled action kinds and fitted short-rollout values. The user
requested greedy top-level decisions with conditional planting PPO and actual
complete returns. The new controller has 47 critic outputs and an eight-species,
eight-tile-map actor, each with independent temporal features and Adam state.
Collection uses 32 complete games followed by separate critic and actor phases.
Short-rollout GAE, periodic overlap and initial actor warm-up are removed.

Game 1.5.0 corrects production/recharge/combat/headless mechanics and adds a
portable per-game RNG. Research adds five frontmost headless flags and compact
phase mapping, giving 286 observations and 289-value raw temporal tokens.
Old files remain preserved; changed signatures require fresh models. The old
five-flower/three-shooter saving control now loses; a revised public controller
uses a mine while saving for defense without changing the lesson configuration.

The architecture is rewritten around exact inputs and training responsibilities.
Derivations and source limits are in docs/math; verification evidence is recorded
in validation. Greedy untried values remain a coverage risk. No claim is made
that learning collapse is resolved, and no formal learning run was launched.

## 0.20.0 — 2026-09-25

- Problem: training exposes numerical curves but no live games; users cannot see
  sampled plant/dig behavior or follow individual games. The console also omitted
  cumulative net value and discounted return despite recording them in JSONL.
- Cause: full game state remains on CUDA and existing rendering serves offline
  recordings; per-tick synchronous rendering would interfere with the collector.
- Changes: one spawned four-panel renderer, private random subscriptions, manual
  per-panel switching, final-board retention and automatic replacement. Two pinned
  staging buffers sample selected present state before resets; bounded queues and
  generation/sequence checks discard stale frames. Closing/failing the UI leaves
  training active. The terminal adds explicitly scoped game reward/economy means.
- Organization: replace the flat implementation directory with `envs`, `policy`,
  `learning`, `evaluation`, `monitoring` and `presentation` packages. Update imports,
  resource ownership, tests and package data. Narrow deserialization aliases keep
  compatible 0.19.0 class references loadable without duplicate source files.
- Compatibility: output-only settings preserve 0.19.0 resume/weight transfer and the
  engine pin. Existing runs/artifacts remain intact. No formal training launched.
- Verification and overhead results are recorded in [validation](validation.md).
  The final median viewer overhead was 0.55% (2,466 → 2,452 transitions/s), with
  matching final policy hashes in all three pairs. Individual comparisons ranged
  from −2.74% to +8.91%; performance remains hardware/load dependent.
  The viewer samples wall-clock frames, so it cannot display every intermediate
  state and is not a statistical replacement for completed-game diagnostics.

## 0.19.0 — 2026-09-25

- Problem: placement remained around 20% wins in the reported long run. Equal
  wait/plant-kind initialization did not give equal species probabilities, and
  injected branch exploration retained learned tile preferences. Ordinary training
  did not save hardware load samples, so stopping left no utilization history.
- Changes: remove placement and start with unrestricted, three-lane saving at 100
  sun. A checked no-income argument rules out winning without flower production;
  five public-state flower investments and reactive shooters win all ten lane
  combinations at 136.52 seconds on CPU and CUDA. These controls do not prove PPO
  will learn the lesson.
- Action initialization: wait weight 1.2, weight one per available species,
  exp(−12) total digging weight, and uniform legal tiles. A complete-action injected
  prior also explores uniform tiles. Joint likelihood, entropy and KL use that same
  mixture. Reward, architecture, engine pin, environment count and phase schedules
  are unchanged. New distribution signature requires fresh models.
- Hardware: one bounded background monitor for training/benchmarks; flushed raw
  JSONL, UUID-selected device metrics, session/activity labels, atomic hardware
  panels after evaluation attempts and graceful stopping, and model-free report
  regeneration. The terminal distinguishes recent task, window and run averages.
- Verification discovered unavailable compiler tracing interfering with same-seed
  reproducibility. The installed Torch compiler restores global CUDA RNG state;
  entering it beside collection is unsafe when tracing is unnecessary. Missing
  Triton is now detected before tracing. The strict output-enabled/disabled
  parameter-equality control then passed without relaxing its assertion.
- Current verification and telemetry overhead are recorded in
  [validation](validation.md): 606 research regressions and 224 pinned game tests
  passed. Three paired telemetry checks retained identical policy hashes, with a
  2.16% median sampling cost. All existing runs and artifacts remain preserved.
  No formal training or learning comparison was launched. Lesson mastery and
  improved learned competence remain unmeasured.

## 0.18.0 — 2026-09-25

- Problem: the stopped placement run reached 3,781 games without mastery. Injected
  exploration fell from 10% during critic warm-up to about 0.016%, while recent wins
  declined to 8–10%. Actor learning also reduced the observed rate from about 4,068
  to 2,680 transitions/s.
- Exploration change: separate warm-up and formal schedules. Formal exploration starts
  at 5%, entropy starts at full strength, both decay over 3,000 actor-learning games,
  and floors of 0.1% injected noise and 10% entropy remain until mastery. Stage entry
  resets the clock. Validation explicitly disables injected exploration.
- Runtime change: CUDA sequence index templates avoid rebuilding full Python index
  lists; frozen evaluation uses inference mode; masked categorical hot paths use
  tensor-only operations; fixed-shape feature paths can use `torch.compile` without
  changing checkpoint keys. Host/device phase telemetry is recorded per rollout slot.
- Compatibility: retired-schedule checkpoints remain readable for inference but cannot
  resume or initialize new training. Same-protocol resume and stage handoff remain
  available. Existing `runs/` and artifacts are preserved.
- Verification: the final research regression passed 618 tests; CUDA doctor,
  dependency checks, packaging and Ruff passed. Three warmed eager benchmark
  repetitions reached a median 2,608 transitions/s, while the compiled path did not
  complete its bounded third repetition. The required 20% throughput gain was not
  demonstrated, so the result is inconclusive. No formal training comparison was
  launched by this change.

## 0.17.0 — 2026-09-24

- Problem: rare digging probabilities could rise abruptly despite sampled KL
  stopping; the 100 Hz discount weakened late outcomes enough for failed saving
  controls to outrank a successful control. Completed-game curves lagged policy
  changes and sparse action statistics could disappear during slot aggregation.
- Math first: documented and independently checked accounting, finite-game outcome
  bounds, GAE timing, exact hierarchical KL and a rare-action counterexample in
  [the math record](math/training-objective.md) before changing training behavior.
- Objective: gamma 1 preserves late outcomes; smaller development weight and mower
  expenditure prioritize natural wins under the documented default bounds. GAE
  retains a calibrated time-based trace. Values remain configurable, and no
  plant-retention rule or special digging reward is added.
- Updates: exact KL checks each available slot against frozen behavior. Excessive
  or non-finite movement restores actor weights and Adam state for the whole window;
  critic updates and collection continue. Sampled stopping persists across slots,
  advantages normalize once per rollout, and entropy decays with stage progress.
- Diagnostics: action-category sums/counts and pooled explained variance preserve
  sparse evidence; attempted/retained actor steps, rejection reasons, effective
  entropy and discounted reward components separate policy updates from past games.
- Compatibility: one architecture, one recipe, 128×128 collection and the same game
  pin. Existing runs are preserved. Earlier compatible weights support inference
  and initialization; full resume requires the new optimizer protocol.
- Verification: mathematical controls, CUDA integration, independent PPO/Adam,
  checkpoint/replay checks and complete regressions are recorded in validation.
  No learning comparison or formal training was launched; collapse resolution and
  throughput effects remain unmeasured. The user will run learning experiments.

## 0.16.0 — 2026-09-24

- Problem: the synchronous scheduler left the CUDA simulator idle during PPO updates
  and the learner idle during rollout collection; live traces showed roughly 1.7 s
  collection followed by 2.7–2.8 s updates.
- Change: periodic two-slot on-policy collection is now the only training scheduler.
  A frozen behavior snapshot fills both slots while the learner consumes the first;
  policy-version hashes, CUDA-stream events, queue waits and overlap are recorded.
- Compatibility: checkpoints require the periodic pipeline signature and fresh models
  are required. Files under `runs/` are preserved for the next session; historical
  generated artifacts outside `runs/` are cleanup candidates.
- Verification: compact-policy regression, CUDA smoke, checkpoint reload and periodic
  resume passed. Three short warmed measurements improved median throughput by
  5.8% (2,570 vs 2,430 transitions/s) at 46% more allocated VRAM. No games completed;
  learning effects remain unmeasured. Final review also replaced estimated overlap
  with interval measurements, made slots reusable, and deferred Ctrl+C until both
  updates and the episode ledger are committed.

## 0.15.0 — 2026-09-24

- Problem: plant species competed directly with waiting and digging, and stage runs
  could exhaust their time/game allowance before meeting mastery.
- Changes: a three-way wait/dig/plant head, conditional species head and nine tile
  maps use one forward pass with consistent masked mixture probabilities and PPO
  ratios. The initial dig bias stays -12. Conditional species entropy is explicit.
- Stage execution: optional `--until-stage-complete` disables both training ceilings,
  keeps completed-game probes and episode cutoffs, and stops only after the selected
  stage passes. Interruption/resume retains both optimizers, counters and pending
  validation; ordinary bounded runs remain the default.
- Representation: centralized command dimensions preserve compact simulator IDs.
  The factorized history trial was 4–5% slower and its bounded competence checks
  were inconclusive, so the existing history embedding remains. Temporary comparison
  code is removed; measurements are in [validation](validation.md).
- Compatibility: event_transformer_v2 requires fresh weights; one policy/recipe,
  281 observations, 128 environments, game checkout and dependency pin remain unchanged.
- Verification: independent probability/gradient/PPO controls, stage state-machine
  tests, current regression suites and bounded CUDA checks are recorded in validation.
  Neither the hierarchy nor unlimited stopping establishes improved playing strength.

## 0.14.0 — 2026-09-24

- Problem: placement frequently dug its single shooter during actor warm-up; training
  throughput was dominated by PPO updates rather than simulation.
- Cause/evidence: repeated stochastic choices accumulated a substantial dig hazard at
  100 Hz. Small categorical embedding backward sorted many repeated history indices.
  These findings do not establish the sole cause of poor lesson success.
- Changes: event_v6 has 281 public inputs, omitting projectiles, spawned counts and all
  mower details except spent. Small trainable tables use equivalent matrix products
  during backpropagation. Distribution construction applies masking once. Initial dig
  bias is -12; the user reduced parallelism to 128, with 128 transitions each.
- Compatibility: fresh models; one architecture/recipe, existing runs and reports preserved,
  game checkout and pin unchanged. Warm-up, rewards, learning rates and PPO remain unchanged.
- Verification and limitations: measured checks and bounded comparison results are recorded
  in [validation](validation.md). No formal training or claim of solved lesson learning.

## 0.13.0 — 2026-09-24

- Problem: each region duplicated zombie state/pole counts alongside threat composition.
- Change: nine regional values and 342 total inputs; nearest unused-pole distance
  replaces the pole count and five state counts. Negative distance sentinels preserve
  presence changes without retaining every movement tick. CPU/CUDA share layout offsets.
- Compatibility: fresh event_v5 models; game 1.4.0, simulation 1.1.0 and source pin unchanged.
- Maintenance: remove duplicate historical lesson/observation tests and consolidate
  the standalone dig-bias test into policy controls. Current lesson and math controls remain.
- Verification/results: see validation. Learning benefit and sufficiency of the
  reduced behavior information remain unmeasured; no formal training or comparisons.

## 0.12.0 — 2026-09-24

- Problem: explicit countdowns bypass temporal inference, while dense tick history
  would grow excessively at 100 Hz.
- Change: 417 timer-free inputs, independent actor/critic event Transformers,
  deduplicated raw history, contiguous chunks with public-prefix reconstruction,
  actor-only evaluation and 25 FPS presentation independent of simulation.
- Engine: game 1.4.0 / simulation 1.1.0 at 100 Hz. Fresh models are required.
  Local 20 Hz recordings were removed by metadata; the other game checkout is untouched.
- Checks and measurements: see validation. Retaining gamma/lambda 0.999 per tick
  shortens physical credit; per-tick exploration also occurs more often per second.
  Both are explicit confounds. Correctness is not a learning success claim.
- The timer/history controls passed, but the bounded paired learning check did not:
  the candidate reached about 1.6k transitions/s versus 8.7–9.9k for the archived
  feed-forward source and completed no training games in either 120-second window.
  Saving validation remained 0/3 for both seeds. Keep this method experimental.

## 0.11.1 — 2026-09-23

- Problem: saving collapsed into waiting; the transferred critic was optimistic
  about saving and the final critic undervalued rarely visited planting states.
- Change: configurable wait/plant exploration starts at 10%, holds during 1,024-game
  critic-only adaptation, then exponentially reaches 0.1% at stage game 3,000 and
  continues toward zero. Ordinary masked PPO uses the actual mixed probabilities.
  Digging remains legal and trainable. No reward or network changes.
- Compatibility: 0.11.0 weights still load. Explicit current configuration enables
  the new settings on weights-only transfer; resume retains the saved experiment.
  Warm-up and decay reuse saved stage residency and preserve optimizer identities.
- Verification and bounded learning findings: see [validation](validation.md).
  The initial 256-game adaptation was insufficient; 1,024 games retained placement
  in short checks, but saving still failed and critic errors remained mixed.
  Only three actor-updating rollouts fit each final check. Remains experimental.

## 0.11.0 — 2026-09-23

- Problem: saving-stage destructive digging remained despite tighter lessons.
  Prior logs showed zero saving wins and a late window of 200 digs per 200 plants.
  Digging itself was negatively rewarded, but potential shaping telescoped and
  zero-time actions advanced decision discounting. These are mechanisms to test,
  not a proven sole explanation of the learned behavior.
- Change: one `net_value_v1` account values actual production, effective plant
  damage, health-weighted assets and mower expenditure. Purchases conserve value;
  losses and damage are charged/credited once. Historical maximum/drawdown is
  diagnostic. Gamma and lambda now advance with simulation ticks.
- Implementation: shared CPU/CUDA metric schema, research-local integer event
  counters, duration tensors, terminal-observation timeout targets and physical-time
  GAE. Actor/critic, input, curriculum, PPO settings and game pin remain unchanged.
- Compatibility: fresh 0.11 models required; retired reward and checkpoint
  conversion removed. New-method stage initialization and resume remain supported.
  Old generated outputs are retired; one recipe remains.
- Verification: independent ledger/time controls, complete regressions and the
  bounded two-seed fresh-saving comparison are recorded in [validation](validation.md).
  No formal training or final-test evaluation is launched. Learning remains experimental.

## 0.10.4 — 2026-09-23

- Problem/hypothesis: early digging recurred. Free lesson income and spare time
  may make wasted purchases too recoverable; this is a curriculum trial, not a
  proven explanation of the easy-stage collapse.
- Change: the current recipe disables sky income in placement/saving. Placement
  has 100 sun and ticks 1/21/41; saving has 150 sun, two lanes and repeated waves
  at ticks 860/1100/1340. All quantities remain configurable. No legacy lesson
  mode, behavioral restriction, extra reward or alternative policy is added.
- Implementation: detached CPU episode rules and a narrow checked CUDA adapter
  supply per-game sky amounts, preserving mixed batches, reset boundaries,
  recording hashes and the installed game pin. No game checkout changes.
- Verification: income/rate calculations, all lane cases, exact delay boundaries,
  plant/dig failures, mixed income/cap events, CPU/CUDA agreement, recordings
  and regressions are recorded in [validation](validation.md).
- Limit: verified feasibility does not demonstrate improved learning. Tight
  lessons can hurt exploration and transfer; normal wins and planting must be
  measured alongside digs. No formal training launched.

## 0.10.3 — 2026-09-23

- Problem/cause: rollout wording suggested a 128-action game cap, and the requested
  default parallelism is now 256. The actual collector continues unfinished games
  across updates; waiting was included in the ambiguous decision count.
- Change: 256 environments, 32,768 transitions per rollout; explicit CLI/startup
  wording and separate accepted plant/dig diagnostics in episode records, progress
  and reports. Waits/rejections do not increment that diagnostic. Existing saved
  configurations retain their own parallelism on resume.
- Investigation: archived comparisons contain about 99.2% waiting transitions,
  often with alternatives still legal. Record weighted waiting-sample optimization
  and temporal-abstraction tradeoffs in the existing research/reference documents.
  Neither proposal is enabled; rewards, per-tick choices and PPO losses are unchanged.
- Verification: see [validation](validation.md) for regression and continuation
  controls. No formal training or stronger-learning claim.

## 0.10.2 — 2026-09-23

- Problem/cause: normal easy/standard/hard evaluation ran periodically and at
  completion even while the current curriculum stage had not passed; 500-game
  probes added frequent evaluation work.
- Change: mastery probes every 2,000 completed games; normal validation only
  after each individual stage succeeds. Keep 100/100 mastery and all evaluation
  cases. Save pending stage-success evaluations so resume can finish them before
  further learning; avoid duplicate final evaluation. Reports/checkpoints remain
  available without a validated model, and suites label missing evaluations.
- Compatibility: saved periodic recipes keep their schedules. Explicit current
  config with weights-only initialization adopts the change. Game pin, policy,
  rewards and optimizer settings are unchanged.
- Verification: 388 research, 207 CPU game and 222 CUDA game tests passed;
  scheduling, deadlines, checkpoint/resume, verified demos, reports, packaging
  and documentation checks are recorded in [validation](validation.md).
- Limit: fewer normal validation points delay detection of within-stage
  regressions. No formal training or learned-performance claim.

## 0.10.1 — 2026-09-23

- Problem: free sky sun let the single-lane saving lesson pass without any
  sunflower. Mastery mainly repeated placement; living-resource shaping was
  stronger than requested, and hardware comparisons stopped at 128 games.
- Change: three simultaneous basic zombies in three distinct lanes at tick 1000,
  still 50 starting sun and ordinary rules. The 275-sun pre-breach budget cannot
  buy three shooters; a two-sunflower control supplies a feasible win in every
  lane combination. Lane count is configurable; omitted counts retain old cases.
- Reward: reduce economy weight from 0.5 to 0.1 without adding reward terms,
  planting restrictions or a direct purchase bonus. Death/digging shaping also
  becomes weaker; no win-rate improvement is claimed from this change alone.
- Runtime: configurable CUDA benchmark sizes through 1024 games, three repetitions,
  failed/incomplete-profile exclusion, warmup separation and load/memory records.
  New default is 1024×128 after a 41.87% median throughput gain and a longer
  completion/reset confirmation of 42.40%; peak sampled GPU memory is 1427 MiB.
  Verification: 376 research, 207 CPU game and 222 CUDA game tests pass, plus
  focused controls and a 1024-game batch smoke with three verified demos.
  Full measurements and limits are recorded in [validation](validation.md).
- Compatibility: unchanged network and game pin. Explicit current-config
  weights-only initialization adopts new settings; resume retains saved settings.
- Remaining limits: task feasibility is not learned mastery; the old saving
  scores are not comparable to the new task. Larger rollouts may change learning
  behavior. No formal training or new learning-performance comparison is launched.

## 0.10.0 — 2026-09-22–23

- Problem: easy-stage training rapidly increased early digs and lost lesson skill.
  Mixed completed-game logs initially showed 100% wins from short rehearsals alone.
  Shared learned features let value-loss updates directly alter action probabilities;
  their causal contribution to collapse requires comparison evidence.
- Change: independent actor/value encoders (169,467 total parameters), separate Adam
  states and clipping, critic epochs continuing after actor KL stop. Frozen critic
  inference is batched after action collection. Rewards and exploration are unchanged.
- Diagnostics: separate task windows, plant usage, episode/transition composition,
  legal-dig probabilities, gradient norms, optimizer steps and post-update drift.
- Compatibility: tensor-only 0.9.0 weights conversion duplicates the shared encoder;
  current checkpoints resume both optimizers. Preserve runs and the game pin.
- Sources: inspected DeepSeek report, What Matters architecture findings and SB3
  interfaces; adopted gradient isolation and phase-specific work, not LLM algorithms.
- Verification/results: independent controls, full regressions and paired five-minute
  transfer comparisons are recorded in [validation](validation.md). The first 200
  games are examined separately. No formal training is launched.
- Remaining problem: easy validation improved for both transfer seeds, but saving
  retention failed and seed 102 developed destructive digging later. Throughput
  was lower. The method remains experimental; separation alone is not a fix.

## 0.9.0 — 2026-09-22

- Problems: too many recipes and reward terms; destructive digging, unstable saving
  mastery and large PPO divergence. Choice-only reduction amplified a small subset
  of transitions; the old representation duplicated several signals.
- Changes: one 500-value categorical observation, one 118,611-parameter spatial
  grouped policy, one TOML recipe, three reward components and standard PPO means
  with configurable KL stopping. Remove old policies, recipes, comparison commands,
  event rewards and the unused early-dig penalty proposal. Keep digging legal.
- Checkpoint behavior: structural signatures gate weights-only initialization;
  new optimizer/counters/schedules allow explicit reward/PPO/curriculum changes.
  Resume preserves the saved experiment. Retired weights cannot load; reports and
  recordings remain independent. Obsolete local checkpoints were deleted at the
  user's request, preserving their recorded evidence.
- Verification: complete research/game regression, CPU/CUDA math and gameplay
  controls, reload/transfer/stage/report/replay tests, packaging and bounded fresh
  comparisons. Exact counts and results are in validation.md.
- Results/limits: both compact pilots scored 0/20 deterministic placement wins,
  versus 20/20 for both references; the improvement criterion failed. The method
  remains experimental. Regional compression is partial observation, resume resets
  active episodes, and the short comparison cannot isolate individual changes.

## 0.8.1 — 2026-09-22

- Follow-up: The user requested perfect 100-game mastery and less frequent validation.
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
