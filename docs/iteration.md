# Iteration history

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
