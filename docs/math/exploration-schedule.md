# Phase-aware exploration schedule

The training policy uses two separate clocks within each teaching stage. Let

- (w) be the critic warm-up length, 1,024 completed games;
- (n) be completed games in the current stage;
- (D) be the formal decay length, 3,000 games;
- (q = min(1, max(0, (n-w)/D))) after warm-up.

During warm-up, the injected exploration rate is 0.10 and the entropy factor is
1.0. At actor unlock, formal exploration starts at 0.05 and formal entropy also
starts at 1.0. During formal learning,

[
  epsilon(n) = 0.05(0.001/0.05)^q,
  qquad
  h(n) = 1.0(0.1/1.0)^q.
]

At (n=w), this gives (epsilon=0.05) and (h=1). At
(n=w+D=4,024), it gives (epsilon=0.001) and (h=0.1). Both values then
remain at their floors until the stage passes. The next stage starts a new
warm-up clock.

For each legal complete action, the injected prior changes
the executed distribution to

[
  P(a)=(1-epsilon)P_\text{learned}(a)+epsilon P_\text{prior}(a).
]

The prior contains legal wait and planting branches and excludes digging. Thus
raising the injected rate explores productive alternatives without directly
sampling digs. Entropy regularization is calculated from the same mixed
hierarchical distribution, so its floor can still preserve alternative legal
tiles and action kinds.

The schedule is resolved once before each periodic synchronization window. Every
transition in that window therefore uses one epsilon, one entropy factor, and one
collection/PPO distribution. Deterministic validation sets epsilon to zero and
selects greedy learned heads; it never uses the training prior.

The floor is deliberate: a stage that has not passed its mastery gate must not
become effectively greedy solely because a fixed game-count deadline was reached.
These values are project-specific hypotheses. They establish the implemented
clock and probability calculation, not a guaranteed learning improvement.

The prior includes uniform legal tiles within each species and a configurable wait
weight. See [initialization and complete-action mixture](saving-and-actions.md).
