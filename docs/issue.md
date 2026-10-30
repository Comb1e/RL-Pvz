## Residual-sun incentives and tile-first training — 2026-10-29

- Reported problem and impact: winning games do not produce useful learning.
  Training appears to eliminate choices from losing games and can also eliminate
  correct actions, preventing reliable reinforcement of winning play despite
  improved incentives and additional placement training. The user suspects
  backpropagation or related learning rules; the faulty component is unconfirmed.
- Possible causes below are hypotheses, not confirmed findings:
  - **Outcome credit may confuse useful decisions with the trajectory around
    them.** A correct move in an eventual loss can inherit an unfavorable target,
    while incidental moves in a win can share its reward. The resulting learning
    signal may suppress good decisions without identifying what caused victory.
  - **Fitting values may fail to preserve the ranking of winning choices.**
    Reducing prediction error on successful games need not make their useful
    actions outrank alternatives. Updates from losing games may also generalize
    across similar states and overwrite preferences learned from wins.
  - **Placement-to-action supervision may sustain a pessimistic feedback loop.**
    Inaccurate placement estimates can undervalue a useful plant; choosing it
    less often then reduces opportunities to correct that estimate through
    successful play. Extra placement training may reduce noise without resolving
    this coupled credit problem.
