## Rule-guided scenario training and unresolved exploration — 2026-11-01

- Reported problem and impact: exploration remains unresolved in scenarios
  without a clearly correct answer. There is no satisfactory criterion for
  judging better behavior or assigning useful rewards. Strong results on taught
  scenarios therefore do not yet establish reliable learning of general play.
- Possible causes, all speculative:
  - **The missing signal may be a meaningful preference rather than action
    diversity alone.** Correct-answer exercises supply an external ranking;
    ambiguous situations lack that reference, so more exploration or better
    reward fitting may still fail to establish useful choices.
  - **Strategic value may depend on coordinated sequences and later policy
    behavior.** Resource preparation, timing and complementary plants can make
    individually weak choices valuable together. Incomplete follow-through can
    make useful preparation appear wasteful, obscuring its credit even when
    alternative placements are compared from the same board.
