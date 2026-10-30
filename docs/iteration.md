## Initial learning gains and passive late-game behavior — 2026-10-28

- Reported improvements: the agent begins to wait when planting is unavailable
  and plant when a plant becomes usable. It also learns the first-frame Sunflower
  opening the user identifies as optimal, showing initial availability awareness
  and early economic planning.
- Reported time trend: average game duration clearly decreases during training,
  suggesting that time incentives influence behavior. This does not establish
  stronger play, since shorter games can also mean earlier defeats.
- Immediate-predecessor comparison: the preceding recorded version, Saved-run
  cohort reward reports, added reward inspection without changing policy learning.
  Species selection, conditional placement, and complete-game learning retain
  their existing responsibilities and flow. Better reporting may help interpret
  outcomes, but no new structural change explains the observed gains. Existing
  availability supervision, time incentives, and early-income credit are possible
  explanations whose effects have not been isolated.
- Discovered problem: later play remains passive, tending to plant Sunflowers
  with only one or two other plants. The initial gains have not yet produced a
  varied, sustained defensive strategy.
- Evidence: outcomes are user-reported observations, consistent with the current
  learning and reward rules; no new training or controlled comparison was run.

## Residual-sun incentives and tile-first training — 2026-10-29

- Reported improvements: settling remaining sunlight according to victory or
  defeat prevents the near-always-wait-and-lose behavior. Placement requires more
  training than action selection; two placement pretraining scenarios and repeated
  placement-only training before action fitting reportedly reduce misleading
  placement-value information passed to the action stage.
- Immediate-predecessor comparison: relative to Initial learning gains and passive
  late-game behavior, reward accounting now links unused sunlight to the outcome.
  The independent action and placement stages retain their roles, while the
  workflow adds Peashooter/Cherry Bomb prerequisites and cycles several
  placement-only cohorts before a joint action/placement cohort. These changes
  may discourage passive defeat and improve placement estimates before they
  supervise action selection; their individual effects have not been isolated.
- Discovered problem: the user reports that winning games do not produce useful
  learning. Training appears to eliminate choices from losing games and can also
  eliminate correct actions, preventing reliable reinforcement of winning play.
  Backpropagation or the surrounding learning rules remain under suspicion;
  the source of the failure is unconfirmed.
- Evidence: improvements and failures are user-reported. Review confirms the
  reward and training workflow, not their behavioral effects; no new training
  or controlled comparison was run.
