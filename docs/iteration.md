## Initial learning gains and passive late-game behavior — 2026-10-28

- Observed progress: training begins to teach the agent to wait when planting
  is unavailable and to plant when a plant becomes usable. It also learns the
  desired opening of planting a Sunflower in the first frame, showing an initial
  ability to distinguish action availability and invest in early income.
- Time incentives: average game duration clearly decreases with training,
  consistent with completion-time rewards and penalties influencing behavior.
  Shorter games alone do not establish better play, since they can also end in
  earlier defeat.
- Comparison and possible reasons: compared with the previous recorded version,
  the separation between species selection and placement, followed by learning
  from complete games, is unchanged. Recent corrections to how invalid choices
  train the responsible stage may have helped distinguish unavailable planting
  from waiting. Existing time incentives and extra credit for early Sunflower
  income may help explain the duration trend and opening choice; their individual
  effects have not been isolated.
- Evidence: these outcomes are the user's training observations. The current
  learning and reward rules support the explanations above, but no new training
  experiment or controlled comparison was performed for this record.
- Remaining issue: later play remains passive, tending to plant Sunflowers with
  only one or two other plants. The initial learning gains have not yet
  produced a varied, sustained defensive strategy; the cause remains unconfirmed.
