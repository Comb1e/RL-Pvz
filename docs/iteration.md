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
