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

## Separate tile stages and comparative search — 2026-10-31

- Completed changes: Cherry Bomb and Peashooter scenario training now have
  separate progression and validation. Process parallelism, CUDA acceleration,
  and workflow optimizations reportedly make Cherry Bomb training nearly 100
  times faster; useful bomb placement is largely learned after about 50,000
  environments, although the stage has not completed.
- A main-environment/sub-environment search now compares alternative placements
  from the same public board and supplies confirmed comparative learning targets.
  Action annotation and absolute-unacceptable-option annotation/fitting were
  removed as inefficient; manual supervision retains positive tile sets.
- Immediate-predecessor comparison: relative to Residual-sun incentives and
  tile-first training, the shared placement-prerequisite workflow now separates
  the two skills and their mastery gates. Placement learning gains branch
  comparisons alongside the earlier complete-game feedback, while manual
  supervision narrows to acceptable placements. Separating workloads may permit
  specialized acceleration, and comparing alternatives may better distinguish
  a decision's value from its eventual game outcome; learning benefits remain
  unverified.
- Discovered problems: the user confirms the preceding version's failure to
  learn reliably from winning games; a correction has been attempted but has
  not yet been checked through actual training. Cherry Bomb still selects plots
  containing plants, preventing completion despite useful placement learning.
  Whether comparable performance gains extend to other stages remains untested.
- Evidence: speed and behavioral observations are user-reported. Code review
  confirms the revised workflow, not the reported speedup or the correction's
  effectiveness; no training or benchmark was run for this record update.

## Rule-guided scenario training and unresolved exploration — 2026-11-01

- Reported improvements: a dedicated Cherry Bomb obstacle stage resolves the
  occupied-plot blockage, reaching 100% accuracy after 2 hours 40 minutes of
  training. Attacker preparation and Peashooter training teach attackers to
  favor lanes containing zombies and positions from which they can kill them,
  with balanced defensive preparation when no zombies are present. Both attacker
  stages reportedly reach 100% success quickly.
- Reported efficiency gains: cross-entropy supervision replaces sparse feedback
  in scenario fitting, while simple strategies with clearly correct targets
  replace unguided exploration in those exercises. Learning efficiency improves
  substantially; scenario-training performance improves by more than 100 times
  in the best stage, without establishing that speedup for every stage.
- Immediate-predecessor comparison: relative to Separate tile stages and
  comparative search, the independently validated curriculum adds obstacle and
  attacker-preparation stages. Dedicated teaching pipelines supply scenario
  targets; normal placement now uses factual same-board comparisons in place of
  the preceding search. Species and tile decisions retain separate roles, while
  specialized preparation and fitting workflows extend scenario acceleration.
  Clearer targets may reduce ambiguous feedback, and focused workloads may
  reduce training cost; their individual contributions have not been isolated.
- Discovered problem: exploration remains unresolved for scenarios without a
  clearly correct answer. How to judge better behavior and provide useful reward
  signals is still unclear, limiting the transition from taught scenario skills
  to general play.
- Evidence: accuracy, training time and performance gains are user-reported.
  Repository review confirms the workflow, not those measured outcomes; no new
  training or benchmark was run for this record update.
