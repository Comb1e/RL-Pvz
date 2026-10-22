# System architecture

This project learns plant placement in a pinned Plants vs. Zombies simulator.
It has one stateless two-stage Transformer Q policy and one autonomous learning
framework. Human annotations and verified demonstrations provide optional
starting weights for that same policy.

The project owns learning, reward interpretation and experimental evidence.
The separately installed game owns combat, legal actions and game rules.
Autonomous learning requires an NVIDIA GPU with native BF16 support; CPU
simulation supports annotation, recording, baselines and reference checks.

## Project framework

| Component | Responsibility |
|---|---|
| Simulation and public-state boundary | Execute proposals, expose the current board and report actual outcomes |
| Policy and exploration | Choose an action category and location, with controlled exploration during collection |
| Complete-game learner | Turn finished trajectories into targets and update the shared policy |
| Human supervision | Collect reachable-state labels and verified games, then produce compatible starting weights |
| Experiment control | Resolve settings, manage curriculum and budgets, validate progress and recover interrupted runs |
| Evidence and presentation | Preserve provenance, evaluate frozen policies, report results and display actual gameplay |

```mermaid
flowchart LR
    Game[Pinned game] --> World[Simulation adapter]
    World -->|public board| Policy[Two-stage policy]
    Policy -->|proposal| World
    World -->|execute| Game
    World --> Trajectory[Actual game trajectories]
    Trajectory --> Learn[Complete-game learning]
    Learn -->|updated weights| Policy
    Human[Annotations and demonstrations] --> Init[Verified initialization]
    Init -->|starting weights| Policy
    Policy -->|frozen weights| Evaluate[Evaluation]
    Evaluate --> Evidence[Results and reports]
```

The interaction loop produces actual experience; the learning loop consumes
completed games. Human supervision feeds the same model through initialization.
Evaluation uses frozen weights and supplies evidence rather than training data.
Experiment control coordinates these workflows, while viewers observe them.

A single [training configuration](../src/pvz_rl/data/train.toml) defines the
bundled model, learning and task settings. A run resolves its configuration
before execution and records the effective settings with provenance. Hardware
profiles adjust execution without defining a different policy or objective.
The game dependency is installed from a verified source pin, and its installed
sources and rules are checked before experiments.

## What the policy knows and decides

The policy sees only the current public board: plants, zombies, projectiles,
mowers, resources, card cooldowns and public progress counters. A shared encoding
presents individual entities and global facts to both stages. Entity storage is
bounded; truncation is reported, and transport padding carries no information.

Seeds, future spawn schedules, simulator snapshots, source identities and reward
ledgers remain private execution data. Annotation provenance and viewer history
also stay outside model inputs. The policy has no learned memory between
decisions.

The first stage predicts values for wait, each plant species and dig. The
independent second stage receives the chosen non-wait category and the same
board, then scores the 45 plots. Both stages share the public representation
but have separate learned parameters. The first stage predicts category values
directly; it does not run every possible placement to choose a category.

The simulator remains the authority on whether a proposal succeeds. Resource
and cooldown restrictions do not remove categories from ordinary policy
selection, and placement considers occupied plots as well as empty ones.
An impossible category can skip placement inference while preserving its
rejected proposal. This lets learning distinguish a poor category choice from
a poor location within an available category.

Accepted planting and digging change the board without advancing combat time.
Wait or rejection advances one tick; rejection executes wait while retaining
the proposed action and reason. Collection may explore locations or maintain a
commitment to another plant species until it can be placed. Evaluation disables
these exploration mechanisms.

## Learning from complete games

Training collects a fixed cohort of games. Finished games remain paused until
all cohort games finish, so fitting uses complete trajectories from one
collection period. Only executed decisions enter those trajectories; inactive
slots and padding create no experience. Stable episode/action identities keep
observations, outcomes and reward evidence associated during storage and recovery.

```mermaid
stateDiagram-v2
    [*] --> Collect
    Collect --> Finalize: all games and issued decisions finish
    Finalize --> Fit: rewards and complete returns ready
    Fit --> Fit: next complete fitting pass
    Fit --> Boundary: all passes committed
    Boundary --> Collect: continue training
    Boundary --> Evaluate: mastery check or validation due
    Evaluate --> Boundary: required evaluation completes
    Boundary --> [*]: budget or mastery stop
```

Reward finalization precedes fitting. At the boundary, experiment control can
continue, evaluate or stop. An incomplete required validation remains pending
and prevents further updates until it is completed or the run is saved.

Rewards combine the terminal result, changes in resources and living plant
value, and credited damage, blocking, produced sun and mower expenditure.
Completion-time ranking adjusts terminal rewards after the cohort finishes.
The learning target is the complete undiscounted remaining return; it does not
bootstrap a prediction beyond the recorded game. Diagnostic quantities are
kept separate from reward components used for learning.

Credit attribution has an important backend boundary. CPU/reference games and
reconstructed demonstrations can assign delayed owned effects to the originating
placement. The current CUDA collector consumes aggregate effects on the
transition where they occur. Equal aggregate accounting therefore does not imply
identical placement-time targets across those paths.

Fitting assigns responsibility according to the actual proposal and outcome.
Wait learns through the category stage; accepted plant/dig returns train the
selected placement. Impossible categories and bad locations train their
respective stage toward an invalid-action target, which is not an episode
penalty. A slowly updated reference policy supplies consistency targets that
connect accepted category choices to their placement values.

Every fitting pass covers the prepared cohort before committing an optimizer
update. Balancing uses cohort-wide populations so splitting work into batches
does not redefine the objective. Repeated invalid decisions can be represented
more compactly for fitting, while complete returns retain every actual decision.

## Human supervision and initialization

Human supervision is independent of a running autonomous learner. Both routes
produce full-model weights that can start a fresh run.

```mermaid
flowchart TD
    Reachable[Reachable simulator states] --> Labels[Human action or tile labels]
    Labels --> Review[Validate labels and resolve conflicts]
    Review --> Supervised[Fit the selected policy stage]
    Supervised --> Verify[Verify saved training-set requirements]
    Verify --> Weights[Compatible model weights]
    Human[Completed human game] --> Recording[Replay and transition evidence]
    Recording --> Reconstruct[Verify reconstruction and reprice rewards]
    Reconstruct --> DemoFit[Fit both policy stages]
    DemoFit --> Weights
    Weights --> Fresh[Fresh autonomous run]
```

Annotations supervise either the action category or placement stage. Recorded
human games supervise both stages through the shared return-fitting framework.
The two paths can be composed by initializing demonstration fitting from
annotation weights.

Annotation states are reached through simulator actions rather than edits to
resources or combat state. Records retain the public snapshot, model input,
availability evidence and human label sets. Pending, labeled and discarded
records persist across restarts. Contradictory labels for identical model inputs
are reviewed together; revision checks prevent stale edits. Failure to generate
the next board preserves the saved labels and allows retry.

Annotation fitting freezes the other stage. It publishes an initialization only
when freshly reloaded weights satisfy every training record's feasible-winner
and absolute-bottom requirements. Its resumable fitting progress is separate
from portable weights. Held-out annotation results remain diagnostics and do not
establish gameplay quality.

A demonstration must contain a complete natural win or loss. Reconstruction
checks observations, proposals, outcomes, timing and replay identity before
fitting. Verified events can be valued under current reward settings without
rewriting the recording. Demonstration fitting has its own pass/time allowance
and publishes weights only at completed-pass boundaries.

## Curriculum and evaluation

The teaching curriculum progresses from easy games to mixed easy/standard games
and then a shared mix including hard games. Mastery checks use separate,
deterministic cases and require the configured residency and success criteria.
Stage changes affect future game resets and preserve learning state within a
continuous run. A separately initialized stage transfers weights into fresh
learning state.

Under the default schedule, a mastered stage triggers ordinary validation.
Complete validation chooses one shared best policy across difficulties.
Partial results cannot select a best checkpoint, and a run that never reaches
qualifying validation may have no selected best policy. Ending a run or reaching
its game budget does not imply mastery.

Training, mastery, validation, final test and changed-scenario evaluation use
separate case pools. Evaluation freezes the policy and performs no fitting.
It can refill finished simulation slots because it does not need training's
cohort boundary. CUDA traces selected for recording are replayed on CPU and
checked before publication; this check does not cover every unrecorded case.

Reports compare learned policies with fixed baselines on paired scenarios.
They reject incompatible protocols, duplicate cases and incomplete comparison
sets. Formal research orchestration explicitly runs learner attempts, frozen
checkpoint evaluations and reports, preserving a journal for restart.
Availability checks and performance benchmarks are separate evidence: neither
certifies learning quality.

## Storage, recovery and compatibility

The system distinguishes three kinds of durable information:

| Information | Purpose |
|---|---|
| Supervision | Annotation records and verified recordings support review, reconstruction and reusable initialization |
| Learning state | Autonomous checkpoints preserve unfinished trajectories with the rest of the run; supervision-specific progress preserves its own resumable work |
| Provenance and results | Resolved settings, source identities, evaluation cases and reports explain what ran and what the results mean |

Trajectory storage spans bounded RAM and disk and is released after its cohort
is fitted. Checkpoints include unfinished experience needed for recovery.
Temporary batching, transfer and execution caches are derived working state and
are rebuilt; scratch files are not the recovery contract.

Initialization transfers compatible complete model weights into a fresh
optimizer, curriculum, budget and random state. Resume continues an autonomous
run, including simulator state, unfinished experience, optimizer, reference
policy, exploration, schedules and random state. Annotation fitting has its own
resume mechanism. These operations are deliberately distinct.

Checkpoint compatibility checks engine identity, public representation, model
structure and the relevant learning/state contracts before collectors allocate.
Some incompatible run states may still provide structurally compatible starting
weights; they cannot be resumed as though their learning settings matched.

Saving stops new selection and finishes issued work before publishing one
coherent checkpoint. Replacement is atomic, preserving the previous checkpoint
if saving fails. A partial fitting pass restarts from its beginning; a committed
update is not repeated. This boundary prevents recovery from combining model
weights with mismatched optimizer or game state.

## Execution and presentation boundaries

GPU batching, compilation and memory management are execution concerns. They may
change how work is scheduled, but must preserve public inputs, targets and
learning semantics. Memory pressure can spill host storage or trigger bounded
execution fallbacks. Exhausted recovery and nonfinite learning values fail
explicitly rather than silently dropping evidence or changing the objective.

Progress, hardware sampling, journals and viewers observe actual execution.
Bounded queues prevent a slow viewer from holding up training; presentation
failure can disable the viewer while learning continues. Desktop and browser
frontends consume public frames and diagnostic decisions, not ownership of the
model or simulator. Viewer history never becomes policy memory.

Offline reports, replay videos and plots are derived artifacts. Their export
status is tracked separately, so an export failure does not invalidate completed
learning. Performance diagnostics also distinguish completed production cycles
from partial measurements; utilization or an isolated benchmark is not evidence
of total training speedup.

## Further reading

- [Quick start](../README.md) for requirements and installation.
- [Entity inputs](math/entity-inputs.md), [reward economics](math/training-objective.md)
  and [invalid-action attribution](math/invalid-action-penalties.md) for detailed contracts.
- [Execution controls and diagnostics](math/training-throughput.md) for performance work.
- [Code and test ownership](../AGENTS.md#code-structure) for implementation navigation.
- [References](references.md), [iteration history](iteration.md) and
  [retained measurements](evidence/) for research attribution and evidence.
