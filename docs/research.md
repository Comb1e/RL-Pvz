# Research design

The default method fits an entity Transformer–LSTM to complete gamma-one
returns from human demonstrations and autonomous cohorts. It compares ten branch
Q values and conditional tile values using public observations only. Rejected
proposals and zero-tick actions remain part of the sequence. This is Monte Carlo
regression, not a bootstrapped recurrent DQN or recurrent PPO implementation.

The recurrent objective, group normalization, truncated gradients and exploration
probabilities are specified in [recurrent training](math/recurrent-training.md).
The event-memory model remains an explicit alternative configuration with its
own model and optimizer protocols. Weights cannot cross those model families.

Curriculum progression is easy → standard → shared. Net-value reward coefficients,
rejection penalties and the pinned game's 100 Hz mechanics are preserved. The
saving lesson and its custom scenario/rules are removed. Reward derivations are
in [training objective](math/training-objective.md).

The model never sees seeds, future schedules, private snapshots or presentation
artifacts. A completed human recording must reconstruct exactly before fitting.
A one-game initialization does not establish generalization or curriculum mastery.
Shared targets do not guarantee coverage, optimal actions or branch/tile agreement.

[Architecture](architecture.md) documents the implemented data and recovery paths;
[training instructions](training.md) provide commands. Inspected sources and their
limits are recorded in [references](references.md). Release verification belongs
in [iteration history](iteration.md). Bounded integration tests establish correct
mechanics and operability, not a learning improvement.
