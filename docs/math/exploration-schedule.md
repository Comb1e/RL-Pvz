# Plant exploration

At g completed games in the current teaching stage, let p=min(g/3000,1).
Injected exploration is 0.1*(0.001/0.1)^p and the entropy factor is 0.1^p.
The values at 0/1,500/3,000 games are 10%/1%/0.1% noise and
1/sqrt(10)/0.1 entropy factors. Beyond 3,000 the floors remain active.
Stage advancement resets g. There is no separate actor-freeze phase.

Resolve these values before collecting the complete cohort and preserve them
through critic and actor fitting. Resume restores the same values and progress.
Evaluation uses no injected exploration and greedy plant arguments.

Only planting has a stochastic probability distribution. The complete
species/tile mixture and conditional PPO derivation are in
[complete-game learning](complete-game-learning.md). Wait, plant selection and
digging tiles are greedy critic decisions; entropy does not apply to them.
