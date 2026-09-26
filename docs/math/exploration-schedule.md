# Cohort exploration schedule

For completed stage games g, p=min(1,max(0,g)/3000) and
budget=0.1*(0.001/0.1)^p. Games 0/1500/3000 resolve budgets 0.1/0.01/0.001.
Later games hold 0.001. Stage advancement resets g; no warm-up or entropy schedule
exists. Non-curriculum collection uses its completed training-game progress.

Resolve once before each cohort. Collection and fitting keep that value fixed;
resume restores the same saved budget, progress and RNGs. Deterministic validation
sets exploration to zero in a restoring context and does not advance training
progress.

Planting winners use independent species and tile coins with probability
1-sqrt(1-budget) each. Wait/dig winners remain greedy. The probability that either
coin fires equals the budget; actual action changes can be less frequent because
uniform exploration includes the preferred option. See the enumeration and
counterexamples in [sequential Q control](sequential-q-control.md).
