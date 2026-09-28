# Cohort exploration schedule

For completed stage games g, p=min(1,max(0,g)/3000) and
budget=0.1*(0.001/0.1)^p. Games 0/1500/3000 resolve budgets 0.1/0.01/0.001.
Later games hold 0.001. Stage advancement resets g; no warm-up or entropy schedule
exists. Non-curriculum collection uses its completed training-game progress.

Resolve once before each cohort. Collection and fitting keep that value fixed;
resume restores the same saved budget, progress and RNGs. Deterministic validation
sets exploration to zero in a restoring context and does not advance training
progress.

The ten-way branch winner is always greedy. Its tile selector uses one coin with
probability equal to the current budget. Wait has no tile and therefore no coin;
plant tiles use occupancy candidates and dig tiles use all board tiles. A fired
coin can still select the preferred tile, so the number of changed commands can
be lower than the number of tile coins. See the probability controls in
[sequential Q control](sequential-q-control.md).
