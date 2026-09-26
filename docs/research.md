# Research design — 0.24.0

The single supported method is a two-level, complete-return Q controller. A shared
public-history Transformer first values wait, eight species and digging, then
values tiles conditioned on the chosen non-wait branch. One complete command
advances the game. Legal argmax is deterministic except for plant-only exploration.

Collect 128 complete games with frozen weights; finalize gamma-one reward-to-go;
fit selected branch and tile values with one optimizer; then synchronize and run
curriculum/deadline checks. The non-wait loss averages both errors. Populated
wait/plant/dig groups receive equal aggregate weight. The checked equations,
partial-minibatch rule, exploration probabilities and counterexample to maximum
consistency are in [sequential Q control](math/sequential-q-control.md).

The exploration budget is 10% → 0.1% over 3,000 stage games. Splitting it into two
independent coins uses 1−sqrt(1−budget) per head. Exploration can change species
and tiles only after a planting branch wins; it cannot force planting when wait
wins. Coin firing and actual choice changes are separate diagnostics. Evaluation
is entirely greedy. No entropy or policy-likelihood objective remains.

The observation, event memory, widths, net-value reward, saving lesson and pinned
100 Hz clock remain unchanged. Game 1.6.0 corrects numerical gait, chilling and
source-based contact/targeting/blast/vault geometry; this is a community-source
reconstruction with documented limits, not original-executable equivalence. Complete trajectories stay in bounded
CPU/disk storage with disposable raw-token caching and one-minibatch prefetch.
See [architecture](architecture.md) for every input and recovery path.

This adaptation uses Monte Carlo regression, not the bootstrapped/off-policy
algorithm in the inspected sequential-Q paper. Shared targets do not guarantee
optimal choices, branch/tile agreement or coverage. Inaccurate or untried values
can sustain poor decisions. Negative waiting controls establish that planting
can become preferred, not that normal games will be learned.

Validation reports branch/tile errors before fitting, per-species coverage,
accepted actions, accounting rewards, cutoff failures and throughput. Bounded
short-episode integration checks establish operability only. No formal training
or learning comparison is launched for this release. Prior findings and performance
measurements remain version-labelled in [iteration](iteration.md) and
[validation](validation.md); inspected sources and limits are in [references](references.md).
