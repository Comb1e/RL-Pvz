# Research design — 0.27.0

The first delivered workflow is a human-game initialization. `record-demo`
records one easy seed-1000 game into a verified native replay and append-only
transition archive. `initialize-demo` verifies the archive and replay, computes
complete gamma-one returns, and fits the 2-layer entity Transformer, 256-unit
LSTM and two Q heads in 256-decision chunks. The hidden/cell state is reset only
at episode boundaries; chunk detachment limits gradient propagation without
resetting forward memory. Defaults are 20 completed passes, learning rate
0.0003, clip 0.5, seed 101 and a 30-minute budget. Outputs are labelled as a
fit to one demonstration.

Human commands select `configs/demo.toml` by default. The established CUDA
collector retains its own `configs/train.toml` / event_v7 configuration;
recording and initialization require the event_v8 recurrent profile.

The observation protocol is `event_v8`: the former 286 public values plus eight
card cooldowns in plant order, each divided by `recharge_ticks + 1`. CPU and
CUDA encoders use the same offsets and normalization. Seeds, snapshots, future
schedules, and presentation artifacts remain outside policy inputs. A partial
archive cannot produce a terminal target or an initialization checkpoint.

The 128-game frozen-weight autonomous collector is a separate explicit command
using event-memory Q weights. Recurrent initialization checkpoints are not
accepted by this collector. [Training instructions](training.md) give the
commands and artifact requirements for both workflows.

The recurrent demo profile retains tile exploration starting at 50% and decaying
exponentially to a fixed 1% floor by game 5,000. Validation enters the existing
deterministic context, which sets both species and tile exploration coins to
zero for the entire evaluation pass.

The single supported method is a two-level, complete-return Q controller. A shared
public-history Transformer first values wait, eight species and digging, then
values tiles conditioned on the chosen non-wait branch. One complete command
advances the game. The argmax is deterministic across all ten branches; plant-only
exploration can alter species and tile after a plant branch wins. Invalid plant
proposals are retained as actions and are penalized by the reward ledger.

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

The CUDA collector keeps its event-memory widths, net-value accounting,
saving lesson and pinned 100 Hz clock. The current initialization workflow uses
the event_v8 entity/recurrent protocol described above. Rejected-action penalties
are added on top of the net-value objective. Game 1.7.0 corrects collision ordering, chilling and
source-based contact/targeting/blast/vault geometry; this is a community-source
reconstruction with documented limits, not original-executable equivalence.
Complete trajectories stay in bounded CPU/disk storage with disposable raw-token
caching and one-minibatch prefetch.
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

The [full-game memory framework](full-game-memory.md) is a separate researched
proposal for retaining public events and replacing the current encoder with a
joint sparse spatial–temporal transformer. It is not implemented behavior.
