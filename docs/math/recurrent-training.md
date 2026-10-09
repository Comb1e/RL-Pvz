# Recurrent complete-return Q regression and alternative-action supervision

Demonstration initialization and autonomous training use the `complete_return_probe_v3`
objective. The raw ledger keeps terminal, development, rejection, house-entry and
victory-time components separately. Fitting multiplies only development by the
configured dense multiplier before constructing complete returns.

Each living zombie is charged once when it first enters the second house-side column
and once again when it enters the innermost column. Victory time shaping is applied
only after a cohort median is known, so every victory and counterfactual terminal
transition uses the same median.
A verified demonstration supplies one naturally completed episode; autonomous
collection supplies a cohort of completed games. Both visit chronological chunks
with fixed weights for the whole pass and publish only completed optimizer updates.

For an episode of T decisions, including rejected proposals and zero-tick
operations, the target is

\[
G_t = \sum_{u=t}^{T-1} r^{train}_u, \qquad G_T=0.
\]

Gamma is one. A simulator cutoff receives its configured terminal penalty once;
there is no bootstrap. Different episode slots never share a return accumulator.
For rewards [1, 2, -1, 4] and [3, -1], independent targets are [6, 5, 3, 4]
and [2, -1]. Inactive rows with arbitrary rewards cannot alter these targets.

The branch value estimates remaining return after choosing a branch and following
the collecting tile policy. For a fixed continuation policy,
`Q_branch(s,b) = sum_t p(t|s,b) Q_tile(s,b,t)`. During tile exploration this is an
expectation, not necessarily the best tile value. Shared targets do not enforce
exact equality between heads or provide information about unvisited alternatives.

Let N_{g,o} count decisions in group g (wait, plant or dig), with outcome o
(accepted or rejected), over the complete cohort. Let K count nonempty groups.
If both strata exist alpha_{g,o}=0.5 each; the sole represented stratum otherwise
gets alpha=1. Missing strata get zero. Selected predictions are q_b and q_l. Define

\[
e_i = \begin{cases}
(q_b-G_i)^2 & a_i=\mathrm{wait},\\
\tfrac12\big((q_b-G_i)^2+(q_l-G_i)^2\big) & \text{otherwise}.
\end{cases}
\qquad
L=\frac1K\sum_{g:N_g>0}\sum_{o:N_{g,o}>0}
\frac{\alpha_{g,o}}{N_{g,o}}\sum_{i\in(g,o)}e_i.
\]

Each chunk contributes real decisions with coefficient alpha_{g,o}/(K N_{g,o}).
Denominators cover the complete cohort, never individual chunks. Summing chunks
recovers L regardless of episode partitioning. Padding creates neither wait samples
nor outcome counts. Empty groups and missing strata do not dilute the objective.
Demonstration selected-action regression uses this rule too; accepted-only ranking
is unchanged.

At the start of every pass h_0=c_0=0 independently for each episode. Forward
memory crosses chunk boundaries, but gradients treat the carried h,c as constants
at each boundary. Chunk gradients sum before clipping and one Adam step:

\[
D=\sum_j\nabla_\theta L_j(\theta;\operatorname{stopgrad}(h_j,c_j)),
\quad \bar D=D\min(1,C/\lVert D\rVert_2).
\]

Both fitting paths use `C = training.max_grad_norm`, with default 5.
PyTorch applies a small denominator epsilon and leaves gradients below the
limit unamplified. Clipping occurs once after the complete pass accumulation.

The test compares accumulated gradients when two episodes are processed together
versus separately with the same two-decision detach boundaries. Changing chunk
length preserves the scalar forward objective at fixed weights (within numeric
precision), but generally changes truncated gradients; it is not claimed to equal
full-episode backpropagation. An uncommitted interrupted pass must clear its
partial gradient before recomputing from episode starts.

## Event-only recurrent memory

Transitions supply seven gross public facts: actual sunlight gained/spent, zombies
spawned/defeated/physically removed, and plants added/removed. Any positive counter
triggers a write. Actual capped gains, additions and losses remain separate.
Initial entities are not fabricated events. One transition makes one entry;
separate zero-time operations at the same tick remain separate entries.

An entry is consumed once before the next decision, using the resulting board.
Otherwise hidden/cell remain exactly unchanged. Time, movement, nonlethal damage,
cooldown and rejection alone never write; a rejection concurrent with a qualifying
world event does. Memory resets to zero. Recovery includes pending facts and time
since the previous write, preventing lost or repeated event consumption.

At writes, the 32-wide entity summary, 64-wide scalar features and 32-wide event
projection feed the 256-unit LSTM. Sun uses the existing cost scale, counts the
public count scale, and elapsed time log1p(ticks/tick_rate), without clipping.
Every decision fuses its current summary/scalars with latest memory into 256 Q-head
features. Action/outcome embeddings are absent; evidence remains in trajectories.

Fitting packs genuine event rows per episode chronologically, excludes padding,
then gathers latest memory onto all decision rows. Quiet rows still supervise
current-state predictions. Detach boundaries count decisions, not events:
defaults stay 256 decisions, four whole-cohort passes, Adam learning rate 0.0003
and clipping limit 5. Quiet chunks carry state forward. Deterministic memory-read
backward avoids repeated-index atomic accumulation during recovery.

Tile exploration uses epsilon(g)=0.5*(0.01/0.5)^min(g/10000,1).
Outside commitments the ten-way branch remains greedy; the tile coin applies only to the selected
non-wait tile and can still select the greedy tile. Evaluation sets the tile
probability to zero. The proposal remains the action selected for Q fitting.
For n candidate tiles and greedy tile t*, the conditional probability is
`p(t|s,b) = (1-epsilon) 1[t=t*] + epsilon/n`. Plants use empty tiles and digs
use all tiles. Waiting has no tile coin; a full-board plant uses tile zero.
Thus epsilon zero is greedy and epsilon one is uniform over the selected branch's
tiles.

After each accepted normal planting, a separate coin uses
epsilon_plant(g)=0.1*(0.01/0.1)^min(g/10000,1). On firing, choose uniformly from
the other seven species, independent of sun/cooldown. Commit until it can be
planted or the episode ends: resource, cooldown and geometry blockers cause actual
one-tick waits, never a paused simulator or timeout. Select its tile from current
features with tile epsilon. An exploratory success does not fire another species
coin. Original policy proposals are diagnostic; waits and exploratory proposals
supervise their actual branch/tile Q-values with ordinary complete returns.
The controller is not part of LSTM input or the network's action schema.

Resolve both probabilities once per cohort from completed stage games and retain them through
fitting and recovery. Stage promotion resets that progress. Deterministic
evaluation temporarily disables exploration without advancing training counters.
Entity encoder microbatching preserves frame order when reconstructing the LSTM
sequence; it changes neither the group denominators nor optimizer boundaries.

## Counterfactual supervision

At every active decision, deterministic round-robin cursors select two distinct
nonselected branches and up to two alternative tiles of the behavior branch.
Alternative branch tiles are greedy under occupancy-only geometry; dig considers
all tiles and a full-board plant proposes tile zero. Waiting has no tile probes.
Sun and cooldown never filter these branches, and probing consumes no behavior RNG.

An isolated simulator copies pre-decision state, RNG and proximity ledger for each
probe. The frozen FP32 EMA teacher maintains its own history along actual execution.
Each probe clones complete event-memory state after consuming actual pending facts,
then observes only its own public events and duration. Its current board remains
available even without a write. Forks are discarded, never updating actual or EMA
history. Deduplication requires matching boards and event/timing inputs.

\[
y_p=r^{train}_p+(1-d_p)\gamma^{\Delta t_p}
       \max_b Q^-_b(s'_p,h'_p).
\]

Both probe roles bootstrap from the next **branch** head. Terminal probes have no
bootstrap, including cutoffs; their time component uses the actual cohort median.
Bootstrap values are collected with fixed EMA weights, and finalized targets remain
fixed through all four passes. A branch probe supervises its branch and non-wait
tile; a tile probe supervises only its proposed tile.

For head \(h\), let \(P_{h,b}\) contain all cohort probes supervising branch \(b\),
and let \(B_h\) be its represented branches. Within each head/branch split outcomes
50/50 when both exist; otherwise the sole stratum receives full weight. Define
alpha_{h,b,o} accordingly, then average across represented branches and heads:

\[
L_{probe}=\frac1{|H|}\sum_{h\in H}\frac1{|B_h|}
\sum_{b\in B_h}\sum_{o:|P_{h,b,o}|>0}
\frac{\alpha_{h,b,o}}{|P_{h,b,o}|}
\sum_{p\in P_{h,b,o}}\operatorname{Huber}_{1}(q_{h,p}-y_p),\qquad
L_{auto}=L+0.25L_{probe}.
\]

All denominators cover the whole cohort, so summing chunk contributions preserves
weights when execution groups change. After one successful FP32 optimizer commit,
\(\theta^-\leftarrow0.95\theta^-+0.05\theta\), exactly once. Its half-life is about
14 updates; failed attempts never advance the EMA version. No optimizer owns EMA.

## Demonstration preferences

Verified replay facts are repriced with current coefficients; a single replay has
zero time adjustment. For each accepted demonstration, compare the chosen branch
against executable alternatives, and its tile against other executable tiles:

\[
\ell_{i,h}=\frac1{|C_{i,h}|}\sum_{a\in C_{i,h}}
\operatorname{softplus}(0.05-q_{i,h,selected}+q_{i,h,a}),\qquad
L_{demo}=L+0.10(L_{rank,branch}+L_{rank,tile}).
\]

Each ranking head averages decisions within each demonstrated branch, then averages
the represented branches. Empty competitor sets and rejected actions contribute
zero; rejected actions retain complete-return regression. Ranking masks affect
supervision only. Behavior branches remain unmasked and greedy. With both auxiliary
weights zero, the shared selected-action loss remains exactly \(L\).

`tests/test_recurrent_training.py` checks hand-computed returns, independently
computed group means, episode-batch gradient invariance, unequal lengths,
inactive rewards, quiet/event/zero-tick inputs, isolated resets, exact weight
transfer and exact collection/fitting recovery. The existing step/sequence and
future-input controls check causality and forward agreement. These are numerical
and implementation controls, not evidence of learning quality.

Demonstration controls independently average wait/plant/dig errors over a
five-decision, three-chunk episode and compare the published loss. Replay
verification checks observation, action, outcome, public reward facts and terminal
reconstruction; manifest identity, count and hash must agree. Initialization
recomputes rewards from native replay observations/events using current coefficients
before summing returns. Independent controls check asset losses of 50 and 100 sun,
new rejection prices on a terminal tick, and changed win/loss rewards. Historical
ledger prices never supply targets. Failed writes preserve the previous
completed pass. Recording and verification commands are in [training](../training.md).
