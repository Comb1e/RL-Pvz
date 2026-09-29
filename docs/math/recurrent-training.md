# Recurrent complete-return Q regression

Demonstration initialization and autonomous training use this same objective.
A verified demonstration supplies one naturally completed episode; autonomous
collection supplies a cohort of completed games. Both visit chronological chunks
with fixed weights for the whole pass and publish only completed optimizer updates.

For an episode of T decisions, including rejected proposals and zero-tick
operations, the target is

\[
G_t = \sum_{u=t}^{T-1} r_u, \qquad G_T=0.
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

Let N_g count actual decisions in group g (wait, plant or dig) over the complete
cohort, and let K be the number of nonempty groups. Selected branch and tile
predictions are q_b and q_l. Define

\[
e_i = \begin{cases}
(q_b-G_i)^2 & a_i=\mathrm{wait},\\
\tfrac12\big((q_b-G_i)^2+(q_l-G_i)^2\big) & \text{otherwise}.
\end{cases}
\qquad
L=\frac1K\sum_{g:N_g>0}\frac1{N_g}\sum_{i\in g}e_i.
\]

Each chronological chunk contributes only its real decisions with coefficient
1/(K N_g). Thus summing all chunk losses exactly recovers L regardless of episode
batch partitioning. The shared loss interface uses the entire cohort count as
its normalization budget on this path. Padding has coefficient zero; it does not
create wait samples or alter N_g. Empty groups are excluded instead of diluting
the loss with a third zero contribution.

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

Previous inputs are the executed action and actual accepted/ticks outcome.
A rejected proposal 120 with one tick becomes (0, 0, 1); an accepted zero-tick
dig 361 becomes (361, 1, 0). Neither equals a recurrent reset. Finished collection
slots retain their final hidden/cell state until replaced by a new episode.

Tile exploration uses epsilon(g)=0.5*(0.01/0.5)^min(g/5000,1).
The ten-way branch remains greedy; this single coin applies only to the selected
non-wait tile and can still select the greedy tile. Evaluation sets the tile
probability to zero. The proposal remains the action selected for Q fitting.
For n candidate tiles and greedy tile t*, the conditional probability is
`p(t|s,b) = (1-epsilon) 1[t=t*] + epsilon/n`. Plants use empty tiles and digs
use all tiles. Waiting has no tile coin; a full-board plant uses tile zero.
Thus epsilon zero is greedy, epsilon one is uniform over the selected branch's
tiles, and exploration never replaces the greedy branch.

Resolve epsilon once per cohort from completed stage games and retain it through
fitting and recovery. Stage promotion resets that progress. Deterministic
evaluation temporarily disables exploration without advancing training counters.
Entity encoder microbatching preserves frame order when reconstructing the LSTM
sequence; it changes neither the group denominators nor optimizer boundaries.

`tests/test_recurrent_training.py` checks hand-computed returns, independently
computed group means, episode-batch gradient invariance, unequal lengths,
inactive rewards, rejected/zero-tick inputs, isolated resets, exact weight
transfer and exact collection/fitting recovery. The existing step/sequence and
future-input controls check causality and forward agreement. These are numerical
and implementation controls, not evidence of learning quality.

Demonstration controls independently average wait/plant/dig errors over a
five-decision, three-chunk episode and compare the published loss. Replay
verification checks observation, action, outcome, reward and terminal reconstruction;
manifest identity, count and hash must agree. Failed writes preserve the previous
completed pass. Recording and verification commands are in [training](../training.md).
