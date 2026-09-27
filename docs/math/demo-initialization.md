# Demonstration initialization controls

For decision `t`, the complete-return target is

\[
G_t = \sum_{k=t}^{T-1} r_k,
\]

because the research clock uses `gamma = 1`. The implementation computes this
sum once from the final transition backward; it does not bootstrap from an
unfinished game. A recording is eligible only when both the archive and native
replay end in `won` or `lost`.

The eight cooldown inputs use the engine's public countdown:

\[
c_{t,j}=\frac{\operatorname{cooldown\_ticks}_{t,j}}
                  {\operatorname{recharge\_ticks}_{j}+1}.
\]

The denominator includes the engine's immediate post-placement value. Therefore
ready is exactly zero, a successful instantaneous placement is exactly one, and
one simulation tick later is `recharge_ticks / (recharge_ticks + 1)`.
The CPU and CUDA controls compare these values after reset, placement and a
one-tick wait.

The demonstration pass visits the sequence in chronological chunks of 256
decisions. The recurrent state is detached between chunks, so gradients do not
cross a chunk boundary while forward state still carries across it. Parameters
remain fixed until every chunk has contributed to the pass gradient. Gradient
norm clipping and one Adam step then publish the next checkpoint atomically.

Let the nonempty action groups be wait, plant and dig, with K nonempty groups
and N_g decisions in group g. A wait has error e_t=(Q_branch−G_t)². A non-wait
has e_t=((Q_branch−G_t)²+(Q_tile−G_t)²)/2. The complete pass loss is

\[
L = \frac{1}{K}\sum_g\frac{1}{N_g}\sum_{t\in g} e_t.
\]

Each chunk contributes its errors divided by the complete group's K·N_g;
chunk length and a short final chunk never change a sample's weight. Averaging
separately inside each chunk would overweight small chunks. Initialization
reuses the collector's shared balanced loss with the whole demonstration as
its denominator. A five-decision, three-chunk control independently averages
the three groups and compares the published loss, then reloads the checkpoint.
Gradient truncation at chunk boundaries still affects recurrent credit assignment.

Replay verification reconstructs the reward components and terminal state as
well as observations and action outcomes. Manifest hash, episode identity,
decision count and final outcome must agree. Partial or altered inputs cannot
create an initialization checkpoint; a failed checkpoint write preserves the
previous completed pass.

Independent controls cover step-versus-sequence outputs, zero-state episode
resets, future-observation isolation, branch/tile dimensions, replay action
order, pre-action and final observation reconstruction, and rejection of
unfinished archives. These controls establish protocol correctness only; they
do not establish one-game generalization.
