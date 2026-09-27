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

Independent controls cover step-versus-sequence outputs, zero-state episode
resets, future-observation isolation, branch/tile dimensions, replay action
order, pre-action and final observation reconstruction, and rejection of
unfinished archives. These controls establish protocol correctness only; they
do not establish one-game generalization.
