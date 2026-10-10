# Joint action values, complete returns and bounded probes

The sole model is `transformer_lstm_q_v4`, with objective
`complete_return_joint_tile_probe_v1`. The public observation/action schemas and
deterministic event-memory rules are unchanged. All older weights require fresh
demonstration fitting; recordings remain reusable.

## Complete-action attribution

The network predicts a scalar wait value and a conditional table for eight plant
species plus dig. There is no independent species prediction. For branch b:

\[
Q_b(s,h)=\max_{\ell\in C_b(s)}Q_{\rm joint}(s,h,b,\ell).
\]

Plant candidates are empty tiles, dig candidates are all tiles. A full-board
plant compares tile zero, retaining the rejection fallback. Sun and cooldown
do not hide branches. Choose the greedy branch, then apply tile exploration or
the existing committed-species controller. Exploration may sample a worse tile
without lowering the species' best-tile value.

Actual decisions, including waits/rejections and zero-time actions, have targets:

\[
G_t=\sum_{u=t}^{T-1}r^{\rm train}_u,\qquad G_T=0.
\]

Gamma is one. The six components are terminal, development, invalid plant,
empty dig, house proximity and victory time. Only development receives the
dense multiplier (10), exactly once. Actual cutoff penalties occur once.
Victories finalize against the median duration of actual cohort games; probe
durations never contribute to that median.

For groups g = wait/plant/dig and initial acceptance strata o, count all actual
decisions in the complete cohort as N(g,o). Let K count represented groups.
Both outcome strata get alpha=0.5 when both exist; a sole stratum gets alpha=1.
The selected complete action alone receives return regression:

\[
L_{\rm selected}=\frac1K\sum_g\sum_{o:N(g,o)>0}
 \frac{\alpha_{g,o}}{N(g,o)}
 \sum_{i\in(g,o)}(Q(s_i,h_i,a_i)-G_i)^2.
\]

Chunk contributions use these global denominators. Padding has no counts or
loss. A bad sampled tile does not directly regress its species maximum toward
that tile's return; shared parameters still allow indirect coupling. Report
selected-action MSE and best-tile minus sampled-tile Q as separate quantities.

## Event-only memory and fitting

Gross public sunlight gains/spending, zombie spawns/defeats/removals and plant
additions/removals write memory once, before the next decision, using the
resulting board. Initial entities are not fabricated events. Capped gains count;
simultaneous gains/losses do not cancel. Separate zero-time transitions retain
their order. Movement, nonlethal damage, cooldown, elapsed time and rejection
alone preserve hidden/cell exactly.

At writes, concatenate 32-wide entity summary, 64-wide scalar features and
32-wide event projection into the 256-unit LSTM. Normalize events with existing
public resource/count scales and log1p(elapsed seconds), without clipping.
Every decision fuses its current board features with memory into 256 features
for wait/tile values. Quiet decisions still supervise current-state predictions.

Fitting packs real event rows chronologically and gathers the latest memory
onto every real decision. Episode states are independent. Gradients detach at
256-decision boundaries, not event boundaries, while forward memory continues.
Whole-pass gradients accumulate before one clipping and Adam update:

\[
D=\sum_j\nabla_\theta L_j(\theta;\operatorname{stopgrad}(h_j,c_j)),
\quad \bar D=D\min(1,5/\lVert D\rVert_2).
\]

Autonomous training makes four passes; demonstration fitting makes twenty.
Adam learning rate remains 0.0003. BF16 feature controls, FP32 core and existing
execution grouping remain unchanged. An interrupted uncommitted pass restarts
from episode beginnings. Different chunk lengths can change truncated gradients,
even when the forward loss agrees.

## Persistent exploration

Both exponential probabilities freeze throughout each cohort and restart on
curriculum promotion. After g stage games:

\[
\epsilon_{\rm tile}=0.5(0.01/0.5)^{\min(g/10000,1)},\qquad
\epsilon_{\rm species}=0.1(0.01/0.1)^{\min(g/10000,1)}.
\]

The tile coin samples uniformly among candidate tiles. After an accepted normal
planting, a species coin may choose uniformly among the other seven species.
Ordinary one-tick waits persist until sun, cooldown and geometry permit planting.
There is no timeout or defensive cancellation. Exploratory success never triggers
another commitment. Actual controller proposals receive regression; original
policy proposals remain diagnostic. Evaluation, demonstrations and probes do not
run commitments or tile randomness.

## Bounded counterfactuals

Every accepted planting, normal or exploratory, queues up to four distinct other
tiles of its actual species. There are no alternative-species probes. Alternative
tiles exclude the actual tile. Waits, digs, rejected plants and initial entities
do not trigger probes. Separate zero-time plantings retain their order.
Invalid rows execute nothing and consume no behavior RNG.

Retain pre-plant simulator state, accounting/proximity ledgers and complete EMA
memory after consuming source events. Gate copies with the pinned read-only
legality mask, then authorize probes using actual acceptance after execution.
Execute each alternative initial proposal ordinarily from the immutable source.
Continue greedily with frozen EMA weights, without exploration. Stop on true
terminal/cutoff, 30 simulated seconds from the pre-action tick (3000 ticks), or
4096 executed transitions including the initial proposal. Limits do not invent
defeat, force waits, or advance time. Horizon/cap endpoints bootstrap:

\[
y_p=\sum_{k=0}^{K_p-1}r^{\rm train}_{p,k}
       +(1-d_p)\max_aQ^-_{\rm joint}(s_{p,K_p},h_{p,K_p},a).
\]

Rewards accumulate in transition order; terminal penalties occur once and
early Sunflower eligibility remains event-local. True terminal/cutoff endpoints
do not bootstrap. Probe victories use the actual cohort's median.

Each valid probe supervises its initial complete action once. Initial proposal
acceptance determines its stratum, never continuation acceptance. For represented
branches B and cohort counts P(b,o):

\[
L_{\rm probe}=\frac1{|B|}\sum_{b\in B}\sum_{o:P(b,o)>0}
 \frac{\alpha_{b,o}}{P(b,o)}
 \sum_{p\in(b,o)}{\rm Huber}_1(Q(s_p,h_p,a_p)-y_p),\qquad
L_{\rm auto}=L_{\rm selected}+0.25L_{\rm probe}.
\]

After each successful Adam commit, EMA updates once as
theta-minus = 0.95 theta-minus + 0.05 theta. Its own history follows actual
execution; fork events never reach actual histories or trajectories.

Auxiliary environments advance in short slices between real-decision rounds;
their events and rewards never enter actual histories. Completed lanes retire
immediately, with heavy control reporting every 64 continuation rounds.
No ongoing simulations merge. Endpoint deduplication compares exact public
entities/globals, gross events and elapsed timing, hidden and cell states before
tail inference. Each proposal/outcome retains its own evidence and loss.

Only another eligible planting lacking source capacity is held, with its
preselected action and random draws unchanged. Canonical inactive storage rows
during such gaps carry no supervision. Sequence preparation gathers each slot's
actual decisions first; detach boundaries remain every 256 actual decisions,
not scheduler rounds. Missing activity cannot end an unfinished history.

Endpoint patches are exact-once by source row and tile-probe slot. Cohort targets
and optimizer updates require all expected endpoints to be complete. Saving
finishes issued decisions and drains auxiliary work; no temporary fork is saved.

## Demonstration preferences

Verified replay facts are repriced under current rewards without rewriting the
recording. Regress the demonstrated complete-action value and compare that same
value against other branches' best executable actions and other executable tiles
of its branch:

\[
\ell_{i,h}=\frac1{|C_{i,h}|}\sum_{a\in C_{i,h}}
 {\rm softplus}(0.05-Q(s_i,h_i,a_i)+Q(s_i,h_i,a)),
\qquad L_{\rm demo}=L_{\rm selected}+0.10(L_{\rm rank,branch}+L_{\rm rank,tile}).
\]

Each ranking category averages decisions within demonstrated branches, then
represented branches. Rejected demonstrations and empty competitor sets contribute
no ranking; they retain selected regression. This normalization is unchanged.

## Controls

`test_joint_q.py` independently checks maxima, species/placement attribution,
gradient routing, unified probe strata, partition invariance and ranking anchors.
`test_probe_rollouts.py` compares serial CPU with CUDA delayed shooting, mine
arming, Sunflower production, limits, initial rejection, isolation and interrupted
replay. Event recurrence, sparse gradients and current-board responsiveness belong
in `test_transformer_lstm.py`; returns, optimizer/recovery and storage have
their own suites. Passing controls establish semantics, not win-rate improvement.
