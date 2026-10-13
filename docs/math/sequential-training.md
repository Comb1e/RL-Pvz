# Stateless sequential Q learning

The sole policy is `two_stage_transformer_q_v1`; its objective is
`complete_return_sequential_tile_probe_v1`. Independent current-board networks
predict species/wait/dig values and conditional tile values. Both have EMA copies.
All previous weights/resumes require fresh initialization; verified recordings
remain reusable. No learned history or partial-weight migration exists.

## Values and action selection

Let \(B_\theta(s,b)\) be the first Transformer's ten direct branch values and
\(T_\phi(s,b,\ell)\) the second Transformer's 45 conditional tile values.
Parameters \(\theta\) and \(\phi\) are disjoint. The second network receives raw
board \(s\) and branch \(b\), not the first network's features or values.
Discrete branch choice carries no gradient between stages.

Normal behavior selects \(b=\arg\max_b B_\theta(s,b)\). A non-wait branch then
selects its highest-valued geometry candidate, unless tile exploration samples
another candidate or the commitment controller requests a different species.
Plants use empty tiles; dig uses all 45 tiles. A full-board plant uses tile zero
as rejection fallback. Sun/cooldown never mask normal branches. Wait skips the
tile network. Duplicate requested board/branch pairs share computation.

The value regressed for an actual proposal is

\[
q(s,a)=\begin{cases}
B_\theta(s,0),&a=\text{wait},\\
T_\phi(s,b,\ell),&a=(b,\ell).
\end{cases}
\]

## Actual complete returns and balancing

Every actual decision, including rejections and separate zero-time actions, has
target \(G_t=\sum_{u=t}^{T-1}r_u^{\rm train}\), with \(G_T=0\) and gamma one.
The six signed reward components and dense development multiplier are defined
in [reward accounting](training-objective.md). Finalization applies victory-time
shaping against actual cohort durations before returns are constructed.

For any represented group set \(H\), outcome \(o\in\{accepted,rejected\}\),
whole-cohort count \(N(h,o)\), and per-row error \(e_i\), define

\[
\mathcal W_H(e;N)=\frac1{|H|}\sum_{h\in H}
\sum_{o:N(h,o)>0}\frac{\alpha_{h,o}}{N(h,o)}
\sum_{i\in(h,o)}e_i.
\]

When both strata exist, \(\alpha_{h,o}=0.5\); a sole stratum receives 1.
Absent groups contribute nothing. Selected regression is
\(L_{selected}=\mathcal W_{wait/plant/dig}((q_i-G_i)^2;N)\).
The cohort uses 3x2 counts, not per-batch counts. Capacity gaps and padding have
no supervision. Flat-batch contributions add to the same whole-pass loss and
gradient regardless of partitioning, within numerical tolerance.

## Detached species consistency

For every actual non-wait proposal, the first network's target is the best
geometry-candidate value from the frozen EMA tile network:

\[
y_i^{species}=\operatorname{stopgrad}
\left[\max_{\ell\in C_{b_i}(s_i)}T^-_{\phi}(s_i,b_i,\ell)\right],\qquad
L_{consistency}=\mathcal W_{plant/dig}
((B_\theta(s_i,b_i)-y_i^{species})^2;N^{nonwait}).
\]

Candidates and full-board fallback match selection; affordability/cooldown do
not mask this target. Counts are 2x2 over the complete cohort. The configurable
`species_consistency_weight` is 1.0.

For a poor explored tile, selected regression directly updates only the tile
network. Consistency directly updates only the species network toward a detached
EMA maximum, not the poor tile's return. This removes direct shared-target blame,
not estimation error: tile generalization and later EMA updates can still change
species targets. Wait regression is the exception that trains the species wait
output directly from return.

Report selected-action error, species-consistency error and placement gap
separately. Placement gap is
\(\max_{\ell\in C_b(s)}T_\phi(s,b,\ell)-T_\phi(s,b,\ell_{sampled})\),
not species Q minus sampled-tile Q; it is diagnostic, not another loss.

## Background tile targets

Each accepted actual normal/exploratory planting queues up to four distinct other
empty tiles of its species from the immutable pre-action board. Waits, digs,
rejected plants and initial plants trigger none; auxiliary plants never recurse.
Selection is deterministic and consumes no behavior RNG.

Fork simulator/accounting state, execute the alternative placement, then follow
frozen greedy EMA with no exploration. Stop at true terminal/cutoff, 30 elapsed
simulated seconds, or 4,096 executed transitions including the initial proposal.
Limits do not force waits, advance time or fabricate defeat. For endpoint \(s_K\),

\[
y_p=\sum_{k=0}^{K-1}r_{p,k}^{train}+(1-d_p)\max_b B^-_\theta(s_K,b).
\]

True terminal/cutoff has \(d_p=1\); horizon/cap endpoints bootstrap. Rewards
accumulate chronologically, with event-local early Sunflower eligibility and
terminal penalties applied once. Probe victories use the actual cohort's median
duration and never contribute to that median.

Initial proposal acceptance determines the probe stratum, not continuation
acceptance. With represented initial-action species and 10x2 cohort counts,

\[
L_{probe}=\mathcal W_{species}
(\operatorname{Huber}_1(T_\phi(s_p,b_p,\ell_p)-y_p);P),\qquad
L_{auto}=L_{selected}+L_{consistency}+0.25L_{probe}.
\]

Each endpoint supervises its initial tile once. Exact public-input/EMA-version
deduplication reuses endpoint bootstrap only; conditional tile requests also
compare branch. Proposals, rewards and outcomes remain distinct, and ongoing
simulations never merge.

Scheduler rounds interleave real decisions and occupied auxiliary transitions.
Source pressure holds only the affected eligible planting, preserving its proposal
and draws. Endpoint patches attach exactly once to stable source rows, including
spilled rows. Fitting, finalization and EMA changes require every patch complete;
saving drains auxiliary work and transfers instead of serializing transient forks.

## Demonstration preferences

Verified replay facts are repriced under current rewards without rewriting
recordings. Demonstrations use the same selected/consistency losses but no probes.
Accepted demonstrations additionally rank their branch using \(B_\theta\), and
their chosen tile using \(T_\phi\). Competitors are other executable branches or
other executable tiles of the demonstrated branch, respectively:

\[
\ell_{i,h}=\frac1{|C_{i,h}|}\sum_{c\in C_{i,h}}
\operatorname{softplus}(0.05-q_{i,h}^{demo}+q_{i,h}^{c}),\qquad
L_{demo}=L_{selected}+L_{consistency}
+0.10(L_{rank,branch}+L_{rank,tile}).
\]

Each ranking category averages eligible decisions within demonstrated branches,
then represented branches. Rejected demonstrations/empty competitor sets have
no ranking term but retain regression. Branch and tile anchors use their own
network, not a shared complete-action value.

## Flat fitting and EMA

Both EMA networks initialize from online weights, remain frozen within a pass,
and update after each committed step:
\((\theta^-,\phi^-)\leftarrow0.95(\theta^-,\phi^-)+0.05(\theta,\phi)\).
Autonomous training uses four passes and demonstrations twenty. Each pass
accumulates all active flat batches, clips total gradient norm to 5 and makes
one Adam update at learning rate 0.0003. Base batch size is 1,024;
`fit_batch_groups=4` is execution grouping, not additional optimizer updates.
There are no recurrent states or temporal detach boundaries. Chronology matters
for returns and source evidence, not for model execution. An uncommitted pass
restarts from evidence with cleared gradients; committed updates are retained.

[Validation](../validation.md) owns independent attribution, stopped-gradient,
global-denominator, missing-stratum and partition controls. Passing them does
not establish improved win rate or curriculum mastery.
