# Sequential complete-return training

The official objective is `complete_return_sequential_no_probe_v1`. It uses
complete remaining returns with gamma one and no counterfactual rollouts.

For each actual decision (t),

\[
G_t=\sum_{u=t}^{T-1}r_u^{\mathrm{train}}.
\]

Wait proposals regress the species-stage wait value. Plant and dig proposals
regress only the tile-stage value at the actual proposed tile. A poor sampled
tile therefore does not directly train the species estimate.

## Species consistency

For a non-wait branch (b), let (C_b(s)) be empty plant tiles or all dig
tiles, with tile zero as the full-board fallback. The online tile table selects

\[
\ell^*=\arg\max_{\ell\in C_b(s)}Q_{\mathrm{tile}}(s,b,\ell),
\]

and the frozen BF16 EMA tile table evaluates that selected tile:

\[
y=\operatorname{stopgrad}(Q^-_{\mathrm{tile}}(s,b,\ell^*)).
\]

The species value for (b) uses (y) with twice the unreduced Huber loss.
Selection and target evaluation are detached. Consistency gradients reach only
the species stage; the actual-action tile loss reaches only the conditional stage.
The target helper reuses the online tile table and the existing EMA forward.

## Balancing and passes

Selected losses use equal nonempty wait/plant/dig groups and a 50/50 accepted /
rejected split when both strata exist. If one stratum is absent, the represented
stratum receives its group share. Species consistency balances plant/dig and the
same outcome strata using complete-cohort counts, never batch-local counts.
Demonstration ranking remains coefficient `0.10`, margin `0.05`.

Autonomous training commits four optimizer steps per cohort; demonstration
initialization commits twenty. Each pass accumulates the whole cohort, clips
FP32 master gradients once at `training.max_grad_norm=5`, updates FP32 Adam
masters and copies them to BF16 runtime weights. The BF16 EMA teacher is updated
only after the commit with decay `0.95`. An interrupted uncommitted pass is
discarded.

## Reward and provenance

Reward ledgers retain physical net-value diagnostics and dense development
scaling. Credited early Sunflower income is tripled while
`3 * cumulative_spawned < total`; sky income, capped/discarded sunlight and
zero-total episodes are excluded. Actual proposals, rejected actions,
commitments and exploration provenance remain separate from diagnostic greedy
proposals. No rollout row, probe target or auxiliary reward exists.

## Recovery

Canonical rows retain public entities, masks, globals, action, acceptance,
duration, reward, return and exploration provenance. Active packing, width
buckets, fitting spools, prefetch workers and graph captures are transient.
Checkpoints contain FP32 online masters, FP32 Adam state, BF16 EMA state and the
no-probe trajectory/recovery contracts. Old probe-enabled or FP32-runtime
contracts are rejected before simulator construction.
