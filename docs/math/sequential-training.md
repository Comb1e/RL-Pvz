# Sequential complete-return training

The official objective is `complete_return_selective_invalid_v2`. It uses
complete remaining returns with gamma one and no counterfactual rollouts.

For each actual decision (t),

\[
G_t=\sum_{u=t}^{T-1}r_u^{\mathrm{train}}.
\]

Ordinary wait proposals regress the species-stage wait value. Accepted plant
and dig proposals regress the tile value at the actual proposed tile. Invalid
proposals use the [shared attribution table](invalid-action-penalties.md): an
impossible branch trains species to -1; a bad tile in a usable branch trains
only that tile to -1. This target never enters episode rewards or returns.

## Species consistency

For an accepted non-wait branch (b), let (C_b(s)) be empty plant tiles or all dig
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
accepted-only strata using complete-cohort counts, never batch-local counts.
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
selective-invalid trajectory/recovery v2 contracts. Old-objective resume is
rejected before simulator construction; structurally compatible BF16-master
weights remain valid for inference and fresh weight-only initialization.

## Annotation convergence

For annotated legal tiles P and legal unannotated tiles N, each record uses
`max(0, margin + max(Q[N]) - max(Q[P]))`, with margin 1. Only one acceptable tile
needs to win; other annotated tiles have no imposed ordering. Empty sets and
identical model inputs with collectively disjoint feasible sets fail before
fitting. All-legal-positive records are correct and have zero loss.

Gradients accumulate with record weight `1 / training_records`, including
partial batches and zero-loss records. The frozen species stage is untouched.
After each FP32 Adam commit, the BF16 runtime evaluates the entire training
partition with legal masking and lowest-index tie breaking. Exact training hits
must equal training count; held-out metrics cannot terminate fitting. Grouped
80/10/10 splits remain isolated. Test metrics run only after convergence.

Default passes 0 means unlimited. A cap or interrupt yields separate incomplete
progress; weights, Adam, RNG and counters resume only under the same dataset,
split and fitting settings. Progress commits atomically per pass. A fresh
full-training evaluation of reloaded saved weights gates initialization
publication, including honestly recorded zero-pass success. No claim about
held-out accuracy or game improvement follows from training convergence.
