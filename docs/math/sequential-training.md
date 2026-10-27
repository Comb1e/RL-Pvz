# Sequential complete-return training

The official objective is `complete_return_action_global_v5`. It uses
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

Repeated waits and invalid proposals are compacted only in the fitting input. Within each
environment episode, consecutive rows with the same action, route, and rejection
reason are partitioned into 100-row windows and the first row of each window is
used once. Episode/decision/inactive gaps and terminal boundaries split runs;
other environments' interleaved rows do not. Accepted non-wait rows are retained.
Complete returns remain chronological over every actual row. Fitting uses
representative counts without frequency weights.

## Species consistency

For an accepted non-wait branch (b), let (C_b(s)) contain all 45 tiles. Occupancy
is post-selection legality and never masks these candidates. The online table selects

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
action-global trajectory/recovery v5 contracts. Old-objective resume is
rejected before simulator construction; structurally compatible BF16-master
weights remain valid for inference and fresh weight-only initialization.

## Annotation convergence

Tile and action annotation fitting use fixed grouped training/validation/test
splits, feasible top-1 and absolute-bottom ranking across all outputs, and a
frozen opposite head. Explicit FP32 L2 begins only after training first converges
while validation remains incomplete. Both training and validation must be exact
before publication or test evaluation. Validation never supplies gradients.

Progress commits atomically per pass, retaining optimizer/RNG/phase for resume
and complete weights for fresh initialization elsewhere, including incomplete
or zero-pass snapshots. Pass caps count additional commits per invocation; zero
is unlimited. Annotation protocols are unchanged by autonomous v5. The
[annotation objective and recovery contract](training-objective.md#annotation-convergence-and-l2-phase)
owns the detailed requirements.
