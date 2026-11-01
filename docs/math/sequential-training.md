# Sequential complete-return training

The official objective is `complete_return_action_global_v7`. It uses
complete remaining returns with gamma one and no counterfactual rollouts.

For each actual decision (t),

\[
G_t=\sum_{u=t}^{T-1}r_u^{\mathrm{train}}.
\]

Ordinary wait proposals regress the species-stage wait value. Accepted plant
and dig proposals regress the tile value at the actual proposed tile. Invalid
proposals use the [shared attribution table](invalid-action-penalties.md): an
impossible branch trains species to its detached current ten-output minimum
plus `invalid_action_q_offset`; a bad tile in a usable branch trains only that
tile to its detached current 45-output conditional minimum plus the same offset.
Dig with no plants is an impossible branch and skips tile inference; Dig with
plants remains a 45-tile proposal with post-selection tile validation.
The configured offset is `-0.1`, independent of the selected invalid output.
The baseline is recomputed from each pass's online forward, without an EMA
forward or gradient through the minimum. Returns remain unchanged, and the
offset never enters episode rewards or earlier chronological returns. Prefit
diagnostics use the same helper with collection-time minima.

Repeated waits and invalid proposals are compacted only in the fitting input. Within each
environment episode, consecutive rows with the same action, route, and rejection
reason are partitioned into configured `invalid_fit_group_size` windows and the first row is
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
scaling. Credited early Sunflower action income uses the configured extra multiplier while
`3 * cumulative_spawned < total`; sky income, capped/discarded sunlight and
zero-total episodes are excluded. Actual proposals, rejected actions,
commitments and exploration provenance remain separate from diagnostic greedy
proposals. No rollout row, probe target or auxiliary reward exists.

Residual-sun passivity shaping is finalized at the cohort boundary. A win adds
`0.2 * clamp(remaining_sun / 1000, 0, 1)^2` to the terminal outcome row. A loss
distributes `-0.3 * clamp((remaining_sun - 200) / 800, 0, 1)^2` over accepted
ordinary Wait rows by executed duration; commitment waits, rejected proposals,
plants and digs are excluded. A loss with no eligible waits has no passivity
penalty. The rows remain chronological and complete returns are calculated after
this update. Older in-flight cohorts retain their saved settings, while resumed
runs apply current passivity values to the next cohort.

## Plant exploration

Normal accepted planting and prolonged ordinary waiting share the same
cohort-frozen `plant_epsilon` probability check. After normal planting, success
chooses uniformly among the seven other species. After waiting, success chooses
uniformly among all eight, without filtering by sunlight, cooldown or occupancy.
Either starts the existing commitment on the next decision: wait for branch
availability, select among all 45 tiles using tile exploration, and retain the
commitment until a planting succeeds or the episode ends. Completing a commitment
does not immediately draw another one. Evaluation and demonstrations do not use
this controller.

`training.exploration.plant_wait_check_seconds=15.0` is converted to exact engine
ticks. Each environment counts only actual transition ticks from accepted Wait
proposals in normal mode with no pending commitment. Any executed plant/dig
proposal, including a rejection, breaks the streak. Commitment modes and episode
completion/reset clear it; inactive and deferred slots neither advance nor clear
it. Reaching the interval performs one check and resets the counter even when
the probability check fails, requiring another full interval for a retry.

Probability and species draws retain original-slot shape and are owned by issued
scheduler tickets. Deferred tickets reuse their saved draws. Wait-triggered checks
have a separate all-eight-species draw; the existing post-planting mapping stays
unchanged. Collection reports `plant_exploration_checks`,
`wait_exploration_checks`, and matching `*_commitments` counters, alongside the
existing aggregate exploration counts. `plant_wait_check_seconds` reports the
active interval; `requested_plant_wait_check_seconds` identifies a pending change.
`plant_exploration_epsilon` reports the probability actually used.

Controller state v2 saves elapsed waiting ticks and the active interval. Matching
v1 autonomous recovery remains supported: pending species and probabilities are
preserved, and the old active cohort retains its disabled wait trigger. Newly
introduced counters start at zero. Resume imports only the wait interval from
current configuration; all probability schedules and other learning settings
retain their existing compatibility checks. Requested settings persist in model
configuration while the active interval persists in controller state. The next
collection cohort activates the request with zero waiting counters. Reward,
objective, optimizer and trajectory identities remain unchanged.

## Recovery

Canonical rows retain public entities, masks, globals, action, acceptance,
duration, reward, return and exploration provenance. Active packing, width
buckets, fitting spools, prefetch workers and graph captures are transient.
Checkpoints contain FP32 online masters, FP32 Adam state, BF16 EMA state and the
action-global trajectory/recovery v7 contracts. Old-objective resume is
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
is unlimited. Annotation protocols are unchanged by autonomous v7. The
[annotation objective and recovery contract](training-objective.md#annotation-convergence-and-l2-phase)
owns the detailed requirements.
