# Recording and training

Fresh weights are required for `two_stage_transformer_q_v1` and
`complete_return_sequential_tile_probe_v1`. All previous weights/resumes fail
before simulator allocation. Verified archives/native replays remain reusable:
refit into a new directory without overwriting recordings, runs or checkpoints.

The policy is stateless. A 4-layer/8-head Transformer predicts ten species/wait/dig
values from the current board. An independent 6-layer/12-head Transformer scores
45 tiles conditioned on the requested plant/dig branch. Both use `entity_v1`,
with raw `[n,11]` entities and 18 globals; no learned history or scalar bypass.
See [architecture](architecture.md) and [entity inputs](math/entity-inputs.md).

Install using the [quick start](../README.md); run commands from the project root.

## Record and initialize

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human-1000.jsonl --replay runs\human-1000.pvzdemo `
  --output runs\human-init-stateless
```

Recording uses one easy attempt, default seed 1000. Choose unused replay/archive/
sidecar paths and finish naturally. Pause retains queued actions; restarting and
stage switching are disabled. Closing early leaves an incomplete recording.

Initialization checks the manifest, native replay hash, observations, proposals,
acceptance, durations and terminal state. It verifies configuration-independent
reward facts/accounting, then reconstructs rewards under current settings without
changing the archive/replay. Complete returns use reconstructed rewards, not old
recorded prices. `replay-verification.json` reports changed reward counts/totals.
Unsupported archive/engine/schema and reconstruction failures remain errors.
Repricing recordings is not checkpoint conversion or partial-weight migration.

`--config` can change learning/reward settings while preserving recording schemas.
Initialization defaults to `--device cpu`; `--device cuda` changes execution, not
verification. `--passes`, `--learning-rate` and `--seed` override fitting defaults.
Twenty demonstration passes (`training.demo.passes`) are independent of four
autonomous passes (`training.n_epochs`). Both use the shared clipping limit 5,
learning rate 0.0003 and flat batches; there is no demo-only clipping setting.
Initialization's configured time budget is 30 minutes. Checkpoints publish only
after complete passes, with coverage, curves and verification reports.
Demonstrations initialize/maintain EMA copies of both stages, use detached tile-max
species consistency and accepted branch/tile ranking, and run no probes/exploration.

## Autonomous training

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --init-from runs\human-init-stateless\initialization.pt `
  --output runs\stateless-trained
```

`--init-from` accepts only current-contract demonstration `.pt` or autonomous
`.zip` weights after full model/protocol checks. It takes current `train.toml`
settings (or explicit `--config`) and starts fresh optimizer, counters, curriculum,
controller and RNG state. The source supplies full compatible weights only.
There is no partial transfer or silent fallback to random initialization.
Omitting both `--init-from` and `--resume` creates a fresh two-stage policy.
Metadata records source provenance and effective settings.

Autonomous training requires CUDA. The curriculum is easy → standard → shared:
easy-only, equal easy/standard, then 20%/40%/40% easy/standard/hard. Held-out gates
and residency determine mastery; promotion affects future resets only.
`--stage easy --until-stage-complete` is intentionally unbounded until that gate
passes; do not use it for availability/integration checks.

Up to 128 games form a fixed-weight cohort. Finished slots stay inactive and are
not refilled; the final cohort respects the requested remaining game count.
Active-only packing preserves original identities and original-slot RNG order.
Species/tile inference uses separate public-width buckets and token overheads;
padding is execution-only. Trajectories retain actual proposed action, execution,
acceptance, duration, original policy proposal and exploration provenance.

Actual complete remaining returns supervise wait through the species network and
plant/dig through the proposed conditional tile. Species consistency instead
matches the frozen EMA tile network's geometry-candidate maximum. Its coefficient
is 1.0; a poor sampled tile's return is not regressed directly into species values.
Full-cohort nonempty wait/plant/dig groups receive equal selected-loss weight.
Accepted/rejected strata split 50/50 when both exist, or give a sole stratum full
weight. Consistency balances plant/dig strata; tile probes balance represented
initial-action species and initial acceptance with Huber coefficient 0.25.

Each of four passes processes every active flat decision, accumulates gradients,
clips once and commits one Adam step. Both EMA networks then update with decay
0.95. Base batch size is 1,024; `training.performance.fit_batch_groups=4` groups
flat batches for execution without extra optimizer steps. Complete returns stay
chronological per episode; capacity gaps/padding have no counts/loss. There are
no learned-history states, sequence chunks or temporal detach boundaries.
See [learning mathematics](math/sequential-training.md).

## Exploration and background probes

Tile exploration decays exponentially from 50% to 1% over 10,000 stage games;
accepted-normal-plant species exploration decays from 10% to 1% over the same
span. Both probabilities freeze per cohort and restart on stage promotion.
Ordinary ten-way branch selection stays greedy; sun/cooldown do not hide branches.
Plants use empty tiles, dig all tiles, and a full-board plant proposes tile zero
for ordinary rejection. Rejection executes a one-tick wait; accepted plant/dig
advances zero ticks.

After accepted normal planting, the species coin can select uniformly from the
other seven species. The agent waits ordinarily until resources, cooldown and an
empty tile permit it, without timeout or defensive cancellation. Current-board
inference continues; tile selection uses that species and current tile exploration.
Exploratory success clears commitment without recursively drawing a new coin.
Evaluation, demonstrations and auxiliary games disable both exploration behaviors.

Every accepted actual planting, including exploration, creates up to four
distinct alternative empty tiles of the same species, excluding its actual tile.
The pre-action simulator/accounting source is retained before eligible execution;
actual acceptance authorizes probes. Separate zero-time plantings remain separate
jobs. Waits, digs, rejections and initial/auxiliary plants trigger none.

Defaults under `training.objective` are `probe_trigger="accepted_plant"`,
`tile_probes=4` (supported 1–44), `probe_rollout_seconds=30`,
`probe_rollout_max_decisions=4096` and control interval 64. Forks follow frozen
greedy EMA without exploration. Horizon/cap endpoints bootstrap maximum EMA
species Q; true terminal/cutoff endpoints do not. Rewards accumulate ordinarily
and probe victories use the actual cohort's median duration.

Real games, including the planting owner, continue while auxiliary games advance.
Execution defaults under `training.performance` allow 256 lanes, 128 outstanding
source jobs and one auxiliary transition per round. Capacity pressure retains only
the affected eligible planting proposal/draws for FIFO admission, without artificial
waits or discarded probes. Allocation fallback reduces capacity to one source/lane;
minimum failure is explicit. Independence does not remove shared-GPU contention.
Rows append immediately and receive exact-once delayed endpoint patches in RAM/spill.
Finalization/fitting/EMA updates require complete evidence. Save/interruption stops
selection, finishes issued decisions and drains jobs/copies/patches before saving.

Early credited Sunflower sunlight has triple development value while
`3 * cumulative_spawned < total`. Sky income, cap-discarded sun and zero-total
scenarios receive no bonus. Dense scaling applies once; physical net-value
diagnostics remain separate. See [reward accounting](math/training-objective.md).

## Progress, storage and execution

Compact progress shows completed games/wins out of the current cohort, mean raw
return (`Rraw`) and training return (`Rtrain`) for completed games only. Before
any finish these are `n/a`; `~` marks provisional victory-time shaping. Finalized
means stay visible during fitting/recovery. Interval active decisions/game/s and
transitions/s are separate from lifetime rates and the rolling 100-game window.
Detailed reports include auxiliary queues/backpressure/drain, stop reasons,
workspace and both-stage timing. Selected error, species-consistency error and
conditional best-tile minus sampled-tile placement gap are separate diagnostics.

Defaults request 10,000 games with `training.max_minutes=120` and a 15-minute
finalization reserve. `--games`, `--n-envs`, `--batch-size` and `--no-live-view`
are overrides; change time budgets in TOML. Press F in the viewer to follow the
latest actual action. Journals/viewers are diagnostics, never model inputs.

`encoding.max_entities=256` preserves individual records up to the cap; globals
count omitted entities and journals disclose omissions. `training.storage.ram_gib`
covers trajectory slabs and pinned staging; excess slabs/index scratch spill.
Both stages use encoder microbatch 128, token budget 65,536 and fallback query
chunk 64. Allocation retries reduce work size, never the entity cap or coverage.

Each online/EMA encoder owns a bounded collection cache: eight graph entries and
512 MiB headroom. Pressure/capture failure uses exact eager execution. Separate
collection/fitting phase owners release outgoing workspaces at quiescent boundaries.
CUDA fitting defaults to temporary feature BF16 with FP32 master weights, heads,
losses and gradients; `fp32` is a reference. Ordered double-buffer prefetch and
execution grouping preserve whole-pass updates. See
[precision and execution](math/training-throughput.md) for ownership/retry details.
Neither historical benchmarks nor configured capacities establish current speed.

## Resume and evaluate

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\stateless-trained\latest.zip --output runs\stateless-trained
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\stateless-trained\latest.zip --split validation `
  --output runs\stateless-trained\validation --record
```

Current recovery restores online/EMA networks, optimizer, counters, curriculum,
RNGs, simulator/accounting, controller commitments, scheduler/cursors and complete
trajectory evidence at a canonical decision boundary. Pools/maps/caches rebuild;
transient forks and learned-history state are not serialized. Saved learning and
reward settings remain authoritative; execution/logging controls refresh.
Model/slot-count changes and previous protocols are rejected before allocation.
Interrupted fitting discards partial gradients and restarts the uncommitted pass;
committed updates are retained. `interrupted.zip` is the recovery artifact;
`latest.zip`/`final.zip` publish on normal completion. In-place resume appends history.

Evaluation accepts current checkpoint formats and uses the stateless runner
without exploration/probes/updates. `simulation.backend="cpu"` chooses CPU
evaluation while preserving model semantics. CUDA action traces are CPU-verified
before export. Use a fresh evaluation directory; validation/test seeds stay separate.
