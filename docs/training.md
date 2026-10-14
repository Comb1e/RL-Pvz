# Recording and training

The official objective is `complete_return_sequential_no_probe_v1`. Previous
weights and resumes are rejected before simulator allocation. Verified replay
archives remain reusable and can be repriced into a new initialization directory.
The current protocols are demo checkpoint v9, autonomous state v1 and
`stateless-no-probe-trajectory-v1`.

## Record and initialize

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human-1000.jsonl --replay runs\human-1000.pvzdemo `
  --output runs\human-init
```

Replay verification checks observations, proposals, outcomes, timing and
configuration-independent facts. Current reward settings reprice the verified
recording without rewriting it. Demonstration fitting uses 20 passes, while
autonomous fitting uses four; both keep learning rate `0.0003`, clipping `5` and
the configured batch groups.

## Stateless policy and precision

The first Transformer predicts eight plants, wait and dig from the current board
using width 128, four layers and eight heads. The second independently scores 45
tiles for a selected branch using width 192, six layers and twelve heads. No
history, event memory, LSTM or scalar bypass is used.

Runtime collection, evaluation, demonstration fitting and autonomous fitting all
use BF16 model parameters and activations. The online policy owns separate
unregistered FP32 master parameters; Adam and its first/second moments own only
those FP32 masters. Runtime BF16 gradients are accumulated into FP32 master
gradients, clipped there, stepped once per pass, and copied back to BF16. The
BF16 EMA teacher updates after each committed step. Checkpoints store FP32 online
masters, FP32 optimizer state and BF16 EMA state. BF16 support is checked before
simulator allocation; nonfinite loss or gradients fail explicitly. There is no
FP32 inference fallback and no INT8/INT4 path.

## Autonomous collection

Up to 128 games form a cohort. Finished slots are never refilled. Active packing
uses current public entity widths and restores original slot identities; padding
has no RNG, action, loss or storage evidence. Every actual decision records its
board, proposal, acceptance, duration, reward and exploration provenance.

Ordinary tile exploration decays from 50% to 1%. Accepted normal plantings can
start a persistent uniformly selected alternative-species commitment, decaying
from 10% to 1%. Both schedules span 10,000 stage games, freeze within a cohort
and restart on a stage change. A commitment waits ordinarily until affordable,
cooled down and placeable, then clears after successful planting. Evaluation,
demonstrations do not use exploration commitments.

Plant and dig candidates are geometry-only: empty tiles for plants and all tiles
for dig. Full-board plants use tile zero for the ordinary rejection fallback;
sunlight and cooldown do not hide branches. Rejected proposals advance one tick,
accepted plants/digs advance zero ticks. Planting immediately returns to normal
collection. No simulator state is copied, no 30-second rollout is scheduled and
no probe evidence is stored or learned.

## Fitting and recovery

Complete returns are calculated chronologically per original episode. Flat active
rows are prefetched through the double buffer and width-bucketed into padded
batches. A fitting spool can be materialized once and reused across all passes;
it is RAM/disk-budgeted and never serialized as simulator state. Padding and
inactive rows are excluded from denominators and diagnostics.

Wait values regress through the species stage. Plant/dig values regress only the
actual proposed tile through the tile stage. Species consistency selects a
geometry candidate with detached online tile values and evaluates it with the
frozen EMA tile values. Its MSE-matched Huber loss updates only the species stage;
the poor sampled tile's return does not directly train species values. Accepted /
rejected and wait/plant/dig weights use whole-cohort denominators.

Saving or interruption stops new selection and keeps only complete actual-game
state. An uncommitted fit pass discards gradients and restarts; committed steps
are not repeated. Runtime maps, prefetch workers, fitting spools and graph
captures rebuild after recovery. Old probe-enabled, FP32-runtime and partial-
BF16 contracts are rejected before simulator construction. Curriculum mastery
probes remain separate evaluation controls and are not counterfactual gameplay.

See [architecture](architecture.md), [sequential-training](math/sequential-training.md),
[throughput](math/training-throughput.md) and [validation](validation.md).
