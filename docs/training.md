# Recording and training

The objective is `complete_return_action_global_v1`. Old-objective optimizer
resume is rejected. Structurally compatible previous BF16-master two-stage
checkpoints remain valid for inference and fresh weight-only initialization.
Current recovery uses `action-global-state-v1` and
`action-global-trajectory-v1`; the model weight protocol remains v2.
Verified recordings remain reusable and are repriced without rewriting files.

The learning reward has two channels. Global reward contains only the terminal
win/loss pair and the change in `sun + living-plant value`, with sky-generated
sun removed. Action reward credits plant/projectile body and armor damage,
plant-produced sun, authoritative plant-bite blocking and mower activation cost
to the action that caused them. Provenance stays private and never enters an
observation. Ordinary plants receive a signed bite penalty, while Wall-nut's
configured positive blocker coefficient can outweigh its lost asset value.
Home proximity and victory-time remain diagnostics only. A reward protocol
change invalidates autonomous and optimizer recovery, while weight-only model
transfer remains portable across supported GPUs.

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
sunlight and cooldown do not hide branches. Rejected proposals execute ordinary
wait with zero rejection reward penalties and advance one tick,
accepted plants/digs advance zero ticks. Planting immediately returns to normal
collection. No simulator state is copied, no 30-second rollout is scheduled and
no probe evidence is stored or learned.

## Fitting and recovery

Complete returns are calculated chronologically per original episode. Flat active
rows are prefetched through the double buffer and width-bucketed into padded
batches. A fitting spool can be materialized once and reused across all passes;
it is RAM/disk-budgeted and never serialized as simulator state. Padding and
inactive rows are excluded from denominators and diagnostics.

Ordinary wait regresses the complete return through species. Accepted plant/dig
regresses the complete return at the proposed tile, and detached online selection /
EMA evaluation supplies species consistency. Invalid proposals instead use the
[attribution table](math/invalid-action-penalties.md): branch impossibility trains
species to `-1`; an invalid tile in a usable branch trains only that tile to
`-1`. No rejected row performs EMA inference. The fixed target never changes
episode rewards or earlier returns. Whole-cohort wait/plant/dig and accepted /
rejected balancing remains; consistency uses accepted plant/dig counts only.
CPU plans group by entity bucket and route before transfer, skipping irrelevant
online encoders as well as EMA. Diagnostics report routed counts and actual
online/EMA row workloads.

Saving or interruption stops new selection and keeps only complete actual-game
state. An uncommitted fit pass discards gradients and restarts; committed steps
are not repeated. Runtime maps, prefetch workers, fitting spools and graph
captures rebuild after recovery. Old probe-enabled, FP32-runtime and partial-
BF16 contracts are rejected before simulator construction. Curriculum mastery
probes remain separate evaluation controls and are not counterfactual gameplay.

See [architecture](architecture.md), [sequential-training](math/sequential-training.md),
[throughput](math/training-throughput.md) and [validation](validation.md).
For cloud GPU stalls during fitting, wrap the same training command with
[`tools/collect_fit_diagnostics.py`](../tools/collect_fit_diagnostics.py);
[the diagnostic workflow](math/training-throughput.md#diagnose-cloud-fitting-stalls)
explains lightweight collection, bounded CPU/CUDA traces and the files to share.

## Tile annotation warm-start

The same annotator can label either output branch. `annotate-tiles --kind tile`
writes `data/tile-annotations.jsonl`; `--kind action` writes
`data/action-annotations.jsonl`. Tile labels contain an unordered feasible set
and an unordered absolute-bottom set over all 45 outputs. Action labels use
`wait`, the eight configured plant names, and `dig`. A feasible set must be
non-empty. Simulator-impossible branches and tiles are automatically absolute
and locked; human labels may add more absolute entries. Runtime gameplay masks
are unchanged, while annotation fitting deliberately scores every output.

Tile fitting freezes the species stage and action fitting freezes the tile
stage. Both use the top-1/bottom-k margin objective, require exact feasible
membership and exact absolute-bottom membership on the training split, and can
start from random weights. `initialize-demo --init-from` accepts either
annotation artifact and creates a fresh demo optimizer and EMA state, enabling
annotation before human/demo initialization.

The optional annotation workflow trains the existing conditional tile Transformer
before complete-return learning. `annotate-tiles` is the single on-demand
workflow: it starts a CPU web server, prints the relative JupyterLab entry point,
and lets the browser choose `easy`, `standard`, or `hard`. It advances the
pinned simulator until a complete reachable observation contains a present,
affordable, cooled-down plant card with at least one legal placement. Seeds,
levels, and generation metadata are audit fields and never enter policy tensors.

```bash
python -m pvz_rl annotate-tiles --port 6006
```

Records are stored in `data/tile-annotations.jsonl` by default. The current map
is persisted as `pending`, so a restart resumes it exactly. Hold the left mouse
button while dragging to paint feasible plots, then choose **Save and next**.
The next map is generated automatically at the selected difficulty. Use
**Discard state** for a meaningless board; discarded records remain available for
audit but are never fitted. A difficulty change is applied to the next map only
after the current map is handled. The server accepts `--data-dir`, `--seed`,
`--family`, `--max-steps`, and `--max-attempts` overrides.

Use **Previous** and **Next** to browse earlier maps and edit their feasible
plots with the same Paint/Erase/Clear controls. **Save changes** updates the
displayed record in place and stays on that map. A discarded map can be restored
by painting and saving it; a labeled map can also be discarded. **Return to
current** resumes the pending map. Reviewing or editing a completed record does
not generate a map or change the pending record. Leaving unsaved edits requires
a browser confirmation. Difficulty changes preserve the current painted draft
and still apply only to the next generated map. Stale saves from another browser
tab are rejected; use **Reload map** to fetch the saved labels before retrying.
Saved edits survive restart and update the annotation digest and label counts;
they affect subsequent annotation fits, not existing initialization artifacts.

Before normal annotation resumes, the server checks completed labels for
contradictory duplicate inputs. Records are grouped by plant type plus their
canonical encoded entities and globals. If a group has no tile common to every
member, the browser opens it as **Duplicate annotation conflict** and shows the
existing label sets. Paint one corrected non-empty legal set; saving applies that
set atomically to every member, increments their revisions, and advances to the
next conflict. Groups with a common tile are already compatible and are not
queued. A pending map is preserved while conflicts are reviewed and resumes when
the queue is empty. **Save and exit** persists only the displayed map or group and
stops the annotator without generating another map.

Annotation generation draws an inclusive uniform limit of 0–20,000 actions per
attempt. It simulates exactly that many actions, then accepts only a running
game with a usable plant card. Lost, won, truncated, or unusable endpoints are
discarded and retried within the attempt limit. Each planting attempt is spaced
500–1,500 steps after the previous attempt. Each interval chooses equally from
wait and all eight configured plant types (1/9 each), before checking availability.
A plant on cooldown, with insufficient sun, without a card or legal plot, or
blocked by the eight-plant cap becomes wait without drawing another choice.
Usable choices select uniformly among their legal plots. Actual wait probability
therefore rises above 1/9 when some plants are unavailable. Sun has time
to accumulate instead of being spent at every opportunity. The sampling settings
live in `training.annotation` in the central TOML. `--max-steps` changes the upper
bound of the random limit. Plant actions need not advance time, so action steps
and simulator ticks can differ.

Later endpoints preserve actual zombie health/armor, mower use, sunlight, and
card cooldown progress from the simulator. Early endpoints, including zero,
can have no zombies or plants. Accepted maps are conditioned on survival and
card availability; they are not a uniform sample of all possible board states.

Fit the unordered feasible-tile sets with a fresh or compatible weight
initialization:

```bash
python -m pvz_rl fit-tile-annotations \
  --annotations data/tile-annotations.jsonl \
  --output artifacts/tile-initialization.pt \
  --init-from runs/initialization.pt
```

The fit freezes the species encoder/head and trains the existing tile branch
with a top-1/bottom-k margin hinge over all 45 outputs: one feasible tile must
beat every non-feasible tile and the absolute set must occupy the bottom k.
Action fitting applies the same objective to the ten species outputs while
freezing the tile stage. Margin stays `1.0`; Adam uses `0.0003`, batch size
`256`, FP32 masters, BF16 runtime and shared clipping. Every training record
has equal weight across partial batches. Empty labeled sets and contradictory
identical inputs identify their record IDs and fail before fitting. Pending and
discarded records remain excluded. Grouped 80/10/10 splits prevent source leakage.

`training.annotation.passes=0` means unlimited. Every completed pass prints its
update count, full-training hits/total, remaining failures, loss and validation
metrics. Training completion requires exact 100% legal top-1 membership using
BF16 runtime and lowest-index ties. Validation is diagnostic; test is evaluated
only after convergence. `--passes N` adds a total pass cap; reaching it short of
100% is incomplete. This training-set requirement does not establish held-out
accuracy or game performance.

Formal fitting keeps all collected rows for complete-return targets and reward
statistics. To avoid spending encoder work on repeated invalid proposals, the
shared fitting spool groups consecutive invalid `(action, route, rejection_reason)`
runs per environment episode into windows of 100 and retains only each window's
first row. A run shorter than 100 contributes one representative; a run of 201
contributes three. The representative rows define fitting denominators, so no
dropped invalid row receives an implicit frequency weight. Accepted actions and
waits are unchanged, and invalid proposals still train only their responsible
head at `-1` without EMA inference. Raw and representative counts are reported
in fitting diagnostics.

Progress is atomically saved to `artifacts/tile-initialization.progress.pt` after
each pass. Metrics stream to `.history.jsonl`; the bounded summary is `.report.json`.
Interrupting discards uncommitted gradients and retains the last saved pass.
Resume with the same data and fitting settings (repeat any explicit overrides):

```bash
.venv/bin/python -m pvz_rl fit-tile-annotations \
  --annotations data/tile-annotations.jsonl \
  --output artifacts/tile-initialization.pt \
  --resume-from artifacts/tile-initialization.progress.pt
```

`--resume-from` restores annotation weights, Adam, RNG and pass counters and is
mutually exclusive with `--init-from`. Incomplete progress cannot initialize
formal training. The requested initialization is published only after reloading
saved weights and freshly verifying all training records at 100%. Already-correct
initial weights may qualify with zero completed passes. A previous output file
is preserved during an incomplete refit; use the report to identify the current
result. The final artifact uses `stateless-two-stage-bf16-master-state-v2` and
loads via `train --init-from`, with fresh autonomous optimizer and EMA initialized
from transferred weights. It teaches spatial choices; formal complete-return
training still learns timing, resources and long-term values.

## GPU presets and bounded calibration

`--gpu` accepts a visible index or complete CUDA UUID. The selected device is applied to PyTorch, CuPy, the simulator and telemetry, and must provide native BF16. `--gpu-profile auto` matches an exact device name when a measured/provisional preset exists; otherwise `generic-bf16` is used. The RTX 4080 SUPER profile is calibrated on the reported 32 GiB cloud device and retains a 6,554 MiB collection reserve. Presets only alter execution settings. `--learning-profile` applies a fresh-run learning overlay; resume keeps the checkpoint's learning settings while refreshing execution settings.

Fitting uses masked entity buckets `[8, 16, 32, 64, 128, 256]`, CPU-owned row selections and a fitting-only EMA graph owner. The architecture, entity cap, objective, action semantics and checkpoint weight contract are unchanged. Calibrate a new machine with:

```bash
python tools/tune_gpu.py --gpu 0 --output artifacts/tune --minutes 60 --stage easy
```

When telemetry shows spare VRAM, short GPU busy intervals and a saturated
training CPU thread, run the full candidate set with larger grouped cases:

```bash
python tools/tune_gpu.py --gpu 0 --output artifacts/tune-extended --minutes 60 \
  --stage easy --rows-per-width 32
```

This diagnoses fitting underfeeding; it does not select a learning rate or
turn utilization into a training objective.

For an 8 GiB laptop card, use `--no-compile` for the first one-hour rehearsal
if warming all compiled workload shapes exceeds the available memory. The
eager path still measures preparation, transfers and complete fitting work;
remove the flag on a larger card when calibrating graph reuse and compilation.

Trials are isolated and incomplete trials cannot be recommended. Use the resulting raw JSON to decide whether a preset is measured for the tested initialization and stage; cloud presets shipped in TOML remain provisional until calibrated.

After execution calibration, an explicit short learning pilot can compare the
current, half and double learning rates with paired seeds 101/102 and equal
two-cohort budgets: append `--learning-study --initial-weights <checkpoint>`.
The script records incomplete pilots but never promotes a rate from fitting loss;
win-rate evidence and confirmation cases are required.

## JupyterLab viewer

Use `--viewer web` for a CPU viewer process. It shares panel selection, latest-frame delivery and paged history mailboxes with the desktop viewer; the browser can reconnect without stopping training. Install the optional proxy package in the Jupyter environment, then open the printed relative `./proxy/<port>/` path. Use `--viewer-port 6006` when the cloud port-forwarding service requires a fixed remote port; omit it to keep the default ephemeral port. `--viewer auto` uses the desktop window where a display exists and the browser viewer on headless Linux; `--no-live-view` remains an alias for `--viewer off`.
