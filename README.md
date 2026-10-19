# PVZ plant research

Record a human Plants vs. Zombies game, initialize a stateless two-stage
Transformer Q policy, then train and evaluate it with the pinned simulator.
The species Transformer selects a plant, wait or dig; the independent tile
Transformer scores 45 tiles for the selected branch. Both read only the current
public board. Planting creates no counterfactual rollout.

Training uses `complete_return_action_global_v1` with a causal action/global
reward ledger. Previous compatible
BF16-master two-stage weights remain usable for inference and fresh `--init-from`;
previous-objective runs cannot resume. Parameters live in
[src/pvz_rl/data/train.toml](src/pvz_rl/data/train.toml).

## Requirements

Python 3.12, Git and the pinned game package. Recording needs pygame.
Autonomous training requires NVIDIA CUDA with BF16 support.

## Installation

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

For Linux/WSL or Git Bash, use an existing `pvz-cuda-work` checkout:

```bash
bash tools/bootstrap.sh --game-repo ../pvz-cuda-work --python python3.12
```

Use `--python /path/to/python` if Python 3.12 has a different executable name.
Both installers verify and install the pinned game without changing its checkout,
install PyTorch `2.8.0+cu128` / CUDA `12.8`, and run the availability check.
The Bash installer removes its temporary game archive on success or failure.

The [engine lock](src/pvz_rl/data/engine-lock.json) fixes source and package hashes.

## First useful command

Record one easy game using unused output paths; finish with a natural win or loss:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
```

Use [recording and training](docs/training.md) to initialize, train, resume or
evaluate. Demonstration fitting uses 20 passes; autonomous cohorts use four.
Select a physical CUDA GPU independently from execution and learning presets. Native BF16 devices are detected by capability; the catalogue includes measured/provisional presets for the 4070 Laptop, 4080 SUPER, 4090, 5090, RTX PRO 6000 and RTX 6000D, while arbitrary user profiles remain supported:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --output runs\fit --gpu 0 --gpu-profile auto --learning-profile current --viewer off
.\.venv\Scripts\python.exe tools\tune_gpu.py --gpu 0 --output artifacts\tune-4070 --minutes 60
```

For JupyterLab, install `pip install -e .[web]` (or `jupyter-server-proxy`) and use `--viewer web`; training prints a relative `./proxy/<port>/` URL. Add `--viewer-port 6006` when the cloud port-forwarding service needs a fixed port. Checkpoint weights remain portable; destination GPU execution caches rebuild.

Open the standalone annotator with one command. It serves a local web page,
stores records in `data/tile-annotations.jsonl`, and lets you choose `easy`,
`standard`, or `hard` in the browser. Each generated map contains a reachable
state with a plant card that is present, affordable, cooled down, and legally
placeable.

Generation draws a simulation limit uniformly from 0–20,000 steps and retries
finished games. Planting attempts occur every random 500–1,500 steps, choosing
equally from wait and the eight plant types (1/9 each). Unavailable plant choices
become wait without redrawing; usable choices select a random legal plot, with
at most eight plants on the board:

The annotator uses a fresh entropy-derived session seed by default, so restarting
it does not replay the same map sequence. Pass `--seed N` when a reproducible
sequence is needed.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl annotate-tiles --port 6006
```

Open the printed `./proxy/6006/` path in JupyterLab, hold the left mouse button
while dragging to paint feasible plots, then choose **Save and next**. The next
map is generated at the same difficulty. Use **Discard state** for a meaningless
map; discarded states stay in the audit file and are excluded from fitting.
Use **Previous** and **Next** to revisit maps, edit labels, and **Save changes**
in place. Discarded maps can be labeled again. **Return to current** resumes the
pending map; reviewing earlier maps does not generate another one.
Pending maps are resumed after a restart. Fit only completed labels, then use the
resulting weight-only file with a fresh training run:

If saved labels contain contradictory duplicates, the annotator opens the first
duplicate conflict before generating a new map. It shows every member's old label
set and lets one corrected set be applied to the whole identical-input group;
groups are handled in order. Resolve these conflicts before fitting. **Save and
exit** saves only the displayed map or conflict group and stops the annotator
without generating another map.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl fit-tile-annotations `
  --annotations data\tile-annotations.jsonl `
  --output artifacts\tile-initialization.pt --init-from runs\initialization.pt
.\.venv\Scripts\python.exe -m pvz_rl train --output runs\annotated `
  --init-from artifacts\tile-initialization.pt --gpu 0 --viewer off
```

Annotation fitting defaults to unlimited passes until every training board's
highest-scoring legal tile is annotated. `--passes N` adds an explicit cap;
incomplete runs save `tile-initialization.progress.pt` without publishing an
initialization. Resume with `--resume-from artifacts\tile-initialization.progress.pt`
using the same dataset and fitting settings. Training accuracy is separate from
held-out accuracy and game performance.

Illegal proposals execute ordinary wait with zero rejection reward penalty.
Fitting targets only their responsible output at `-1`, skipping EMA inference.
See [the routing table and convergence workflow](docs/training.md).

Repeated illegal proposals are compacted only for fitting: consecutive identical
`(action, route, rejection_reason)` runs are split into windows of 100 and the
first row of each window is fitted once. Complete returns and reward accounting
still retain every collected row.

Annotation fitting preserves the existing two-stage architecture and checkpoint
contract. It initializes placement values only; complete-return training still
learns timing, resource management, and long-term action ranking.

The browser viewer receives the same JSON snapshots used by the live protocol and
renders the boards locally with Canvas 2D. It needs no simulator package or GPU;
expand the raw diagnostics sections when inspecting the original JSON.

Runtime inference and fitting use BF16 model parameters; optimization uses
separate FP32 master weights and FP32 Adam state. INT8/INT4 paths are not
implemented.

## Common checks

Static checks do not start the simulator or training:

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
```

- [Validation and test ownership](docs/validation.md)
- [Architecture and workflows](docs/architecture.md)
- [Viewer controls](docs/live-view.md)
- [Entity schema and attention](docs/math/entity-inputs.md)
- [Sequential Q objective](docs/math/sequential-training.md)
- [Reward accounting](docs/math/training-objective.md)
- [Execution, compiler troubleshooting and throughput](docs/math/training-throughput.md)
- [Collect cloud fitting diagnostics](docs/math/training-throughput.md#diagnose-cloud-fitting-stalls)
- [Sources used](docs/references.md)
- [Iteration history and measurements](docs/iteration.md)
