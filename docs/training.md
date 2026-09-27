# Recording and training

Install with the [quick start](../README.md). Run commands from the project root.
The human demonstration and CUDA collector have separate profiles and checkpoint
formats. Availability checks and replay verification do not launch training.

| Workflow | Profile | Inputs and output |
|---|---|---|
| Human recording | `configs/demo.toml` | Easy seed 1000 → replay, transition archive, manifest, history |
| One-game initialization | `configs/demo.toml` | Complete replay/archive → `initialization.pt`, curves, coverage |
| Autonomous CUDA training | `configs/train.toml` | Fresh or compatible event-memory Q model → resumable ZIP checkpoints |

## Human demonstration

```powershell
.\.venv\Scripts\python.exe -m pvz_rl record-demo `
  --output runs\human-1000.pvzdemo --archive runs\human-1000.jsonl
```

Choose unused paths. Replay, archive, manifest and history files are checked
before the window opens; even empty files block reuse, and overlapping paths
are rejected. A session contains one easy-stage attempt. Pause preserves queued
actions; restart and stage switching are disabled. Closing early preserves an
incomplete recording. A natural win or loss produces reusable complete inputs.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl replay runs\human-1000.pvzdemo
.\.venv\Scripts\python.exe -m pvz_rl initialize-demo `
  --archive runs\human-1000.jsonl --replay runs\human-1000.pvzdemo `
  --output runs\human-init
```

Initialization verifies action order, execution outcomes and observation
reconstruction. It computes complete returns and fits the Transformer–LSTM in
256-decision chunks, carrying state between chunks. Defaults are 20 passes,
learning rate 0.0003, gradient clip 0.5 and a 30-minute ceiling. It writes
`initialization.pt`, `learning-curves.json`, `action-coverage.json` and
`replay-verification.json`. This is a fit to one demonstration; recurrent
autonomous training is not implemented by the CUDA collector.

Use the same `--config` for recording and initialization if changing demo
settings. The demo profile retains the requested 50% to 1% tile-exploration
schedule over 5,000 games for the recurrent protocol; fitting the recorded
actions does not sample exploration.

## Explicit autonomous training

The following command starts CUDA training and is separate from availability
checks and human initialization. The default curriculum is saving → easy →
standard → shared, with 128 complete games collected at fixed weights per cohort.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --config configs\train.toml `
  --seed 101 --output runs\cuda-101
```

The default budget is 10,000 completed games or 120 minutes, with a finalization
reserve. `--games`, `--max-minutes`, `--n-envs` and `--no-live-view` override these
execution settings. `--stage saving --until-stage-complete` removes the game/time
ceiling and stops at that stage's mastery gate. Use it only for an intentional
unbounded mastery run.

The collector's exploration budget decays from 10% to 0.1% over 3,000 stage
games. Species and tile coins each use `1-sqrt(1-budget)`; validation disables
both. This preserves the independent [probability controls](math/sequential-q-control.md).
The viewer's **F** shortcut follows the latest actions in visible panels;
[viewer controls](live-view.md) explain history and board switching.

## Resume and evaluation

Resume from an existing run's `last.zip`; metadata next to the checkpoint supplies
the original configuration, seed and progress. Full resume restores optimizer,
RNGs, unfinished games and collection/fitting position.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\cuda-101\last.zip --output runs\cuda-101
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\cuda-101\last.zip --split validation `
  --output runs\cuda-101\validation.jsonl
```

`--init-from` transfers structurally compatible CUDA weights into a fresh run
with fresh optimizer and counters. It does not accept `initialization.pt`.
Retired model protocols require their original source revision for inference;
they are rejected before deserialization here. Recordings remain preserved and
require their recorded engine version. Historical reports can be rebuilt with
`visualize --run <directory> --report-only`.

The [architecture](architecture.md) explains data ownership and failure paths;
[validation](validation.md) records checks without claiming training success.
