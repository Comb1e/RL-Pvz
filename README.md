# PVZ plant-placement research

Research **0.16.0** trains one shared CUDA PPO policy across easy, standard and hard.
The policy uses a **281-value timer-free observation** and independent actor/critic
Transformers with bounded event memory. It chooses wait, dig or plant first, then
the required plant type and tile. Simulation runs at **100 Hz**. Learning
results are experimental; **all earlier models require fresh training**. Regional
inputs retain zombie type counts, health, armor and
nearest zombie/unused-pole distances; explicit zombie behavior labels are absent.

## Requirements

Python 3.12, Git, an NVIDIA CUDA GPU and a compatible driver. Training and neural
optimization require CUDA. The CPU game remains a correctness and replay reference.
FFmpeg/libx264 is optional, only for MP4 export.

## Installation

Run from this project root. The external game source is
[E:/Projects/pvz-cuda-work](https://github.com/Comb1e/pvz-cuda-work), package **1.4.0**,
simulation **1.1.0**. The exact source commit and manifest are in
[src/pvz_rl/data/engine-lock.json](src/pvz_rl/data/engine-lock.json).

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

The installer reuses `.venv`, stages a verified Git archive under `build/`, and
installs it non-editably. It does not switch the game repository's branch or create
another environment. `E:/Projects/pvz` is not used or modified by these commands.
`requirements-lock.txt` locks common dependencies; `requirements-cuda-lock.txt`
locks CuPy/CUDA runtime/NVRTC. The installer separately selects CUDA PyTorch 2.8.0.
The compatibility switch `-Cuda` is accepted; CUDA dependencies are always installed.

## First training run

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --config configs\train.toml `
  --seed 101 --games 10000 --max-minutes 120 --output runs\event-memory-101
```

`runs/event-memory-101` is a **new generated directory**, not a shipped example.
Choose another unused name if it already exists. One recipe is shipped and is also
the installed default. All settings, including memory sizes and PPO parameters,
are configurable; alternate algorithms and legacy model conversions are absent.

Defaults are **128 environments × 128 transitions each** (16,384 per rollout),
batch size 1,024 and four epochs. Training uses a two-rollout periodic on-policy
window: one frozen policy snapshot collects both rollouts while the learner updates
the first. Probes, validation, stopping and checkpoint saves run only after both
updates complete. A final one-slot window handles a smaller remaining decision
budget. Waiting remains a transition. **128 never limits game length**;
unfinished episodes and their histories continue across updates. A wait/rejection
advances one 0.01-second tick; accepted planting/digging is instantaneous.

Use `--stage placement`, `saving`, `easy`, `standard` or `shared` to train one stage.
A compatible `--init-from CHECKPOINT` copies weights and starts fresh optimizers,
counters and episode histories. `--resume CHECKPOINT` restores the experiment and
remaining budget; interrupted games restart with empty memories. Both options need
the source checkpoint beside its `metadata.json`. See [training details](docs/research.md).

To train only placement until it passes, with no training time or game ceiling:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --config configs\train.toml `
  --stage placement --until-stage-complete --seed 101 --output runs\placement-101
```

This stops after mastery and saves `final.zip`; it does not advance automatically.
Use that checkpoint with `--stage saving --init-from runs\placement-101\final.zip`
and a fresh output directory for the next stage. Ctrl+C finishes the active two-rollout
window, then saves `interrupted.zip`. Resume it with `--resume CHECKPOINT` and a fresh output directory. The saved mode and
mastery progress are restored. Do not combine `--until-stage-complete` with explicit
`--games`, `--steps` or `--max-minutes`; per-game cutoffs remain active.

Mastery probes default to every **2,000 completed games**, requiring **100/100 wins**.
Normal easy/standard/hard checkpoint validation runs after each stage passes.
`best.zip` uses normal-game macro win rate with earliest-checkpoint ties. If no stage
passes, there is a final checkpoint and report but no validated best checkpoint or
automatic demos. Demonstrations use one selected checkpoint and CPU-verified hashes.

## Common checks

```powershell
Get-Content runs\event-memory-101\train.log -Wait
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\event-memory-101
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```

The offline report is generated under the run's `visualizations/` directory.
Open a generated **100 Hz** `.pvzdemo` with `pvz-rl replay FILE --watch`; use the
actual file path reported by training. Old 20 Hz recordings are incompatible and
have been removed locally. MP4 export is optional and samples 25 frames/second
without changing 100 Hz simulation or replay verification.

For a small CUDA pipeline check with a one-second simulation cutoff and three
verified demos, use a fresh artifact directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_sc2_training.py::test_spatial_cuda_report_and_three_verified_shared_demos `
  --basetemp artifacts\cuda-smoke-v0160
```

`src/pvz_rl/` contains implementation; `configs/` the single recipe; `tools/` installation
and diagnostics; `tests/` independent controls; `docs/` technical records. `.venv/`,
`runs/`, `artifacts/`, `build/` and `dist/` are generated, untracked local content.

- [Architecture and workflows](docs/architecture.md)
- [Training, reward, curriculum and limitations](docs/research.md)
- [Inspected research sources](docs/references.md)
- [Measured verification and learning results](docs/validation.md)
- [Iteration history](docs/iteration.md)
