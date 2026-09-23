# PVZ plant-placement research

Research **0.10.2** trains **one shared CUDA MaskablePPO policy** for easy, standard,
and hard. There is one recipe, [configs/train.toml](configs/train.toml), also used
when `--config` is omitted. Policy and value learning use independent encoders and
Adam states. In the 0.10.0 saving-to-easy comparison, easy validation improved
for both seeds, but saving skill was not reliably retained and throughput fell.
The method remains experimental; the collapse is not resolved. See
[validation](docs/validation.md).

## Project layout

Run commands from this project root, `PVZ-plant`.

| Path | Contents |
|---|---|
| `src/pvz_rl/` | CUDA learning, CPU reference adapter, evaluation, reports and replay tools |
| `configs/train.toml` | The single configurable training recipe |
| `tools/` | Installer, pinned-source staging and diagnostic plotting |
| `tests/` | Independent mathematical, simulation and integration controls |
| `docs/` | Architecture, research, references, validation and iteration history |
| `requirements-lock.txt` | Common research, development and presentation dependencies |
| `requirements-cuda-lock.txt` | Required CuPy, CUDA runtime and NVRTC dependencies |
| `.venv/` | Local environment; generated, not committed |
| `runs/` | User training outputs; generated, not committed |
| `artifacts/`, `build/`, `dist/` | Local checks, staged dependency source and built packages; generated |

The game is an **external dependency**, not a second training project. This
machine's source repository is `E:/Projects/pvz-cuda-work`; `E:/Projects/pvz` is
another game/viewer checkout. Both stay unchanged. The installed game remains
package **1.3.0**, simulation **1.0.0**, commit
`8861824df6893a34c2cd4df7f9b68613376d7964`.

## Install and check availability

Requires Python 3.12, Git, an NVIDIA CUDA GPU and a compatible driver. From the
project root:

```powershell
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz-cuda-work
```

The installer reuses `.venv`, stages the pinned Git commit under `build/`, verifies
its source manifest and installs the game non-editably. The source repository
need only contain the commit; its checked-out branch is irrelevant. No extra
clone, environment, global CUDA Toolkit or Visual Studio is needed.

The two locks complement CUDA PyTorch **2.8.0+cu128**, installed separately by the
installer. CUDA uses CuPy **13.6.0**, runtime **12.8.90**, NVRTC **12.8.93**.
The old `-Cuda` installer switch is accepted; CUDA is always installed.

For an existing environment, update this research package and check it:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
```

`doctor` verifies the engine pin, kernels, tensor sharing and recording support.
CPU simulation remains for non-learning baselines, replay verification and tests.
Training and optimization require CUDA; no CPU fallback exists. FFmpeg/libx264 is
optional and needed only for MP4 export, not reports or compact recordings.

## Train

Start with a fresh directory:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --config configs\train.toml `
  --seed 101 --games 10000 --max-minutes 120 --output runs\saving-economy-101
```

`runs/saving-economy-101` is created by this command. It is not shipped. If it already
exists, choose another name and use it in the examples below. Training never
overwrites a run.

Defaults are **1,024 parallel GPU games × 128 decisions per game per rollout**,
minibatches of 1,024 and four PPO epochs. Completed games drive budgets, curriculum
and validation. Planting/digging is immediate; waiting or rejection advances one
tick. The same actor, critic and their optimizers continue through all five curriculum stages.
This collects 131,072 transitions per rollout. On the measured RTX 4070 laptop,
collection plus PPO updates were about **42% faster** than 128 environments,
using about 1.4 GB of GPU memory. Validation/export are outside that measurement;
larger rollouts are not evidence of better learning. See [measurements](docs/validation.md).

Normal checkpoint validation runs **once after each individual curriculum stage
passes**, on 50 seeds per difficulty. It does not run periodically while learning
that stage or merely because the game/time budget ends. `best.zip` maximizes the
equally weighted win rate across the three difficulties; earlier checkpoints win
ties. Checks use the same post-update weights that passed the mastery probe.
Interrupted evaluations retain a pending checkpoint for resume before more learning.
The 120-minute budget includes startup, validation and presentation, reserving
15 minutes for finalization. Stops occur after complete PPO updates, so game
counts can overshoot. Unfinished validation, exports and mastery are explicit.

Edit a copy of the TOML to change rewards, learning rate, discount, GAE lambda,
clipping, epochs, batch size, value coefficient, gradient clipping, advantage
normalization, exploration coefficients or architecture widths. `profile` is just
a label. `target_kl = 0.01` stops actor updates when approximate KL exceeds 0.015;
the independent critic finishes its epochs. Zero disables the check. PPO averages
over **all transitions**, including forced waits. `training.learning_rate` controls
the actor; optional `critic_learning_rate` defaults to the same rate. Gradient
clipping applies independently to both parameter groups.

The 500 inputs use categorical plant/state embeddings, three enemy/projectile
regions per lane and global state. Defaults are 8/4 embedding dimensions, a
64-unit scalar encoder, two 32-channel convolutions and 128×128 policy/value heads.
Each learned encoder has its own embeddings, scalar MLP and convolutions. The actor
selects a legal action type, then a tile, retaining all 406 actions. The critic
estimates remaining discounted reward and has no gradient path into the actor.
The default training model has 169,467 parameters. During collection only the
actor runs per action; the frozen critic evaluates stored observations afterward
in `training.value_batch_size = 1024` chunks, before timeout bootstrapping and GAE.
Evaluation and demonstrations use the actor alone.
The initial dig-logit bias is −6 and remains trainable. There are no plant-retention
rules, savings rules, early-dig penalties or action delays.

Only outcome, mower cost and potential shaping contribute to reward:

\[
r = r_{outcome} - 0.2\,newMowerActivations + \gamma\Phi(next)-\Phi(current)
\]

\[
\Phi(o)=0.5\frac{defeated}{\max(1,initialZombies)}
 +0.1\frac{sun+livingPlantPurchaseValue}{300}.
\]

Victory gives +1 and defeat −2. Natural terminal potential is zero; time-limit
truncation retains potential and bootstraps the critic. **300 is a scale, not a
sun cap**. Purchase value is preserved when planting and lost when digging or a
plant dies. Damage, kills and early digs remain diagnostics without separate rewards.
The economy weight is reduced from 0.5 to 0.1: resource income and living-plant
value contribute one fifth as much potential. There is no direct planting bonus.
Buying preserves total resource value; buying and immediately digging loses value.
This also weakens the shaping loss from digging or plant death; it is not an
additional anti-digging rule or a demonstrated learning improvement.

CLI overrides include `--n-envs`, `--rollout-steps-per-env`, `--batch-size`,
`--games` and `--max-minutes`. Total rollout size is `n_envs × rollout_steps_per_env`;
conflicting explicit totals fail. `--device cuda` and `--simulator cuda` remain
accepted. `--eval-games` controls periodic validation for diagnostic/fixed runs and
archived periodic configurations; it does not trigger teaching-stage evaluation.
A short pipeline check is:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --family diagnostic `
  --games 2 --n-envs 2 --rollout-steps-per-env 32 --batch-size 32 `
  --eval-games 1 --validation-count 1 --max-minutes 2 `
  --output artifacts\cuda-smoke-v0102
```

This tests integration, not game-playing competence. `suite` repeats this same
method for the configured learner seeds and performs the configured evaluations;
jobs without a validated checkpoint are explicitly skipped in suite evaluation.
The suite can be expensive and is user initiated. `benchmark-gpu` changes parallelism
only. Old recipes, alternate policies, CPU/hybrid training, `pilot`, `benchmark`
and `compare-sc2` were removed.

## Stages, initialization and resume

| Stage | Training tasks | Default mastery requirement |
|---|---|---|
| `placement` | Placement lesson | 100/100 placement wins |
| `saving` | 80% saving, 20% placement | 100/100 saving wins |
| `easy` | 80% easy, 10% each lesson | 100/100 easy wins |
| `standard` | 45% easy, 45% standard, 10% saving | 100/100 each on easy and standard |
| `shared` | 20% easy, 40% standard, 40% hard | 100/100 on each difficulty |

The saving lesson starts with **50 sun**, permits sunflower/peashooter and legal
digging, disables mowers, and spawns **one basic zombie in each of three randomly
chosen lanes at 50 seconds**. Sky sun remains 25 every 10 seconds. An undefended
lane loses at tick 1999 (99.95 seconds), before the tenth sky payment. Without
sunflowers there can be only `50 + 9 × 25 = 275` sun, less than the 300 required
to place even one peashooter in all three lanes. Sunflowers are therefore necessary.
A verification control with two sunflowers wins all ten lane combinations; it
never supplies learner actions. See [the calculation](docs/research.md#saving-lesson).
Placement remains the original one-lane lesson. All lesson quantities, including
`lanes_per_spawn`, are configured in the TOML.

Mastery probes run every **2,000 training games** (previously 500), using seeds 100050–100149,
separate from normal checkpoint validation (100000–100049). At least 100 games
started in the current stage must finish before it can pass. One passing probe
is required by default. Counts, intervals, consecutive passes and thresholds are
configurable. A loss, truncation or incomplete probe prevents mastery.
`--validation-count` never shrinks mastery probes.
Change `curriculum.probe_interval_games` to adjust this frequency. Crossed intervals
coalesce into one probe after a complete PPO update; no extra probe is forced
before its interval when training stops. `training.validation_schedule = "stage_success"`
controls normal checkpoint validation. Saved recipes without this setting retain
their original periodic schedule on resume.

Use `--stage` to train a stage alone. It stops on mastery or budget and never
promotes itself. For example:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --stage placement --seed 101 `
  --games 10000 --max-minutes 30 --output runs\compact-stages-101\placement

# Run after the placement checkpoint exists:
.\.venv\Scripts\python.exe -m pvz_rl train --config configs\train.toml --stage saving `
  --init-from runs\compact-stages-101\placement\final.zip `
  --games 10000 --max-minutes 30 --output runs\compact-stages-101\saving
```

`--init-from` copies **weights only**. Both optimizers, counters, mastery,
checkpoint selection and time allowance start fresh. It also works without
`--stage` for a new automatic-curriculum experiment. Without `--config`, it inherits
the source settings; specify an edited TOML to change rewards, PPO or curriculum.
Network dimensions, categorical layout and engine must match. Metadata records
the source checkpoint hash, structural signature and parameter changes.
Use `--config configs\train.toml --init-from ...` to adopt the new saving lesson,
reward weight and parallelism with compatible old weights. `--resume` deliberately
keeps the saved lesson, reward and environment count. Earlier saving mastery is
not evidence of passing the new lesson.
The same explicit-config initialization adopts the new evaluation schedule;
resuming an older run preserves its recorded intervals and periodic evaluations.

`--resume` restores the saved actor, critic, both Adam states, configuration, counts, mastery
and validation schedule with the remaining cumulative budget. It cannot change
research parameters. Use a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --resume runs\saving-economy-101\latest.zip `
  --output runs\saving-economy-101-resumed
```

The checkpoint must remain beside `metadata.json`. Interrupted episodes restart;
continuation is not bit-for-bit. Use `--init-from` for changed experiments or an
additional allowance. `final.zip` is suitable for stage handoffs; `best.zip` is
selected exclusively by normal-game validation and supplies demonstrations.
A budget-limited stage is not certified as mastered. If no stage passes, training
still saves `final.zip` and its report, but has no `best.zip` or normal-game demos.
`latest.zip` is saved after mastery probes as well as completed validation; before
the first probe use `final.zip` or an available `interrupted.zip` for recovery.

A **0.9.0 shared-encoder checkpoint** can initialize this version through
`--init-from`: its encoder weights are copied into the independent actor and critic,
preserving starting predictions. This conversion starts fresh Adam states and is
recorded in metadata. Shared-encoder checkpoints cannot be resumed or directly
loaded for new evaluation. Earlier architectures remain unsupported. Existing
reports and recordings can still be regenerated without loading their models.

To recover from the easy-stage collapse, initialize from a mastered saving-stage
checkpoint and choose a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train --config configs\train.toml --stage easy `
  --init-from runs\compact-stages-101\saving\final.zip `
  --games 10000 --max-minutes 120 --output runs\separated-easy-101
```

This example requires that saving checkpoint to exist beside its metadata.

## Logs, curves and replay demos

Every 15 seconds, `train.log` and the terminal show phase, games, throughput,
elapsed time, rolling performance, stage and kills. Episode records are saved
without printing every episode. Optimization metrics include KL, actual optimizer
steps for each optimizer, gradient norms, entropy, actor KL stopping and policy
drift measured after the update. Task-specific rolling windows report easy,
placement and saving separately, including accepted plant usage. Started, active,
completed-game and transition counts show the actual data mixture.

An **early dig** is an accepted voluntary dig within five simulated seconds
(100 ticks, inclusive) of that plant's placement, including the same tick. The
window is configurable. Digs per game and digs per accepted planting are both
diagnostics; no purchases produces a missing ratio rather than a misleading zero.
Overall rolling win rate mixes tasks and can change when short lessons finish
before normal games. Use task-specific curves and normal validation to assess it.

```powershell
Get-Content runs\saving-economy-101\train.log -Wait
# Open after the report exists:
Start-Process runs\saving-economy-101\visualizations\index.html
.\.venv\Scripts\tensorboard.exe --logdir runs
```

| Generated path inside a run | Contents |
|---|---|
| `metadata.json`, `config.json`, `status.json` | Source pins, resolved settings, provenance and current state |
| `training-episodes.jsonl`, `training-metrics.jsonl` | Episodes, rolling behavior and completed-update PPO metrics |
| `learning-curve.jsonl`, `validation/` | Normal-game results after stage success; absent before the first success |
| `curriculum.json`, `curriculum-probes.jsonl`, `curriculum-probes/` | Mastery state and probe evidence |
| `best.zip`, `best.json`, `latest.zip`, `final.zip` | Best requires completed validation; latest requires a probe/validation; final is always saved on completion |
| `visualizations/index.html`, `visualizations/*.png` | Offline report and curves |
| `visualizations/demos.json`, `visualizations/games/` | Three CPU-verified compact recordings sharing one checkpoint |
| `visualizations/videos/` | Optional MP4 exports |

All three normal-game demos use the same `best.zip` and seed 100000, showing
actual outcomes. Diagnostic runs produce only a diagnostic demo. Reports refresh
after validation and at completion. With no validated checkpoint, regeneration
builds a report without gameplay or model loading. Replay examples require
`visualizations/demos.json` to exist. Rebuild or optionally export videos:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\saving-economy-101
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\saving-economy-101 --videos

# Resolve the first recording from the generated manifest:
$run = Resolve-Path runs\saving-economy-101
$demo = (Get-Content "$run\visualizations\demos.json" -Raw | ConvertFrom-Json).demos[0]
$recording = Join-Path "$run\visualizations" $demo.replay
.\.venv\Scripts\python.exe -m pvz_rl replay $recording --watch --speed 2
.\.venv\Scripts\python.exe -m pvz_rl replay $recording --video artifacts\demo.mp4
```

The replay command supports `.pvzdemo`, legacy JSON/gzip and immediate research
actions. Space pauses, arrows seek and Home/End jump to boundaries. Supported
speeds are 0.5, 1, 2, 4 and 8. Use this viewer without upgrading the simulator pin.
`--videos`/`--no-videos` affect MP4 only; compact demos remain enabled. Failed
exports preserve training results and can be retried. Archived report regeneration
does not load retired weights or regenerate their gameplay.

## Evaluate and verify

```powershell
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\saving-economy-101\best.zip `
  --split validation --count 10 --record --output artifacts\saving-economy-validation
.\.venv\Scripts\python.exe -m pvz_rl benchmark-gpu --minutes 15 --output artifacts\cuda-benchmark-v0101
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests tools
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build --no-isolation
```

Benchmarking tests 128/256/512/1024 parallel games with three repetitions; override
the candidates with `--env-counts 128 256 512`. It measures collection **and PPO
updates**, separates setup/warmup, and selects the smallest stable count within
5% of the fastest median. Failed or incomplete profiles cannot be selected.
Larger environment counts also enlarge the rollout and can change learning
dynamics; throughput is not win-rate evidence. Tests include short CUDA updates and independent
CPU mathematical/game controls. They do not launch formal research training.

Details: [architecture](docs/architecture.md), [research](docs/research.md),
[references](docs/references.md), [validation](docs/validation.md),
[iteration history](docs/iteration.md).
