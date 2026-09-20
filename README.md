# PVZ plant-placement research

Train a policy to choose **which plant, which tile, and when to act** in the existing
[Lawn Lab game](../../pvz/README.md). This project contains the research code; the game remains
an independent dependency at `E:/Projects/pvz`.

Implemented: a Gymnasium adapter, MaskablePPO and PPO training, five research
conditions, four non-learning baselines, checkpoint recovery, deterministic replays,
held-out evaluations, changed-wave scenarios, statistical analysis, progress logs,
automatic offline reports, learning curves, compact game demos, and optional MP4 export.
**Each run trains one shared policy for easy, standard, and hard.** The same
`best.zip` is used for all three difficulties and their demonstration recordings.
Research version **0.4.1** uses game package **1.2.1** (simulation version **1.0.0**).
The new pure-RL profiles add tactical observations, grouped plant/tile decisions,
and introductory lessons. They are experimental: see [paper adaptation and pilot
evidence](docs/paper-adaptation.md). The baseline remains the default configuration.
Compatible 1.2.1 baseline checkpoints retain their original encoder, policy, and
fixed curriculum. Improved profiles require fresh runs; weights are not migrated.
Checkpoints from older game source pins cannot be resumed or evaluated. Archived
reports and replay files remain readable. No formal research suite was launched.

## 1. Install and check availability

The workspace already has a Python 3.12 virtual environment with CUDA-enabled
PyTorch. From this directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl doctor --output artifacts\availability.json
.\.venv\Scripts\python.exe -m pytest -q
```

For a fresh installation, use Python 3.12 and Git:

```powershell
# NVIDIA GPU installation; downloads the CUDA-enabled PyTorch wheel.
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz -Cuda

# CPU-only installation, suitable for these small MLP policies.
.\tools\bootstrap.ps1 -GameRepo E:\Projects\pvz
```

Training uses **CUDA by default**. The current environment detects an NVIDIA
GeForce RTX 4070 Laptop GPU with PyTorch `2.8.0+cu128`. For a CPU-only installation,
add `--device cpu` to training and suite commands. If CUDA is unavailable, training
stops with an actionable error instead of silently switching devices.

The installer checks that the game checkout is clean and at commit
`6fd1f54706369915013a49eab5c1790f8c55ab0a`, stages a verified Git archive under
this project's `build` directory, and installs the game non-editably from that copy.
Packaging inputs and build artifacts stay outside the game repository.
Every training/evaluation command verifies the installed game's source manifest
and rules hash, package version 1.2.1, and simulation version 1.0.0. Source pins
must match the checkpoint even when two releases share the same combat rules.

For an existing research virtual environment, upgrade only the game and research package:

```powershell
$gameStage = Join-Path 'build' ('game-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -B tools\stage_game.py --repo E:\Projects\pvz --output $gameStage
.\.venv\Scripts\python.exe -m pip install --no-deps --force-reinstall $gameStage
.\.venv\Scripts\python.exe -m pip install --no-deps -e ".[dev,ui]"
.\.venv\Scripts\python.exe -m pvz_rl doctor
```

Use a new output directory and the bundled configuration for new training.
Do not edit old run metadata to bypass version checks. The original game checkout
and existing run directories are preserved; old models require their original
research environment. No checkpoint migration is provided.

The commands below use `python -m pvz_rl`; `pvz-rl.exe` in the virtual environment
is an equivalent entry point. No environment activation is required.

Video export needs `pygame-ce` (included in the installer) and FFmpeg with the
`libx264` encoder. FFmpeg 8.1 is available on this machine. For a fresh Windows
installation, install FFmpeg, for example with `winget install --id Gyan.FFmpeg`,
then open a new terminal. `doctor` reports rendering and encoder availability.
Alternatively set `visualization.ffmpeg` to an executable path in your TOML file.
Compact recording and report generation require neither pygame nor FFmpeg.
The native viewer and frame/video rendering need pygame; optional MP4 export also
needs FFmpeg. Missing video dependencies do not block default training or demos.

## 2. Run a short smoke test

This executes 128 training decisions on the restricted diagnostic task, evaluates
one validation case, and saves real checkpoints. It checks integration, not skill.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --condition masked --family diagnostic --seed 101 `
  --steps 128 --n-envs 1 --rollout-size 64 --batch-size 32 `
  --eval-interval 64 --validation-count 1 `
  --output runs\smoke
```

Output directories must be new. Existing runs and checkpoints are not overwritten.
The smoke run also produces a report and a diagnostic `.pvzdemo` recording.
Add `--videos` to export an MP4 as well.
The diagnostic uses one threatened lane, 100 starting sun, no mowers, and a
peashooter/wait action mask. A manually placed peashooter provides an independently
tested winning control. Diagnostic policies are not benchmark results and cannot
be passed to the formal test evaluator.

## 3. Train a policy

To try the new experimental method, start **one shared model** with:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --config configs\pure-rl.toml --condition masked --seed 101 `
  --device cuda --output runs\pure-rl-101
```

This is a fresh model, not a continuation of an old flat-action checkpoint.
`best.zip` is selected only by the average normal-game validation win rate across
easy, standard, and hard. All three automatic `.pvzdemo` recordings use that same
checkpoint. Lessons never select the shared checkpoint, and no scripted actions,
imitation data, or warm-start weights enter learning.

| Configuration | Observation | Action policy | Training curriculum |
|---|---|---|---|
| `configs/baseline.toml` | 2,719 spatial values | Flat masked | Fixed schedule |
| `configs/tactical.toml` | 1,140 tactical values | Flat masked | Fixed schedule |
| `configs/grouped.toml` | 1,140 tactical values | Masked type, then tile | Fixed schedule |
| `configs/pure-rl.toml` | 1,140 tactical values | Masked type, then tile | Lesson mastery gates |

Grouped profiles use `--condition masked` (or `sparse` for an explicit reward
experiment). They do not support the unmasked or scripted-hybrid conditions.
Use the baseline configuration for the original five-condition `suite` command.
The new recipe comparison is:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl pilot --output runs\paper-pilot --minutes 30
```

This spends at most five minutes of the budget on placement/saving learning
diagnostics, then compares four profiles with learner seeds 101 and 102. Each run
has at most 262,144 decisions and validates every 32,768 on the first five
validation seeds per difficulty. Configuration order reverses for seed 102.
Remaining time is divided among remaining jobs; collection stops between complete
PPO updates when its slot expires. An in-progress probe/validation and checkpoint/
report cleanup can finish afterward. It never uses final-test seeds.

`report.md`, `comparison.json`, and `comparison.png` compare only validation budgets
completed by every profile and both seeds. Missing runs/budgets are explicit.
Diagnostics must pass and the candidate must improve over baseline for both seeds
at a shared budget before the runner recommends a larger development study.
Otherwise it remains experimental. Per-run offline HTML reports are under
`PROFILE-SEED/visualizations/index.html`; pilot demos are disabled to reserve time
for learning. To generate a pilot checkpoint's demos afterward, use a copy of its
configuration with `visualization.demos = true` and `visualize --config ... --run ...`.

The teaching stages are placement → saving/economy → easy → easy/standard → shared
20/40/40 easy/standard/hard training. Probes run every 16,384 collected decisions,
using 20 fixed validation cases. Advancement requires two consecutive passes and
at least 16,384 decisions in the stage. Thresholds and rehearsal distributions are
in the configuration. New stages affect only episode resets and retain the same
policy and optimizer. `curriculum.json` and `curriculum-probes.jsonl` record progress;
checkpoints embed the same state for resume. Budget exhaustion before the shared
stage is reported as `curriculum_incomplete`; stages are never skipped to meet a
deadline. Normal-game evaluation always permits all eight plants.

Episode records also include sustained-attacker purchases, first-attacker time,
maximum sun, spending by plant, and affordable-attacker opportunities. The report
adds economy and grouped-entropy curves. Waiting frequency alone is not treated
as failure. Logging remains throttled to 15 seconds, with immediate curriculum,
validation, checkpoint, and completion messages.

### Rewards and penalties

All supplied profiles use the following reward. `N` is the initial number of
zombies in the episode, guarded by `max(1, N)` for empty scenarios.

| Event | Reward |
|---|---:|
| Win / loss | +1 / −1 |
| Plant kills a zombie | +1/N |
| Mower kills a zombie | −2/N |
| Plant damages a zombie without that hit killing it | +actual HP removed / (full starting zombie HP × N) |
| Mower activates and kills nothing on that tick in its lane | −1/5 |

Armor-only damage earns zero. The lethal hit earns only the kill reward;
earlier nonlethal hits still earn damage rewards, even if a mower later finishes
the zombie. Starting HP means the base health from the active game rules, not
remaining health or armor. For example, a nonlethal 20-HP hit on a basic zombie
in a 10-zombie game earns `20/(200*10) = 0.01`; a mower kill costs `0.2`.
The pinned engine immediately kills its triggering zombie when a mower activates,
so the empty-activation penalty normally remains zero.

Shaped conditions additionally receive `gamma * Phi(next) - Phi(current)`, with:

```text
Phi = 0.5 * defeated / max(1, N)
    + 0.5 * (stored sun + sum of living plant purchase costs) / 300
```

The economy sum has no cap and uses full purchase costs, regardless of plant
health. Buying a plant preserves value; digging or losing it removes its value.
The difference of potentials still uses `gamma = 0.999`. Genuine wins/losses set
`Phi(next) = 0`; external time cutoffs retain the potential for bootstrapping.

```toml
[reward]
plant_kill_weight = 1.0
mower_kill_weight = 2.0
normalize_kills = true
damage_weight = 1.0
empty_mower_activation_penalty = 0.2
gamma = 0.999
potential_mode = "plant_value"
defeated_weight = 0.5
economy_weight = 0.5
economy_scale = 300
```

The legacy `sparse` condition disables potential shaping but keeps these event
rewards. For terminal-only reward, also set both kill weights, `damage_weight`,
and `empty_mower_activation_penalty` to zero. Checkpoint selection remains the
equal-weight normal-game **win rate** across easy, standard, and hard.

Episode records separate kill counts, nonlethal HP damage, empty activations, and
all four event-reward contributions. Progress logs and curves show kill counts,
damage reward, and empty activations. Detailed semantics and numerical examples
are in [docs/reward-design.md](docs/reward-design.md).

**Start a fresh run to use the revised reward.** Old saved configurations retain
their original potential and reward defaults; existing compatible checkpoints
remain readable. Resume requires identical research settings and rejects mixing
reward versions. The earlier 0.4.0 pilot does not evaluate this revised reward.

Use one training command for all difficulties. Curriculum stages change which
games are sampled, while the same policy weights and optimizer continue learning.
There are no difficulty-specific model files or difficulty-specific checkpoint
selection. The five conditions below are optional research comparisons, and learner
seeds are independent repetitions; each individual run still learns one shared policy.

Start with a pilot on the real game:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --condition masked --seed 101 --steps 100000 `
  --n-envs 4 --device cuda --eval-interval 25000 --validation-count 5 `
  --output runs\pilot-masked-101
```

Then train the full direct-placement condition with the default protocol:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --condition masked --seed 101 --output runs\masked-101
```

Defaults: 3 million decisions, eight environments, CUDA policy training, separate
256–256 policy/value networks, learning rate `3e-4`, discount `0.999`, GAE `0.98`,
4,096 decisions per rollout, minibatches of 256, four optimization epochs,
clipping `0.2`, and entropy coefficient `0.01`.

The policy and value networks run on the GPU; game simulation workers run on the
CPU. Use `--device cuda` to select CUDA explicitly or `--device cpu` for CPU training.
Small VRAM usage is expected for this MLP: memory usage is not GPU utilization.
Runtime optimizations are enabled by default; no larger network, batch, rollout,
worker count, changed precision, or new learning strategy is needed to use them.

Each decision applies one action and advances 10 ticks (0.5 simulated seconds).
`--steps` counts **aggregate policy decisions across all workers**, not ticks,
episodes, or decisions per worker. Complete PPO rollouts round the 3,000,000 target
up to **3,002,368 actual decisions**; both counts are recorded. Validation happens
after completed PPO updates when its interval is reached, and after the final update.

| `--condition` | Actions | Reward | Difficulty sampling |
|---|---|---|---|
| `masked` | 406 actions, legal-action mask | Potential-shaped | Curriculum |
| `unmasked` | Same 406 actions, no mask | Same shaped reward | Curriculum |
| `sparse` | 406 actions, legal-action mask | Win/loss + configured combat rewards; no potential | Curriculum |
| `mixed` | 406 actions, legal-action mask | Potential-shaped | 20% easy / 40% standard / 40% hard throughout |
| `hybrid` | Five masked strategies with scripted placement | Potential-shaped | Curriculum |

The curriculum spends the first 10% of decisions on easy, the next 30% on an equal
easy/standard mixture, and the remaining 60% on the 20/40/40 mixture. Stage changes
affect the next episode reset. One policy learns across all three difficulties.
All eight plants remain available under the normal game rules.

The hybrid chooses wait, economy, sustained attack, blocking defense, or emergency
placement. It uses scripted placement knowledge, so evaluate it alongside the
`random_strategy` baseline as well as the original heuristic.

### Monitor training

The terminal and `train.log` show startup settings immediately, progress every
15 seconds, curriculum changes, validation progress/results, checkpoint saves,
and video export progress. Completed episodes are stored as research data rather
than printed individually. Example progress format (illustrative values):

```text
2026-09-20T12:00:15+00:00 [collecting] 8,192/100,352 decisions (8.2%); 550 decisions/s; elapsed 00:00:15; training ETA 00:02:47; last 18 games win 22.2%, reward 0.145; best validation pending
2026-09-20T12:01:00+00:00 [validating] Validation at 28,672 decisions
2026-09-20T12:01:12+00:00 [validating] Saved shared best.zip at 20.0%
```

The ETA estimates remaining collection/optimization time; validation and final
video export add time. Rolling statistics use up to 100 completed training games.
The rolling win rate changes with the curriculum's difficulty mix, so use the
fixed validation curves for progress comparisons.

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs
Get-Content runs\masked-101\status.json
# In a separate terminal, follow the persistent log:
Get-Content runs\masked-101\train.log -Wait
```

The main artifacts in a run are:

| Artifact | Meaning |
|---|---|
| `metadata.json`, `config.json` | Resolved settings, seeds, dependency versions, Git state, engine and rules hashes |
| `status.json` | Lifecycle state, collected decisions, elapsed time, validation result |
| `train.log` | Timestamped operational progress and significant events |
| `training-metrics.jsonl` | Rolling training statistics, throughput, and metrics after each PPO update |
| `training-episodes.jsonl` | Outcomes, plant usage, mowers, invalid actions, and episode lengths |
| `learning-curve.jsonl` | Validation win rates against training decisions and wall time |
| `validation/<steps>/` | Full validation episode records and summaries |
| `best.zip`, `best.json` | Best validation macro win rate, with earlier checkpoints winning ties |
| `latest.zip` | Most recent validation checkpoint |
| `final.zip` | Policy after the final optimization update |
| `interrupted.zip` | Recovery checkpoint when an interruption/error can be handled |
| `visualizations/index.html` | Offline curves, compact-demo links/viewer commands, and available videos |
| `visualizations/*-curves.png` | Validation, training behavior, throughput, and PPO figures |
| `visualizations/demos.json` | Fixed demo cases, outcomes, replay paths, and shared checkpoint hash |
| `visualizations/games/attempt-*/replays/*.pvzdemo` | Verified compact recordings with embedded outcome and provenance |
| `visualizations/videos/*.mp4` | Optional videos, created with `--videos`; omitted by default |
| `visualizations/status.json` | Export outcome and duration, separately from training success |

The checkpoint criterion averages easy, standard, and hard win rates equally.
It does not use shaped return or final-test results.

### Open the curves and game demos

```powershell
Start-Process runs\masked-101\visualizations\index.html
```

The report refreshes after validation and at completion; reload the browser to see
updates. It works offline and includes validation win rates overall/per difficulty
against decisions and wall time, rolling training statistics, PPO losses/entropy/KL,
and throughput. TensorBoard remains available for detailed inspection.

After training, one load of the selected `best.zip` plays easy, standard, and hard
using deterministic actions on the first validation seed (100000 by default).
Compact demos show the actual outcome, including losses and cutoffs, and all three
carry identical checkpoint hashes. Each `.pvzdemo` contains compressed timed actions
and an initial snapshot, with no video frames or model weights. Games are recorded
to completion or the configured cutoff and hash-verified. Follow the report's
viewer command to watch, pause, seek, inspect entities, or change playback speed.

Optional MP4 export uses the game's native 1280×820 renderer at 20 frames/second,
with a two-second final outcome hold. Available videos appear in the HTML report
with play/pause, seeking, and speed controls; they do not autoplay.

These fixed validation demonstrations are for inspection, not held-out evidence.
Diagnostic checkpoints produce only a diagnostic demonstration. Demo generation
begins after training workers close and does not update model weights.

Regenerate reports/demos or request MP4 export without training:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\masked-101
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\masked-101 --videos
.\.venv\Scripts\python.exe -m pvz_rl visualize --run runs\masked-101 --no-videos
```

Older runs may lack aggregated optimizer/training metrics; the report marks those
panels unavailable. Archived runs rebuild reports and can export existing recordings
without loading old models; missing gameplay cannot be regenerated from an old
checkpoint. Use `--no-videos` to rebuild an old report without encoding, since older
saved configurations may have enabled video by default.
Resumed runs display labeled segments with separate elapsed
times and exclude ancestor data beyond the checkpoint used to resume. If an older
resume has no recorded boundary, the report displays only that run's segment.
Reports and videos are derived artifacts and can be rebuilt. Export failures are
logged separately and do not invalidate completed training or saved checkpoints.

Optional settings in a copied configuration file:

```toml
[logging]
progress_seconds = 15
rolling_window = 100

[visualization]
enabled = true
demos = true
videos = false
video_size = [1280, 820] # positive even dimensions for H.264
ffmpeg = "" # PATH lookup; or a literal path such as 'E:\Tools\ffmpeg.exe'
crf = 23
final_hold_seconds = 2
```

`train`, `suite`, and `visualize` accept mutually exclusive `--videos` / `--no-videos`;
with neither flag they respect configuration. Disabling videos retains compact demos.
Set `demos = false` and `videos = false` for reports only, or `enabled = false` to
disable automatic visualization entirely. Explicit video generation also creates
the necessary demos, even when `demos = false`.
Logging/visualization settings are saved in metadata but excluded from experiment
compatibility checks. Changing these preferences alone does not prevent resuming.

### Resume an interrupted run

Resume is supported only for checkpoints using the current game source pin.

Resume into a **new directory**, using the same configuration, learner seed,
condition, diagnostic setting, and validation count as the original run:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --condition masked --seed 101 `
  --resume runs\masked-101\interrupted.zip `
  --output runs\masked-101-resumed
```

If the process was killed before saving `interrupted.zip`, use `latest.zip`.
Repeat any original pilot overrides when resuming a pilot. `--steps` remains the
original total budget; it is not an additional-step count. The earlier best checkpoint
is retained when applicable. Resume restores the policy and optimizer but starts
fresh game episodes; it is not a bit-for-bit continuation of rollout/RNG state.
Device settings must also match: when resuming a compatible run created with the
previous CPU default, pass `--device cpu` (or its original configuration file).
Only checkpoints with the current **1.2.1 source pin** can resume. Game 1.2.1 adds
the defeated/total HUD without changing combat, but the strict source-pin rule
still requires fresh training when upgrading from 1.2.0. Existing compact recordings
can still be watched or exported, and archived reports remain readable.
Runtime settings can change on same-pin resume; all research settings must match.
An already running Python process keeps its loaded code. Use the improvements in
your next run, or interrupt normally with Ctrl+C and resume the saved checkpoint
into a new directory. Do not overwrite or edit the previous run's metadata.

## 4. Choose CPU or GPU and worker count

This environment is mostly Python simulation plus a modest neural network, so GPU
availability alone does not establish faster training. Measure the complete loop:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl benchmark `
  --steps 16384 --workers 1 4 8 --devices cpu cuda --repeats 3 `
  --output artifacts\hardware-benchmark
```

This performs short real training runs and writes raw measurements and a
`recommendation.json`. CUDA configurations are recorded as unavailable if needed.
Each measurement first warms up with one rollout. Worker/model startup and warmup
times are recorded separately; reported throughput includes collection and PPO
updates, but excludes validation, reports, and demos. Actual rounded decisions,
phase timings, and peak PyTorch CUDA memory are saved in `measurements.json`.
Use the recommendation consistently in later comparisons:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl train `
  --hardware artifacts\hardware-benchmark\recommendation.json `
  --condition masked --seed 101 --output runs\selected-hardware-101
```

`--hardware` overrides the bundled CUDA default; an explicit `--device` overrides
the hardware recommendation.

Parallel environments use Windows-compatible `spawn`; each worker owns its game.
When invoking the Python API from a script, put training calls under
`if __name__ == "__main__":`.

### Measure runtime improvements with the same training strategy

```powershell
.\.venv\Scripts\python.exe -m pvz_rl benchmark `
  --workers 8 --devices cuda --steps 32768 --repeats 3 --compare-runtime `
  --output artifacts\runtime-benchmark
```

This compares the stock data path with the configured optimizations using the same
learner seeds, environments, PPO settings, and warmup budget. Pair order alternates.
`runtime-comparison.json` records each speed ratio and whether the final policy
weights match exactly. Avoid overlapping other training or tests while measuring.
This command runs short benchmarks only; it does not start the formal suite.

On the RTX 4070 Laptop GPU, the local three-pair benchmark measured **2,176 → 2,664
decisions/s median (22% faster)**, with exactly equal final policy weights in every
pair. This measures early curriculum collection and updates, excluding validation
and reports. See [validation details](docs/validation.md) for raw evidence and limits.

The training log now includes `last rollout 1.80s collect / 0.30s update` (example
values). Collection includes simulation, policy inference, and worker communication;
update time covers PPO optimization. These timings also appear in
`training-metrics.jsonl`. Validation and report time remain separate.

In the optional `[runtime]` TOML section, `coalesce_masks = true` sends legal masks
with observations instead of requesting them separately from every worker.
`cache_legal_actions = true` reuses the engine's legality query while occupied
tiles, card availability, and game status are unchanged. It does not predict new
legal moves: changes trigger a fresh engine query. This also accelerates evaluation.
`cache_rollout_on_device = true` copies the completed rollout to CUDA once and
reuses it across optimization epochs; it adds about 49 MiB at default settings.
CPU training uses the stock buffer sampler. These settings preserve NumPy minibatch
permutations, arithmetic precision, PPO losses, and the policy/optimizer.
Set any option to `false` for diagnosis. Older configs receive current defaults;
resolved runtime settings are recorded in metadata but excluded from research
compatibility checks. Mask and numeric validation checks stay enabled.

## 5. Evaluate baselines and checkpoints

First reproduce development cases without touching final-test seeds:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --baseline heuristic --split development --count 10 --record `
  --output runs\baseline-development
```

Available baselines are `wait`, `random_legal`, `heuristic`, and `random_strategy`.
The heuristic is the frozen controller from the pinned game commit. Random policies
use reproducible scenario-derived RNGs. Every policy receives the same decision rate.

Evaluate a pilot checkpoint on validation seeds:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\pilot-masked-101\best.zip `
  --split validation --count 10 --record --output runs\pilot-validation
```

Once settings and checkpoints are frozen, run the held-out evaluation:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\masked-101\best.zip --split test --record `
  --output runs\masked-101-test

.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --baseline heuristic --split test --record --output runs\heuristic-test
```

Default held-out evaluation uses 300 seeds for each difficulty. A checkpoint's own
configuration is restored automatically; supplying a conflicting `--config` is rejected.
The evaluator passes action masks to masked policies and preserves unmasked behavior
for the unmasked ablation. An invalid action advances time without an extra penalty.

| Split | Seeds | Purpose |
|---|---|---|
| Development | 0–9; seed 42 also exists in regression tests | Already inspected cases |
| Training | 1,000–99,999 | Sampled only by training environments |
| Validation | 100,000–100,049 | Checkpoint and pilot selection |
| Test | 200,000–200,299 | Final familiar-distribution results |
| OOD | 300,000–300,099 | Changed-scenario results, namespaced by family |

### Test changed wave patterns

```powershell
.\.venv\Scripts\python.exe -m pvz_rl evaluate `
  --checkpoint runs\masked-101\best.zip --split ood `
  --family redistributed --record --output runs\masked-101-redistributed
```

Repeat with `--family faster` and `--family concentrated`, including each baseline.
OOD evaluations default to standard and hard, 100 seeds each:

- `redistributed`: shuffle the zombie roster across waves after the first three,
  preserving total types and each wave's size.
- `faster`: reduce wave spacing from 25 to 20 seconds with the same jitter draws.
- `concentrated`: use only three seeded lanes, preserving spawn times and types.

Every generated case is retained. The hard preset shares opening compositions with
standard, so success across the normal presets alone is not structural generalization.

### Inspect replays

With `--record`, evaluation saves and hash-verifies the first ten wins, losses, and
truncations per difficulty in seed order. Paths and whole-file checksums are included
in episode records. The default output is a compact `.pvzdemo` file.

```powershell
.\.venv\Scripts\python.exe -m pvz_rl replay runs\baseline-development\replays\easy-0-won.pvzdemo
.\.venv\Scripts\python.exe -m pvz_rl replay runs\baseline-development\replays\easy-0-won.pvzdemo --watch --speed 2
.\.venv\Scripts\python.exe -m pvz_rl replay runs\baseline-development\replays\easy-0-won.pvzdemo --video artifacts\easy-0.mp4
```

Replay files deliberately contain complete engine snapshots for verification. They
are not policy observations. A truncated replay ends with the engine still running;
embedded metadata supplies the wrapper's cutoff outcome. Verification prints both
the engine `status` and the presentation `outcome`. Natural wins/losses always take
precedence. Caller-supplied metadata is outside simulation hashes, so the research
manifest also hashes the complete recording file.

The native viewer supports timeline dragging, Space to pause, period for one tick,
Left/Right for five-second seeks, Home/End, R to restart, and I to inspect entities.
`--speed` requires `--watch` and accepts 0.5, 1, 2, 4, or 8. Viewing and optional
video export also accept `.json.gz` and legacy `.json` files. Legacy sidecars fill
missing metadata; embedded fields take precedence. A raw older replay ending in
`running` is not labeled as a loss or an inferred cutoff.

## 6. Run the full research protocol when ready

This is an explicit long-running command. It is not launched by installation,
availability checks, or the short smoke example:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl suite `
  --hardware artifacts\hardware-benchmark\recommendation.json `
  --output runs\formal
```

The suite runs five conditions × five learner seeds (`101–105`), with the same
budget per condition. Nominal total: 75 million decisions; rollout-rounded total:
75,059,200. It then evaluates the selected checkpoints and all four baselines on
the final and OOD splits, captures predetermined replays, and generates a report.
All training finishes before final evaluation begins.

The suite records `suite.json` as a restart journal. Resume with exactly the same
configuration/hardware arguments:

```powershell
.\.venv\Scripts\python.exe -m pvz_rl suite `
  --hardware artifacts\hardware-benchmark\recommendation.json `
  --output runs\formal --resume
```

Completed jobs are retained. Failed or interrupted attempts remain inspectable;
new attempts use new directories. Training resumes from an available checkpoint,
while an interrupted evaluation restarts its affected evaluation group.

## 7. Generate a report yourself

```powershell
.\.venv\Scripts\python.exe -m pvz_rl report `
  --episodes runs\masked-101-test\episodes.jsonl runs\heuristic-test\episodes.jsonl `
  --curves runs\masked-101\learning-curve.jsonl `
  --output artifacts\comparison
```

For research conclusions, include results from all five independent learner seeds.
Do not combine multiple checkpoints for the same learner/case. The report rejects
duplicate or missing pairs and mismatched scenario sets in paired comparisons.
It rejects incompatible game/reward protocols. Cross-observation profile comparisons
require explicit matching game-protocol hashes and distinct profile labels; older
records without those hashes still require matching observation protocols. Different
training configurations cannot be pooled under the same policy name.

The implementation pilot remains inconclusive: at 65,536 matched decisions, the
full method scored 0% and 6.7% macro validation win rate for seeds 101/102, versus
0%/0% for baseline; neither learning diagnostic passed. See
[validation evidence](docs/validation.md) and the
[documented plant/dig failure and follow-ups](docs/paper-adaptation.md).

Outputs include `report.md`, `statistics.json`, `win-rates.png`, and optionally
`learning-curves.png`. Bootstrap intervals resample learner runs and scenario seeds
and preserve pairing against the heuristic. Always report per-difficulty results,
sample counts, and variation across runs; all-win/all-loss samples can give degenerate
bootstrap intervals. The target win rates (95% easy, 90% standard, 75% hard) are goals,
not implemented guarantees or current results.

Episode records also include invalid/waiting rates, plant usage, mowers consumed,
inference time, wins/loss durations, truncations, and final hashes. House breaches
and time cutoffs are classified automatically; strategic failure explanations require
replay inspection. A positive lower confidence bound on paired win-rate improvement
is required to claim that a method beats the heuristic.

## Configuration and interfaces

The canonical configuration is [research.toml](src/pvz_rl/data/research.toml).
To customize a study, copy it and pass the copy explicitly:

```powershell
Copy-Item src\pvz_rl\data\research.toml local-research.toml
# Edit local-research.toml, then:
.\.venv\Scripts\python.exe -m pvz_rl train --config local-research.toml --output runs\custom
```

The Gymnasium environment exposes `reset`, `step`, `action_masks`, and optional
`rgb_array` rendering. Public observations are encoded into **2,719 float32 features**:
plant tiles, aggregated zombie/projectile spatial bins, and global resource/timer data.
Counts are accumulated without truncating crowds. Scenario names, seeds, future
spawns, and entity IDs are excluded. Spatial aggregation loses some information;
this is a partially observed representation, not a claim of full state observability.

The 406-action mapping is fixed: wait `0`; placement `1 + 45*p + 9*r + c`; digging
`361 + 9*r + c`. Plant order follows the game API. Masking excludes illegal actions,
not strategically undesirable placements. Games stop naturally on win/loss; a
1,200-second external cutoff is a truncation with value bootstrapping, counted as
unsuccessful during evaluation.

Rewards combine terminal outcomes, plant/mower kills, nonlethal plant damage,
and empty mower activations as described above. Shaped conditions add
`gamma * potential(next) - potential(current)` using defeated fraction and total
plant-plus-sun value. True terminals have zero potential; external truncations
retain it. Tests independently check the discounted telescoping identity,
planting/digging counterexamples, damage normalization, and killing-blow attribution.

## Development and research notes

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests tools
.\.venv\Scripts\python.exe -m build
```

The full test suite includes a tiny end-to-end protocol, short real PPO updates,
serialization/replay checks, a synthetic interruption/resume, and a two-worker
Windows/CUDA check when CUDA is available. These tests do not launch formal runs.

- [Architecture, data flow, and state machines](docs/architecture.md)
- [Version history and remaining limitations](docs/iteration.md)
- [Papers and projects actually used](docs/references.md)
- [Availability and verification results](docs/validation.md)
- [Notes and optional suggestions for the separate game](docs/engine-notes.md)

Use feature branches, Conventional Commits, and PRs; never push directly to main.
The initial study is about winning with normal actions in this daytime clone.
Human imitation, pixel input, DQN, and recurrent policies are follow-up work.
