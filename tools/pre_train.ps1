$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe -m pvz_rl fit-tile-annotations --annotations data\tile-annotations.jsonl --output runs\tile-init\tile.pt

.\.venv\Scripts\python.exe -m pvz_rl initialize-demo --init-from runs\tile-init\tile.pt --archive data\human-1000.jsonl --replay data\human-1000.pvzdemo --output runs\human-init --device cuda

#.\.venv\Scripts\python.exe -m pvz_rl initialize-demo --init-from runs\tile-init\tile.progress.pt --archive data\human-1000.jsonl --replay data\human-1000.pvzdemo --output runs\human-init --device cuda
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage easy --seed 101 --output runs\compact-stages-101\easy --init-from runs\human-init\initialization.pt --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage easy --seed 101 --output runs\compact-stages-101-new\easy --resume runs\compact-stages-101-new\easy\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop
