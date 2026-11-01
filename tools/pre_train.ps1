$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe -m pvz_rl fit-tile-annotations --annotations data\tile-annotations.jsonl --output runs\tile-init\tile.pt

.\.venv\Scripts\python.exe -m pvz_rl initialize-demo --init-from runs\tile-init\tile.pt --archive data\human-1000.jsonl --replay data\human-1000.pvzdemo --output runs\human-init --device cuda

# cherry_bomb
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb --seed 101 --output runs\compact-stages-101\cherry_bomb --init-from runs\human-init\initialization.pt --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb --seed 101 --output runs\compact-stages-101\cherry_bomb --resume runs\compact-stages-101\cherry_bomb\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# cherry_bomb_obstacle
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb_obstacle --seed 101 --output runs\compact-stages-101\cherry_bomb_obstacle --init-from runs\compact-stages-101\cherry_bomb\final.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb_obstacle --seed 101 --output runs\compact-stages-101\cherry_bomb_obstacle --resume runs\compact-stages-101\cherry_bomb_obstacle\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# attacker_pre
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage attacker_pre --seed 101 --output runs\compact-stages-101\attacker_pre --init-from runs\compact-stages-101\cherry_bomb_obstacle\final.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage attacker_pre --seed 101 --output runs\compact-stages-101\attacker_pre --resume runs\compact-stages-101\attacker_pre\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# peashooter
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage peashooter --seed 101 --output runs\compact-stages-101\peashooter --init-from runs\compact-stages-101\attacker_pre\final.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage peashooter --seed 101 --output runs\compact-stages-101\peashooter --resume runs\compact-stages-101\peashooter\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage easy --seed 101 --output runs\compact-stages-101\easy --init-from runs\human-init\initialization.pt --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage easy --seed 101 --output runs\compact-stages-101\easy --resume runs\compact-stages-101\easy\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe tools\plot_annotation_loss.py runs\tile-init\tile.history.jsonl --output runs\tile-loss.html --serve
