$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe -m pvz_rl initialize-demo --archive data\human-1000.jsonl --replay data\human-1000.pvzdemo --output runs\human-init --device cuda

.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb --output runs\compact-stages\cherry_bomb --gpu 0 --gpu-profile rtx4070-laptop

.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb_obstacle --output runs\compact-stages\cherry_bomb_obstacle --init-from runs\compact-stages\cherry_bomb\final.zip --gpu 0 --gpu-profile rtx4070-laptop

.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage attacker_pre --output runs\compact-stages\attacker_pre --init-from runs\compact-stages\cherry_bomb_obstacle\final.zip --gpu 0 --gpu-profile rtx4070-laptop

.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage prediction --output runs\compact-stages\prediction --init-from runs\compact-stages\attacker_pre\final.zip --gpu 0 --gpu-profile rtx4070-laptop

.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage attacker --output runs\compact-stages\attacker --init-from runs\compact-stages\prediction\final.zip --gpu 0 --gpu-profile rtx4070-laptop

.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage peashooter --output runs\compact-stages\peashooter --init-from runs\compact-stages\attacker\final.zip --gpu 0 --gpu-profile rtx4070-laptop
# cherry_bomb
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb --output runs\compact-stages\cherry_bomb --init-from runs\human-init\initialization.pt --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb --output runs\compact-stages\cherry_bomb --resume runs\compact-stages\cherry_bomb\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# cherry_bomb_obstacle
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb_obstacle --output runs\compact-stages\cherry_bomb_obstacle --init-from runs\compact-stages\cherry_bomb\final.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage cherry_bomb_obstacle --output runs\compact-stages\cherry_bomb_obstacle --resume runs\compact-stages\cherry_bomb_obstacle\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# attacker_pre
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage attacker_pre --output runs\compact-stages\attacker_pre --init-from runs\compact-stages\cherry_bomb_obstacle\final.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage attacker_pre --output runs\compact-stages\attacker_pre --resume runs\compact-stages\attacker_pre\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# prediction
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage prediction --output runs\compact-stages\prediction --init-from runs\compact-stages\attacker_pre\final.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage prediction --output runs\compact-stages\prediction --resume runs\compact-stages\prediction\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# attacker
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage attacker --output runs\compact-stages\attacker --resume runs\compact-stages\attacker\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# peashooter
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage peashooter --output runs\compact-stages\peashooter --init-from runs\compact-stages\attacker\final.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage peashooter --output runs\compact-stages\peashooter --resume runs\compact-stages\peashooter\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop

# easy
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage easy --output runs\compact-stages\easy --init-from runs\compact-stages\peashooter\final.zip --gpu 0 --gpu-profile rtx4070-laptop
#.\.venv\Scripts\python.exe -m pvz_rl train --until-stage-complete --stage easy --output runs\compact-stages\easy --resume runs\compact-stages\easy\interrupted.zip --gpu 0 --gpu-profile rtx4070-laptop
