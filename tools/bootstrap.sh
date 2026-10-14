#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash tools/bootstrap.sh --game-repo PATH [--python COMMAND]

Install the pinned pvz-cuda-work game and this project's CUDA environment.
Requires Git and Python 3.12. The game checkout is never changed.
--python defaults to python3.12 (or the PYTHON environment variable).
USAGE
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

game_repo=''
python_command="${PYTHON:-python3.12}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --game-repo|--python)
            [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || fail "$1 requires a value"
            if [[ "$1" == --game-repo ]]; then
                game_repo="$2"
            else
                python_command="$2"
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) fail "Unknown argument: $1; use --help" ;;
    esac
done
[[ -n "$game_repo" ]] || fail '--game-repo is required; use --help'
command -v git >/dev/null || fail 'Git is required'
command -v "$python_command" >/dev/null || fail "Python command not found: $python_command"
[[ -d "$game_repo" ]] || fail "Game repository not found: $game_repo"

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
game_root="$(cd -- "$game_repo" && pwd -P)"
"$python_command" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 is required by the dependency lock")'

temp_root="$(cd -- "${TMPDIR:-/tmp}" && pwd -P)"
stage_root="$(mktemp -d "$temp_root/pvz-rl-game.XXXXXXXX")"
cleanup() {
    if [[ "$stage_root" == "$temp_root"/pvz-rl-game.* && -d "$stage_root" ]]; then
        rm -rf -- "$stage_root"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$python_command" -B "$project_root/tools/stage_game.py" \
    --repo "$game_root" --output "$stage_root/game"

venv_root="$project_root/.venv"
if [[ ! -e "$venv_root" ]]; then
    "$python_command" -m venv "$venv_root"
fi
if [[ -x "$venv_root/bin/python" ]]; then
    venv_python="$venv_root/bin/python"
elif [[ -x "$venv_root/Scripts/python.exe" ]]; then
    venv_python="$venv_root/Scripts/python.exe"
else
    fail "Existing virtual environment has no usable Python: $venv_root"
fi
"$venv_python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "The virtual environment must use Python 3.12")'

"$venv_python" -m pip install --no-cache-dir 'torch==2.8.0+cu128' \
    --index-url https://download.pytorch.org/whl/cu128
"$venv_python" -m pip install --no-cache-dir -r "$project_root/requirements-lock.txt"
"$venv_python" -m pip install --no-deps --force-reinstall "$stage_root/game"
"$venv_python" -m pip install --no-deps -e "${project_root}[dev,ui]"
"$venv_python" -m pip install -r "$project_root/requirements-cuda-lock.txt"
"$venv_python" -m pip check
"$venv_python" -m pvz_rl doctor
