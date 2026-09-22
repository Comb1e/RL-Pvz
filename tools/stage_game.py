"""Stage a verified Git archive outside the game checkout for non-editable installation."""

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path


def stage_game(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    project = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        raise ValueError("Staging output must be outside the game checkout")
    expected = json.loads((project / "src/pvz_rl/data/engine-lock.json").read_text("utf-8"))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    try:
        resolved = git("rev-parse", "--verify", expected["commit"] + "^{commit}").decode().strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Game repository does not contain pinned commit {expected['commit']}"
        ) from exc
    if resolved != expected["commit"]:
        raise ValueError("Pinned game commit did not resolve exactly")
    archive = git("archive", "--format=zip", expected["commit"])
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        manifest = {
            name.removeprefix("src/pvz_game/"): hashlib.sha256(
                source.read(name).replace(b"\r\n", b"\n")
            ).hexdigest()
            for name in source.namelist()
            if name.startswith("src/pvz_game/")
            and Path(name).suffix in (".py", ".toml", ".cu", ".cuh")
        }
        if manifest != expected["files"]:
            raise ValueError("Archived game source differs from the research manifest")
        output.mkdir(parents=True, exist_ok=False)
        for name in source.namelist():
            target = (output / name).resolve()
            if not target.is_relative_to(output):
                raise ValueError("Archive contains a path outside the staging directory")
        source.extractall(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(stage_game(args.repo, args.output))
