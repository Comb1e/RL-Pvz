"""Record a clean local game commit and its complete Python/TOML/CUDA sources."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def pin_engine(repo):
    import sys

    repo = Path(repo).resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"]
    ).strip():
        raise ValueError("Commit the game changes before pinning them")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    sys.path.insert(0, str(repo / "src"))
    from pvz_game import ENGINE_VERSION, PACKAGE_VERSION, Rules

    root = repo / "src/pvz_game"
    manifest = {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.suffix in (".py", ".toml", ".cu", ".cuh")
    }
    data = dict(
        commit=commit,
        version=ENGINE_VERSION,
        package_version=PACKAGE_VERSION,
        rules_hash=Rules().digest,
        files=manifest,
    )
    project = Path(__file__).resolve().parents[1]
    destination = project / "src/pvz_rl/data/engine-lock.json"
    previous = json.loads(destination.read_text("utf-8"))
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in [project / "src/pvz_rl/data/research.toml", *(project / "configs").glob("*.toml")]:
        text = (
            path.read_text("utf-8")
            .replace(previous["commit"], commit)
            .replace(
                f'engine_version = "{previous["version"]}"', f'engine_version = "{ENGINE_VERSION}"'
            )
            .replace(
                f'engine_package_version = "{previous["package_version"]}"',
                f'engine_package_version = "{PACKAGE_VERSION}"',
            )
        )
        path.write_text(text, encoding="utf-8")
    print(
        json.dumps({"commit": commit, "package_version": PACKAGE_VERSION, "files": len(manifest)})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    pin_engine(parser.parse_args().repo)
