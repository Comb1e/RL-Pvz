"""Atomic metadata and verification of the pinned installed game source."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

import pvz_game
from pvz_game import ENGINE_VERSION, Rules

from .config import digest


def write_json(path: str | Path, value: object):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(stream, value: dict):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def source_manifest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.suffix in (".py", ".toml")
    }


def verify_engine(cfg: dict) -> dict:
    expected = json.loads(files("pvz_rl").joinpath("data/engine-lock.json").read_text("utf-8"))
    actual = source_manifest(Path(pvz_game.__file__).parent)
    if ENGINE_VERSION != cfg["engine_version"] or expected["commit"] != cfg["engine_commit"]:
        raise RuntimeError("Engine version/commit does not match the research configuration")
    if actual != expected["files"]:
        changed = sorted(
            k
            for k in set(actual) | set(expected["files"])
            if actual.get(k) != expected["files"].get(k)
        )
        raise RuntimeError(f"Installed game differs from pinned engine: {changed}")
    return {
        "commit": expected["commit"],
        "version": ENGINE_VERSION,
        "source_hash": digest(actual),
        "rules_hash": Rules().digest,
    }


def git_metadata() -> dict:
    root = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)

    def run(*args):
        if root is None:
            return None
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def metadata(cfg: dict, **extra) -> dict:
    versions = {}
    for name in (
        "pvz-plant-research",
        "pvz-research-game",
        "numpy",
        "torch",
        "gymnasium",
        "stable-baselines3",
        "sb3-contrib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "research": git_metadata(),
        "engine": verify_engine(cfg),
        "config_hash": digest(cfg),
        "config": cfg,
        "research_source_hash": digest(source_manifest(Path(__file__).parent)),
        **extra,
    }


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
