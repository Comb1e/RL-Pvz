"""One append-only JSONL series per run log, rolled past a byte limit.

Parts live in a folder named for the log: ``hardware-metrics/00001.jsonl``.
The highest index is the open tail. A root file already larger than the limit
is moved into that folder whole, and a new file takes later records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_BATCH_BYTES = 5_000_000
_PART_LIMIT = 99_999
_PART_FILE = re.compile(r"^(?P<index>\d{5})\.jsonl$")


def _stem(path: Path) -> str:
    name = Path(path).name
    if not name.endswith(".jsonl"):
        raise ValueError(f"JSONL series path must end in .jsonl: {path}")
    return name[: -len(".jsonl")]


def series_dir(path) -> Path:
    """Folder that holds every part of this series."""
    path = Path(path)
    return path.parent / _stem(path)


def numbered_parts(path) -> list[Path]:
    """Parts of this series, in index order."""
    path = Path(path)
    folder = series_dir(path)
    found = []
    if folder.is_dir() and not folder.is_symlink():
        for candidate in folder.iterdir():
            match = _PART_FILE.fullmatch(candidate.name)
            if match and candidate.is_file():
                found.append((int(match.group("index")), candidate))
    found.sort()
    return [item for _, item in found]


def jsonl_series(path) -> list[Path]:
    """Unmigrated root file first, then folder parts in index order."""
    path = Path(path)
    parts = numbered_parts(path)
    if path.is_file():
        return [path, *parts]
    return parts


def _next_part(path: Path) -> Path:
    highest = 0
    for part in numbered_parts(path):
        highest = max(highest, int(part.stem))
    index = highest + 1
    folder = series_dir(path)
    while index <= _PART_LIMIT:
        candidate = folder / f"{index:05d}.jsonl"
        if not candidate.exists():
            return candidate
        index += 1
    raise ValueError(f"JSONL series {path.name} has no free part index")


def _adopt_root(path: Path, limit: int) -> None:
    """Move a root file into the series folder. An oversized file gets a new tail."""
    path = Path(path)
    if not path.is_file():
        return
    folder = series_dir(path)
    folder.mkdir(parents=True, exist_ok=True)
    destination = _next_part(path)
    path.replace(destination)
    if destination.stat().st_size > limit:
        _next_part(path).touch()


class JsonlAppender:
    """Append to the open part, rolling only before a record that follows an oversized file."""

    def __init__(self, path, limit: int = DEFAULT_BATCH_BYTES):
        self.series = Path(path)
        self.limit = limit
        self.series.parent.mkdir(parents=True, exist_ok=True)
        _adopt_root(self.series, limit)
        self.path = self._active_path()
        self._stream = self.path.open("a", encoding="utf-8")

    def _active_path(self) -> Path:
        parts = numbered_parts(self.series)
        if parts and parts[-1].stat().st_size <= self.limit:
            return parts[-1]
        target = _next_part(self.series)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return target

    def write(self, text: str) -> int:
        self._roll_if_needed()
        return self._stream.write(text)

    def _roll_if_needed(self) -> None:
        self._stream.flush()
        if self.path.is_file() and self.path.stat().st_size > self.limit:
            self._stream.close()
            self.path = _next_part(self.series)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8")

    def flush(self) -> None:
        if not self._stream.closed:
            self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def append_jsonl_file(path, value: dict, limit: int = DEFAULT_BATCH_BYTES) -> None:
    """Open the series, append one object, and close it."""
    from pvz_rl.provenance import append_jsonl

    with JsonlAppender(path, limit) as stream:
        append_jsonl(stream, value)


def read_jsonl_series(path, *, tolerate_partial_tail: bool = False) -> list:
    """Parse every part. An unfinished line is ignored only on the open tail."""
    rows = []
    parts = jsonl_series(path)
    for index, part in enumerate(parts):
        tail = index == len(parts) - 1
        with part.open(encoding="utf-8") as stream:
            lines = stream.readlines()
        for line_number, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                unfinished = line_number == len(lines) - 1 and not line.endswith("\n")
                if not (tolerate_partial_tail and tail and unfinished):
                    raise
    return rows
