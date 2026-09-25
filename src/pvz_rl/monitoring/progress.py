"""One throttled, timestamped reporter for training, evaluation, and exports."""

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from time import perf_counter


class Phase(StrEnum):
    STARTING = "starting"
    COLLECTING = "collecting"
    UPDATING = "updating"
    VALIDATING = "validating"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class ProgressReporter:
    """Phase changes are explicit; routine messages share one wall-time throttle.

    Rollout collection/update transitions do not force console output. Significant
    events (validation, checkpoint saves, completion) bypass the throttle.
    """

    def __init__(self, path: Path, interval=15, *, clock=perf_counter):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8")
        self.interval, self.clock = interval, clock
        self.last = -float("inf")
        self.state = Phase.STARTING

    def phase(self, state: Phase, message=None):
        self.state = Phase(state)
        if message is not None:
            self.emit(message, force=True)

    def emit(self, message: str, *, force=False):
        now = self.clock()
        if not force and now - self.last < self.interval:
            return False
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{stamp} [{self.state}] {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")
        self.stream.flush()
        self.last = now
        return True

    def due(self):
        return self.clock() - self.last >= self.interval

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def duration(seconds):
    if seconds is None:
        return "pending"
    hours, rest = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
