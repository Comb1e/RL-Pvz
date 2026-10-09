"""Compact, event-driven progress for terminals and matching durable log records."""

import re
import shutil
import sys
import warnings
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite
from numbers import Integral, Real
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


class CollectionProgress:
    """Host-counter interval rates, weighted by games active before each callback.

    Observe every committed collection callback and each phase boundary. Sampling
    does not advance the interval; call :meth:`advance` after publishing a status
    snapshot. Phase/cohort changes and restored counters start a fresh interval,
    excluding fitting, validation and recovery downtime. No device values are read.
    """

    def __init__(self, *, clock=perf_counter):
        self.clock = clock
        self._last = None
        self._totals = [0, 0.0, 0.0]
        self._baseline = tuple(self._totals)

    def observe(self, *, transitions, active_games, cohort, phase):
        for name, value in (("transitions", transitions), ("active_games", active_games)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative host integer")
        now = self.clock()
        if not isinstance(now, Real) or not isfinite(now):
            raise ValueError("Collection clock must return a finite host time")
        phase = str(phase)
        current = (now, int(transitions), int(active_games), cohort, phase)
        if self._last is not None:
            before, steps, games, previous_cohort, previous_phase = self._last
            reset = now < before or transitions < steps or cohort != previous_cohort
            if not reset and previous_phase == "collect":
                seconds = now - before
                self._totals[0] += int(transitions) - steps
                self._totals[1] += seconds
                self._totals[2] += seconds * games
            if reset or phase != previous_phase:
                self.advance()
        self._last = current

    def advance(self):
        """Start the next reporting interval without reading the clock or device."""
        self._baseline = tuple(self._totals)

    def snapshot(self):
        transitions, seconds, game_seconds = (
            total - baseline for total, baseline in zip(self._totals, self._baseline)
        )
        collecting = self._last is not None and self._last[-1] == "collect"
        return dict(
            active_games=self._last[2] if collecting else 0,
            interval_seconds=seconds,
            interval_active_transitions=transitions,
            interval_active_game_seconds=game_seconds,
            interval_mean_active_games=game_seconds / seconds if seconds > 0 else None,
            interval_decisions_per_game_per_second=(
                transitions / game_seconds if collecting and game_seconds > 0 else None
            ),
            interval_active_transitions_per_second=(
                transitions / seconds if collecting and seconds > 0 else None
            ),
            lifetime_active_transitions=self._totals[0],
            lifetime_collection_seconds=self._totals[1],
            lifetime_active_game_seconds=self._totals[2],
        )

    @staticmethod
    def text(row, games):
        """Format explicit units; an unmeasured interval is not a zero rate."""
        per_game = row["interval_decisions_per_game_per_second"]
        transitions = row["interval_active_transitions_per_second"]
        return (
            f"active {row['active_games']}/{games} | "
            + ("n/a" if per_game is None else f"{per_game:,.2f}")
            + " decisions/game/s | "
            + ("n/a" if transitions is None else f"{transitions:,.0f}")
            + " active transitions/s"
        )


class ProgressReporter:
    """One reporter for training, evaluation, benchmarks and exports.

    ``interval`` paces machine-readable snapshots through :meth:`due`; terminal
    output has its own cadence. Routine updates (``emit`` without ``force``)
    print at most once per ``terminal_interval`` after the previous line and
    only when their ``change`` key (or text) differs from the last printed
    summary. Significant events use ``force=True`` and print immediately; every
    print passes through consecutive-duplicate suppression, optionally scoped
    by ``key``.

    An interactive terminal rewrites one routine line in place with carriage
    returns and starts a new line for events. Redirected output receives one
    timestamped line per update without control characters. The log file
    receives exactly the printed records, never a multiline dashboard.
    """

    def __init__(
        self,
        path: Path,
        interval=15,
        *,
        terminal_interval=60,
        mode="compact",
        clock=perf_counter,
        stream=None,
        interactive=None,
    ):
        if mode != "compact":
            raise ValueError("ProgressReporter only supports compact terminal output")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.path.open("a", encoding="utf-8")
        self.interval, self.terminal_interval, self.clock = interval, terminal_interval, clock
        self.stream = sys.stdout if stream is None else stream
        isatty = getattr(self.stream, "isatty", None)
        self.interactive = bool(isatty and isatty()) if interactive is None else interactive
        self.state = Phase.STARTING
        self.warning = None
        self._showwarning = None
        self._status_last = self._terminal_last = -float("inf")
        self._last_line = None
        self._last_change = None
        self._keyed = {}
        self._open_width = 0

    @classmethod
    def from_settings(cls, path, cfg, **kwargs):
        """Construct from the run's resolved ``[logging]`` settings."""
        from pvz_rl.config import output_settings

        logging = output_settings(cfg)["logging"]
        return cls(
            path,
            logging["progress_seconds"],
            terminal_interval=logging["terminal_progress_seconds"],
            mode=logging["terminal_mode"],
            **kwargs,
        )

    def phase(self, state: Phase, message=None):
        """Record the phase; a message makes the transition a printed event."""
        self.state = Phase(state)
        if message is not None:
            self.emit(message, force=True)

    def warn(self, message):
        """Print a warning immediately and keep it in later routine summaries."""
        self.warning = " ".join(str(message).split())
        return self.emit(f"Warning: {self.warning}", force=True)

    def route_warnings(self):
        """Send Python warnings through this reporter until :meth:`close`."""
        if self._showwarning is None:
            self._showwarning = warnings.showwarning
            warnings.showwarning = self._show_warning

    def _show_warning(self, message, category, filename, lineno, file=None, line=None):
        self.warn(f"{category.__name__}: {message}")

    def emit(self, message: str, *, force=False, change=None, key=None):
        """Print one compact record; return whether it was written."""
        clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(message))
        text = " | ".join(" ".join(part.split()) for part in clean.splitlines() if part.strip())
        if not text:
            return False
        now = self.clock()
        if key is not None and self._keyed.get(key) == text:
            return False
        line_key = (self.state, text)
        if line_key == self._last_line:
            return False
        if not force:
            change = text if change is None else change
            if now - self._terminal_last < self.terminal_interval or change == self._last_change:
                return False
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{stamp} [{self.state}] {text}"
        self._print(line, routine=not force)
        self.log.write(line + "\n")
        self.log.flush()
        self._terminal_last = now
        self._last_line = line_key
        if change is not None:
            # Forced summaries also set the progress later routine lines must exceed.
            self._last_change = change
        if key is not None:
            self._keyed[key] = text
        return True

    def _print(self, line, *, routine):
        if self.interactive and routine:
            width = max(20, shutil.get_terminal_size((120, 20)).columns - 1)
            visible = line.split("] ", 1)[-1]
            if len(visible) > width:
                visible = visible.replace("decisions/game/s", "dec/g/s").replace(
                    "active transitions/s", "active t/s"
                )
            visible = visible[:width]
            self.stream.write("\r" + visible + " " * max(0, self._open_width - len(visible)))
            self.stream.flush()
            self._open_width = len(visible)
            return
        self._close_line()
        self.stream.write(line + "\n")
        self.stream.flush()

    def _close_line(self):
        if self._open_width:
            self.stream.write("\n")
            self.stream.flush()
            self._open_width = 0

    def due(self):
        """Machine-readable snapshot cadence, independent of terminal output."""
        now = self.clock()
        if now - self._status_last < self.interval:
            return False
        self._status_last = now
        return True

    def close(self):
        if self._showwarning is not None:
            if warnings.showwarning == self._show_warning:
                warnings.showwarning = self._showwarning
            self._showwarning = None
        self._close_line()
        if not self.log.closed:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
