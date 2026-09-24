"""Window states and interruption/timing controls for the CUDA pipeline."""

import signal
import threading
from contextlib import contextmanager
from enum import Enum


class WindowState(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    OVERLAPPING = "overlapping"
    DRAINING = "draining"
    FAILED = "failed"


@contextmanager
def finish_window_on_interrupt():
    """Defer Ctrl+C until both optimizers and the episode ledger are committed."""
    interrupted = []
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: interrupted.append(True))
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if interrupted:
        raise KeyboardInterrupt


def interval_overlap(first, second):
    """Measured host interval intersection, not a claim of simultaneous GPU kernels."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
