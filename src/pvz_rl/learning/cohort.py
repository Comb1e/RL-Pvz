"""Explicit complete-game fitting and atomic interruption boundaries."""

import signal
import threading
from contextlib import contextmanager
from enum import StrEnum

OPTIMIZER_PROTOCOL = "sequential_q_mc_v1"


class CohortPhase(StrEnum):
    IDLE = "idle"
    COLLECT = "collect"
    RETURNS = "returns"
    FIT = "fit"
    SYNCHRONIZE = "synchronize"


@contextmanager
def atomic_transition():
    """Finish one simulation/ledger or optimizer step before honoring Ctrl+C."""
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
