"""Explicit complete-game fitting and atomic interruption boundaries."""

import signal
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum


class CohortPhase(StrEnum):
    IDLE = "idle"
    COLLECT = "collect"
    FINALIZE_REWARDS = "finalize_rewards"
    RETURNS = "returns"
    FIT = "fit"
    SYNCHRONIZE = "synchronize"


_interruption = ContextVar("cohort_interruption", default=None)


def check_transition_interruption():
    interrupted = _interruption.get()
    if interrupted:
        raise KeyboardInterrupt


@contextmanager
def atomic_transition():
    """Finish one simulation/ledger or optimizer step before honoring Ctrl+C."""
    interrupted = []
    token = _interruption.set(interrupted)
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: interrupted.append(True))
    try:
        yield
    finally:
        _interruption.reset(token)
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if interrupted:
        raise KeyboardInterrupt
