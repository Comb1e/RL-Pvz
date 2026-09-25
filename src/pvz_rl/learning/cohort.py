"""Explicit complete-game phases and atomic optimization/interruption boundaries."""

import math
import signal
import threading
from contextlib import contextmanager
from copy import deepcopy
from enum import StrEnum

import torch

OPTIMIZER_PROTOCOL = "complete_game_mc_conditional_ppo_v1"


class CohortPhase(StrEnum):
    IDLE = "idle"
    COLLECT = "collect"
    RETURNS = "returns"
    CRITIC = "critic"
    ACTOR = "actor"
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


class ActorTransaction:
    def __init__(self, policy, target_kl, state=None):
        self.policy, self.target_kl = policy, target_kl
        self.state = state or {
            "weights": [p.detach().clone() for p in policy.actor_parameters()],
            "optimizer": deepcopy(policy.optimizer.state_dict()),
            "stopped": False,
            "rejected": False,
            "reason": None,
        }

    def check(self, divergence):
        if not math.isfinite(divergence) or self.target_kl and divergence > self.target_kl:
            self.reject("nonfinite_kl" if not math.isfinite(divergence) else "kl_limit")

    def reject(self, reason):
        with torch.no_grad():
            for p, old in zip(self.policy.actor_parameters(), self.state["weights"], strict=True):
                p.copy_(old)
        self.policy.optimizer.load_state_dict(self.state["optimizer"])
        self.policy.optimizer.zero_grad(set_to_none=True)
        self.state.update(stopped=True, rejected=True, reason=reason)
