"""A cumulative wall-clock allowance; game counts still schedule learning."""

from time import perf_counter

from pvz_rl.learning.budget import effective_limits


class BudgetExpired(Exception):
    """An operation can be regenerated without collecting additional experience."""


def check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise BudgetExpired("Run time allowance exhausted")


class RunBudget:
    def __init__(self, cfg, *, elapsed=0.0, started=None, clock=perf_counter):
        self.clock = clock
        self.started = clock() if started is None else started
        self.prior_elapsed = elapsed
        minutes = effective_limits(cfg)["minutes"]
        self.limit = None if minutes is None else minutes * 60
        # Short availability runs reserve the same fraction rather than spending
        # their entire allowance on a fixed fifteen-minute reserve.
        self.reserve = (
            0
            if self.limit is None
            else min(cfg["training"].get("finalization_minutes", 15) * 60, self.limit / 8)
        )

    @property
    def deadline(self):
        return None if self.limit is None else self.started + self.limit - self.prior_elapsed

    @property
    def learning_deadline(self):
        return None if self.deadline is None else self.deadline - self.reserve

    def can_collect(self, estimated_cycle=0):
        return self.learning_deadline is None or (
            self.clock() + 1.2 * estimated_cycle < self.learning_deadline
        )

    def state(self):
        elapsed = self.prior_elapsed + self.clock() - self.started
        return {
            "elapsed_seconds": elapsed,
            "limit_seconds": self.limit,
            "reserve_seconds": self.reserve,
            "remaining_seconds": None if self.limit is None else max(0, self.limit - elapsed),
        }
