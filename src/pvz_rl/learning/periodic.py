"""Window states and interruption/timing controls for the CUDA pipeline."""

import math
import signal
import threading
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum

import torch

OPTIMIZER_PROTOCOL = "periodic_exact_kl_v1"


class ActorUpdateState(str, Enum):
    ACTIVE = "active"
    STOPPED = "stopped"
    REJECTED = "rejected"


class ActorWindow:
    """One transaction for actor weights and Adam, independent of the critic."""

    def __init__(self, policy, target_kl):
        self.policy, self.target_kl = policy, target_kl
        self.state = ActorUpdateState.ACTIVE
        self.rejection_reason = None
        self.parameters = tuple(policy.actor_parameters())
        self.weights = [p.detach().clone() for p in self.parameters] if target_kl else []
        self.optimizer = deepcopy(policy.optimizer.state_dict()) if target_kl else None

    def stop(self):
        if self.state == ActorUpdateState.ACTIVE:
            self.state = ActorUpdateState.STOPPED

    def check(self, divergences):
        if self.target_kl and any(not math.isfinite(x) or x > self.target_kl for x in divergences):
            self.reject(
                "nonfinite_kl" if any(not math.isfinite(x) for x in divergences) else "kl_limit"
            )

    def reject(self, reason="nonfinite_actor"):
        if self.state == ActorUpdateState.REJECTED:
            return
        if self.optimizer is None:
            raise RuntimeError("Non-finite actor update with KL guarding disabled")
        with torch.no_grad():
            for parameter, saved in zip(self.parameters, self.weights, strict=True):
                parameter.copy_(saved)
        self.policy.optimizer.load_state_dict(self.optimizer)
        self.policy.optimizer.zero_grad(set_to_none=True)
        self.state = ActorUpdateState.REJECTED
        self.rejection_reason = reason


def aggregate_update_metrics(slots):
    """Merge disjoint rollout statistics without losing sparse action categories."""
    totals = {
        "optimizer_steps",
        "actor_optimizer_steps",
        "critic_optimizer_steps",
        "epochs_completed",
        "critic_epochs_completed",
        "dig_legal_observations",
        "dig_probability_sum",
        "actor_metric_samples",
        "critic_metric_samples",
        "value_sample_count",
        "return_sum",
        "return_squared_sum",
        "residual_sum",
        "residual_squared_sum",
    }
    actor_means = {
        "policy_gradient_loss",
        "entropy_loss",
        "clip_fraction",
        "joint_entropy",
        "exploration_bonus",
        "choice_fraction",
        "approx_kl",
        "actor_loss",
        "kind_exploration_bonus",
        "plant_exploration_bonus",
        "tile_exploration_bonus",
    }
    result = {}
    for key in slots[-1]:
        name = key.removeprefix("train/")
        values = [s[key] for s in slots]
        if name in totals or name.startswith(
            (
                "action_count_",
                "value_target_error_sum_",
                "raw_advantage_sum_",
                "positive_advantage_count_",
            )
        ):
            value = sum(values)
        elif name == "kl_stopped":
            value = max(values)
        elif name == "n_updates":
            value = values[-1]
        else:
            weight = (
                "actor_metric_samples"
                if name in actor_means
                else {
                    "value_loss": "critic_metric_samples",
                    "actor_grad_norm": "actor_optimizer_steps",
                    "critic_grad_norm": "critic_optimizer_steps",
                }.get(name)
            )
            pairs = [
                (v, s.get("train/" + weight, 1) if weight else 1)
                for v, s in zip(values, slots)
                if math.isfinite(v)
            ]
            count = sum(n for _, n in pairs)
            value = sum(v * n for v, n in pairs) / count if count else float("nan")
        result[key] = value
    n = result.get("train/value_sample_count", 0)
    if n:
        variance = result["train/return_squared_sum"] / n - (result["train/return_sum"] / n) ** 2
        error_variance = (
            result["train/residual_squared_sum"] / n - (result["train/residual_sum"] / n) ** 2
        )
        result["train/explained_variance"] = (
            1 - max(0, error_variance) / variance if variance > 0 else float("nan")
        )
    for action in ("wait", "plant", "dig"):
        count = result[f"train/action_count_{action}"]
        for mean, total in (
            ("value_target_error", "value_target_error_sum"),
            ("raw_advantage_mean", "raw_advantage_sum"),
            ("positive_advantage_fraction", "positive_advantage_count"),
        ):
            result[f"train/{mean}_{action}"] = (
                result[f"train/{total}_{action}"] / count if count else float("nan")
            )
    count = result["train/dig_legal_observations"]
    result["train/dig_probability_when_legal"] = (
        result["train/dig_probability_sum"] / count if count else float("nan")
    )
    return result


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
