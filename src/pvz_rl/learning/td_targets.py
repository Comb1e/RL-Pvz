"""Frozen-cohort n-step Double-Q targets with simulation-tick discounting."""

import numpy as np
import torch

from pvz_rl.envs.actions import ActionSchema
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.learning.action_routing import FitRoute
from pvz_rl.learning.objective import selected_action_targets, training_rewards
from pvz_rl.policy.forecast_input import prior_from_rows


def nstep_returns(rewards, durations, bootstrap, terminal, *, n, gamma):
    """Per-start n-step sums, vectorized across starts in the original step order."""
    rewards, durations = np.asarray(rewards), np.asarray(durations)
    bootstrap, terminal = np.asarray(bootstrap), np.asarray(terminal, dtype=bool)
    size = len(rewards)
    result = np.zeros(size, np.float64)
    if not size:
        return result
    # Python float powers, matching the scalar gamma ** duration exactly.
    values = np.unique(durations)
    decay = np.array([gamma ** int(value) for value in values])[np.searchsorted(values, durations)]
    starts = np.arange(size)
    factor = np.ones(size)
    stop = starts.copy()
    alive = np.ones(size, dtype=bool)
    for offset in range(n):
        position = starts + offset
        step = alive & (position < size)
        if not step.any():
            break
        at = position[step]
        result[step] += factor[step].astype(rewards.dtype) * rewards[at]
        factor[step] *= decay[at]
        stop[step] = at
        alive[step] = ~terminal[at]
    bootstrapped = ~terminal[stop] & (stop + 1 < size)
    # The scalar loop multiplies a Python float by a bootstrap element in its own dtype.
    result[bootstrapped] += (
        factor[bootstrapped].astype(bootstrap.dtype) * bootstrap[stop[bootstrapped] + 1]
    )
    return result


def fork_potential(outcomes, tau, weights):
    outcomes = np.asarray(outcomes, np.float64)
    urgency = np.stack(
        (
            outcomes[..., 0] / 10,
            np.where(outcomes[..., 1] < 0, 0, 1 / (1 + np.maximum(0, outcomes[..., 1]) / tau)),
            outcomes[..., 2] / 10,
        ),
        -1,
    )
    return -(urgency * np.asarray(weights)).sum(axis=(-1, -2))


class TDTargetBuilder:
    """Each advance commits one bounded inference slab or one complete episode."""

    def __init__(self, model, state=None):
        self.model, self.buffer = model, model._buffer
        self.cursor, self.episode_cursor = 0, 0
        self.bootstrap = np.zeros(self.buffer.size, np.float32)
        self.keys = sorted(self.buffer._source_index)
        if state:
            self.cursor, self.episode_cursor = state["cursor"], state["episode_cursor"]
            self.bootstrap[:] = state["bootstrap"]

    def advance(self):
        model, buffer = self.model, self.buffer
        if self.cursor < buffer.size:
            last = min(buffer.size, self.cursor + model.cfg["training"]["batch_size"])
            rows = buffer.take(np.arange(self.cursor, last))
            obs = buffer.observations(rows, device=model.device)
            prior = prior_from_rows(rows, model.device)
            with torch.no_grad():
                online = model.policy.species_values(obs, prior=prior).float()
                selected = online.argmax(-1)
                target = model._teacher.species_values(obs, prior=prior).float()
                self.bootstrap[self.cursor : last] = (
                    target.gather(1, selected[:, None]).flatten().cpu().numpy()
                )
            self.cursor = last
            return False
        if self.episode_cursor < len(self.keys):
            blocks, length = buffer._source_index[self.keys[self.episode_cursor]]
            indices = np.concatenate(blocks)[:length]
            rows = buffer.take(indices)
            rewards = training_rewards(rows["components"], model.cfg)
            settings = model._resolved_forecast_schedule
            shaping = model.cfg["training"]["forecasts"]
            if shaping["shaping_weight"]:
                phi = shaping["shaping_weight"] * fork_potential(
                    rows["baseline_outcomes"],
                    reference_tau(model.env.batch.rules),
                    shaping["shaping_weights"],
                )
                next_phi = np.r_[phi[1:], 0]
                next_phi[rows["done"]] = 0
                rewards = (
                    rewards + np.power(settings["gamma_tick"], rows["duration"]) * next_phi - phi
                )
            targets = nstep_returns(
                rewards,
                rows["duration"],
                self.bootstrap[indices],
                rows["done"],
                n=settings["n_step"],
                gamma=settings["gamma_tick"],
            )
            buffer.put(indices, "target", targets)
            self.episode_cursor += 1
            return False
        return True

    def snapshot(self):
        return dict(
            cursor=self.cursor, episode_cursor=self.episode_cursor, bootstrap=self.bootstrap.copy()
        )

    def diagnostics(self):
        """Measure the finalized TD targets, leaving economic-return reports intact."""
        counts = np.zeros(3, np.int64)
        sums = np.zeros((3, 3), np.float64)
        offset = self.model.cfg["training"]["objective"]["invalid_action_q_offset"]
        total = square = 0.0
        for block, rows in enumerate(self.buffer.blocks):
            rows = rows[
                : min(self.buffer.block_rows, self.buffer.size - block * self.buffer.block_rows)
            ]
            rows = rows[rows["active"]]
            kinds = np.where(
                rows["action"] == 0, 0, np.where(rows["action"] < ActionSchema.dig_start, 1, 2)
            )
            for group in range(3):
                subset = rows[kinds == group]
                species_only = np.isin(
                    subset["route"], [FitRoute.WAIT_RETURN, FitRoute.SPECIES_INVALID]
                )
                selected = np.where(species_only, subset["branch_value"], subset["tile_value"])
                target = selected_action_targets(
                    subset["target"], subset["route"], subset["minimum_q"], offset
                )
                error = selected.astype(np.float64) - target
                counts[group] += len(error)
                sums[group] += [np.square(error).sum(), error.sum(), subset["placement_gap"].sum()]
            total += rows["target"].sum(dtype=np.float64)
            square += np.square(rows["target"].astype(np.float64)).sum()
        count = max(1, int(counts.sum()))
        self.buffer.reward_summary["td_targets"] = dict(
            count=int(counts.sum()),
            mean=total / count,
            std=float(np.sqrt(max(0, square / count - (total / count) ** 2))),
            **self.model._resolved_forecast_schedule,
        )
        return {
            name: dict(
                count=int(counts[i]),
                mse=float(sums[i, 0] / counts[i]) if counts[i] else None,
                signed_error=float(sums[i, 1] / counts[i]) if counts[i] else None,
                placement_gap=float(sums[i, 2] / counts[i]) if i and counts[i] else None,
            )
            for i, name in enumerate(("wait", "plant", "dig"))
        }
