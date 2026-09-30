"""Stateful public-input inference shared by collection, evaluation and replay."""

import torch

from pvz_rl.policy.transformer_lstm import RecurrentState


class PolicyRunner:
    """Recurrent decision state for a batch of games.

    Updates are masked with ``torch.where`` so a batched collection step never
    waits for the device. Decision validity accumulates on the device and is
    checked by :meth:`check` at the caller's host boundary.
    """

    def __init__(self, policy, cfg, rules, count, device):
        self.policy, self.cfg, self.device = policy, cfg, torch.device(device)
        self.state = policy.initial_state(count, device=device)
        self.previous_actions = torch.zeros(count, dtype=torch.long, device=device)
        self.outcomes = torch.zeros(count, 2, device=device)
        self.resets = torch.ones(count, dtype=torch.bool, device=device)
        self.valid = torch.ones((), dtype=torch.bool, device=device)

    def reset(self, indices):
        self.resets[indices] = True
        self.previous_actions[indices] = 0
        self.outcomes[indices] = 0
        self.state = self.policy.reset_state(self.state, self.resets)

    @torch.no_grad()
    def decide(self, obs, masks, ticks, *, active=None, deterministic=True, defer_check=False):
        if active is None:
            active = torch.ones(len(obs), dtype=torch.bool, device=self.device)
        self.policy.set_training_mode(False)
        actions, state, details = self.policy.decide(
            obs,
            self.state,
            action_masks=masks,
            previous_action=self.previous_actions,
            execution_outcome=self.outcomes,
            deterministic=deterministic,
            active=active,
        )
        keep = active.view(1, -1, 1)
        self.state = RecurrentState(
            torch.where(keep, state.hidden, self.state.hidden),
            torch.where(keep, state.cell, self.state.cell),
        )
        self.resets &= ~active
        self.valid &= details["valid"]
        if not defer_check:
            self.check()
        return actions, details["branch_value"], details["tile_value"], details

    def check(self):
        """Raise at a host boundary when any queued decision was invalid."""
        if not bool(self.valid):
            raise ValueError("Q values must be finite and every decision needs a legal action")

    def observe_result(self, proposals, accepted, ticks, *, active=None):
        if active is None:
            active = torch.ones(len(self.previous_actions), dtype=torch.bool, device=self.device)
        proposals = torch.as_tensor(proposals, device=self.device).long().reshape(-1)
        accepted = torch.as_tensor(accepted, device=self.device).bool().reshape(-1)
        ticks = torch.as_tensor(ticks, device=self.device, dtype=torch.float32).reshape(-1)
        # The proposal is the learning target and remains in the trajectory,
        # but recurrent history must describe what the simulator actually
        # executed.  A rejected proposal advances time as an automatic wait,
        # so every policy family receives the wait marker (action zero).
        self.previous_actions = torch.where(
            active, torch.where(accepted, proposals, 0), self.previous_actions
        )
        self.outcomes = torch.where(
            active[:, None], torch.stack((accepted.float(), ticks), -1), self.outcomes
        )

    def snapshot(self):
        return dict(
            hidden=self.state.hidden.cpu(),
            cell=self.state.cell.cpu(),
            previous_actions=self.previous_actions.cpu(),
            outcomes=self.outcomes.cpu(),
            resets=self.resets.cpu(),
        )

    def restore(self, saved):
        self.state = RecurrentState(saved["hidden"].to(self.device), saved["cell"].to(self.device))
        for key in ("previous_actions", "outcomes", "resets"):
            setattr(self, key, saved[key].to(self.device))
