"""Stateful public-input inference shared by collection, evaluation and replay."""

import torch

from pvz_rl.policy.event_memory import EventMemory
from pvz_rl.policy.transformer_lstm import RecurrentState


class PolicyRunner:
    def __init__(self, policy, cfg, rules, count, device):
        self.policy, self.cfg, self.device = policy, cfg, torch.device(device)
        self.recurrent = cfg["policy"]["kind"] == "transformer_lstm_q_v1"
        self.capacity = 0 if self.recurrent else None
        self.state = policy.initial_state(count, device=device) if self.recurrent else None
        self.memory = None if self.recurrent else EventMemory(cfg, rules, count, device)
        self.previous_actions = torch.zeros(count, dtype=torch.long, device=device)
        self.outcomes = torch.zeros(count, 2, device=device)
        self.resets = torch.ones(count, dtype=torch.bool, device=device)

    def reset(self, indices):
        self.resets[indices] = True
        self.previous_actions[indices] = 0
        self.outcomes[indices] = 0
        if self.recurrent:
            self.state = self.policy.reset_state(self.state, self.resets)

    @torch.no_grad()
    def decide(self, obs, masks, ticks, *, active=None, deterministic=True):
        if active is None:
            active = torch.ones(len(obs), dtype=torch.bool, device=self.device)
        self.policy.set_training_mode(False)
        if self.recurrent:
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
            result = actions, details["branch_value"], details["tile_value"], details
        else:
            context = self.memory.observe(obs, masks, self.previous_actions, self.resets, ticks)
            result = self.policy.decide(
                obs, masks, deterministic=deterministic, context=context, active=active
            )
        self.resets[active] = False
        return result

    def observe_result(self, proposals, accepted, ticks, *, active=None):
        if active is None:
            active = torch.ones(len(self.previous_actions), dtype=torch.bool, device=self.device)
        proposals = torch.as_tensor(proposals, device=self.device).long().reshape(-1)
        accepted = torch.as_tensor(accepted, device=self.device).bool().reshape(-1)
        ticks = torch.as_tensor(ticks, device=self.device, dtype=torch.float32).reshape(-1)
        previous = proposals if self.recurrent else torch.where(accepted, proposals, 0)
        self.previous_actions[active] = previous[active]
        self.outcomes[active, 0] = accepted[active].float()
        self.outcomes[active, 1] = ticks[active].float()

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
