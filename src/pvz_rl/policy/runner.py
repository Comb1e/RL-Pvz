"""Event-memory lifecycle shared by collection, evaluation, EMA and recovery."""

import torch

from pvz_rl.envs.history import HISTORY_PROTOCOL
from pvz_rl.policy.transformer_lstm import EventMemoryState


class PolicyRunner:
    def __init__(self, policy, cfg, rules, count, device):
        self.policy, self.cfg, self.device = policy, cfg, torch.device(device)
        self.state = policy.initial_state(count, device=device)
        self.resets = torch.ones(count, dtype=torch.bool, device=device)
        self.valid = torch.ones((), dtype=torch.bool, device=device)

    def reset(self, indices):
        reset = torch.zeros_like(self.resets)
        reset[indices] = True
        self.resets |= reset
        self.state = self.policy.reset_state(self.state, reset)

    @torch.no_grad()
    def advance(self, obs, active):
        events = self.state.inputs()
        writes = events[:, :7].ne(0).any(-1) & active
        output = self.policy.forward_step(obs, self.state, events=events, memory_write=writes)
        self.state = output.state
        self.resets &= ~active
        return output, events, writes

    @torch.no_grad()
    def decide(self, obs, masks, ticks, *, active=None, deterministic=True, defer_check=False):
        if active is None:
            active = torch.ones(len(obs), dtype=torch.bool, device=self.device)
        self.policy.train(False)
        events = self.state.inputs()
        writes = events[:, :7].ne(0).any(-1) & active
        actions, state, details = self.policy.decide(
            obs,
            self.state,
            action_masks=masks,
            events=events,
            memory_write=writes,
            deterministic=deterministic,
            active=active,
        )
        self.state = state
        self.resets &= ~active
        self.valid &= details["valid"]
        details.update(events=torch.where(active[:, None], events, 0), memory_write=writes)
        if not defer_check:
            self.check()
        return actions, details["branch_value"], details["tile_value"], details

    def check(self):
        if not bool(self.valid):
            raise ValueError("Q values must be finite and every decision needs a legal action")

    def observe_result(self, facts, ticks, *, active=None):
        if active is None:
            active = torch.ones(self.state.pending.shape[0], dtype=torch.bool, device=self.device)
        facts = torch.as_tensor(facts, device=self.device).reshape(-1, 7)
        ticks = torch.as_tensor(ticks, device=self.device).reshape(-1)
        self.state = self.state.observe(facts, ticks, active)

    def snapshot(self):
        return dict(
            protocol=HISTORY_PROTOCOL,
            **{
                key: getattr(self.state, key).cpu()
                for key in ("hidden", "cell", "pending", "elapsed_ticks")
            },
            resets=self.resets.cpu(),
        )

    def restore(self, saved):
        if saved.get("protocol") != HISTORY_PROTOCOL:
            raise ValueError("Unsupported event-memory recovery protocol")
        values = []
        for key in ("hidden", "cell", "pending", "elapsed_ticks"):
            value = saved[key]
            expected = getattr(self.state, key)
            if (
                value.shape != expected.shape
                or value.dtype != expected.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid event-memory recovery state")
            if key in ("pending", "elapsed_ticks") and torch.any(value < 0):
                raise ValueError("Negative event-memory facts or timing")
            values.append(value.to(self.device))
        if saved["resets"].shape != self.resets.shape or saved["resets"].dtype != torch.bool:
            raise ValueError("Invalid event-memory reset mask")
        self.state = EventMemoryState(*values)
        self.resets = saved["resets"].to(self.device)
