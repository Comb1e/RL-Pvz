"""Stateless policy control shared by collection and evaluation."""

import torch


class PolicyRunner:
    def __init__(self, policy):
        self.policy = policy
        self.valid = torch.ones((), dtype=torch.bool, device=policy.device)

    @torch.no_grad()
    def decide(
        self,
        obs,
        masks,
        *,
        active=None,
        deterministic=True,
        defer_check=False,
        plan=None,
        branch_usable=None,
    ):
        self.policy.train(False)
        actions, details = self.policy.decide(
            obs,
            action_masks=masks,
            deterministic=deterministic,
            active=active,
            active_plan=plan,
            branch_usable=branch_usable,
        )
        self.valid &= details["valid"]
        if not defer_check:
            self.check()
        return actions, details["branch_value"], details["tile_value"], details

    def check(self):
        if not bool(self.valid):
            raise ValueError("Q values must be finite and every decision needs a legal action")
