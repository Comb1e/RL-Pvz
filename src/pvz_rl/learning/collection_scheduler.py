"""Canonical decision tickets and local capacity backpressure."""

import numpy as np
import torch


class CollectionScheduler:
    def __init__(self, count, device):
        self.pending = np.zeros(count, dtype=bool)
        self.resets = np.ones(count, dtype=bool)
        self.decisions = np.zeros(count, dtype=np.int64)
        self.order = torch.zeros(count, dtype=torch.long, device=device)
        self.columns = None
        self.species_coins = torch.zeros(count, device=device)
        self.species_choices = torch.ones(count, dtype=torch.long, device=device)
        self.round = 0
        self.draining = False

    @property
    def retained_bytes(self):
        return sum(
            value.numel() * value.element_size()
            for value in (self.columns, self.order, self.species_coins, self.species_choices)
            if value is not None
        ) + sum(value.nbytes for value in (self.pending, self.resets, self.decisions))

    def ready(self, unfinished):
        return (
            np.zeros_like(self.pending)
            if self.draining
            else np.asarray(unfinished, bool) & ~self.pending
        )

    def retain(self, columns, issued):
        pending = torch.as_tensor(self.pending, device=columns.device)
        if self.columns is None:
            self.columns = torch.zeros_like(columns)
        self.columns.copy_(torch.where(pending[:, None], self.columns, columns))
        new_order = self.round * len(issued) + torch.arange(len(issued), device=columns.device)
        self.order.copy_(torch.where(pending, self.order, new_order))
        self.round += 1
        return self.columns, torch.as_tensor(issued, device=columns.device) | pending

    def retain_species_draws(self, new, coins, choices):
        selected = torch.as_tensor(new, device=self.order.device)
        self.species_coins.copy_(torch.where(selected, coins, self.species_coins))
        self.species_choices.copy_(torch.where(selected, choices, self.species_choices))

    def commit(self, issued, executing):
        executing = np.asarray(executing, bool)
        self.pending = np.asarray(issued, bool) & ~executing
        self.resets[executing] = False
        self.decisions += executing

    def snapshot(self):
        if self.pending.any():
            raise RuntimeError("Finish issued decisions before saving scheduler counters")
        return dict(decisions=self.decisions.copy(), resets=self.resets.copy(), round=self.round)

    def restore(self, state):
        for name, dtype in (("decisions", np.int64), ("resets", bool)):
            value = np.asarray(state.get(name))
            if value.shape != getattr(self, name).shape or value.dtype != dtype:
                raise ValueError("Invalid collection scheduler recovery counters")
            setattr(self, name, value.copy())
        if np.any(self.decisions < 0) or type(state.get("round")) is not int or state["round"] < 0:
            raise ValueError("Invalid collection scheduler recovery timing")
        self.round = state["round"]
