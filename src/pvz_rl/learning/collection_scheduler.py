"""Canonical decision tickets and local capacity backpressure."""

import numpy as np
import torch


def unissued_collection(state, phase, steps, cohort_start):
    """Setup failed before tickets: owned boards need no reconstructed decisions."""
    buffer = state.get("buffer")
    return (
        phase == "collect"
        and isinstance(buffer, dict)
        and buffer.get("size") == 0
        and type(steps) is int
        and steps >= 0
        and steps == cohort_start
        and state.get("pending_effect") is None
        and state.get("pending_episodes") == {}
    )


class CollectionScheduler:
    def __init__(self, count, device):
        self.pending = np.zeros(count, dtype=bool)
        self.resets = np.ones(count, dtype=bool)
        self.decisions = np.zeros(count, dtype=np.int64)
        self.order = torch.zeros(count, dtype=torch.long, device=device)
        self.columns = None
        self.species_coins = torch.zeros(count, device=device)
        self.species_choices = torch.ones(count, dtype=torch.long, device=device)
        self.wait_species_choices = torch.ones(count, dtype=torch.long, device=device)
        self.round = 0
        self.draining = False
        self.forecast_classes = np.zeros((count, 5, 3), np.int8)
        self.forecast_valid = np.zeros(count, dtype=bool)
        self.held_classes = np.zeros((count, 5, 3), np.int8)
        self.held_valid = np.zeros(count, dtype=bool)

    @property
    def retained_bytes(self):
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.columns,
                self.order,
                self.species_coins,
                self.species_choices,
                self.wait_species_choices,
            )
            if value is not None
        ) + sum(
            value.nbytes
            for value in (
                self.pending,
                self.resets,
                self.decisions,
                self.forecast_classes,
                self.forecast_valid,
                self.held_classes,
                self.held_valid,
            )
        )

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

    def retain_species_draws(self, new, coins, choices, wait_choices):
        selected = torch.as_tensor(new, device=self.order.device)
        self.species_coins.copy_(torch.where(selected, coins, self.species_coins))
        self.species_choices.copy_(torch.where(selected, choices, self.species_choices))
        self.wait_species_choices.copy_(
            torch.where(selected, wait_choices, self.wait_species_choices)
        )

    def hold_forecast(self, classes, valid):
        """Keep a pending decision's prior and replace every other slot."""
        if classes.shape != self.forecast_classes.shape or valid.shape != self.forecast_valid.shape:
            raise ValueError("Forecast prior must match the env batch")
        self.held_classes = np.where(
            self.pending[:, None, None], self.held_classes, classes
        ).astype(np.int8, copy=False)
        self.held_valid = np.where(self.pending, self.held_valid, valid).astype(bool, copy=False)

    def note_forecast(self, index, classes):
        index = np.asarray(index, dtype=np.int64)
        classes = np.asarray(classes, dtype=np.int8)
        if not len(index):
            return
        if classes.shape != (len(index), 5, 3):
            raise ValueError("Played forecast classes must be [slots, 5, 3]")
        self.forecast_classes[index] = classes
        self.forecast_valid[index] = True

    def clear_forecast(self, index):
        index = np.asarray(index, dtype=np.int64)
        if not len(index):
            return
        self.forecast_valid[index] = False
        self.forecast_classes[index] = 0

    def commit(self, issued, executing):
        executing = np.asarray(executing, bool)
        self.pending = np.asarray(issued, bool) & ~executing
        self.resets[executing] = False
        self.decisions += executing

    def snapshot(self):
        return dict(
            decisions=self.decisions.copy(),
            resets=self.resets.copy(),
            round=self.round,
            pending=self.pending.copy(),
            columns=None if self.columns is None else self.columns.cpu(),
            draws={
                name: getattr(self, name).cpu()
                for name in ("order", "species_coins", "species_choices", "wait_species_choices")
            },
            forecast_classes=self.forecast_classes.copy(),
            forecast_valid=self.forecast_valid.copy(),
            held_classes=self.held_classes.copy(),
            held_valid=self.held_valid.copy(),
        )

    def restore(self, state):
        for name, dtype in (("decisions", np.int64), ("resets", bool)):
            value = np.asarray(state.get(name))
            if value.shape != getattr(self, name).shape or value.dtype != dtype:
                raise ValueError("Invalid collection scheduler recovery counters")
            setattr(self, name, value.copy())
        if np.any(self.decisions < 0) or type(state.get("round")) is not int or state["round"] < 0:
            raise ValueError("Invalid collection scheduler recovery timing")
        self.round = state["round"]
        pending = np.asarray(state.get("pending"))
        if pending.shape != self.pending.shape or pending.dtype != bool:
            raise ValueError("Invalid pending decision tickets")
        self.pending = pending.copy()
        columns = state.get("columns")
        if columns is not None and (
            columns.shape != (len(pending), 25) or not torch.isfinite(columns).all()
        ):
            raise ValueError("Invalid retained decision evidence")
        if pending.any() and columns is None:
            raise ValueError("Pending decisions have no retained evidence")
        self.columns = None if columns is None else columns.to(self.order.device)
        names = ("order", "species_coins", "species_choices", "wait_species_choices")
        if set(state.get("draws", {})) != set(names):
            raise ValueError("Incomplete retained exploration draws")
        for name in names:
            value, target = state["draws"][name], getattr(self, name)
            if (
                value.shape != target.shape
                or value.dtype != target.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid retained exploration draw")
            target.copy_(value.to(self.order.device))
        for name, shape, dtype in (
            ("forecast_classes", (len(self.pending), 5, 3), np.int8),
            ("forecast_valid", (len(self.pending),), bool),
            ("held_classes", (len(self.pending), 5, 3), np.int8),
            ("held_valid", (len(self.pending),), bool),
        ):
            value = np.asarray(state.get(name))
            if value.shape != shape or value.dtype != dtype:
                raise ValueError("Invalid forecast carry")
            getattr(self, name)[...] = value
