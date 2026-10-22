"""Array-backed host transitions and lazy compatibility dictionaries."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pvz_game.cuda.schema import REASONS


class StepInfos(Sequence):
    def __init__(self, result, slots=None):
        self.result = result
        self.slots = np.arange(len(result.compact)) if slots is None else np.asarray(slots)

    def __len__(self):
        return len(self.slots)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return StepInfos(self.result, self.slots[index])
        return self.result.info(int(self.slots[index]))


@dataclass
class PackedStepResult:
    compact: np.ndarray
    actions: np.ndarray
    outcomes: np.ndarray
    truncation: np.ndarray
    active: np.ndarray
    completed: dict

    @property
    def ticks(self):
        return int(self.compact[self.active, 2].sum())

    @property
    def instant_actions(self):
        return int(np.count_nonzero(self.active & (self.compact[:, 2] == 0)))

    def infos(self, *, active_only=False):
        return StepInfos(self, np.flatnonzero(self.active) if active_only else None)

    def info(self, index):
        accepted = bool(self.outcomes[index, 0])
        reason = int(self.outcomes[index, 1])
        omitted = self.truncation[index]
        result = dict(
            entity_truncation=dict(
                total=int(omitted.sum()),
                **dict(zip(("plants", "zombies", "projectiles"), map(int, omitted))),
            ),
            proposal_action=int(self.actions[index, 0]),
            executed_action=int(self.actions[index, 1]),
            accepted=accepted,
            rejection_reason=None
            if accepted or not 0 <= reason < len(REASONS)
            else REASONS[reason],
            ticks_advanced=int(self.compact[index, 2]),
            causal_credits={},
            causal_diagnostics=dict(source_action_credited_rows=0, unattributable_events=0),
        )
        result.update(self.completed.get(index, {}))
        return result
