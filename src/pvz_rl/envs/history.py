"""Gross public transition facts for deterministic event-only memory."""

from dataclasses import dataclass

import numpy as np
import torch

HISTORY_PROTOCOL = "public_event_history_v1"
EVENT_FIELDS = (
    "sun_gained",
    "sun_spent",
    "zombies_spawned",
    "zombies_defeated",
    "zombies_removed",
    "plants_added",
    "plants_removed",
)
EVENT_WIDTH = len(EVENT_FIELDS) + 1


@dataclass(frozen=True)
class HistoryEvent:
    sun_gained: int = 0
    sun_spent: int = 0
    zombies_spawned: int = 0
    zombies_defeated: int = 0
    zombies_removed: int = 0
    plants_added: int = 0
    plants_removed: int = 0
    elapsed_ticks: int = 0

    def array(self):
        return np.asarray([getattr(self, name) for name in EVENT_FIELDS], dtype=np.int64)

    def inputs(self):
        return np.append(self.array(), self.elapsed_ticks)

    @property
    def writes(self):
        return bool(self.array().any())


def public_history_event(events, rules):
    values = dict.fromkeys(EVENT_FIELDS, 0)
    for event in events:
        kind = event.kind
        if kind == "SunProduced":
            values["sun_gained"] += event.get("amount")
        elif kind == "PlantPlaced":
            values["plants_added"] += 1
            values["sun_spent"] += rules.plants[event.get("plant_type")]["cost"]
        elif kind == "PlantRemoved":
            values["plants_removed"] += 1
        elif kind == "ZombieSpawned":
            values["zombies_spawned"] += 1
        elif kind == "ZombieDefeated":
            values["zombies_defeated"] += 1
        elif kind == "ZombieRemoved":
            values["zombies_removed"] += 1
    return HistoryEvent(**values)


def normalize_events(events, layout):
    scales = events.new_tensor([layout.cost_scale, layout.cost_scale, *([layout.count_scale] * 5)])
    return torch.cat(
        (events[..., :7] / scales, torch.log1p(events[..., 7:8] / layout.rules.game["tick_rate"])),
        -1,
    )
