"""Shared configurable counterfactual shapes for collection, storage and fitting."""

from dataclasses import asdict, dataclass

from pvz_rl.envs.actions import ActionSchema


@dataclass(frozen=True)
class ProbeLayout:
    tile_probes: int

    def __post_init__(self):
        if type(self.tile_probes) is not int or not 1 <= self.tile_probes < ActionSchema.tiles:
            raise ValueError("objective.tile_probes must be an integer in [1, 44]")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            from pvz_rl.config import load_config

            cfg = load_config()
        objective = cfg["training"]["objective"]
        if "branch_probes" in objective:
            raise ValueError("Branch probes have been removed; use tile probes only")
        return cls(objective["tile_probes"])

    @property
    def count(self):
        return self.tile_probes

    @property
    def metadata_width(self):
        return 15 + self.count * 4

    def metadata(self):
        return asdict(self)
