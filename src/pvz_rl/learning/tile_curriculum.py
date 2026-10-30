"""Prerequisite mastery state, independent of normal curriculum/game counters."""

from dataclasses import asdict, dataclass, field

from pvz_rl.envs.tile_scenarios import SKILLS

TILE_SCENARIO_PROTOCOL = "tile-skill-prerequisites-v1"


@dataclass
class TileSkillState:
    phase: str = "complete"
    episodes: dict = field(default_factory=lambda: dict.fromkeys(SKILLS, 0))
    last_probe: int = 0
    scores: dict = field(default_factory=dict)
    cohorts: int = 0
    generation_failures: int = 0

    def __post_init__(self):
        self.episodes = dict(self.episodes)
        self.scores = dict(self.scores)
        if self.phase not in (*SKILLS, "complete"):
            raise ValueError("Unknown tile prerequisite phase")
        if set(self.episodes) != set(SKILLS) or any(
            type(value) is not int or value < 0 for value in self.episodes.values()
        ):
            raise ValueError("Invalid tile prerequisite episode counters")
        if any(
            type(value) is not int or value < 0
            for value in (self.last_probe, self.cohorts, self.generation_failures)
        ):
            raise ValueError("Invalid tile prerequisite progress")
        if self.last_probe > self.total:
            raise ValueError("Tile prerequisite probe exceeds completed episodes")
        if self.scores and (
            set(self.scores) != set(SKILLS)
            or any(type(value) is not int or value < 0 for value in self.scores.values())
        ):
            raise ValueError("Invalid tile prerequisite scores")

    @property
    def active(self):
        return self.phase != "complete"

    @property
    def total(self):
        return sum(self.episodes.values())

    def due(self, cfg):
        return (
            self.active
            and self.total - self.last_probe >= cfg["curriculum"]["probe_interval_games"]
        )

    def observe(self, scores, cfg):
        count = cfg["curriculum"]["probe_cases"]
        if set(scores) != set(SKILLS) or any(
            type(value) is not int or not 0 <= value <= count for value in scores.values()
        ):
            raise ValueError("Incomplete or invalid tile prerequisite probe")
        failed = [skill for skill in SKILLS if scores[skill] != count]
        self.scores = dict(scores)
        self.last_probe = self.total
        if not failed:
            self.phase = "complete"
        elif len(failed) == 1:
            self.phase = failed[0]
        return self.phase

    def to_dict(self):
        return asdict(self)


def initial_tile_state(cfg, family="preset"):
    enabled = (
        cfg["training"].get("tile_scenarios", {}).get("enabled", False)
        and family == "preset"
        and cfg["curriculum"].get("mode") == "teaching"
        and cfg["curriculum"].get("run_stage") in (None, "pre")
    )
    return TileSkillState(phase="peashooter" if enabled else "complete").to_dict()
