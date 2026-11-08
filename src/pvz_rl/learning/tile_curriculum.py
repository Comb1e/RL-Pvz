"""Prerequisite mastery state, independent of normal curriculum/game counters."""

from dataclasses import asdict, dataclass, field

from pvz_rl.envs.tile_scenarios import LEGACY_SKILLS, SKILLS, attacker_mix, prediction_settings
from pvz_rl.learning.budget import validation_interval_games
from pvz_rl.learning.curriculum import TILE_STAGES

TILE_SCENARIO_PROTOCOL = "tile-skill-prerequisites-v1"


def tile_validation_count(cfg, skill):
    if skill == "prediction":
        return prediction_settings(cfg)["validation_cases"]
    return (
        attacker_mix(cfg)["validation_cases"]
        if skill == "attacker"
        else cfg["curriculum"]["probe_cases"]
    )


def required_validation_wins(cfg, skill):
    """Strict per-skill thresholds; absent settings and other skills require every case."""
    count = tile_validation_count(cfg, skill)
    threshold = (
        cfg["curriculum"].get(f"{skill}_validation_success_threshold")
        if skill in ("cherry_bomb", "peashooter")
        else None
    )
    if threshold is None:
        return count
    return next(wins for wins in range(1, count + 1) if wins / count > threshold)


@dataclass
class TileSkillState:
    phase: str = "complete"
    episodes: dict = field(default_factory=lambda: dict.fromkeys(SKILLS, 0))
    last_probe: int = 0
    scores: dict = field(default_factory=dict)
    cohorts: int = 0
    generation_failures: int = 0
    stage_order: list | None = None
    stage_probes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.episodes = dict(self.episodes)
        # Old saved stage orders and joint probes never acquire a new lesson.
        self.episodes.setdefault("cherry_bomb_obstacle", 0)
        self.episodes.setdefault("attacker_pre", 0)
        self.episodes.setdefault("prediction", 0)
        self.episodes.setdefault("attacker", 0)
        self.scores = dict(self.scores)
        self.stage_probes = dict(self.stage_probes)
        if self.stage_order is not None:
            self.stage_order = list(self.stage_order)
            if (
                len(set(self.stage_order)) != len(self.stage_order)
                or any(skill not in TILE_STAGES for skill in self.stage_order)
                or self.active
                and self.phase not in self.stage_order
            ):
                raise ValueError("Invalid independent tile-stage order")
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
        invalid_score_keys = (
            set(self.scores) != set(LEGACY_SKILLS)
            if self.stage_order is None
            else bool(set(self.scores) - set(SKILLS))
        )
        if self.scores and (
            invalid_score_keys
            or any(type(value) is not int or value < 0 for value in self.scores.values())
        ):
            raise ValueError("Invalid tile prerequisite scores")
        if any(
            skill not in SKILLS or type(value) is not int or not 0 <= value <= self.episodes[skill]
            for skill, value in self.stage_probes.items()
        ):
            raise ValueError("Invalid tile-stage probe counters")

    @property
    def active(self):
        return self.phase != "complete"

    @property
    def total(self):
        return sum(self.episodes.values())

    @property
    def validation_skills(self):
        return LEGACY_SKILLS if self.stage_order is None else (self.phase,) if self.active else ()

    @property
    def validation_domain(self):
        return "pre" if self.stage_order is None else self.phase

    @property
    def progress(self):
        return self.total if self.stage_order is None else self.episodes.get(self.phase, 0)

    def due(self, cfg):
        previous = (
            self.last_probe if self.stage_order is None else self.stage_probes.get(self.phase, 0)
        )
        return self.active and self.progress - previous >= validation_interval_games(
            cfg, self.validation_domain
        )

    def observe(self, scores, cfg):
        count = tile_validation_count(cfg, self.validation_domain)
        if set(scores) != set(self.validation_skills) or any(
            type(value) is not int or not 0 <= value <= count for value in scores.values()
        ):
            raise ValueError("Incomplete or invalid tile prerequisite probe")
        if self.stage_order is not None:
            skill = self.phase
            self.stage_probes[skill] = self.episodes[skill]
            self.scores.update(scores)
            self.last_probe = self.total
            if scores[skill] >= required_validation_wins(cfg, skill):
                index = self.stage_order.index(skill) + 1
                self.phase = (
                    self.stage_order[index] if index < len(self.stage_order) else "complete"
                )
            return self.phase
        failed = [skill for skill in LEGACY_SKILLS if scores[skill] != count]
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
    stage = cfg["curriculum"].get("run_stage")
    if stage == "pre":
        raise ValueError(
            "The combined pre stage is retired for fresh runs; select an independent tile stage"
        )
    enabled = (
        cfg["training"].get("tile_scenarios", {}).get("enabled", False)
        and family == "preset"
        and cfg["curriculum"].get("mode") == "teaching"
        and stage in (None, *TILE_STAGES)
    )
    order = list(TILE_STAGES) if stage is None else [stage] if stage in TILE_STAGES else []
    if "attacker" not in cfg["training"].get("tile_scenarios", {}):
        order = [skill for skill in order if skill != "attacker_pre"]
    if attacker_mix(cfg) is None:
        order = [skill for skill in order if skill != "attacker"]
    if prediction_settings(cfg) is None:
        order = [skill for skill in order if skill != "prediction"]
    return TileSkillState(phase=order[0] if enabled else "complete", stage_order=order).to_dict()
