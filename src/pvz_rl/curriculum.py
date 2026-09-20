"""Mastery gates change future resets, never a running game or optimizer."""

from dataclasses import asdict, dataclass

STAGES = ("placement", "saving", "easy", "standard", "shared")
LESSONS = ("placement", "saving")


def teaching_enabled(cfg):
    return cfg["curriculum"].get("mode", "fixed") == "teaching"


@dataclass
class CurriculumState:
    stage: int = 0
    entered_steps: int = 0
    last_probe: int = 0
    consecutive_passes: int = 0

    @property
    def name(self):
        return STAGES[self.stage]

    def to_dict(self):
        return asdict(self)

    def due(self, steps, cfg):
        return (
            self.stage < len(STAGES) - 1
            and steps - self.last_probe >= cfg["curriculum"]["probe_interval"]
        )

    def requirements(self, cfg):
        return cfg["curriculum"]["stages"][self.name]["requirements"]

    def observe(self, wins, steps, cfg):
        if steps < self.last_probe:
            raise ValueError("Curriculum progress cannot move backwards")
        c = cfg["curriculum"]
        self.last_probe = steps
        requirements = self.requirements(cfg)
        passed = bool(requirements) and all(
            wins.get(task, 0) >= n for task, n in requirements.items()
        )
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        if (
            self.stage < len(STAGES) - 1
            and self.consecutive_passes >= c["consecutive_passes"]
            and steps - self.entered_steps >= c["minimum_stage_steps"]
        ):
            self.stage += 1
            self.entered_steps = steps
            self.consecutive_passes = 0
            return True
        return False


def stage_distribution(cfg, stage):
    spec = cfg["curriculum"]["stages"][STAGES[stage]]
    return spec["tasks"], spec["weights"]
