"""Mastery gates change future resets, never a running game or optimizer."""

from dataclasses import asdict, dataclass

from .budget import uses_games

STAGES = ("placement", "saving", "easy", "standard", "shared")
LESSONS = ("placement", "saving")


def teaching_enabled(cfg):
    return cfg["curriculum"].get("mode", "fixed") == "teaching"


def selected_stage(cfg):
    return cfg["curriculum"].get("run_stage")


def initial_state(cfg):
    stage = selected_stage(cfg)
    return CurriculumState(stage=STAGES.index(stage) if stage else 0)


@dataclass
class CurriculumState:
    stage: int = 0
    entered_steps: int = 0
    last_probe: int = 0
    consecutive_passes: int = 0
    entered_games: int = 0
    last_probe_games: int = 0
    completed_stage_games: int = 0
    mastered: bool = False

    def completed_episode(self, episode_stage):
        if episode_stage == self.stage:
            self.completed_stage_games += 1

    @property
    def name(self):
        return STAGES[self.stage]

    def to_dict(self):
        return asdict(self)

    def due(self, steps, cfg):
        last = self.last_probe_games if uses_games(cfg) else self.last_probe
        key = "probe_interval_games" if uses_games(cfg) else "probe_interval"
        return (
            not self.mastered
            and bool(self.requirements(cfg))
            and steps - last >= cfg["curriculum"][key]
        )

    def requirements(self, cfg):
        return cfg["curriculum"]["stages"][self.name]["requirements"]

    def complete(self, cfg):
        # Archived recipes ended their curriculum upon entering the shared stage.
        return self.stage == len(STAGES) - 1 and (self.mastered or not self.requirements(cfg))

    def observe(self, wins, steps, cfg, *, advance=True):
        games = uses_games(cfg)
        last = self.last_probe_games if games else self.last_probe
        entered = self.entered_games if games else self.entered_steps
        if steps < last:
            raise ValueError("Curriculum progress cannot move backwards")
        c = cfg["curriculum"]
        if self.mastered:
            return False
        if any(type(n) is not int or not 0 <= n <= c["probe_cases"] for n in wins.values()):
            raise ValueError("Probe wins must be integer counts within probe_cases")
        if games:
            self.last_probe_games = steps
        else:
            self.last_probe = steps
        requirements = self.requirements(cfg)
        passed = bool(requirements) and all(
            wins.get(task, 0) >= n for task, n in requirements.items()
        )
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        if (
            self.consecutive_passes >= c["consecutive_passes"]
            and (
                self.completed_stage_games
                if games and c.get("residency") == "episode_start_stage"
                else steps - entered
            )
            >= c["minimum_stage_games" if games else "minimum_stage_steps"]
        ):
            if not advance or self.stage == len(STAGES) - 1:
                self.mastered = True
                return False
            self.stage += 1
            if games:
                self.entered_games = steps
            else:
                self.entered_steps = steps
            self.consecutive_passes = 0
            self.completed_stage_games = 0
            return True
        return False


def stage_distribution(cfg, stage):
    spec = cfg["curriculum"]["stages"][STAGES[stage]]
    return spec["tasks"], spec["weights"]
