"""Mastery gates change future resets, never a running game or optimizer."""

from dataclasses import asdict, dataclass

from pvz_rl.learning.budget import uses_games

STAGES = ("easy", "standard", "shared")
TILE_STAGES = (
    "prediction",
    "cherry_bomb",
    "cherry_bomb_obstacle",
    "attacker_pre",
    "attacker",
    "peashooter",
)
SELECTABLE_STAGES = (*TILE_STAGES, *STAGES)
# These five score every cohort, then take one fitting pass. A saved
# cohorts_per_epoch still selects its memorized cohort. Peashooter and the
# three normal stages keep a fixed pass count. Ten full cohorts on the stage
# bar leave the stage.
COHORT_PERFECT_STAGES = (
    "prediction",
    "cherry_bomb",
    "cherry_bomb_obstacle",
    "attacker_pre",
    "attacker",
)
FIXED_EPOCH_STAGES = ("peashooter", *STAGES)


def pre_stage(cfg):
    """Selected tile stage; ``pre`` is accepted only for saved legacy workflows."""
    return cfg["curriculum"].get("run_stage") in ("pre", *TILE_STAGES)


def teaching_enabled(cfg):
    return cfg["curriculum"].get("mode", "fixed") == "teaching"


def selected_stage(cfg):
    return cfg["curriculum"].get("run_stage")


def validation_after_stage(cfg, family="preset"):
    """Archived recipes retain periodic validation; lessons need no full-game gate."""
    return (
        teaching_enabled(cfg)
        and not pre_stage(cfg)
        and family == "preset"
        and cfg["training"].get("validation_schedule", "periodic") == "stage_success"
    )


def initial_state(cfg):
    stage = selected_stage(cfg)
    return CurriculumState(stage=STAGES.index(stage) if stage in STAGES else 0)


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
    cohort_streak: int = 0

    def __post_init__(self):
        if type(self.cohort_streak) is not int or self.cohort_streak < 0:
            raise ValueError("Invalid curriculum cohort streak")

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

    def note_full_cohort(self, passed):
        """Count one finished full cohort. A short cohort does not call this."""
        if self.mastered:
            return
        self.cohort_streak = self.cohort_streak + 1 if passed else 0

    def advance_after_streak(self, limit, *, advance=True):
        """Leave this stage once ``note_full_cohort`` has reached ``limit``."""
        if self.mastered or type(self.cohort_streak) is not int or self.cohort_streak < limit:
            return False
        self.cohort_streak = 0
        if not advance or self.stage == len(STAGES) - 1:
            self.mastered = True
            return False
        self.stage += 1
        self.consecutive_passes = 0
        self.completed_stage_games = 0
        return True


def stage_distribution(cfg, stage):
    spec = cfg["curriculum"]["stages"][STAGES[stage]]
    return spec["tasks"], spec["weights"]
