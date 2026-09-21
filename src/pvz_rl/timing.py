"""Phase timings shared by operational logs and throughput benchmarks."""

from time import perf_counter

from stable_baselines3.common.callbacks import BaseCallback


class TrainingTimings:
    def __init__(self, clock=perf_counter):
        self.clock = clock
        self.collection_seconds = 0.0
        self.optimization_seconds = 0.0
        self.last_collection_seconds = None
        self.last_optimization_seconds = None
        self.collecting_since = None
        self.updating_since = None

    def begin_collection(self):
        self.collecting_since = self.clock()

    def end_collection(self):
        now = self.clock()
        self.last_collection_seconds = now - self.collecting_since
        self.collection_seconds += self.last_collection_seconds
        self.collecting_since = None
        self.updating_since = now

    def end_update(self):
        if self.updating_since is not None:
            self.last_optimization_seconds = self.clock() - self.updating_since
            self.optimization_seconds += self.last_optimization_seconds
            self.updating_since = None

    def snapshot(self):
        return {
            "collection_seconds": self.collection_seconds,
            "optimization_seconds": self.optimization_seconds,
            "last_collection_seconds": self.last_collection_seconds,
            "last_optimization_seconds": self.last_optimization_seconds,
        }


class TimingCallback(BaseCallback):
    def __init__(self, *, deadline=None, load_monitor=None):
        super().__init__()
        self.timings = TrainingTimings()
        self.games, self.ticks = 0, 0
        self.deadline, self.load_monitor = deadline, load_monitor

    def _on_step(self):
        for info in self.locals.get("infos", []):
            self.games += "episode_metrics" in info
            self.ticks += info.get("ticks_advanced", 0)
        return True

    def _on_rollout_start(self):
        self.timings.end_update()
        if self.deadline is not None and perf_counter() >= self.deadline:
            from .training import TrainingDeadline

            raise TrainingDeadline()
        if self.load_monitor:
            self.load_monitor.phase = (
                self.load_monitor.phase.split("/phase-")[0] + "/phase-collecting"
            )
        self.timings.begin_collection()

    def _on_rollout_end(self):
        self.timings.end_collection()
        if self.load_monitor:
            self.load_monitor.phase = (
                self.load_monitor.phase.split("/phase-")[0] + "/phase-updating"
            )

    def _on_training_end(self):
        self.timings.end_update()
