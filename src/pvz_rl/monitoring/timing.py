"""Phase timings shared by operational logs and throughput benchmarks."""

from time import perf_counter

from stable_baselines3.common.callbacks import BaseCallback


class TrainingTimings:
    def __init__(self):
        self.collection_seconds = 0.0
        self.optimization_seconds = 0.0
        self.last_collection_seconds = None
        self.last_optimization_seconds = None
        self.window_seconds = 0.0
        self.last_window_seconds = None
        self.overlap_seconds = 0.0
        self.queue_wait_seconds = 0.0

    def record_window(self, metrics):
        self.last_collection_seconds = sum(metrics["slot_collection_seconds"])
        self.last_optimization_seconds = sum(metrics["slot_optimization_seconds"])
        self.collection_seconds += self.last_collection_seconds
        self.optimization_seconds += self.last_optimization_seconds
        self.last_window_seconds = metrics["window_seconds"]
        self.window_seconds += self.last_window_seconds
        self.overlap_seconds += metrics["overlap_seconds"]
        self.queue_wait_seconds += metrics["queue_wait_seconds"]

    def snapshot(self):
        return {
            "collection_seconds": self.collection_seconds,
            "optimization_seconds": self.optimization_seconds,
            "last_collection_seconds": self.last_collection_seconds,
            "last_optimization_seconds": self.last_optimization_seconds,
            "window_seconds": self.window_seconds,
            "last_window_seconds": self.last_window_seconds,
            "overlap_seconds": self.overlap_seconds,
            "queue_wait_seconds": self.queue_wait_seconds,
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
        if self.deadline is not None and perf_counter() >= self.deadline:
            from pvz_rl.learning.training import TrainingDeadline

            raise TrainingDeadline()
        if self.load_monitor:
            from pvz_rl.learning.curriculum import STAGES

            self.load_monitor.update(
                activity="training",
                phase="warmup" if getattr(self.model, "critic_warmup_active", False) else "formal",
                stage=STAGES[getattr(getattr(self.model.env, "queue", None), "stage", 0)],
            )

    def _on_rollout_end(self):
        self.timings.record_window(self.model.pipeline_metrics)
        if self.load_monitor:
            self.load_monitor.record_window(self.model.pipeline_metrics)
            self.load_monitor.update(training_steps=self.model.num_timesteps)
