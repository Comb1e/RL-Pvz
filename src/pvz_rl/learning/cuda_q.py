"""One CUDA trainer: complete games and sequential Q Monte Carlo regression."""

import io
import json
import random
import shutil
import tempfile
import warnings
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm

from pvz_rl.learning.cohort import (
    CohortPhase,
    atomic_transition,
)
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.exploration import EXPLORATION_PROTOCOL, configure_exploration
from pvz_rl.policy.sequential_q import ACTION_DISTRIBUTION, POLICY_SIGNATURE


def checkpoint_metadata(path):
    """Read plain JSON before any SB3/cloudpickle or torch model deserialization."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".zip")
    with ZipFile(path) as archive:
        if "protocol.json" not in archive.namelist():
            raise ValueError("Sequential Q controller requires fresh models; retired checkpoint")
        metadata = json.loads(archive.read("protocol.json"))
    from pvz_rl.learning.checkpoints import protocol_for

    if metadata != protocol_for(metadata.get("policy")):
        raise ValueError("Incompatible model/optimizer checkpoint protocol")
    return metadata


def checkpoint_optimizer_protocol(path):
    return checkpoint_metadata(path)["optimizer"]


class CudaCohortLifecycle(BaseAlgorithm):
    policy_protocol = POLICY_SIGNATURE
    optimizer_version = "complete_return_lstm_v1"

    def __init__(
        self,
        policy,
        env=None,
        *,
        learning_rate=3e-4,
        batch_size=1024,
        n_epochs=4,
        max_grad_norm=0.5,
        policy_kwargs=None,
        seed=None,
        device="cuda",
        verbose=0,
        tensorboard_log=None,
        _init_setup_model=True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            support_multi_env=True,
        )
        self.batch_size, self.n_epochs = batch_size, n_epochs
        self.max_grad_norm = max_grad_norm
        self.optimizer_protocol = self.optimizer_version
        self.action_distribution_protocol = ACTION_DISTRIBUTION
        self.phase = CohortPhase.IDLE
        self.training_games = self._n_updates = 0
        self.cohort_metrics = {}
        self.runtime_state = None
        self._buffer = self._memory = None
        self._fit_epoch = 0
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        ).to(self.device)
        configure_exploration(self, self.policy.features_extractor.cfg)

    @classmethod
    def load(cls, path, *args, **kwargs):
        if checkpoint_metadata(path)["policy"] != cls.policy_protocol:
            raise ValueError("Incompatible checkpoint model family for this loader")
        model = super().load(path, *args, **kwargs)
        model.phase = CohortPhase(model.phase)
        # SB3 maps every saved tensor to the requested device. Ordinary Adam
        # keeps its scalar step on CPU; retain that original optimizer protocol.
        for optimizer in (model.policy.optimizer,):
            for group in optimizer.param_groups:
                if not group.get("capturable", False) and not group.get("fused", False):
                    for parameter in group["params"]:
                        state = optimizer.state.get(parameter, {})
                        if "step" in state:
                            state["step"] = state["step"].cpu()
        model._checkpoint_source = Path(path).resolve()
        if not model._checkpoint_source.suffix:
            model._checkpoint_source = model._checkpoint_source.with_suffix(".zip")
        with ZipFile(model._checkpoint_source) as archive:
            if (
                "observation-schema.json" not in archive.namelist()
                or json.loads(archive.read("observation-schema.json"))
                != model.policy.layout.schema()
            ):
                raise ValueError("Checkpoint entity schema disagrees with model")
            if "cohort-state.pt" not in archive.namelist():
                raise ValueError("Checkpoint is missing its complete-game runtime state")
            model.runtime_state = torch.load(
                io.BytesIO(archive.read("cohort-state.pt")), map_location="cpu", weights_only=False
            )
        return model

    @property
    def cfg(self):
        return self.policy.features_extractor.cfg

    def predict(
        self, observation, state=None, episode_start=None, deterministic=False, action_masks=None
    ):
        return self.policy.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )

    def _new_buffer(self):
        root = Path(getattr(self, "trajectory_root", tempfile.gettempdir()))
        path = Path(tempfile.mkdtemp(prefix="cohort-", dir=root))
        spec = self.cfg["training"]["storage"]
        buffer = CompleteGameBuffer(
            path,
            self.n_envs,
            ram_bytes=spec["ram_gib"] * 1024**3,
            block_rows=spec["block_rows"],
            schema=self.policy.layout.schema(),
        )
        return buffer

    def _phase(self, phase, callback):
        self.phase = phase
        if hasattr(callback, "hardware_context"):
            callback.hardware_context(phase.value)
        if hasattr(callback, "progress"):
            from pvz_rl.monitoring.progress import Phase

            callback.progress.phase(
                Phase.COLLECTING if phase == CohortPhase.COLLECT else Phase.UPDATING,
                f"Complete-game cohort: {phase.value}",
            )

    def _begin(self, callback):
        callback.on_rollout_start()
        from pvz_rl.learning.budget import budget_target, uses_games

        count = self.n_envs
        target = budget_target(self.env.cfg)
        if uses_games(self.env.cfg) and target is not None:
            count = min(count, max(0, target - self.training_games))
        if count < 1:
            from pvz_rl.learning.training import TrainingGamesComplete

            raise TrainingGamesComplete
        self._last_obs = self.env.reset_cohort(count)
        self._memory = self._new_memory()
        self._first = True
        self._buffer = self._new_buffer()
        self._fit_epoch = 0
        self._phase_times = {"collect": 0.0, "returns": 0.0, "fit": 0.0}
        self._device_seconds = 0.0
        self._simulation_ticks = 0
        self._planting_samples = 0
        for key in ("q_loss", "branch_loss", "tile_loss", "q_grad_norm"):
            self.logger.record("train/" + key, None)
        self._stats = dict(
            q_optimizer_steps=0,
            species_exploration_coins=0,
            tile_exploration_coins=0,
            exploratory_changes=0,
        )
        self._phase(CohortPhase.COLLECT, callback)

    def _synchronize(self, callback):
        torch.cuda.synchronize(self.device)
        elapsed = sum(self._phase_times.values())
        transitions = int(self._buffer.group_counts.sum())
        self.cohort_metrics = {
            **self._stats,
            "prefit_errors": self.prefit_errors,
            **self._buffer.transport_metrics,
            "device_compute_seconds": self._device_seconds,
            "simulation_speed": self._simulation_ticks
            / self.env.batch.rules.game["tick_rate"]
            / max(self._phase_times["collect"], 1e-9),
            "planting_samples": int(self._buffer.group_counts[1]),
            "species_counts": self._buffer.species_counts.tolist(),
            "collection_seconds": self._phase_times["collect"],
            "fit_seconds": self._phase_times["fit"],
            "returns_seconds": self._phase_times["returns"],
            "window_seconds": elapsed,
            "optimization_seconds": self._phase_times["fit"],
            "transitions_per_second": transitions / max(elapsed, 1e-9),
            "transitions": transitions,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(self.device),
        }
        for key, value in self._stats.items():
            self.logger.record("train/" + key, value)
        self.logger.record("train/learning_rate", self.policy.optimizer.param_groups[0]["lr"])
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates)
        self._buffer.close()
        self._buffer = self._memory = None
        self.phase = CohortPhase.IDLE
        callback.on_rollout_end()

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1,
        tb_log_name="run",
        reset_num_timesteps=True,
        progress_bar=False,
    ):
        if self.optimizer_protocol != self.optimizer_version:
            raise ValueError("Incompatible optimizer protocol")
        if reset_num_timesteps:
            self.num_timesteps = 0
        # BaseAlgorithm initializes callbacks/logger without resetting private games.
        if self._last_obs is None:
            from pvz_rl.envs.encoding import EntityBatch

            self._last_obs = EntityBatch(
                torch.zeros(self.n_envs, 0, 11, dtype=torch.int32, device=self.device),
                torch.zeros(self.n_envs, 0, dtype=torch.bool, device=self.device),
                torch.zeros(self.n_envs, 18, device=self.device),
            )
            self._last_episode_starts = np.ones(self.n_envs, dtype=bool)
        total, callback = self._setup_learn(
            total_timesteps, callback, False, tb_log_name, progress_bar
        )
        if self.runtime_state:
            self._restore_runtime()
        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total or self.phase != CohortPhase.IDLE:
            if self.phase == CohortPhase.IDLE:
                from pvz_rl.learning.budget import budget_target, uses_games

                target = budget_target(self.env.cfg)
                if (
                    uses_games(self.env.cfg)
                    and target is not None
                    and self.training_games >= target
                ):
                    break
                with atomic_transition():
                    self._begin(callback)
                continue
            phase = self.phase
            start = perf_counter()
            try:
                with atomic_transition():
                    if phase == CohortPhase.COLLECT:
                        self._collect_step(callback)
                    elif phase == CohortPhase.RETURNS:
                        self.prefit_errors = self._buffer.finalize()
                        for name, values in self.prefit_errors.items():
                            self.logger.record("train/action_count_" + name, values["count"])
                            self.logger.record("train/value_target_error_" + name, values["mse"])
                            self.logger.record(
                                "train/tile_target_error_" + name, values["tile_mse"]
                            )
                        self._phase(CohortPhase.FIT, callback)
                    elif phase == CohortPhase.FIT:
                        self._fit_step(callback)
                    elif phase == CohortPhase.SYNCHRONIZE:
                        self._synchronize(callback)
            finally:
                if phase.value in self._phase_times:
                    self._phase_times[phase.value] += perf_counter() - start
        callback.on_training_end()
        return self

    def _restore_runtime(self):
        state = self.runtime_state
        if state.get("environment") is None:
            self._restore_rng(state)
            self.runtime_state = None
            return
        self._last_obs = self.env.restore_training(state["environment"])
        with ZipFile(self._checkpoint_source) as archive:
            self.env.action_journal.restore_archive_safe(archive)
        self.resumed_cohort = True
        if self.phase == CohortPhase.IDLE:
            self._restore_rng(state)
            self.runtime_state = None
            return
        self._memory = self._new_memory()
        self._memory.restore(state["memory"])
        self._buffer = self._new_buffer()
        workspace = self._buffer.path
        with ZipFile(self._checkpoint_source) as archive:
            self._buffer = CompleteGameBuffer.restore_archive(archive, state["buffer"], workspace)
        if self._buffer.schema != self.policy.layout.schema():
            raise ValueError("Trajectory entity schema disagrees with model")
        self._restore_rng(state)
        self.runtime_state = None
        self.resumed_cohort = True

    @staticmethod
    def _restore_rng(state):
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])

    def save(self, path, *args, **kwargs):
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".zip")
        runtime = self.runtime_state or {
            "environment": self.env.snapshot_training()
            if self.env is not None and self.env._tasks
            else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        if self.phase != CohortPhase.IDLE and self._buffer is not None:
            torch.cuda.synchronize(self.device)
            runtime = {
                "environment": self.env.snapshot_training(),
                "memory": self._memory.snapshot(),
                "buffer": self._buffer.metadata(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }
        # One atomic archive ties weights, optimizers, RNG and unfinished games
        # together. A failed save leaves the previous checkpoint intact.
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as file:
            temporary = Path(file.name)
        try:
            super().save(temporary, *args, **kwargs)
            with ZipFile(temporary, "a") as archive:
                archive.writestr(
                    "protocol.json",
                    json.dumps(
                        dict(
                            policy=self.policy_protocol,
                            optimizer=self.optimizer_version,
                            exploration=EXPLORATION_PROTOCOL,
                        )
                    ),
                )
                archive.writestr("observation-schema.json", json.dumps(self.policy.layout.schema()))
                archive.writestr(
                    "run.json",
                    json.dumps(
                        getattr(
                            self,
                            "research_metadata",
                            {
                                "config": self.cfg,
                                "condition": "masked",
                                "family": "preset",
                                "learner_seed": self.seed,
                                "validation_limit": None,
                                "exploration_protocol": EXPLORATION_PROTOCOL,
                            },
                        )
                    ),
                )
                state_bytes = io.BytesIO()
                torch.save(runtime, state_bytes)
                archive.writestr("cohort-state.pt", state_bytes.getvalue())
                if self.env is not None:
                    try:
                        self.env.action_journal.write_archive(archive)
                    except (OSError, MemoryError, ValueError) as exc:
                        # Journal metadata is the last entry; incomplete optional
                        # blocks have no manifest and cannot invalidate training state.
                        warnings.warn(
                            f"Checkpoint saved without complete action history: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                elif hasattr(self, "_checkpoint_source"):
                    with ZipFile(self._checkpoint_source) as source:
                        for name in source.namelist():
                            if name.startswith("journal/"):
                                with (
                                    source.open(name) as inp,
                                    archive.open(name, "w", force_zip64=True) as out,
                                ):
                                    shutil.copyfileobj(inp, out, length=1024 * 1024)
                if self._buffer is not None:
                    self._buffer.write_archive(archive)
                elif runtime.get("buffer") is not None:
                    with ZipFile(self._checkpoint_source) as source:
                        for name in source.namelist():
                            if name.startswith("cohort/"):
                                with (
                                    source.open(name) as inp,
                                    archive.open(name, "w", force_zip64=True) as out,
                                ):
                                    shutil.copyfileobj(inp, out, length=1024 * 1024)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _excluded_save_params(self):
        return [
            *super()._excluded_save_params(),
            "_buffer",
            "_memory",
            "_checkpoint_source",
            "runtime_state",
        ]

    def _get_torch_save_params(self):
        return ["policy", "policy.optimizer"], []
