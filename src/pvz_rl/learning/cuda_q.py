"""One CUDA trainer: complete games and sequential Q Monte Carlo regression."""

import io
import json
import random
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.learning.cohort import (
    OPTIMIZER_PROTOCOL,
    CohortPhase,
    atomic_transition,
)
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.exploration import EXPLORATION_PROTOCOL, configure_exploration
from pvz_rl.policy.event_memory import EventMemory
from pvz_rl.policy.sequential_q import ACTION_DISTRIBUTION, POLICY_SIGNATURE, balanced_q_loss


def checkpoint_metadata(path):
    """Read plain JSON before any SB3/cloudpickle or torch model deserialization."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".zip")
    with ZipFile(path) as archive:
        if "protocol.json" not in archive.namelist():
            raise ValueError("Sequential Q controller requires fresh models; retired checkpoint")
        metadata = json.loads(archive.read("protocol.json"))
    if metadata != dict(
        policy=POLICY_SIGNATURE, optimizer=OPTIMIZER_PROTOCOL, exploration=EXPLORATION_PROTOCOL
    ):
        raise ValueError("Incompatible sequential Q checkpoint protocol; start fresh")
    return metadata


def checkpoint_optimizer_protocol(path):
    return checkpoint_metadata(path)["optimizer"]


class CudaSequentialQ(BaseAlgorithm):
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
        self.optimizer_protocol = OPTIMIZER_PROTOCOL
        self.action_distribution_protocol = ACTION_DISTRIBUTION
        self.phase = CohortPhase.IDLE
        self.training_games = self._n_updates = 0
        self.cohort_metrics = {}
        self.runtime_state = None
        self._buffer = self._memory = None
        self._fit_order = None
        self._fit_epoch = self._fit_cursor = 0
        self._cohort_elapsed = 0.0
        self._prefetch = None
        self._collection_host = None
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
        checkpoint_metadata(path)
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
            self.observation_space.shape[0],
            self._memory.capacity,
            self.n_envs,
            ram_bytes=spec["ram_gib"] * 1024**3,
            block_rows=spec["block_rows"],
        )
        buffer.configure_cache(
            self.device, self.env.cfg["training"].get("performance", {}).get("token_cache_gib", 2)
        )
        return buffer

    def _drain_transfers(self):
        if self._prefetch is not None:
            self._prefetch.close()
            self._prefetch = None

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
        self._last_obs = self.env.reset()
        self._memory = EventMemory(self.cfg, self.env.batch.rules, self.n_envs, self.device)
        self._previous = torch.zeros(self.n_envs, device=self.device, dtype=torch.long)
        self._first = True
        self._buffer = self._new_buffer()
        self._fit_epoch = self._fit_cursor = 0
        self._fit_order = None
        self._cohort_elapsed = 0.0
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
        self._loss_sums = dict(branch=0.0, tile=0.0, branch_count=0, tile_count=0)
        self._phase(CohortPhase.COLLECT, callback)

    def _collect_step(self, callback):
        env = self.env
        self.policy.set_training_mode(False)
        with torch.no_grad(), env.device_context():
            active = env.enabled_envs.copy()
            obs = self._last_obs
            masks = env.action_masks().clone()
            inactive = ~torch.as_tensor(active, device=self.device)
            masks[inactive] = False
            masks[inactive, 0] = True
            ticks = env.header_tensor[:, 0]
            resets = torch.full((self.n_envs,), self._first, device=self.device, dtype=torch.bool)
            ids = torch.arange(
                self._buffer.size, self._buffer.size + self.n_envs, device=self.device
            )
            context = self._memory.observe(obs, masks, self._previous, resets, ticks, token_ids=ids)
            viewer = env.live_view
            watched = None
            if viewer is not None and viewer.enabled:
                try:
                    watched = viewer.decision_indices()
                except Exception as exc:
                    viewer.fail(exc)
            result = self.policy.decide(obs, masks, context=context, diagnostic_indices=watched)
            actions, values, tile_values = result[:3]
            if watched is not None:
                try:
                    viewer.capture_decision(result[3]["viewer"], ticks)
                except Exception as exc:
                    viewer.fail(exc)
            # One bounded packed transfer; copied before the simulator mutates observations.
            packed = torch.cat(
                (
                    obs,
                    masks.float(),
                    self._previous[:, None].float(),
                    ticks[:, None].float(),
                    actions[:, None].float(),
                    values[:, None],
                    tile_values[:, None],
                    result[3]["greedy_actions"][:, None].float(),
                    result[3]["coins"].float(),
                    # Bit-preserve 32-bit references, including indices above 2**24.
                    self._memory.ids.to(torch.int32).view(torch.float32),
                    self._memory.counts,
                    self._memory.starts,
                ),
                -1,
            )
            if self._collection_host is None or self._collection_host.shape != packed.shape:
                self._collection_host = torch.empty_like(packed, device="cpu", pin_memory=True)
            self._collection_host.copy_(packed, non_blocking=True)
            raw_tokens = torch.cat(
                (obs, self._previous[:, None], resets[:, None], ticks[:, None]), -1
            )
            next_obs, rewards, dones, _, _, infos = env.step_tensors(actions, autoreset=False)
            # step_tensors' required completion readback also completes pre-action copy.
            host = self._collection_host.numpy()
            reward = env.last_transition_host[:, 3]
            duration = env.last_transition_host[:, 2]
            rows = np.zeros(self.n_envs, dtype=self._buffer.dtype)
            d = self.observation_space.shape[0]
            rows["observation"] = host[:, :d]
            rows["mask"] = np.packbits(
                host[:, d : d + A.size].astype(bool), axis=-1, bitorder="little"
            )
            base = d + A.size
            for j, key in enumerate(
                (
                    "previous",
                    "tick",
                    "action",
                    "branch_value",
                    "tile_value",
                    "greedy_action",
                    "species_coin",
                    "tile_coin",
                )
            ):
                rows[key] = host[:, base + j]
            base += 8
            rows["ids"] = host[:, base : base + self._memory.capacity].view(np.uint32)
            base += self._memory.capacity
            for key in ("counts", "starts"):
                rows[key] = host[:, base : base + self._memory.capacity]
                base += self._memory.capacity
            rows["reset"] = self._first
            rows["active"], rows["env"] = active, np.arange(self.n_envs)
            rows["reward"], rows["duration"] = reward, duration
            self._buffer.append(rows, raw_tokens)
            self._stats["species_exploration_coins"] += int(rows["species_coin"][active].sum())
            self._stats["tile_exploration_coins"] += int(rows["tile_coin"][active].sum())
            self._stats["exploratory_changes"] += int(
                np.count_nonzero(active & (rows["action"] != rows["greedy_action"]))
            )
            self._planting_samples += int(
                np.count_nonzero(active & (rows["action"] > 0) & (rows["action"] < A.dig_start))
            )
            self._simulation_ticks += int(duration[active].sum())
            self._first = False
            self._last_obs = next_obs
            self._previous = env.executed_actions.clone()
            self.num_timesteps += int(active.sum())
            # Paused workers contribute no action, reward, duration or optimizer sample.
            active_infos = [info for info, on in zip(infos, active) if on]
            if not hasattr(callback, "cfg"):
                self.training_games += sum("episode_metrics" in x for x in active_infos)
            callback.update_locals({"infos": active_infos, "dones": dones})
            if not callback.on_step():
                raise KeyboardInterrupt
            if not env.enabled_envs.any():
                self._phase(CohortPhase.RETURNS, callback)

    def _prepare_fit(self):
        if self._fit_epoch >= self.n_epochs:
            if self._fit_order is None:
                self._fit_order = np.empty(0, dtype=np.int64)
            return
        if self._fit_order is None:
            indices = self._buffer.valid_indices()
            self._fit_order = indices[np.random.permutation(len(indices))]
            self._fit_cursor = 0
        if self._prefetch is None and len(self._fit_order) > self._fit_cursor:
            from pvz_rl.learning.transfers import BatchPrefetch

            torch.cuda.current_stream(self.device).wait_stream(self.env.stream)
            self._prefetch = BatchPrefetch(
                self._buffer, self._fit_order[self._fit_cursor :], self.batch_size, self.device
            )

    def _q_step(self, data):
        branch, tile = self.policy.selected_values(
            data["observation"], data["action"], data["context"]
        )
        loss, branch_error, tile_error = balanced_q_loss(
            branch, tile, data["target"], data["action"], self._buffer.group_counts, self.batch_size
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Q loss")
        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True
        )
        self.policy.optimizer.step()
        self._stats["q_optimizer_steps"] += 1
        self.logger.record("train/q_loss", float(loss.detach()))
        nonwait = data["action"] != 0
        totals = self._loss_sums
        totals["branch"] += float(branch_error.detach().sum())
        totals["branch_count"] += len(branch_error)
        totals["tile"] += float(tile_error[nonwait].detach().sum())
        totals["tile_count"] += int(nonwait.sum())
        self.logger.record("train/branch_loss", totals["branch"] / totals["branch_count"])
        self.logger.record(
            "train/tile_loss",
            totals["tile"] / totals["tile_count"] if totals["tile_count"] else None,
        )
        self.logger.record("train/q_grad_norm", float(norm))

    def _fit_step(self, callback):
        self._prepare_fit()
        if self._fit_epoch >= self.n_epochs or not len(self._fit_order):
            self._drain_transfers()
            self._phase(CohortPhase.SYNCHRONIZE, callback)
            self._fit_epoch = self._fit_cursor = 0
            self._fit_order = None
            return
        indices = self._fit_order[self._fit_cursor : self._fit_cursor + self.batch_size]
        data = next(self._prefetch)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.policy.set_training_mode(True)
        self._q_step(data)
        end.record()
        end.synchronize()
        self._device_seconds += start.elapsed_time(end) / 1000
        self._fit_cursor += len(indices)
        if self._fit_cursor == len(self._fit_order):
            self._drain_transfers()
            self._fit_epoch += 1
            self._fit_order = None
        if hasattr(callback, "log_progress"):
            callback.log_progress()

    def _synchronize(self, callback):
        self._drain_transfers()
        torch.cuda.synchronize(self.device)
        elapsed = sum(self._phase_times.values())
        transitions = int(self._buffer.group_counts.sum())
        self.cohort_metrics = {
            **self._stats,
            "prefit_errors": self.prefit_errors,
            **self._buffer.transport_metrics,
            "cache_hit_rate": self._buffer.transport_metrics["cache_hits"]
            / max(1, self._buffer.transport_metrics["cache_requests"]),
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
        if self.optimizer_protocol != OPTIMIZER_PROTOCOL:
            raise ValueError("Incompatible optimizer protocol")
        if reset_num_timesteps:
            self.num_timesteps = 0
        # BaseAlgorithm initializes callbacks/logger without resetting private games.
        if self._last_obs is None:
            self._last_obs = torch.zeros(
                self.n_envs, *self.observation_space.shape, device=self.device
            )
            self._last_episode_starts = np.ones(self.n_envs, dtype=bool)
        total, callback = self._setup_learn(
            total_timesteps, callback, False, tb_log_name, progress_bar
        )
        if self.runtime_state:
            self._restore_runtime()
        callback.on_training_start(locals(), globals())
        try:
            while self.num_timesteps < total or self.phase != CohortPhase.IDLE:
                if self.phase == CohortPhase.IDLE:
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
                                self.logger.record(
                                    "train/value_target_error_" + name, values["mse"]
                                )
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
        except KeyboardInterrupt:
            self._drain_transfers()
            raise
        finally:
            self._drain_transfers()

    def _restore_runtime(self):
        state = self.runtime_state
        if state.get("environment") is None:
            self._restore_rng(state)
            self.runtime_state = None
            return
        self._last_obs = self.env.restore_training(state["environment"])
        self.resumed_cohort = True
        if self.phase == CohortPhase.IDLE:
            self._restore_rng(state)
            self.runtime_state = None
            return
        self._memory = EventMemory(self.cfg, self.env.batch.rules, self.n_envs, self.device)
        self._memory.restore(state["memory"])
        self._previous = state["previous"].to(self.device)
        self._buffer = self._new_buffer()
        workspace = self._buffer.path
        with ZipFile(self._checkpoint_source) as archive:
            self._buffer = CompleteGameBuffer.restore_archive(archive, state["buffer"], workspace)
        self._buffer.configure_cache(
            self.device, self.env.cfg["training"].get("performance", {}).get("token_cache_gib", 2)
        )
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
        self._drain_transfers()
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
                "previous": self._previous,
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
                            policy=POLICY_SIGNATURE,
                            optimizer=OPTIMIZER_PROTOCOL,
                            exploration=EXPLORATION_PROTOCOL,
                        )
                    ),
                )
                state_bytes = io.BytesIO()
                torch.save(runtime, state_bytes)
                archive.writestr("cohort-state.pt", state_bytes.getvalue())
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
            "_previous",
            "_checkpoint_source",
            "runtime_state",
            "_prefetch",
            "_collection_host",
        ]

    def _get_torch_save_params(self):
        return ["policy", "policy.optimizer"], []
