"""One CUDA trainer: complete games, Monte Carlo critic, conditional plant PPO."""

import io
import random
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.learning.cohort import (
    OPTIMIZER_PROTOCOL,
    ActorTransaction,
    CohortPhase,
    atomic_transition,
)
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.exploration import configure_exploration, exploration_loss
from pvz_rl.policy.event_memory import EventMemory
from pvz_rl.policy.grouped_policy import ACTION_DISTRIBUTION, selected_value_indices


def checkpoint_optimizer_protocol(path):
    data, _, _ = load_from_zip_file(path, device="cpu", load_data=True)
    return data.get("optimizer_protocol")


class CudaCompleteGamePPO(BaseAlgorithm):
    def __init__(
        self,
        policy,
        env=None,
        *,
        learning_rate=1e-4,
        batch_size=1024,
        n_epochs=4,
        clip_range=0.2,
        max_grad_norm=0.5,
        normalize_advantage=True,
        target_kl=0.01,
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
        self.batch_size, self.n_epochs, self.clip_range = batch_size, n_epochs, clip_range
        self.max_grad_norm, self.normalize_advantage = max_grad_norm, normalize_advantage
        self.target_kl = target_kl or 0.0
        self.optimizer_protocol = OPTIMIZER_PROTOCOL
        self.action_distribution_protocol = ACTION_DISTRIBUTION
        self.phase = CohortPhase.IDLE
        self.training_games = self._n_updates = 0
        self.cohort_metrics = {}
        self.runtime_state = None
        self._buffer = self._memory = self._behavior = self._transaction = None
        self._fit_order = None
        self._fit_epoch = self._fit_cursor = 0
        self._cohort_elapsed = 0.0
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
        data, _, _ = load_from_zip_file(path, device="cpu")
        if (
            data.get("optimizer_protocol") != OPTIMIZER_PROTOCOL
            or data.get("action_distribution_protocol") != ACTION_DISTRIBUTION
        ):
            raise ValueError("Complete-game controller requires fresh models")
        model = super().load(path, *args, **kwargs)
        model.phase = CohortPhase(model.phase)
        # SB3 maps every saved tensor to the requested device. Ordinary Adam
        # keeps its scalar step on CPU; retain that original optimizer protocol.
        for optimizer in (model.policy.optimizer, model.policy.critic_optimizer):
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
        return CompleteGameBuffer(
            path,
            self.observation_space.shape[0],
            self._memory.capacity,
            self.n_envs,
            ram_bytes=spec["ram_gib"] * 1024**3,
            block_rows=spec["block_rows"],
        )

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
        self._phase_times = {"collect": 0.0, "returns": 0.0, "critic": 0.0, "actor": 0.0}
        for key in (
            "policy_gradient_loss",
            "joint_entropy",
            "exploration_bonus",
            "plant_exploration_bonus",
            "tile_exploration_bonus",
            "actor_grad_norm",
            "approx_kl",
        ):
            self.logger.record("train/" + key, None)
        self._stats = {
            "actor_attempted_steps": 0,
            "actor_retained_steps": 0,
            "critic_optimizer_steps": 0,
            "exact_kl": 0.0,
            "actor_window_rejected": 0,
            "exact_kl_attempted_max": 0.0,
        }
        self._behavior = self._snapshot_policy()
        self._behavior.set_training_mode(False)
        self._transaction = None
        self._phase(CohortPhase.COLLECT, callback)

    def _snapshot_policy(self):
        # Module construction initializes weights; preserve both CPU/CUDA RNG.
        devices = (
            [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=devices):
            policy = self.policy_class(
                self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
            ).to(self.device)
        policy.load_state_dict(self.policy.state_dict())
        policy.exploration_settings = dict(self.policy.exploration_settings)
        policy.set_training_mode(False)
        return policy

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
            actions, values, logs = self.policy.decide(obs, masks, context=context)
            # One bounded packed transfer; copied before the simulator mutates observations.
            packed = torch.cat(
                (
                    obs,
                    masks.float(),
                    self._previous[:, None].float(),
                    ticks[:, None].float(),
                    actions[:, None].float(),
                    values[:, None],
                    logs[:, None],
                    # Bit-preserve 32-bit references, including indices above 2**24.
                    self._memory.ids.to(torch.int32).view(torch.float32),
                    self._memory.counts,
                    self._memory.starts,
                ),
                -1,
            )
            host = packed.cpu().numpy()
            next_obs, rewards, dones, _, _, infos = env.step_tensors(actions, autoreset=False)
            reward = rewards.cpu().numpy()
            duration = env.transition_ticks.cpu().numpy()
            rows = np.zeros(self.n_envs, dtype=self._buffer.dtype)
            d = self.observation_space.shape[0]
            rows["observation"] = host[:, :d]
            rows["mask"] = np.packbits(
                host[:, d : d + A.size].astype(bool), axis=-1, bitorder="little"
            )
            base = d + A.size
            for j, key in enumerate(("previous", "tick", "action", "baseline", "log_prob")):
                rows[key] = host[:, base + j]
            base += 5
            rows["ids"] = host[:, base : base + self._memory.capacity].view(np.uint32)
            base += self._memory.capacity
            for key in ("counts", "starts"):
                rows[key] = host[:, base : base + self._memory.capacity]
                base += self._memory.capacity
            rows["reset"] = self._first
            rows["active"], rows["env"] = active, np.arange(self.n_envs)
            rows["reward"], rows["duration"] = reward, duration
            self._buffer.append(rows)
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

    def _prepare_fit(self, planting):
        if self._fit_order is None:
            indices = self._buffer.valid_indices(planting)
            self._fit_order = indices[np.random.permutation(len(indices))]
            self._fit_cursor = 0

    def _critic_step(self, data):
        values = self.policy.predict_values(data["observation"], data["context"])
        indices = selected_value_indices(data["action"])
        predicted = values.gather(1, indices[:, None]).flatten()
        groups = torch.where(indices < 2, indices, 2)
        counts = torch.as_tensor(self._buffer.group_counts, device=self.device, dtype=values.dtype)
        weight = counts.sum() / (counts.clamp_min(1)[groups] * (counts > 0).sum())
        # Keep the same per-sample scale in the final short minibatch, so each
        # populated group has equal aggregate weight over a complete epoch.
        loss = ((predicted - data["target"]).square() * weight).sum() / min(
            self.batch_size, int(self._buffer.group_counts.sum())
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite critic loss")
        self.policy.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy.critic_parameters(), self.max_grad_norm, error_if_nonfinite=True
        )
        self.policy.critic_optimizer.step()
        self._stats["critic_optimizer_steps"] += 1
        self.logger.record("train/value_loss", float(loss.detach()))
        self.logger.record("train/critic_grad_norm", float(norm))

    def _actor_step(self, data):
        distribution = self.policy.get_distribution(
            data["observation"], data["masks"], data["context"]
        )
        logs = distribution.log_prob(data["action"])
        delta = logs - data["log_prob"]
        ratio = delta.exp()
        approx = ((ratio - 1) - delta).mean()
        if not torch.isfinite(approx):
            self._transaction.reject("nonfinite_likelihood")
            return
        if self.target_kl and float(approx.detach()) > 1.5 * self.target_kl:
            self._transaction.state["stopped"] = True
            return
        advantage = data["advantage"]
        if self.normalize_advantage and self._buffer.group_counts[1] > 1:
            advantage = (advantage - self._buffer.advantage_mean) / (
                self._buffer.advantage_std + 1e-8
            )
        policy_loss = -torch.minimum(
            ratio * advantage, ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * advantage
        ).mean()
        entropy_loss, metrics = exploration_loss(self.policy, distribution.entropy(), logs)
        loss = policy_loss + entropy_loss
        self._stats["actor_attempted_steps"] += 1
        self.policy.optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            self._transaction.reject("nonfinite_actor_loss")
            return
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.max_grad_norm)
        if not torch.isfinite(norm):
            self._transaction.reject("nonfinite_actor_gradient")
            return
        self.policy.optimizer.step()
        self._stats["actor_retained_steps"] += 1
        for key, value in {
            "policy_gradient_loss": policy_loss,
            "actor_grad_norm": norm,
            "approx_kl": approx,
            **metrics,
        }.items():
            self.logger.record("train/" + key, float(value.detach()))

    @torch.no_grad()
    def _exact_kl(self):
        indices = self._buffer.valid_indices(True)
        total = 0.0
        for start in range(0, len(indices), self.batch_size):
            data = self._buffer.batch(indices[start : start + self.batch_size], self.device)
            old = self._behavior.get_distribution(
                data["observation"], data["masks"], data["context"]
            )
            new = self.policy.get_distribution(data["observation"], data["masks"], data["context"])
            total += float(old.kl_divergence(new).sum())
        return total / max(1, len(indices))

    def _guard(self):
        if self._transaction is None:
            return
        kl = self._exact_kl()
        self._stats["exact_kl_attempted_max"] = max(self._stats["exact_kl_attempted_max"], kl)
        self._transaction.check(kl)
        self._stats["exact_kl"] = 0.0 if self._transaction.state["rejected"] else kl
        self._stats["actor_window_rejected"] = int(self._transaction.state["rejected"])
        if self._transaction.state["rejected"]:
            self._stats["actor_retained_steps"] = 0

    def _fit_step(self, callback):
        actor = self.phase == CohortPhase.ACTOR
        if actor and self._transaction is None:
            self._transaction = ActorTransaction(self.policy, self.target_kl)
        self._prepare_fit(actor)
        if (
            self._fit_epoch >= self.n_epochs
            or not len(self._fit_order)
            or actor
            and self._transaction.state["stopped"]
        ):
            if actor:
                self._guard()
                self._phase(CohortPhase.SYNCHRONIZE, callback)
            else:
                self._phase(CohortPhase.ACTOR, callback)
            self._fit_epoch = self._fit_cursor = 0
            self._fit_order = None
            return
        indices = self._fit_order[self._fit_cursor : self._fit_cursor + self.batch_size]
        data = self._buffer.batch(indices, self.device)
        self.policy.set_training_mode(True)
        if actor:
            self._actor_step(data)
        else:
            self._critic_step(data)
        self._fit_cursor += len(indices)
        if self._fit_cursor == len(self._fit_order):
            self._fit_epoch += 1
            self._fit_order = None
            if actor:
                self._guard()
        if hasattr(callback, "log_progress"):
            callback.log_progress()

    def _synchronize(self, callback):
        torch.cuda.synchronize(self.device)
        elapsed = sum(self._phase_times.values())
        transitions = int(self._buffer.group_counts.sum())
        self.cohort_metrics = {
            "collection_seconds": self._phase_times["collect"],
            "critic_seconds": self._phase_times["critic"],
            "actor_seconds": self._phase_times["actor"],
            "returns_seconds": self._phase_times["returns"],
            "window_seconds": elapsed,
            "optimization_seconds": self._phase_times["critic"] + self._phase_times["actor"],
            "transitions_per_second": transitions / max(elapsed, 1e-9),
            "transitions": transitions,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(self.device),
        }
        for key, value in self._stats.items():
            self.logger.record("train/" + key, value)
        self.logger.record("train/learning_rate", self.policy.optimizer.param_groups[0]["lr"])
        self.logger.record(
            "train/critic_learning_rate", self.policy.critic_optimizer.param_groups[0]["lr"]
        )
        for name in ("plant", "tile"):
            self.logger.record(
                "train/effective_" + name + "_entropy_coef",
                self.policy.exploration_settings[name + "_coef"],
            )
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates)
        self._buffer.close()
        self._buffer = self._memory = self._behavior = self._transaction = None
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
                                    "train/raw_advantage_mean_" + name,
                                    -values["signed_error"] if values["count"] else None,
                                )
                                self.logger.record(
                                    "train/positive_advantage_fraction_" + name,
                                    values["positive_advantage_fraction"],
                                )
                            self._phase(CohortPhase.CRITIC, callback)
                        elif phase in (CohortPhase.CRITIC, CohortPhase.ACTOR):
                            self._fit_step(callback)
                        elif phase == CohortPhase.SYNCHRONIZE:
                            self._synchronize(callback)
                finally:
                    if phase.value in self._phase_times:
                        self._phase_times[phase.value] += perf_counter() - start
            callback.on_training_end()
            return self
        except KeyboardInterrupt:
            if self.phase == CohortPhase.ACTOR:
                with atomic_transition():
                    self._guard()
            raise

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
        self._behavior = self._snapshot_policy()
        self._behavior.load_state_dict(state["behavior"])
        if state["transaction"] is not None:
            self._transaction = ActorTransaction(self.policy, self.target_kl, state["transaction"])
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
                "previous": self._previous,
                "buffer": self._buffer.metadata(),
                "behavior": self._behavior.state_dict(),
                "transaction": None if self._transaction is None else self._transaction.state,
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
            "_behavior",
            "_transaction",
            "_previous",
            "_checkpoint_source",
            "runtime_state",
        ]

    def _get_torch_save_params(self):
        return ["policy", "policy.optimizer", "policy.critic_optimizer"], []
