"""SB3-compatible PPO with tensor collection and batched optimizer diagnostics.

The PPO ratios and minibatch order follow SB3/SB3-Contrib 2.7.1. Storage,
transport and metric transfers stay on-device where possible. An explicitly
configured exploration bonus can replace joint-entropy regularization.
"""

import hashlib
import json
import math
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from time import perf_counter

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import update_learning_rate
from torch.nn import functional as F

from .actions import ActionSchema as A
from .cuda_buffer import TensorRolloutBuffer
from .event_memory import EventMemory, MemoryContext
from .exploration import configure_exploration, exploration_loss
from .grouped_policy import ACTION_DISTRIBUTION, GroupedDistribution
from .periodic import (
    OPTIMIZER_PROTOCOL,
    ActorUpdateState,
    ActorWindow,
    WindowState,
    aggregate_update_metrics,
    finish_window_on_interrupt,
    interval_overlap,
)


def checkpoint_optimizer_protocol(path):
    if isinstance(path, (str, Path)) and not Path(path).exists():
        path = str(path) + ".zip"
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data")).get("optimizer_protocol")


@dataclass
class RolloutSlotResult:
    """A complete on-policy buffer and the collector state for the next slot."""

    buffer: TensorRolloutBuffer
    observations: torch.Tensor
    episode_starts: torch.Tensor
    memory: EventMemory
    infos: list[list[dict]]
    transitions: int
    collection_interval: tuple[float, float]
    memory_metrics: dict[str, float]
    policy_version: int
    behavior_hash: str


class PeriodicPipelineError(RuntimeError):
    """Raised when a periodic collector or learner slot fails."""


class TensorPPO:
    def __init__(self, *args, **kwargs):
        self.optimizer_protocol = OPTIMIZER_PROTOCOL
        self.action_distribution_protocol = ACTION_DISTRIBUTION
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path, *args, **kwargs):
        protocol = checkpoint_optimizer_protocol(path)
        archive_path = path
        if isinstance(path, (str, Path)) and not Path(path).exists():
            archive_path = str(path) + ".zip"
        with zipfile.ZipFile(archive_path) as archive:
            if json.loads(archive.read("data")).get("action_distribution_protocol") != ACTION_DISTRIBUTION:
                raise ValueError("Action distribution changed; fresh training is required")
        model = super().load(path, *args, **kwargs)
        model.optimizer_protocol = protocol
        configure_exploration(model, model.policy.features_extractor.cfg)
        return model

    def _require_optimizer_protocol(self):
        if self.optimizer_protocol != OPTIMIZER_PROTOCOL:
            raise ValueError("Optimizer protocol changed; use fresh training or --init-from")

    def _setup_learn(
        self,
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="run",
        progress_bar=False,
    ):
        if reset_num_timesteps or self._last_obs is None:
            self.__dict__.pop("_episode_memory", None)
        return super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )

    def _maybe_recommend_cpu(self, mlp_policy_class_name="ActorCriticPolicy"):
        """The upstream warning assumes CPU observations; this path requires CUDA."""

    def _new_rollout_buffer(self):
        """Allocate one independent slot for the periodic producer/consumer loop."""
        return TensorRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            memory_spec=self.rollout_buffer.memory_spec,
        )

    def collect_slot(
        self,
        env,
        policy,
        rollout_buffer,
        observations,
        episode_starts,
        memory,
        policy_version,
        behavior_hash,
        stop_event=None,
    ):
        """Collect one full rollout using an immutable behavior-policy snapshot.

        The collector owns the environment and memory for the duration of this
        call. No callback or live learner state is touched, which lets the
        learner optimize the previous slot concurrently.
        """
        started = perf_counter()
        host_phase_before = dict(getattr(env, "phases", {}))
        infos_by_step = []
        policy.set_training_mode(False)
        rollout_buffer.reset()
        observations = torch.as_tensor(observations, device=self.device)
        episode_starts = torch.as_tensor(episode_starts, device=self.device)
        with env.device_context():
            if memory is None:
                memory = EventMemory(env.cfg, env.batch.rules, env.num_envs, self.device)
                memory.observe(
                    observations,
                    env.action_masks(),
                    torch.zeros(env.num_envs, device=self.device),
                    episode_starts,
                    env.header_tensor[:, 0],
                )
            rollout_buffer.start_memory(memory)
            for _ in range(self.n_steps):
                if stop_event is not None and stop_event.is_set():
                    raise PeriodicPipelineError("periodic collector cancelled")
                obs = observations.clone()
                masks = env.action_masks().clone()
                rollout_buffer.capture_context(memory)
                with torch.inference_mode(), env.features.profiler.track("inference"):
                    actions, log_probs = policy.sample_actions(obs, masks, context=memory.context())
                    rollout_buffer.capture_behavior(policy.action_dist)
                new_obs, rewards, dones, timeouts, terminal, infos = env.step_tensors(actions)
                infos_by_step.append(infos)
                if terminal is not None and any(
                    info.get("TimeLimit.truncated", False) for info in infos
                ):
                    indices = torch.nonzero(timeouts, as_tuple=True)[0]
                    terminal_memory = memory.fork()
                    terminal_memory.observe(
                        terminal,
                        env.terminal_masks,
                        env.executed_actions,
                        torch.zeros_like(dones),
                        env.terminal_ticks,
                    )
                    rollout_buffer.add_timeouts(
                        indices, terminal[indices], terminal_memory.context().select(indices)
                    )
                rollout_buffer.add(
                    obs,
                    actions,
                    rewards,
                    episode_starts,
                    None,
                    log_probs,
                    masks,
                    durations=env.transition_ticks,
                )
                memory.observe(
                    new_obs,
                    env.action_masks(),
                    env.executed_actions * (~dones),
                    dones,
                    env.header_tensor[:, 0],
                )
                observations, episode_starts = new_obs, dones
            with env.features.profiler.track("critic_inference"):
                values = rollout_buffer.evaluate_values(
                    policy,
                    observations,
                    getattr(self, "value_batch_size", 1024),
                    memory.context(),
                )
            with env.features.profiler.track("gae"):
                rollout_buffer.compute_returns_and_advantage(values, episode_starts)
            metrics = {
                "memory_compression_ratio": float(
                    memory.counts.sum() / memory.valid.sum().clamp_min(1)
                ),
                "memory_event_fraction": float(memory.admitted) / max(1, memory.observed),
            }
            weights = policy.pi_features_extractor.temporal.last_attention
            for name, lo, hi in (
                ("local", 0, memory.local),
                ("events", memory.local, memory.local + memory.events),
                ("summaries", memory.local + memory.events, memory.capacity),
            ):
                metrics[f"attention_{name}"] = float(weights[:, lo:hi].sum(-1).mean())
            device_phases = env.features.profiler.flush()
            env.features.profiler.seconds.clear()
            metrics.update(
                {
                    f"device_{key}_seconds": value
                    for key, value in device_phases.items()
                }
            )
            metrics.update(
                {
                    f"host_{key}_seconds": value - host_phase_before.get(key, 0.0)
                    for key, value in getattr(env, "phases", {}).items()
                }
            )
            # The ready queue transfers only completed tensors to the learner.
            completed = torch.cuda.Event()
            completed.record(env.stream)
            completed.synchronize()
        return RolloutSlotResult(
            buffer=rollout_buffer,
            observations=observations,
            episode_starts=episode_starts,
            memory=memory,
            infos=infos_by_step,
            transitions=self.n_steps * env.num_envs,
            collection_interval=(started, perf_counter()),
            memory_metrics=metrics,
            policy_version=policy_version,
            behavior_hash=behavior_hash,
        )

    def _clone_behavior_policy(self):
        """Clone actor and critic weights for one immutable policy window."""
        torch.cuda.current_stream(self.device).synchronize()
        with torch.random.fork_rng(devices=[self.device]):
            return self._construct_behavior_policy()

    def _construct_behavior_policy(self):
        source = self.policy
        policy = type(source)(
            source.observation_space,
            source.action_space,
            lambda _: 0.0,
            net_arch=source.net_arch,
            activation_fn=source.activation_fn,
            ortho_init=False,
            features_extractor_class=type(source.pi_features_extractor),
            features_extractor_kwargs={"layout_cfg": source.pi_features_extractor.cfg},
            share_features_extractor=False,
            normalize_images=source.normalize_images,
            optimizer_class=source.optimizer_class,
            optimizer_kwargs=dict(source.optimizer_kwargs),
            critic_learning_rate=source.critic_learning_rate,
            exploration_epsilon=source.exploration_epsilon,
        ).to(self.device)
        policy.load_state_dict(
            {key: value.detach().clone() for key, value in source.state_dict().items()},
            strict=True,
        )
        policy.set_training_mode(False)
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        policy.optimizer = None
        policy.critic_optimizer = None
        return policy

    @staticmethod
    def _behavior_hash(policy):
        digest = hashlib.sha256()
        digest.update(repr(policy.exploration_epsilon).encode("utf-8"))
        for name, value in sorted(policy.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _periodic_learn(self, total_timesteps, callback, log_interval=1):
        """Two reusable slots, one frozen snapshot, and one empty-window boundary."""
        assert self.env is not None
        env = self.env
        self.pipeline_state = WindowState.IDLE
        callback.on_training_start(locals(), globals())
        learner_stream = torch.cuda.current_stream(self.device)
        previous_stream = env.stream
        collector_stream = torch.cuda.Stream(device=self.device)
        stop_event = Event()
        observations = torch.as_tensor(self._last_obs, device=self.device)
        episode_starts = torch.as_tensor(self._last_episode_starts, device=self.device)
        memory = getattr(self, "_episode_memory", None)
        settings = self.policy.pi_features_extractor.cfg["training"]["pipeline"]
        free, ready = Queue(maxsize=settings["depth"]), Queue(maxsize=settings["queue_size"])
        for slot in (self.rollout_buffer, self._new_rollout_buffer()):
            free.put(slot)
        behavior = self._clone_behavior_policy()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-collector")
        pending = None
        window = 0
        env.stream = collector_stream

        def produce(depth, version, digest):
            nonlocal observations, episode_starts, memory
            for _ in range(depth):
                if stop_event.is_set():
                    return
                slot = free.get()
                with torch.cuda.stream(collector_stream):
                    result = self.collect_slot(
                        env,
                        behavior,
                        slot,
                        observations,
                        episode_starts,
                        memory,
                        version,
                        digest,
                        stop_event,
                    )
                observations, episode_starts, memory = (
                    result.observations,
                    result.episode_starts,
                    result.memory,
                )
                ready.put(result)

        try:
            while self.num_timesteps < total_timesteps:
                # All policy/schedule changes, probes and saves occur here, with
                # no collector active and no unconsumed rollout.
                callback.on_rollout_start()
                with finish_window_on_interrupt():
                    window_started = perf_counter()
                    self.pipeline_state = WindowState.COLLECTING
                    behavior.load_state_dict(self.policy.state_dict(), strict=True)
                    behavior.exploration_epsilon = self.policy.exploration_epsilon
                    behavior.action_dist.epsilon = self.policy.exploration_epsilon
                    digest = self._behavior_hash(behavior)
                    version = getattr(self, "pipeline_version", 0)
                    self._actor_window = ActorWindow(self.policy, self.target_kl)
                    self._window_kl_max = 0.0
                    self._checked_policy_metrics = {}
                    collector_stream.wait_stream(learner_stream)
                    slot_steps = self.n_steps * self.n_envs
                    depth = min(
                        settings["depth"],
                        max(
                            1, (total_timesteps - self.num_timesteps + slot_steps - 1) // slot_steps
                        ),
                    )
                    first_step = self.num_timesteps
                    pending = executor.submit(produce, depth, version, digest)
                    results, updates, update_metrics = [], [], []
                    queue_wait = 0.0
                    for index in range(depth):
                        waiting = perf_counter()
                        while True:
                            try:
                                result = ready.get(timeout=0.05)
                                break
                            except Empty:
                                if pending.done():
                                    pending.result()  # Surface collector failures; never deadlock.
                                    raise PeriodicPipelineError("collector returned no rollout")
                        queue_wait += perf_counter() - waiting
                        if result.policy_version != version or result.behavior_hash != digest:
                            raise PeriodicPipelineError("collector returned a mixed policy window")
                        self.pipeline_state = (
                            WindowState.OVERLAPPING if index + 1 < depth else WindowState.DRAINING
                        )
                        self.rollout_buffer = result.buffer
                        self.num_timesteps += result.transitions
                        self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                        update_started = perf_counter()
                        # New slot contexts can reveal movement missed by slot one.
                        available = [r.buffer for r in results] + [result.buffer]
                        if results:
                            self._check_actor_window([result.buffer])
                        self._checked_policy_metrics.clear()
                        self.train()
                        self._check_actor_window(available)
                        missing = [
                            buffer
                            for buffer in available
                            if id(buffer) not in self._checked_policy_metrics
                        ]
                        if missing:
                            for buffer, metrics in zip(
                                missing, self._post_update_metrics_many(missing)
                            ):
                                self._checked_policy_metrics[id(buffer)] = metrics
                        post = self._checked_policy_metrics[id(result.buffer)]
                        for key, value in post.items():
                            self.logger.record(f"train/{key}", value)
                        learner_stream.synchronize()
                        updates.append((update_started, perf_counter()))
                        results.append(result)
                        update_metrics.append(
                            {
                                key: value
                                for key, value in self.logger.name_to_value.items()
                                if key.startswith("train/")
                            }
                        )
                        free.put(result.buffer)
                    pending.result()
                    pending = None
                    if self._behavior_hash(behavior) != digest:
                        raise PeriodicPipelineError("behavior policy changed during collection")
                    if not ready.empty() or free.qsize() != settings["depth"]:
                        raise PeriodicPipelineError("pipeline did not drain at window boundary")
                    self._last_obs, self._last_episode_starts = observations, episode_starts
                    self._episode_memory = memory
                    self.pipeline_version = version + 1
                    window_seconds = perf_counter() - window_started
                    collections = [r.collection_interval for r in results]
                    self.pipeline_metrics = {
                        "mode": "periodic_on_policy",
                        "policy_version": version,
                        "behavior_hash": digest,
                        "depth": depth,
                        "slot_collection_seconds": [end - start for start, end in collections],
                        "slot_optimization_seconds": [end - start for start, end in updates],
                        "window_seconds": window_seconds,
                        "overlap_seconds": sum(
                            interval_overlap(c, u) for c in collections for u in updates
                        ),
                        "queue_wait_seconds": queue_wait,
                        "transitions_per_second": depth * slot_steps / max(window_seconds, 1e-9),
                        "actor_update_state": self._actor_window.state.value,
                        "actor_rejection_reason": self._actor_window.rejection_reason,
                    }
                    # Commit every episode exactly once, with its real transition
                    # count. Callbacks cannot observe or checkpoint a half-window.
                    keep_going = True
                    self.num_timesteps = first_step
                    for result in results:
                        for infos in result.infos:
                            self.num_timesteps += self.n_envs
                            self._update_info_buffer(infos)
                            callback.update_locals({"infos": infos})
                            keep_going = callback.on_step() and keep_going
                    merged = aggregate_update_metrics(update_metrics)
                    attempted = merged["train/actor_optimizer_steps"]
                    rejected = self._actor_window.state == ActorUpdateState.REJECTED
                    self.rejected_windows = getattr(self, "rejected_windows", 0) + int(rejected)
                    merged.update(
                        {
                            "train/actor_attempted_steps": attempted,
                            "train/actor_retained_steps": 0 if rejected else attempted,
                            "train/actor_optimizer_steps": 0 if rejected else attempted,
                            "train/optimizer_steps": 0 if rejected else attempted,
                            "train/actor_window_rejected": float(rejected),
                            "train/rejected_windows": self.rejected_windows,
                            "train/exact_kl_attempted_max": self._window_kl_max,
                        }
                    )
                    # Refresh current-policy statistics after any rollback.
                    missing = [
                        r.buffer
                        for r in results
                        if id(r.buffer) not in self._checked_policy_metrics
                    ]
                    if missing:
                        for buffer, metrics in zip(
                            missing, self._post_update_metrics_many(missing)
                        ):
                            self._checked_policy_metrics[id(buffer)] = metrics
                    post = [self._checked_policy_metrics[id(r.buffer)] for r in results]
                    count = sum(x["dig_legal_observations"] for x in post)
                    merged["train/exact_kl"] = max(x["exact_kl"] for x in post)
                    merged["train/dig_probability_when_legal"] = (
                        sum(x["dig_probability_sum"] for x in post) / count
                        if count
                        else float("nan")
                    )
                    for name in ("post_update_approx_kl", "post_update_type_kl"):
                        merged["train/" + name] = sum(x[name] for x in post) / len(post)
                    for key, value in merged.items():
                        self.logger.record(key, value)
                    for key in results[-1].memory_metrics:
                        self.logger.record(
                            f"train/{key}", sum(r.memory_metrics[key] for r in results) / depth
                        )
                    self.pipeline_state = WindowState.IDLE
                    self.__dict__.pop("_actor_window", None)
                    self.__dict__.pop("_checked_policy_metrics", None)
                    callback.on_rollout_end()
                    if hasattr(callback, "capture_update"):
                        callback.capture_update()
                    window += 1
                    if log_interval is not None and window % log_interval == 0:
                        self.dump_logs(window)
                if not keep_going:
                    break
        except BaseException:
            if self.pipeline_state != WindowState.IDLE:
                self.pipeline_state = WindowState.FAILED
            raise
        finally:
            stop_event.set()
            if pending is not None:
                # A producer can be waiting to publish its final slot after a
                # learner failure. Drain its bounded queue until it exits.
                while not pending.done():
                    try:
                        ready.get(timeout=0.05)
                    except Empty:
                        pass
            executor.shutdown(wait=True, cancel_futures=True)
            collector_stream.synchronize()
            env.stream = previous_stream
        callback.on_training_end()
        return self

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=1,
        tb_log_name="OnPolicyAlgorithm",
        reset_num_timesteps=True,
        progress_bar=False,
    ):
        """Use the periodic pipeline as the only learner scheduler."""
        self._require_optimizer_protocol()
        if getattr(self, "pipeline_state", WindowState.IDLE) != WindowState.IDLE:
            raise PeriodicPipelineError(
                "Reload a completed-window checkpoint after pipeline failure"
            )
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        return self._periodic_learn(total_timesteps, callback, log_interval)

    @torch.no_grad()
    def _type_probabilities(self):
        buffer = self.rollout_buffer
        observations = buffer.observations.flatten(0, 1)
        masks = buffer.action_masks.flatten(0, 1)
        size = getattr(self, "value_batch_size", 1024)
        return torch.cat(
            [
                self.policy.get_distribution(
                    observations[i : i + size],
                    masks[i : i + size],
                    context=buffer.context(slice(i, i + size)),
                ).types.probs
                for i in range(0, len(observations), size)
            ]
        )

    @torch.no_grad()
    def _exact_kl(self, buffer):
        if not buffer.has_behavior_logits:
            raise RuntimeError("Missing frozen behavior probabilities")
        if not torch.stack([torch.isfinite(p).all() for p in self.policy.actor_parameters()]).all():
            return float("inf")
        try:
            metrics = self._post_update_metrics(None, buffer)
        except ValueError:
            # A non-finite candidate can fail categorical validation before KL.
            return float("inf")
        self._checked_policy_metrics[id(buffer)] = metrics
        return metrics["exact_kl"]

    def _check_actor_window(self, buffers):
        if not self.target_kl or getattr(self, "critic_warmup_active", False):
            return []
        # Keep the explicit single-buffer hook available for diagnostics and
        # tests that inject a controlled KL outcome. Production uses the
        # batched path below.
        injected_exact_kl = self.__dict__.get("_exact_kl")
        if injected_exact_kl is not None:
            values = [injected_exact_kl(buffer) for buffer in buffers]
            self._window_kl_max = (
                max(self._window_kl_max, *values)
                if all(math.isfinite(x) for x in values)
                else float("inf")
            )
            previous = self._actor_window.state
            self._actor_window.check(values)
            if (
                previous != ActorUpdateState.REJECTED
                and self._actor_window.state == ActorUpdateState.REJECTED
            ):
                return values
            return values
        missing = [buffer for buffer in buffers if id(buffer) not in self._checked_policy_metrics]
        if missing:
            metrics = self._post_update_metrics_many(missing)
            for buffer, values_for_buffer in zip(missing, metrics):
                self._checked_policy_metrics[id(buffer)] = values_for_buffer
        values = [self._checked_policy_metrics[id(buffer)]["exact_kl"] for buffer in buffers]
        self._window_kl_max = (
            max(self._window_kl_max, *values)
            if all(math.isfinite(x) for x in values)
            else float("inf")
        )
        previous = self._actor_window.state
        self._actor_window.check(values)
        if (
            previous != ActorUpdateState.REJECTED
            and self._actor_window.state == ActorUpdateState.REJECTED
        ):
            return [self._checked_policy_metrics[id(buffer)]["exact_kl"] for buffer in buffers]
        return values

    @torch.no_grad()
    def _post_update_metrics(self, old_types, buffer=None):
        buffer = self.rollout_buffer if buffer is None else buffer
        return self._post_update_metrics_many([buffer], old_types=old_types)[0]

    @torch.no_grad()
    def _post_update_metrics_many(self, buffers, old_types=None):
        """Evaluate post-update diagnostics in shared policy batches.

        Slots retain independent memory archives, so their contexts are joined
        only for each device batch and split again before aggregation. This
        keeps the metric definitions and per-slot choice-state denominators
        identical while avoiding one policy forward per rollout slot.
        """
        buffers = list(buffers)
        if not buffers:
            return []
        observations = [buffer.observations.flatten(0, 1) for buffer in buffers]
        masks = [buffer.action_masks.flatten(0, 1) for buffer in buffers]
        actions = [buffer.actions.flatten().long() for buffer in buffers]
        old_logs = [buffer.log_probs.flatten() for buffer in buffers]
        behavior = [
            buffer.behavior_logits.flatten(0, 1) if buffer.has_behavior_logits else None
            for buffer in buffers
        ]
        if any(value is not None for value in behavior) and not all(
            value is not None for value in behavior
        ):
            raise RuntimeError("Cannot batch post-update metrics with mixed behavior logits")
        behavior_epsilon = (
            buffers[0].behavior_epsilon
            if all(value is not None for value in behavior)
            else 0.0
        )
        totals = [torch.zeros(13, device=self.device) for _ in buffers]
        size = getattr(self, "value_batch_size", 1024)
        maximum = max(map(len, observations))
        for start in range(0, maximum, size):
            pieces = []
            lengths = []
            contexts = []
            for index, buffer in enumerate(buffers):
                end = min(start + size, len(observations[index]))
                if start >= end:
                    continue
                pieces.append(
                    (
                        observations[index][start:end],
                        masks[index][start:end],
                        actions[index][start:end],
                        old_logs[index][start:end],
                        behavior[index][start:end] if behavior[index] is not None else None,
                    )
                )
                lengths.append((index, end - start))
                contexts.append(buffer.context(slice(start, end)))
            batch_observations = torch.cat([piece[0] for piece in pieces], 0)
            batch_masks = torch.cat([piece[1] for piece in pieces], 0)
            batch_actions = torch.cat([piece[2] for piece in pieces], 0)
            batch_old_logs = torch.cat([piece[3] for piece in pieces], 0)
            if all(context is None for context in contexts):
                batch_context = None
            elif any(context is None for context in contexts):
                raise RuntimeError("Cannot batch post-update metrics with mixed memory contexts")
            else:
                batch_context = MemoryContext(
                    *(torch.cat([getattr(context, name) for context in contexts], 0)
                      for name in MemoryContext.__dataclass_fields__)
                )
            distribution = self.policy.get_distribution(
                batch_observations, batch_masks, context=batch_context
            )
            batch_log_ratio = distribution.log_prob(batch_actions) - batch_old_logs
            current = distribution.types.probs
            exact = current.new_zeros(len(current))
            if behavior[0] is not None:
                old = GroupedDistribution(
                    behavior_epsilon, wait_weight=self.policy.action_dist.wait_weight
                ).proba_distribution(
                    torch.cat([piece[4] for piece in pieces], 0), batch_masks
                )
                previous = old.types.probs
                exact = old.kl_divergence(distribution)
            else:
                previous = current if old_types is None else old_types[start : start + len(current)]
            kl = (
                previous
                * (previous.clamp_min(1e-30).log() - current.clamp_min(1e-30).log())
            ).sum(-1)
            legal_dig = batch_masks[:, A.dig_start :].any(-1)
            type_entropy, plant_entropy, tile_entropy = distribution.entropy_parts()
            _, exploration = exploration_loss(self.policy, distribution.entropy(), batch_log_ratio)
            sample_totals = torch.stack(
                (
                    batch_log_ratio.exp() - 1 - batch_log_ratio,
                    kl,
                    current[:, A.dig] * legal_dig,
                    legal_dig,
                    type_entropy,
                    plant_entropy,
                    tile_entropy,
                    exploration["exploration_bonus"].expand(len(current)),
                    exploration["kind_exploration_bonus"].expand(len(current)),
                    exploration["plant_exploration_bonus"].expand(len(current)),
                    exploration["tile_exploration_bonus"].expand(len(current)),
                    (batch_masks.sum(-1) > 1),
                    exact * (batch_masks.sum(-1) > 1),
                ),
                -1,
            )
            offset = 0
            for index, length in lengths:
                totals[index] += sample_totals[offset : offset + length].sum(0)
                offset += length

        results = []
        for buffer, values in zip(buffers, totals):
            (
                approx,
                types,
                dig,
                count,
                type_entropy,
                plant_entropy,
                tile_entropy,
                bonus,
                kind_bonus,
                plant_bonus,
                tile_bonus,
                choice,
                exact,
            ) = values.cpu().tolist()
            if getattr(self, "critic_warmup_active", False):
                n = buffer.observations.flatten(0, 1).shape[0]
                self.policy._entropy_totals = values[4:7].clone()
                self.policy._entropy_count = n
                self.logger.record("train/joint_entropy", (type_entropy + plant_entropy + tile_entropy) / n)
                self.logger.record("train/entropy_loss", -(type_entropy + plant_entropy + tile_entropy) / n)
                self.logger.record("train/exploration_bonus", bonus / n)
                for head, value in (("kind", kind_bonus), ("plant", plant_bonus), ("tile", tile_bonus)):
                    self.logger.record(f"train/{head}_exploration_bonus", value / n)
                self.logger.record("train/choice_fraction", choice / n)
            results.append(
                {
                    "exact_kl": exact / choice if choice else 0.0,
                    "post_update_approx_kl": approx / len(buffer.observations.flatten(0, 1)),
                    "post_update_type_kl": types / len(buffer.observations.flatten(0, 1)),
                    "dig_probability_when_legal": dig / count if count else float("nan"),
                    "dig_legal_observations": count,
                    "dig_probability_sum": dig,
                }
            )
        return results

    def train(self):
        self._require_optimizer_protocol()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        critic_lr = self.policy.critic_learning_rate
        if critic_lr is None:
            critic_lr = self.lr_schedule(self._current_progress_remaining)
        update_learning_rate(self.policy.critic_optimizer, critic_lr)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = (
            self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        )
        window = getattr(self, "_actor_window", None)
        old_types = None if window else self._type_probabilities()
        actor_metrics, value_losses, actor_norms, critic_norms = [], [], [], []
        warming = getattr(self, "critic_warmup_active", False)
        actor_active = not warming and (window is None or window.state == ActorUpdateState.ACTIVE)
        actor_steps, critic_steps = 0, 0
        raw = self.rollout_buffer.advantages
        normalize = self.normalize_advantage and raw.numel() > 1
        center, scale = (raw.mean(), raw.std() + 1e-8) if normalize else (0, 1)
        kl_stopped = False
        actor_epochs = 0
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                if actor_active:
                    try:
                        log_prob, entropy = self.policy.evaluate_actor(
                            data.observations,
                            data.actions.long().flatten(),
                            data.action_masks,
                            **({"context": data.context} if hasattr(data, "context") else {}),
                        )
                    except ValueError:
                        if window is None:
                            raise
                        window.reject()
                        actor_active = False
                    if actor_active:
                        advantages = data.advantages
                        advantages = (advantages - center) / scale
                        ratio = torch.exp(log_prob - data.old_log_prob)
                        policy_loss = -torch.minimum(
                            advantages * ratio,
                            advantages * ratio.clamp(1 - clip_range, 1 + clip_range),
                        ).mean()
                        regularizer, exploration_metrics = exploration_loss(
                            self.policy, entropy, log_prob
                        )
                        actor_loss = policy_loss + regularizer
                        with torch.no_grad():
                            log_ratio = log_prob - data.old_log_prob
                            kl = (log_ratio.exp() - 1 - log_ratio).mean()
                            actor_metrics.append(
                                torch.stack(
                                    (
                                        policy_loss.detach(),
                                        -entropy.mean().detach(),
                                        ((ratio - 1).abs() > clip_range).float().mean(),
                                        exploration_metrics["joint_entropy"],
                                        exploration_metrics["exploration_bonus"],
                                        (data.action_masks.sum(-1) > 1).float().mean(),
                                        kl,
                                        actor_loss.detach(),
                                        exploration_metrics["kind_exploration_bonus"],
                                        exploration_metrics["plant_exploration_bonus"],
                                        exploration_metrics["tile_exploration_bonus"],
                                    )
                                )
                            )
                        if self.target_kl and (
                            not torch.isfinite(kl) or float(kl) > 1.5 * self.target_kl
                        ):
                            actor_active = False
                            kl_stopped = True
                            if window is not None:
                                window.stop()
                        else:
                            self.policy.optimizer.zero_grad(set_to_none=True)
                            actor_loss.backward()
                            actor_norms.append(
                                torch.nn.utils.clip_grad_norm_(
                                    self.policy.actor_parameters(), self.max_grad_norm
                                ).detach()
                            )
                            self.policy.optimizer.step()
                            actor_steps += 1
                # Critic updates cannot change the actor and continue after actor KL stopping.
                values = self.policy.predict_values(
                    data.observations,
                    **({"context": data.context} if hasattr(data, "context") else {}),
                ).flatten()
                predicted = (
                    values
                    if clip_vf is None
                    else data.old_values + (values - data.old_values).clamp(-clip_vf, clip_vf)
                )
                value_loss = F.mse_loss(data.returns, predicted)
                value_losses.append(value_loss.detach())
                self.policy.critic_optimizer.zero_grad(set_to_none=True)
                (self.vf_coef * value_loss).backward()
                critic_norms.append(
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.critic_parameters(), self.max_grad_norm
                    ).detach()
                )
                self.policy.critic_optimizer.step()
                critic_steps += 1
            actor_epochs += int(actor_active)
        self._n_updates += self.n_epochs
        returns, old_values = self.rollout_buffer.returns, self.rollout_buffer.values
        variance = returns.flatten().var(unbiased=False)
        explained = 1 - (returns - old_values).flatten().var(unbiased=False) / variance
        explained = torch.where(variance == 0, torch.full_like(explained, float("nan")), explained)
        actor = (
            torch.stack(actor_metrics).mean(0)
            if actor_metrics
            else torch.zeros(11, device=self.device)
        )
        value_mean = torch.stack(value_losses).mean()

        def mean_norm(norms):
            return torch.stack(norms).mean() if norms else torch.zeros((), device=self.device)

        numbers = (
            torch.cat(
                (
                    actor,
                    torch.stack(
                        (
                            value_mean,
                            explained,
                            mean_norm(actor_norms),
                            mean_norm(critic_norms),
                        )
                    ),
                )
            )
            .cpu()
            .tolist()
        )
        keys = (
            "policy_gradient_loss",
            "entropy_loss",
            "clip_fraction",
            "joint_entropy",
            "exploration_bonus",
            "choice_fraction",
            "approx_kl",
            "actor_loss",
            "kind_exploration_bonus",
            "plant_exploration_bonus",
            "tile_exploration_bonus",
            "value_loss",
            "explained_variance",
            "actor_grad_norm",
            "critic_grad_norm",
        )
        metrics = dict(zip(keys, numbers))
        targets = returns.double().flatten()
        residuals = targets - old_values.double().flatten()
        moments = (
            torch.stack(
                (
                    targets.sum(),
                    targets.square().sum(),
                    residuals.sum(),
                    residuals.square().sum(),
                )
            )
            .cpu()
            .tolist()
        )
        metrics.update(
            zip(
                ("return_sum", "return_squared_sum", "residual_sum", "residual_squared_sum"),
                moments,
            )
        )
        metrics.update(
            loss=metrics["actor_loss"] + self.vf_coef * metrics["value_loss"],
            optimizer_steps=actor_steps,
            actor_optimizer_steps=actor_steps,
            critic_optimizer_steps=critic_steps,
            epochs_completed=actor_epochs,
            critic_epochs_completed=self.n_epochs,
            kl_stopped=float(kl_stopped),
            critic_warmup=float(warming),
            critic_learning_rate=critic_lr,
            actor_metric_samples=(raw.numel() if warming else len(actor_metrics) * self.batch_size),
            critic_metric_samples=len(value_losses) * self.batch_size,
            value_sample_count=targets.numel(),
        )
        # Target residuals by decision type expose rare investment states hidden
        # by the aggregate fit. These are bootstrapped targets, not Monte Carlo truth.
        actions = self.rollout_buffer.actions.flatten()
        error = (returns - old_values).abs().flatten()
        for name, mask in (
            ("wait", actions == 0),
            ("plant", (actions > 0) & (actions < A.dig_start)),
            ("dig", actions >= A.dig_start),
        ):
            count, residual, advantage, positive = (
                torch.stack(
                    (
                        mask.sum(),
                        error[mask].sum(),
                        raw.flatten()[mask].sum(),
                        (raw.flatten()[mask] > 0).sum(),
                    )
                )
                .cpu()
                .tolist()
            )
            for key, value in (
                ("action_count", count),
                ("value_target_error_sum", residual),
                ("raw_advantage_sum", advantage),
                ("positive_advantage_count", positive),
                ("value_target_error", residual / count if count else float("nan")),
                ("raw_advantage_mean", advantage / count if count else float("nan")),
                ("positive_advantage_fraction", positive / count if count else float("nan")),
            ):
                metrics[f"{key}_{name}"] = value
        for head, key in (("kind", "type_coef"), ("plant", "plant_coef"), ("tile", "tile_coef")):
            metrics[f"effective_{head}_entropy_coef"] = self.policy.exploration_settings[key]
        for key, value in metrics.items():
            self.logger.record(f"train/{key}", value)
        if window is None:
            for key, value in self._post_update_metrics(old_types).items():
                self.logger.record(f"train/{key}", value)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_vf is not None:
            self.logger.record("train/clip_range_vf", clip_vf)

    def save(self, *args, **kwargs):
        if getattr(self, "pipeline_state", WindowState.IDLE) != WindowState.IDLE:
            raise PeriodicPipelineError("Cannot save an in-flight or failed pipeline window")
        return super().save(*args, **kwargs)

    def _excluded_save_params(self):
        return [
            *super()._excluded_save_params(),
            "_episode_memory",
            "_actor_window",
            "_checked_policy_metrics",
        ]

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        return [*state_dicts, "policy.critic_optimizer"], variables


class CudaMaskablePPO(TensorPPO, MaskablePPO):
    pass


def configure_tensor_buffer(model):
    model.rollout_buffer_class = TensorRolloutBuffer
    spec = model.policy.features_extractor.cfg["policy"]["memory"]
    model.rollout_buffer_kwargs = {"memory_spec": spec}
    model.rollout_buffer = TensorRolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        memory_spec=spec,
    )
