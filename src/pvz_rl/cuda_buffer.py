"""Tensor rollout storage, SB3's environment-major shuffle, and device GAE."""

from types import SimpleNamespace

import numpy as np
import torch
from sb3_contrib.common.maskable.buffers import MaskableRolloutBufferSamples
from stable_baselines3.common.buffers import BaseBuffer

from .event_memory import MemoryContext


class TensorRolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="cuda",
        gae_lambda=1,
        gamma=0.99,
        n_envs=1,
        memory_spec=None,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.memory_spec = memory_spec
        self.gae_lambda, self.gamma = gae_lambda, gamma
        self.observations = torch.empty((buffer_size, n_envs, *self.obs_shape), device=self.device)
        self.actions = torch.empty((buffer_size, n_envs, self.action_dim), device=self.device)
        self.action_masks = torch.empty(
            (buffer_size, n_envs, action_space.n), dtype=torch.bool, device=self.device
        )
        for key in ("values", "log_probs", "rewards", "episode_starts", "advantages", "returns"):
            setattr(self, key, torch.empty((buffer_size, n_envs), device=self.device))
        self.durations = torch.ones((buffer_size, n_envs), dtype=torch.int64, device=self.device)
        self.reset()

    def reset(self):
        super().reset()
        self._flat = None
        self._timeouts = []
        self.memory_archive = None
        self._burn_contexts = {}
        self._step_index_templates = {}
        self.has_behavior_logits = False

    def capture_behavior(self, distribution):
        """Copy frozen collector logits; never share its mutable distribution."""
        if not hasattr(self, "behavior_logits"):
            self.behavior_logits = distribution.logits.new_empty(
                self.buffer_size, self.n_envs, distribution.logit_dim
            )
        self.behavior_logits[self.pos].copy_(distribution.logits)
        self.behavior_epsilon = distribution.epsilon
        self.has_behavior_logits = True

    def add(
        self,
        obs,
        action,
        reward,
        episode_start,
        value,
        log_prob,
        action_masks=None,
        *,
        durations=None,
    ):
        for key, value in (
            ("observations", obs),
            ("actions", action.reshape(-1, self.action_dim)),
            ("rewards", reward),
            ("episode_starts", episode_start),
            ("values", None if value is None else value.flatten()),
            ("log_probs", log_prob.flatten()),
        ):
            if value is not None:
                getattr(self, key)[self.pos].copy_(value)
        if durations is None:
            self.durations[self.pos].fill_(1)
        else:
            self.durations[self.pos].copy_(durations)
        self.action_masks[self.pos].copy_(action_masks)
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def start_memory(self, memory):
        """Archive each raw token once; store small per-transition lookup tables."""
        capacity, width = memory.capacity, memory.width
        initial = self.n_envs * capacity
        if not hasattr(self, "_memory_storage"):
            self._memory_storage = torch.empty(
                (initial + self.buffer_size * self.n_envs, width), device=self.device
            )
        self.memory_archive = self._memory_storage
        self.memory_archive[:initial].copy_(memory.tokens.flatten(0, 1))
        memory.ids.copy_(torch.arange(initial, device=self.device).reshape_as(memory.ids))
        self.archive_offset = initial
        shape = (self.buffer_size, self.n_envs, capacity)
        if not hasattr(self, "context_ids"):
            self.context_ids = torch.empty(shape, dtype=torch.long, device=self.device)
            self.context_valid = torch.empty(shape, dtype=torch.bool, device=self.device)
            self.context_counts = torch.empty(shape, device=self.device)
            self.context_starts = torch.empty(shape, device=self.device)
            self.context_events = torch.empty(
                (*shape[:2], memory.local), dtype=torch.bool, device=self.device
            )
        self.memory_cfg, self.memory_rules = memory.cfg, memory.layout.rules

    def capture_context(self, memory):
        start = self.archive_offset
        end = start + self.n_envs
        self.memory_archive[start:end].copy_(memory.tokens[:, memory.local - 1])
        memory.ids[:, memory.local - 1] = torch.arange(start, end, device=self.device)
        self.archive_offset = end
        for name, source in (
            ("ids", memory.ids),
            ("valid", memory.valid),
            ("counts", memory.counts),
            ("starts", memory.starts),
        ):
            getattr(self, "context_" + name)[self.pos].copy_(source)
        self.context_events[self.pos].copy_(memory.event_flags)

    @torch.no_grad()
    def burn_context(self, envs, start):
        """Replay a chunk's public prefix from an exact retained-history boundary.

        There are no learned cached activations: burn-in reconstructs admission
        and compression, then both current encoders re-encode the same raw bank.
        Prefix transitions never contribute PPO losses.
        """
        # Admission/compression has no learned parameters. A boundary is identical
        # across epochs and minibatch permutations; reconstruct it once for all
        # environments, then reuse the raw bank with each encoder's current weights.
        if start not in self._burn_contexts:
            indices = torch.arange(self.n_envs, device=self.device)
            self._burn_contexts[start] = self._reconstruct_context(indices, start)
        return self._burn_contexts[start].select(envs)

    def _reconstruct_context(self, envs, start):
        from .event_memory import EventMemory

        begin = max(0, start - self.memory_spec["burn_in"])
        ix = begin * self.n_envs + envs
        context = self.context(ix)
        if begin == start:
            return context
        memory = EventMemory(self.memory_cfg, self.memory_rules, len(envs), self.device)
        for key in ("tokens", "valid", "counts", "starts"):
            getattr(memory, key).copy_(getattr(context, key))
        memory.event_flags.copy_(self.context_events[begin, envs])
        memory.last_obs.copy_(self.observations[begin, envs])
        memory.last_mask.copy_(self.action_masks[begin, envs])
        memory.have_previous.fill_(True)
        for step in range(begin + 1, start + 1):
            current = self.context(step * self.n_envs + envs).tokens[:, memory.local - 1]
            memory.observe(
                self.observations[step, envs],
                self.action_masks[step, envs],
                current[:, -3],
                current[:, -2],
                current[:, -1],
            )
        return memory.context()

    def context(self, indices):
        if self.memory_archive is None:
            return None
        ids = self.context_ids.flatten(0, 1)[indices]
        return MemoryContext(
            self.memory_archive[ids],
            *(
                getattr(self, "context_" + key).flatten(0, 1)[indices]
                for key in ("valid", "counts", "starts")
            ),
        )

    def add_timeouts(self, indices, terminal_observations, context=None):
        self._timeouts.append(
            (
                self.pos,
                indices.clone(),
                terminal_observations.clone(),
                context.clone() if context is not None else None,
            )
        )

    @torch.inference_mode()
    def evaluate_values(self, policy, last_observations, batch_size, last_context=None):
        if not self.full:
            raise RuntimeError("Evaluate values only for a completed rollout")

        def evaluate(observations, context_at):
            values = []
            for i in range(0, len(observations), batch_size):
                ix = slice(i, i + batch_size)
                context = context_at(ix)
                kwargs = {} if context is None else {"context": context}
                values.append(policy.predict_values(observations[ix], **kwargs).flatten())
            return torch.cat(values)

        self.values.copy_(
            evaluate(self.observations.flatten(0, 1), self.context).reshape_as(self.values)
        )
        for step, indices, obs, context in self._timeouts:
            terminal_values = evaluate(
                obs, lambda ix: context.select(ix) if context is not None else None
            )
            self.rewards[step, indices] += (
                self.gamma ** self.durations[step, indices] * terminal_values
            )
        self._timeouts.clear()
        return evaluate(
            last_observations,
            lambda ix: last_context.select(ix) if last_context is not None else None,
        )

    def compute_returns_and_advantage(self, last_values, dones):
        if self.device.type == "cuda":
            from importlib.resources import files

            from pvz_game.cuda.backend import cupy_runtime

            cp = cupy_runtime()
            if not hasattr(self, "_gae_kernel"):
                self._gae_kernel = cp.RawKernel(
                    files("pvz_rl").joinpath("cuda_gae.cu").read_text("utf-8"),
                    "gae",
                    options=("--fmad=false",),
                )
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                arrays = (
                    self.rewards,
                    self.values,
                    self.episode_starts,
                    self.durations,
                    last_values.flatten().contiguous(),
                    dones.float(),
                    self.advantages,
                    self.returns,
                )
                views = [cp.from_dlpack(x.detach()) for x in arrays]
                self._gae_kernel(
                    ((self.n_envs + 127) // 128,),
                    (128,),
                    (
                        *views,
                        np.int32(self.buffer_size),
                        np.int32(self.n_envs),
                        np.float32(self.gamma),
                        np.float32(self.gae_lambda),
                    ),
                )
            return
        last = torch.zeros(self.n_envs, device=self.device)
        for step in reversed(range(self.buffer_size)):
            nonterminal = 1.0 - (
                dones.float() if step == self.buffer_size - 1 else self.episode_starts[step + 1]
            )
            next_value = (
                last_values.flatten() if step == self.buffer_size - 1 else self.values[step + 1]
            )
            discount = self.gamma ** self.durations[step]
            trace = (self.gamma * self.gae_lambda) ** self.durations[step]
            delta = self.rewards[step] + discount * next_value * nonterminal - self.values[step]
            last = delta + trace * nonterminal * last
            self.advantages[step].copy_(last)
        torch.add(self.advantages, self.values, out=self.returns)

    def get(self, batch_size=None):
        if not self.full:
            raise RuntimeError("Optimize only completed rollouts")
        fields = (
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "action_masks",
        )
        names = (
            "observations",
            "actions",
            "old_values",
            "old_log_prob",
            "advantages",
            "returns",
            "action_masks",
        )
        size = self.buffer_size * self.n_envs if batch_size is None else batch_size
        if self.memory_spec is None or self.memory_archive is None:
            # Independent feed-forward mathematical controls use upstream ordering.
            flat = [getattr(self, k).transpose(0, 1).flatten(0, 1) for k in fields]
            indices = np.random.permutation(self.buffer_size * self.n_envs)
            for start in range(0, len(indices), size):
                selection = torch.as_tensor(indices[start : start + size], device=self.device)
                yield MaskableRolloutBufferSamples(*(x[selection] for x in flat))
            return
        # Shuffle chunks, never transitions within a chunk. Episode boundaries
        # are encoded in each causal context and cannot leak into a new game.
        length = min(self.memory_spec["sequence_length"], self.buffer_size, size)
        per_batch = max(1, size // length)
        chunks = []
        for start in range(0, self.buffer_size, length):
            environments = np.random.permutation(self.n_envs)
            chunks.extend(
                (start, environments[i : i + per_batch]) for i in range(0, self.n_envs, per_batch)
            )
        order = np.random.permutation(len(chunks))
        flat = [getattr(self, k).flatten(0, 1) for k in fields]
        for chunk in order:
            start, environments = chunks[chunk]
            end = min(start + length, self.buffer_size)
            length = end - start
            key = (start, length)
            offsets = self._step_index_templates.get(key)
            if offsets is None:
                offsets = torch.arange(
                    start * self.n_envs,
                    end * self.n_envs,
                    self.n_envs,
                    device=self.device,
                    dtype=torch.long,
                )
                self._step_index_templates[key] = offsets
            environment_ids = torch.as_tensor(
                environments, device=self.device, dtype=torch.long
            )
            # Preserve the existing environment-major ordering while avoiding a
            # Python list and a full host-to-device index transfer per chunk.
            selection = (environment_ids[:, None] + offsets[None, :]).flatten()
            context = self.context(selection)
            prefix = self.burn_context(environment_ids, start)
            for key in ("tokens", "valid", "counts", "starts"):
                getattr(context, key)[:: end - start].copy_(getattr(prefix, key))
            yield SimpleNamespace(**dict(zip(names, (x[selection] for x in flat))), context=context)

    def _get_samples(self, batch_inds, env=None):
        raise NotImplementedError("Use ordered rollout minibatches")
