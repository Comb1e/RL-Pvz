"""Tensor rollout storage, SB3's environment-major shuffle, and device GAE."""

import numpy as np
import torch
from sb3_contrib.common.maskable.buffers import MaskableRolloutBufferSamples
from stable_baselines3.common.buffers import BaseBuffer


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
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.gae_lambda, self.gamma = gae_lambda, gamma
        self.observations = torch.empty((buffer_size, n_envs, *self.obs_shape), device=self.device)
        self.actions = torch.empty((buffer_size, n_envs, self.action_dim), device=self.device)
        self.action_masks = torch.empty(
            (buffer_size, n_envs, action_space.n), dtype=torch.bool, device=self.device
        )
        for key in ("values", "log_probs", "rewards", "episode_starts", "advantages", "returns"):
            setattr(self, key, torch.empty((buffer_size, n_envs), device=self.device))
        self.reset()

    def reset(self):
        super().reset()
        self._flat = None
        self._timeouts = []

    def add(self, obs, action, reward, episode_start, value, log_prob, action_masks=None):
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
        self.action_masks[self.pos].copy_(action_masks)
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def add_timeouts(self, indices, terminal_observations):
        # Only timeout states need storage, never reset observations or natural terminals.
        self._timeouts.append((self.pos, indices.clone(), terminal_observations.clone()))

    @torch.no_grad()
    def evaluate_values(self, policy, last_observations, batch_size):
        if not self.full:
            raise RuntimeError("Evaluate values only for a completed rollout")

        def evaluate(observations):
            return torch.cat(
                [policy.predict_values(chunk).flatten() for chunk in observations.split(batch_size)]
            )

        observations = self.observations.flatten(0, 1)
        self.values.copy_(evaluate(observations).reshape_as(self.values))
        if self._timeouts:
            terminals = torch.cat([obs for _, _, obs in self._timeouts])
            terminal_values = evaluate(terminals)
            offset = 0
            for step, indices, obs in self._timeouts:
                size = len(obs)
                self.rewards[step, indices] += self.gamma * terminal_values[offset : offset + size]
                offset += size
            self._timeouts.clear()
        return evaluate(last_observations)

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
            delta = self.rewards[step] + self.gamma * next_value * nonterminal - self.values[step]
            last = delta + self.gamma * self.gae_lambda * nonterminal * last
            self.advantages[step].copy_(last)
        torch.add(self.advantages, self.values, out=self.returns)

    def get(self, batch_size=None):
        if not self.full:
            raise RuntimeError("Optimize only completed rollouts")
        # Draw exactly the same NumPy permutation as SB3; this is not a new RNG.
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        fields = ("observations", "actions", "values", "log_probs", "advantages", "returns")
        fields += ("action_masks",)
        if self._flat is None:
            self._flat = [
                getattr(self, k)
                .transpose(0, 1)
                .reshape(self.buffer_size * self.n_envs, *getattr(self, k).shape[2:])
                for k in fields
            ]
        size = self.buffer_size * self.n_envs if batch_size is None else batch_size
        sample = MaskableRolloutBufferSamples
        for start in range(0, len(indices), size):
            selection = torch.as_tensor(indices[start : start + size], device=self.device)
            yield sample(*(x[selection] for x in self._flat))

    def _get_samples(self, batch_inds, env=None):
        raise NotImplementedError("Use ordered rollout minibatches")
