"""Faster data transport without changing PPO sampling or optimization."""

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib.common.maskable.buffers import (
    MaskableRolloutBuffer,
    MaskableRolloutBufferSamples,
)
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecEnvWrapper

MASK_INFO_KEY = "_pvz_action_mask"


class MaskTransport(gym.Wrapper):
    """Send the next observation's mask in the existing step/reset response."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info[MASK_INFO_KEY] = self.env.get_wrapper_attr("action_masks")()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            info[MASK_INFO_KEY] = self.env.get_wrapper_attr("action_masks")()
        # VecEnv immediately resets finished workers. Their next mask belongs to
        # reset_info, while terminal_observation remains intact for bootstrapping.
        return obs, reward, terminated, truncated, info


class CachedMaskVecEnv(VecEnvWrapper):
    """Serve SB3's mask queries locally; leave its spawned workers unchanged."""

    def __init__(self, venv):
        super().__init__(venv)
        self._masks = [None] * self.num_envs

    def reset(self):
        obs = self.venv.reset()
        self.reset_infos = self.venv.reset_infos
        self._masks = [info.pop(MASK_INFO_KEY) for info in self.reset_infos]
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.reset_infos = self.venv.reset_infos
        for i, (done, info) in enumerate(zip(dones, infos)):
            source = self.reset_infos[i] if done else info
            self._masks[i] = source.pop(MASK_INFO_KEY)
        return obs, rewards, dones, infos

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        selected = list(self._get_indices(indices))
        if method_name == "action_masks" and not method_args and not method_kwargs:
            missing = [i for i in selected if self._masks[i] is None]
            if missing:
                for i, mask in zip(missing, self.venv.env_method(method_name, indices=missing)):
                    self._masks[i] = mask
            return [self._masks[i].copy() for i in selected]
        if method_name not in ("set_progress", "set_curriculum_stage"):
            for i in selected:
                self._masks[i] = None
        return self.venv.env_method(method_name, *method_args, indices=selected, **method_kwargs)

    def set_attr(self, attr_name, value, indices=None):
        selected = list(self._get_indices(indices))
        for i in selected:
            self._masks[i] = None
        return self.venv.set_attr(attr_name, value, indices=selected)


class _DeviceSamples:
    """Keep one rollout on CUDA across all PPO epochs; use SB3's CPU path as-is.

    SB3 still computes GAE, flattens arrays, and draws each NumPy permutation.
    Only the location of minibatch indexing changes. No new RNG or arithmetic.
    """

    sample_type = RolloutBufferSamples
    sample_fields = ("observations", "actions", "values", "log_probs", "advantages", "returns")

    def reset(self):
        self._device_samples = None
        super().reset()

    def _get_samples(self, batch_inds, env=None):
        if self.device.type != "cuda":
            return super()._get_samples(batch_inds, env)
        if self._device_samples is None:
            data = []
            for name in self.sample_fields:
                value = getattr(self, name)
                if name in ("values", "log_probs", "advantages", "returns"):
                    value = value.flatten()
                if name == "actions" and self.sample_type is RolloutBufferSamples:
                    value = value.astype(np.float32, copy=False)
                data.append(self.to_torch(value))
            self._device_samples = tuple(data)
        indices = torch.as_tensor(batch_inds, dtype=torch.long, device=self.device)
        return self.sample_type(*(tensor[indices] for tensor in self._device_samples))


class DeviceRolloutBuffer(_DeviceSamples, RolloutBuffer):
    """Unmasked PPO rollout with reusable device tensors."""


class MaskableDeviceRolloutBuffer(_DeviceSamples, MaskableRolloutBuffer):
    """Masked PPO rollout with reusable device tensors, including action masks."""

    sample_type = MaskableRolloutBufferSamples
    sample_fields = (*_DeviceSamples.sample_fields, "action_masks")


def rollout_buffer_class(masked, enabled):
    if enabled:
        return MaskableDeviceRolloutBuffer if masked else DeviceRolloutBuffer
    return MaskableRolloutBuffer if masked else RolloutBuffer


def configure_rollout_buffer(model, masked, enabled):
    """Apply runtime preferences on resume without replacing policy or optimizer."""
    model.rollout_buffer_class = rollout_buffer_class(masked, enabled)
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )
