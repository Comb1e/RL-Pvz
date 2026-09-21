"""Batched deterministic evaluation and traces verified by the CPU replay path."""

import copy
from time import perf_counter

import torch

from .cuda_diagnostics import DeviceProfiler
from .cuda_env import CudaVecEnv


def batched_games(cfg, policy, condition, seeds, levels, family, *, record=False, progress=None):
    """Yield complete numeric episode records in the original level/seed order.

    Recordings retain action indices in bounded device chunks, not game frames.
    CPU verification is performed by the shared evaluation writer before saving.
    """
    cases = [(level, family, seed) for level in levels for seed in sorted(seeds)]
    original_device = policy.policy.device
    policy.policy.to("cuda")
    try:
        for offset in range(0, len(cases), cfg["training"]["n_envs"]):
            chunk = cases[offset : offset + cfg["training"]["n_envs"]]
            local = copy.deepcopy(cfg)
            local["training"]["n_envs"] = len(chunk)
            # Evaluation batch size does not change a saved training configuration.
            local["training"]["rollout_size"] = len(chunk) * local["training"].get(
                "rollout_steps_per_env", 128
            )
            env = CudaVecEnv(local, condition, 0, family, training=False, cases=chunk)
            traces = [[] for _ in chunk]
            completed, buffered = {}, []
            inference_timer = DeviceProfiler(env.cp, enabled=True)
            started = perf_counter()
            try:
                obs = env.reset()
                with env.device_context():
                    while len(completed) < len(chunk):
                        with torch.no_grad():
                            kwargs = {}
                            if cfg["conditions"][condition]["masked"]:
                                masks = env.action_masks().clone()
                                # Completed slots are inactive; dummy wait is never
                                # sent to simulation as an active game operation.
                                masks[:, 0] |= env.header_tensor[:, 17] == 0
                                kwargs["action_masks"] = masks
                            with inference_timer.track("inference"):
                                actions, _, _ = policy.policy(obs, deterministic=True, **kwargs)
                        if record:
                            buffered.append(actions)
                        obs, _, _, _, _, infos = env.step_tensors(actions, autoreset=False)
                        # The compact completion transfer has already waited for
                        # inference. Bound event storage and measure execution,
                        # rather than reporting host enqueue time as GPU latency.
                        if len(inference_timer.pending) >= 128:
                            inference_timer.flush()
                        for i, info in enumerate(infos):
                            if "episode_metrics" in info:
                                completed[i] = {
                                    **info["episode_metrics"],
                                    "state_hash": env.batch.state_hash(i),
                                }
                        if record and (len(buffered) >= 128 or len(completed) == len(chunk)):
                            actions_cpu = torch.stack(buffered).cpu().numpy()
                            for i in range(len(chunk)):
                                traces[i].extend(actions_cpu[:, i].tolist())
                            buffered.clear()
                        if progress:
                            progress.emit(
                                f"GPU evaluation: {offset + len(completed)}/{len(cases)} games complete"
                            )
                inference = inference_timer.flush().get("inference", 0.0)
                for i in range(len(chunk)):
                    row = completed[i]
                    row["inference_seconds"] = inference / len(chunk)
                    row["wall_seconds"] = (perf_counter() - started) / len(chunk)
                    if record:
                        row["action_trace"] = traces[i][: row["decisions"]]
                    yield row
            finally:
                env.close()
    finally:
        policy.policy.to(original_device)
