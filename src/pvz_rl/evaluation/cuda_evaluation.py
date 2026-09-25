"""Batched deterministic evaluation and traces verified by the CPU replay path."""

import copy
from contextlib import contextmanager
from time import perf_counter

import torch

from pvz_rl.config import runtime_settings
from pvz_rl.envs.cuda_env import CudaVecEnv
from pvz_rl.learning.deadline import check_deadline
from pvz_rl.monitoring.cuda_diagnostics import DeviceProfiler
from pvz_rl.policy.event_memory import EventMemory


@contextmanager
def deterministic_validation(policy):
    """Validation never uses injected training exploration."""
    actor = policy.policy
    action_dist = getattr(actor, "action_dist", None)
    if action_dist is None:
        yield
        return
    old_epsilon = getattr(action_dist, "epsilon", 0.0)
    old_policy_epsilon = getattr(actor, "exploration_epsilon", old_epsilon)
    old_model_epsilon = getattr(policy, "exploration_rate", None)
    policy_kwargs = getattr(policy, "policy_kwargs", {})
    had_policy_kw_epsilon = "exploration_epsilon" in policy_kwargs
    old_policy_kw_epsilon = policy_kwargs.get("exploration_epsilon")
    action_dist.epsilon = 0.0
    actor.exploration_epsilon = 0.0
    if old_model_epsilon is not None:
        policy.exploration_rate = 0.0
    if had_policy_kw_epsilon:
        policy.policy_kwargs["exploration_epsilon"] = 0.0
    try:
        yield
    finally:
        action_dist.epsilon = old_epsilon
        actor.exploration_epsilon = old_policy_epsilon
        if old_model_epsilon is not None:
            policy.exploration_rate = old_model_epsilon
        if had_policy_kw_epsilon:
            policy.policy_kwargs["exploration_epsilon"] = old_policy_kw_epsilon


def batched_games(
    cfg, policy, condition, seeds, levels, family, *, record=False, progress=None, deadline=None
):
    with deterministic_validation(policy):
        if runtime_settings(cfg)["refill_evaluation"]:
            yield from refilled_games(
                cfg,
                policy,
                condition,
                seeds,
                levels,
                family,
                record=record,
                progress=progress,
                deadline=deadline,
            )
            return
        yield from fixed_batches(
            cfg,
            policy,
            condition,
            seeds,
            levels,
            family,
            record=record,
            progress=progress,
            deadline=deadline,
        )


def refilled_games(
    cfg, policy, condition, seeds, levels, family, *, record=False, progress=None, deadline=None
):
    """Refill finished GPU slots while preserving case order and exact traces."""
    cases = [(level, family, seed) for level in levels for seed in sorted(seeds)]
    if not cases:
        return
    check_deadline(deadline)
    count = min(len(cases), cfg["training"]["n_envs"])
    local = copy.deepcopy(cfg)
    local["training"]["n_envs"] = count
    original_device = policy.policy.device
    env = None
    policy.policy.to("cuda")
    try:
        env = CudaVecEnv(local, condition, 0, family, training=False, cases=cases[:count])
        slots = list(range(count))
        traces, buffered, completed = [[] for _ in slots], [], {}
        next_case, next_yield, finished = count, 0, 0
        timer = DeviceProfiler(env.cp, enabled=True)
        started = [perf_counter()] * count
        inference_start = [0.0] * count
        obs = env.reset()
        memory = EventMemory(cfg, env.batch.rules, count, policy.policy.device)
        previous = torch.zeros(count, device=policy.policy.device)
        resets = torch.ones(count, dtype=torch.bool, device=policy.policy.device)
        with env.device_context():
            while finished < len(cases):
                check_deadline(deadline)
                with torch.inference_mode():
                    kwargs = {}
                    context = memory.observe(
                        obs, env.action_masks(), previous, resets, env.header_tensor[:, 0]
                    )
                    if cfg["conditions"][condition]["masked"]:
                        masks = env.action_masks().clone()
                        masks[:, 0] |= env.header_tensor[:, 17] == 0
                        kwargs["action_masks"] = masks
                    with timer.track("inference"):
                        actions, _ = policy.policy.sample_actions(
                            obs, kwargs.get("action_masks"), deterministic=True, context=context
                        )
                if record:
                    buffered.append(actions)
                obs, _, _, _, _, infos = env.step_tensors(actions, autoreset=False)
                previous = env.executed_actions
                resets = torch.zeros_like(resets)
                done = [i for i, info in enumerate(infos) if "episode_metrics" in info]
                if record and (done or len(buffered) >= 128):
                    raw = torch.stack(buffered).cpu().numpy()
                    for i in range(count):
                        if slots[i] is not None:
                            traces[i].extend(raw[:, i].tolist())
                    buffered.clear()
                if done or len(timer.pending) >= 128:
                    inference = timer.flush().get("inference", 0.0)
                reset, assigned = [], []
                for i in done:
                    row = {
                        **infos[i]["episode_metrics"],
                        "state_hash": env.batch.state_hash(i),
                        "inference_seconds": (inference - inference_start[i]) / count,
                        "wall_seconds": perf_counter() - started[i],
                    }
                    if record:
                        row["action_trace"] = traces[i][: row["decisions"]]
                    completed[slots[i]] = row
                    finished += 1
                    slots[i], traces[i] = None, []
                    if next_case < len(cases):
                        slots[i] = next_case
                        reset.append(i)
                        assigned.append(cases[next_case])
                        next_case += 1
                        started[i], inference_start[i] = perf_counter(), inference
                if reset:
                    env.reset_indices(reset, assigned)
                    resets[reset] = True
                    previous[reset] = 0
                if progress:
                    progress.emit(f"GPU evaluation: {finished}/{len(cases)} games complete")
                while next_yield in completed:
                    yield completed.pop(next_yield)
                    next_yield += 1
    finally:
        if env is not None:
            env.close()
        policy.policy.to(original_device)


def fixed_batches(
    cfg, policy, condition, seeds, levels, family, *, record=False, progress=None, deadline=None
):
    """Yield complete numeric episode records in the original level/seed order.

    Recordings retain action indices in bounded device chunks, not game frames.
    CPU verification is performed by the shared evaluation writer before saving.
    """
    cases = [(level, family, seed) for level in levels for seed in sorted(seeds)]
    original_device = policy.policy.device
    policy.policy.to("cuda")
    try:
        for offset in range(0, len(cases), cfg["training"]["n_envs"]):
            check_deadline(deadline)
            chunk = cases[offset : offset + cfg["training"]["n_envs"]]
            local = copy.deepcopy(cfg)
            local["training"]["n_envs"] = len(chunk)
            # Evaluation batch size does not change a saved training configuration.

            env = CudaVecEnv(local, condition, 0, family, training=False, cases=chunk)
            traces = [[] for _ in chunk]
            completed, buffered = {}, []
            inference_timer = DeviceProfiler(env.cp, enabled=True)
            started = perf_counter()
            try:
                obs = env.reset()
                memory = EventMemory(cfg, env.batch.rules, len(chunk), policy.policy.device)
                previous = torch.zeros(len(chunk), device=policy.policy.device)
                resets = torch.ones(len(chunk), dtype=torch.bool, device=policy.policy.device)
                with env.device_context():
                    while len(completed) < len(chunk):
                        check_deadline(deadline)
                        with torch.inference_mode():
                            kwargs = {}
                            context = memory.observe(
                                obs, env.action_masks(), previous, resets, env.header_tensor[:, 0]
                            )
                            if cfg["conditions"][condition]["masked"]:
                                masks = env.action_masks().clone()
                                # Completed slots are inactive; dummy wait is never
                                # sent to simulation as an active game operation.
                                masks[:, 0] |= env.header_tensor[:, 17] == 0
                                kwargs["action_masks"] = masks
                            with inference_timer.track("inference"):
                                actions, _ = policy.policy.sample_actions(
                                    obs,
                                    kwargs.get("action_masks"),
                                    deterministic=True,
                                    context=context,
                                )
                        if record:
                            buffered.append(actions)
                        obs, _, _, _, _, infos = env.step_tensors(actions, autoreset=False)
                        previous = env.executed_actions
                        resets = torch.zeros_like(resets)
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
