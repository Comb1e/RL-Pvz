"""Frozen prerequisite probes, with CPU verification of selected placements."""

from contextlib import closing
from copy import deepcopy

import numpy as np
import torch

from pvz_rl.envs.cuda_env import CudaVecEnv
from pvz_rl.envs.env import PvZEnv
from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.tile_scenarios import SKILLS, bomb_hit, peashooter_level
from pvz_rl.evaluation.cuda_evaluation import deterministic_validation
from pvz_rl.learning.deadline import check_deadline
from pvz_rl.policy.runner import PolicyRunner


def validate_tile_skills(cfg, model, seeds, *, deadline=None, progress=None):
    results = {}
    with deterministic_validation(model):
        for skill in SKILLS:
            cases = [namespace_seed(f"tile-validation/{skill}", seed) for seed in seeds]
            rows = []
            for offset in range(0, len(cases), cfg["training"]["n_envs"]):
                check_deadline(deadline)
                chunk = cases[offset : offset + cfg["training"]["n_envs"]]
                local = deepcopy(cfg)
                local["training"]["n_envs"] = len(chunk)
                with closing(CudaVecEnv(local, "masked", 0, training=False)) as env:
                    env.tile_scenario = env.queue.tile_scenario = skill
                    env.queue.deadline = deadline
                    traces = [[] for _ in chunk]
                    with env.device_context():
                        env.reset_indices(list(range(len(chunk))), tile_seeds=chunk)
                        obs = env.features.obs_tensor
                        runner = PolicyRunner(model.policy)
                        while env.enabled_envs.any():
                            check_deadline(deadline)
                            with torch.inference_mode():
                                active = torch.as_tensor(
                                    env.enabled_envs.copy(), device=model.device
                                )
                                availability = env.branch_availability(active=active)
                                branch = cfg["environment"]["plants"].index(skill) + 1
                                supplied = torch.where(availability[:, branch], branch, 0).long()
                                actions = runner.decide(
                                    obs,
                                    env.action_masks(),
                                    active=active,
                                    branch_usable=availability,
                                    supplied_branches=supplied,
                                )[0]
                            obs, _, _, _, _, infos = env.step_tensors(actions, autoreset=False)
                            for index in np.flatnonzero(env._step_active):
                                traces[index].append(int(env._record[index, 6]))
                            for index, info in enumerate(infos):
                                if "episode_metrics" not in info:
                                    continue
                                check_deadline(deadline)
                                metrics = info["episode_metrics"]
                                if skill == "cherry_bomb":
                                    action = traces[index][0]
                                    board = env.tile_boards[index]
                                    success = action > 0 and bomb_hit(
                                        board["snapshot"],
                                        (action - 1) % 45,
                                        board["target_id"],
                                        rules=env.queue.rules,
                                    )
                                else:
                                    with closing(PvZEnv(local, training=False)) as cpu:
                                        cpu.reset(
                                            seed=chunk[index],
                                            options={
                                                "scenario": peashooter_level(
                                                    cpu.rules, chunk[index]
                                                ),
                                                "level": "easy",
                                                "family": "preset",
                                            },
                                        )
                                        for action in traces[index]:
                                            check_deadline(deadline)
                                            _, _, terminal, truncated, cpu_info = cpu.step(action)
                                        if (
                                            not (terminal or truncated)
                                            or cpu_info["episode_metrics"]["status"]
                                            != metrics["status"]
                                            or cpu.game.state_hash() != env.batch.state_hash(index)
                                        ):
                                            raise RuntimeError(
                                                "Tile prerequisite CUDA trace differs from CPU"
                                            )
                                    success = metrics["status"] == "won"
                                rows.append(
                                    dict(
                                        skill=skill,
                                        seed=chunk[index],
                                        win=bool(success),
                                        selected_action=traces[index][0]
                                        if skill == "cherry_bomb"
                                        else None,
                                        decisions=len(traces[index]),
                                        cpu_verified=True,
                                    )
                                )
                            if progress:
                                progress.emit(
                                    f"Tile validation {skill}: {len(rows)}/{len(cases)} complete"
                                )
            results[skill] = rows
    return results
