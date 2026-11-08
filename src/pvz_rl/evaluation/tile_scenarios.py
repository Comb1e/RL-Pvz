"""Frozen prerequisite probes, with CPU verification of selected placements."""

from contextlib import closing
from copy import deepcopy

import numpy as np
import torch

from pvz_rl.envs.cuda_env import CudaVecEnv
from pvz_rl.envs.env import PvZEnv
from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.sky_sun import peashooter_sky_sun_rate
from pvz_rl.envs.tile_scenarios import (
    ATTACKER_EXERCISES,
    LEGACY_SKILLS,
    SKILLS,
    attacker_mix,
    bomb_hit,
    is_bomb_skill,
    is_placement_skill,
    peashooter_level,
    shuffled_attacker_exercises,
)
from pvz_rl.evaluation.cuda_evaluation import deterministic_validation
from pvz_rl.learning.deadline import check_deadline
from pvz_rl.policy.runner import PolicyRunner


def _validation_groups(cfg, skills, seeds):
    for stage in skills:
        if stage != "attacker":
            yield stage, stage, list(enumerate(seeds))
            continue
        if len(seeds) != attacker_mix(cfg)["validation_cases"]:
            raise ValueError("Mixed attacker validation requires a full panel")
        assignments = shuffled_attacker_exercises(cfg, seeds, validation=True)
        for skill in ATTACKER_EXERCISES:
            yield (
                stage,
                skill,
                [(i, seeds[i]) for i, source in enumerate(assignments) if source == skill],
            )


def validate_tile_skills(cfg, model, seeds, *, deadline=None, progress=None, skills=LEGACY_SKILLS):
    results = {skill: [] for skill in skills}
    previous_scope = model.policy.execution_scope
    model.policy.execution_scope = "placement"
    try:
        return _validate_tile_skills(
            cfg, model, seeds, results, deadline=deadline, progress=progress, skills=skills
        )
    finally:
        model.policy.execution_scope = previous_scope


def _validate_tile_skills(cfg, model, seeds, results, *, deadline, progress, skills):
    with deterministic_validation(model):
        if (
            not skills
            or len(set(skills)) != len(skills)
            or any(skill not in SKILLS for skill in skills)
        ):
            raise ValueError("Select at least one distinct supported tile skill")
        for stage, skill, indexed_seeds in _validation_groups(cfg, skills, seeds):
            namespace = f"tile-validation/{stage}" + (f"/{skill}" if stage != skill else "")
            cases = [namespace_seed(namespace, seed) for _, seed in indexed_seeds]
            rows = results[stage]
            for offset in range(0, len(cases), cfg["training"]["n_envs"]):
                check_deadline(deadline)
                chunk = cases[offset : offset + cfg["training"]["n_envs"]]
                local = deepcopy(cfg)
                local["training"]["n_envs"] = len(chunk)
                with closing(CudaVecEnv(local, "masked", 0, training=False)) as env:
                    env.tile_scenario = env.queue.tile_scenario = skill
                    env.queue.deadline = deadline
                    if skill == "attacker_pre":
                        settings = local["training"]["tile_scenarios"]["attacker"]
                        kinds = settings["provided_plants"]
                        counts = list(
                            range(settings["zombie_count"][0], settings["zombie_count"][1] + 1)
                        )
                        env.queue.validation_cases = [
                            (
                                kinds[i % len(kinds)],
                                counts[(i // len(kinds) + i % len(kinds)) % len(counts)],
                            )
                            for i in range(offset, offset + len(chunk))
                        ]
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
                                supplied = env.supplied_branches(skill, availability)
                                actions, _, _, _details = runner.decide(
                                    obs,
                                    env.action_masks(),
                                    active=active,
                                    branch_usable=availability,
                                    supplied_branches=supplied,
                                )
                            obs, _, _, _, _, infos = env.step_tensors(actions, autoreset=False)
                            for index in np.flatnonzero(env._step_active):
                                traces[index].append(int(env._record[index, 6]))
                            for index, info in enumerate(infos):
                                if "episode_metrics" not in info:
                                    continue
                                check_deadline(deadline)
                                metrics = info["episode_metrics"]
                                details = {}
                                if skill == "attacker_pre":
                                    from pvz_rl.envs.attacker_scenarios import attacker_result

                                    action = traces[index][0]
                                    board = env.tile_boards[index]
                                    details = (
                                        attacker_result(
                                            local,
                                            board,
                                            (action - 1) % 45 if action else -1,
                                            rules=env.queue.rules,
                                            deadline=deadline,
                                        )
                                        if action
                                        else dict(
                                            win=False, target_correct=False, actual_kill=False
                                        )
                                    )
                                    success = details.pop("win")
                                elif is_bomb_skill(skill):
                                    action = traces[index][0]
                                    board = env.tile_boards[index]
                                    success = action > 0 and bomb_hit(
                                        board["snapshot"],
                                        (action - 1) % 45,
                                        board["target_id"],
                                        rules=env.queue.rules,
                                    )
                                else:
                                    with closing(
                                        PvZEnv(
                                            local,
                                            training=False,
                                            sky_sun_rate=peashooter_sky_sun_rate(local),
                                        )
                                    ) as cpu:
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
                                row = dict(
                                    skill=stage,
                                    seed=chunk[index],
                                    win=bool(success),
                                    selected_action=traces[index][0]
                                    if is_placement_skill(skill)
                                    else None,
                                    decisions=len(traces[index]),
                                    cpu_verified=True,
                                    **details,
                                )
                                if stage != skill:
                                    row["exercise"] = skill
                                rows.append((indexed_seeds[offset + index][0], row))
                            if progress:
                                progress.emit(
                                    f"Tile validation {stage}: {len(rows)}/{len(seeds)} complete"
                                )
    return {
        stage: [row for _, row in sorted(rows, key=lambda item: item[0])]
        for stage, rows in results.items()
    }
