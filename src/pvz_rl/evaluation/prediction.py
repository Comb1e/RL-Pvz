"""Frozen-policy validation of exact pre-action forecasts on a saved scene panel."""

from contextlib import closing
from copy import deepcopy
from math import isfinite

import torch

from pvz_rl.envs.cuda_env import CudaVecEnv
from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.envs.tile_scenarios import prediction_settings
from pvz_rl.learning.deadline import check_deadline
from pvz_rl.learning.forecast_labels import outcome_codes, scene_class_matches
from pvz_rl.learning.forecast_objective import scene_outcome_loss
from pvz_rl.policy.two_stage_transformer import execution_scope_for


@torch.no_grad()
def validate_prediction(cfg, model, seeds, *, deadline=None):
    settings = prediction_settings(cfg)
    if len(seeds) != settings["validation_cases"] or len(set(seeds)) != len(seeds):
        raise ValueError("Prediction validation requires a complete, distinct scene panel")
    policy = model.policy
    previous_stage = getattr(policy, "forecast_stage", None)
    previous_scope = policy.execution_scope
    policy.forecast_stage = "prediction"
    policy.execution_scope = execution_scope_for("prediction")
    results = []
    try:
        local = deepcopy(cfg)
        local["training"]["n_envs"] = len(seeds)
        check_deadline(deadline)
        with closing(CudaVecEnv(local, "masked", 0, training=False)) as env, env.device_context():
            env.tile_scenario = env.queue.tile_scenario = "prediction"
            env.queue.deadline = deadline
            slots = list(range(len(seeds)))
            env.reset_indices(
                slots, tile_seeds=[namespace_seed("prediction-validation", s) for s in seeds]
            )
            observation = env.features.obs_tensor
            policy.encode_board(observation)
            logits = policy._effect_logits
            proposed_host = [0] * len(seeds)
            service = env.fork_service()
            service.begin(slots)
            while True:
                check_deadline(deadline)
                if service.advance():
                    break
            labels = outcome_codes(service.take(), reference_tau(env.batch.rules))
            targets = torch.as_tensor(labels, device=policy.device, dtype=torch.long)
            matched = scene_class_matches(logits, targets)
            losses = scene_outcome_loss(logits, targets).cpu().tolist()
            correct_terms = matched.sum((-2, -1)).cpu().tolist()
            passed = matched.all(-1).all(-1).cpu().tolist()
            for seed, proposal, loss, terms, win in zip(
                seeds,
                proposed_host,
                losses,
                correct_terms,
                passed,
                strict=True,
            ):
                results.append(
                    dict(
                        seed=seed,
                        proposal=proposal,
                        effective_action=0,
                        loss=loss if isfinite(loss) else None,
                        correct_terms=int(terms),
                        win=bool(win),
                    )
                )
    finally:
        policy.forecast_stage = previous_stage
        policy.execution_scope = previous_scope
    return results
