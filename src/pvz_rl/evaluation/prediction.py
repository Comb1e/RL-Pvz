"""Frozen-policy validation of exact post-action forecasts on a saved scene panel."""

from contextlib import closing
from copy import deepcopy
from math import isfinite

import torch

from pvz_rl.envs.cuda_env import CudaVecEnv
from pvz_rl.envs.scenarios import namespace_seed
from pvz_rl.envs.static_fork import reference_tau
from pvz_rl.envs.tile_scenarios import prediction_settings
from pvz_rl.learning.deadline import check_deadline
from pvz_rl.learning.forecast_labels import outcome_codes
from pvz_rl.learning.forecast_objective import scene_outcome_loss
from pvz_rl.policy.runner import PolicyRunner


def scene_passes(loss, maximum):
    return isfinite(loss) and loss <= maximum


@torch.no_grad()
def validate_prediction(cfg, model, seeds, *, deadline=None):
    settings = prediction_settings(cfg)
    if len(seeds) != settings["validation_cases"] or len(set(seeds)) != len(seeds):
        raise ValueError("Prediction validation requires a complete, distinct scene panel")
    policy = model.policy
    previous_stage = getattr(policy, "forecast_stage", None)
    previous_injection = policy.injection_enabled
    policy.forecast_stage, policy.injection_enabled = "prediction", False
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
            proposals, _, _, details = PolicyRunner(policy).decide(
                observation, env.action_masks(), branch_usable=env.branch_availability()
            )
            effective = env.effective_actions(proposals, env.engine_action_masks_device())
            logits = policy.effect_action_logits(effective, details["board_memory"])
            policy._effect_decision_cache = None
            proposed_host, effective_host = proposals.cpu().tolist(), effective.cpu().numpy()
            service = env.fork_service()
            service.begin(slots, effective_host)
            while True:
                check_deadline(deadline)
                if service.advance():
                    break
            labels = outcome_codes(service.take(), reference_tau(env.batch.rules))
            targets = torch.as_tensor(labels, device=policy.device, dtype=torch.long)
            losses = scene_outcome_loss(logits, targets).cpu().tolist()
            for seed, proposal, action, loss in zip(
                seeds, proposed_host, effective_host, losses, strict=True
            ):
                results.append(
                    dict(
                        seed=seed,
                        proposal=proposal,
                        effective_action=int(action),
                        loss=loss if isfinite(loss) else None,
                        win=scene_passes(loss, settings["validation_max_loss"]),
                    )
                )
    finally:
        policy.forecast_stage, policy.injection_enabled = previous_stage, previous_injection
        policy._effect_decision_cache = None
    return results
