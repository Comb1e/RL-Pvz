"""Fresh stateless checkpoints and early rejection of incompatible evidence."""

import copy

import pytest
import torch
from pvz_game import Rules

from pvz_rl.config import digest, load_config, load_demo_config
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.learning.checkpoints import (
    AUTONOMOUS_STATE_PROTOCOL,
    DEMO_PROTOCOL,
    MODEL_STATE_PROTOCOL,
    inspect_checkpoint,
    validate_runtime_identity,
)
from pvz_rl.learning.performance import without_performance


@pytest.fixture
def demo_payload():
    from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy

    cfg = load_demo_config()
    with torch.random.fork_rng(devices=[]):
        policy = TwoStageTransformerPolicy(cfg)
    compact = without_performance(cfg)
    return dict(
        protocol=DEMO_PROTOCOL,
        model_storage_protocol=MODEL_STATE_PROTOCOL,
        precision_contract="bf16_master",
        config=compact,
        config_digest=digest(compact),
        model=policy.state_dict(),
        observation_schema=ObservationEncoder(cfg, Rules()).schema(),
        passes_completed=1,
        learner_seed=101,
    )


def test_fresh_demo_checkpoint_loads_complete_weights_without_old_execution(tmp_path, demo_payload):
    from pvz_rl.learning.training import initial_weights

    path = tmp_path / "initialization.pt"
    torch.save(demo_payload, path)
    saved_bytes = path.read_bytes()
    inspected = inspect_checkpoint(path)
    assert inspected["config"] == load_demo_config()
    cfg = load_config()
    cfg["reward"]["win_reward"] = 3.0
    weights, metadata = initial_weights(path, cfg)
    assert set(weights) == set(demo_payload["model"])
    assert all(torch.equal(weights[name], value) for name, value in demo_payload["model"].items())
    assert metadata["conversion"] == "none"
    assert metadata["updates"] == metadata["steps"] == metadata["games"] == 0
    assert path.read_bytes() == saved_bytes


@pytest.mark.parametrize("fault", ["model_protocol", "digest", "schema", "nan", "passes"])
def test_demo_rejects_incomplete_protocol_and_invalid_weights(tmp_path, demo_payload, fault):
    payload = copy.deepcopy(demo_payload)
    if fault == "model_protocol":
        payload.pop("model_storage_protocol")
    elif fault == "digest":
        payload["config"]["policy"]["tile_width"] = 96
    elif fault == "schema":
        payload["observation_schema"] = {}
    elif fault == "nan":
        next(iter(payload["model"].values())).reshape(-1)[0] = float("nan")
    else:
        payload["passes_completed"] = 0
    path = tmp_path / "incomplete.pt"
    torch.save(payload, path)
    for weights_only in (False, True):
        with pytest.raises(ValueError):
            inspect_checkpoint(path, weights_only=weights_only)


@pytest.mark.parametrize("fault", ["old_autonomous", "missing_model", "hidden", "teacher_hidden"])
def test_runtime_has_no_recurrent_state_and_requires_both_protocols(fault):
    state = dict(
        protocol=AUTONOMOUS_STATE_PROTOCOL,
        model_storage_protocol=MODEL_STATE_PROTOCOL,
        precision_contract="bf16_master",
    )
    validate_runtime_identity(state)
    if fault == "old_autonomous":
        state["protocol"] = "pvz-rl/tile-probe-event-lstm-state-v5"
    elif fault == "missing_model":
        state.pop("model_storage_protocol")
    elif fault == "hidden":
        state["memory"] = {}
    else:
        state["teacher_memory"] = {}
    with pytest.raises(ValueError, match="protocol|learned-history"):
        validate_runtime_identity(state)


def test_previous_objective_weights_load_for_inference_but_not_resume(tmp_path, demo_payload):
    from pvz_rl.learning.training import initial_weights, load_policy

    payload = copy.deepcopy(demo_payload)
    cfg = payload["config"]
    cfg["training"]["method"] = "complete_return_sequential_no_probe_v1"
    cfg["training"]["objective"]["protocol"] = cfg["training"]["method"]
    cfg["training"]["objective"].pop("invalid_action_target")
    cfg["reward"].update(invalid_plant_penalty=0.00001, empty_dig_penalty=0.000003)
    payload["config_digest"] = digest(cfg)
    path = tmp_path / "old-objective.pt"
    torch.save(payload, path)
    weights, _ = initial_weights(path, load_config())
    assert all(torch.equal(value, payload["model"][key]) for key, value in weights.items())
    model, _ = load_policy(path, device="cpu")
    assert model.cfg["training"]["method"] == "complete_return_action_global_v2"
    with pytest.raises(ValueError, match="objective"):
        inspect_checkpoint(path)


def test_gpu_execution_settings_never_enter_annotation_weight_compatibility(tmp_path, demo_payload):
    from pvz_rl.learning.checkpoints import compatible_config
    from pvz_rl.learning.training import initial_weights

    path = tmp_path / "portable.pt"
    torch.save(demo_payload, path)
    target = load_config()
    target["training"]["performance"].update(
        fit_species_microbatch=16,
        fit_tile_microbatch=8,
        fit_token_budget=4096,
        compile_kernels=False,
    )
    target["training"]["torch_threads"] = 2
    compatible_config(demo_payload["config"], target, weights_only=True)
    weights, _ = initial_weights(path, target)
    assert weights.keys() == demo_payload["model"].keys()
