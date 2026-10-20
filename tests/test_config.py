"""The single stateless model contract and execution-only configuration controls."""

import pytest

from pvz_rl.config import load_config, load_demo_config, validate_config
from pvz_rl.learning.checkpoints import compatible_config, execution_config
from pvz_rl.learning.performance import refresh_performance, without_performance
from pvz_rl.learning.training_requirements import (
    current_model_config,
    transfer_protocol,
    weight_transfer_protocol,
)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("species_width", 128),
        ("species_layers", 4),
        ("species_heads", 8),
        ("species_feedforward", 512),
        ("tile_width", 192),
        ("tile_layers", 6),
        ("tile_heads", 12),
        ("tile_feedforward", 768),
    ],
)
def test_two_stage_dimensions_are_exact_checkpoint_contract(key, expected):
    cfg = load_config()
    assert cfg["policy"][key] == expected
    cfg["policy"][key] = expected + 1
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["history", "scalar_width", "lstm_hidden", "chunk_length"])
def test_policy_rejects_unused_history_and_scalar_fields(key):
    cfg = load_config()
    cfg["policy"][key] = 1
    with pytest.raises(ValueError, match="stateless two-stage"):
        validate_config(cfg)


def test_flat_batches_have_no_recurrent_divisibility_constraint():
    cfg = load_config()
    for size in (1, 17, 1023, 1024):
        cfg["training"]["batch_size"] = size
        validate_config(cfg)
    cfg["training"]["batch_size"] = 1025
    with pytest.raises(ValueError, match="1024"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [0, 17, True, 2.5, None])
def test_flat_batch_groups_reject_invalid_values(value):
    cfg = load_config()
    cfg["training"]["performance"]["fit_batch_groups"] = value
    with pytest.raises(ValueError, match="fit_batch_groups"):
        validate_config(cfg)


def test_flat_batch_groups_refresh_without_changing_learning():
    current, saved = load_config(), load_config()
    assert current["training"]["performance"]["fit_batch_groups"] == 4
    saved["training"]["performance"]["fit_batch_groups"] = 1
    saved["training"]["objective"]["species_consistency_weight"] = 2.0
    saved["training"]["objective"]["species_consistency_huber_delta"] = 0.25
    saved["training"]["n_epochs"] = 3
    refreshed = refresh_performance(saved, current)
    assert refreshed["training"]["performance"]["fit_batch_groups"] == 4
    assert refreshed["training"]["objective"]["species_consistency_weight"] == 2.0
    assert refreshed["training"]["objective"]["species_consistency_huber_delta"] == 0.25
    assert refreshed["training"]["n_epochs"] == 3
    saved["training"]["performance"]["fit_sequence_groups"] = 4
    with pytest.raises(ValueError, match="unsupported execution"):
        validate_config(saved)


@pytest.mark.parametrize("value", [0, -1, True, 1.0, None])
def test_invalid_fit_group_size_requires_positive_integer(value):
    cfg = load_config()
    cfg["training"]["objective"]["invalid_fit_group_size"] = value
    with pytest.raises(ValueError, match="invalid_fit_group_size"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [True, -1, 1.5, None, "entropy"])
def test_annotation_sampling_seed_rejects_invalid_values(value):
    cfg = load_config()
    cfg["training"]["annotation"]["sampling_seed"] = value
    with pytest.raises(ValueError, match="sampling_seed"):
        validate_config(cfg)


def test_annotation_sampling_seed_accepts_random_or_integer():
    cfg = load_config()
    assert cfg["training"]["annotation"]["sampling_seed"] == "random"
    validate_config(cfg)
    cfg["training"]["annotation"]["sampling_seed"] = 101
    validate_config(cfg)


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), None])
def test_consistency_weight_requires_finite_nonnegative_number(value):
    cfg = load_config()
    assert cfg["training"]["objective"]["species_consistency_weight"] == 1.0
    cfg["training"]["objective"]["species_consistency_weight"] = value
    with pytest.raises(ValueError, match="species_consistency_weight"):
        validate_config(cfg)


def test_double_estimator_defaults_and_storage_contracts():
    from pvz_rl.learning.checkpoints import (
        AUTONOMOUS_STATE_PROTOCOL,
        DEMO_PROTOCOL,
        MODEL_STATE_PROTOCOL,
        protocol_for,
    )
    from pvz_rl.learning.cuda_buffer import TRAJECTORY_PROTOCOL

    for cfg in (load_config(), load_demo_config()):
        objective = cfg["training"]["objective"]
        assert cfg["policy"]["kind"] == "two_stage_transformer_q_v1"
        assert cfg["training"]["method"] == "complete_return_action_global_v2"
        assert objective["protocol"] == "complete_return_action_global_v2"
        assert objective["invalid_fit_group_size"] == 100
        assert objective["species_consistency_huber_delta"] == 1.0
        assert objective["species_consistency_weight"] == 1.0
        assert current_model_config(cfg)
        assert protocol_for(cfg["policy"]["kind"])["optimizer"] == objective["protocol"]
    assert DEMO_PROTOCOL == "pvz-rl/demo-initialization-checkpoint-v10"
    assert AUTONOMOUS_STATE_PROTOCOL == "pvz-rl/action-global-state-v2"
    assert MODEL_STATE_PROTOCOL == "pvz-rl/stateless-two-stage-bf16-master-state-v2"
    assert TRAJECTORY_PROTOCOL == "pvz-rl/action-global-trajectory-v2"


@pytest.mark.parametrize(
    "value", [0, -1, True, False, float("nan"), float("inf"), -float("inf"), None, "1.0"]
)
def test_consistency_huber_delta_requires_finite_positive_nonboolean_number(value):
    cfg = load_config()
    cfg["training"]["objective"]["species_consistency_huber_delta"] = value
    with pytest.raises(ValueError, match="species_consistency_huber_delta"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [1, 0.25, 1e-12, 1e12])
def test_consistency_huber_delta_accepts_positive_return_unit_scales(value):
    cfg = load_config()
    cfg["training"]["objective"]["species_consistency_huber_delta"] = value
    validate_config(cfg)


def test_consistency_huber_delta_is_mandatory_without_hydration():
    compact = without_performance(load_demo_config())
    compact["training"]["objective"].pop("species_consistency_huber_delta")
    with pytest.raises(ValueError, match="species_consistency_huber_delta"):
        validate_config(execution_config(compact))


@pytest.mark.parametrize("field", ["method", "objective"])
@pytest.mark.parametrize("weights_only", [False, True])
def test_previous_objective_is_rejected_even_for_structurally_identical_weights(
    field, weights_only
):
    source, target = load_config(), load_config()
    if field == "method":
        source["training"]["method"] = "complete_return_sequential_tile_probe_v1"
    else:
        source["training"]["objective"]["protocol"] = "complete_return_sequential_tile_probe_v1"
    assert not current_model_config(source)
    with pytest.raises(ValueError, match="objective protocol|complete-return method"):
        validate_config(source)
    with pytest.raises(ValueError, match="model or objective is incompatible"):
        compatible_config(source, target, weights_only=weights_only)


def test_compact_demo_config_hydrates_only_execution_settings():
    cfg = load_demo_config()
    compact = without_performance(cfg)
    hydrated = execution_config(compact)
    validate_config(hydrated)
    assert hydrated == cfg
    assert "performance" not in compact["training"]
    assert compact["policy"]["tile_width"] == 192
    assert "encoder_microbatch" not in compact["policy"]
    for signature in (transfer_protocol(compact), weight_transfer_protocol(compact)):
        assert "event_history" not in signature
        assert set(signature["policy"]) == {
            "kind",
            "species_width",
            "species_layers",
            "species_heads",
            "species_feedforward",
            "tile_width",
            "tile_layers",
            "tile_heads",
            "tile_feedforward",
        }


def test_objective_is_mandatory():
    cfg = load_config()
    cfg["training"].pop("objective")
    with pytest.raises(ValueError, match="training.objective"):
        validate_config(cfg)
