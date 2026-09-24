"""Independent controls for the one compact observation and reward contract."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from pvz_game import Dig, LevelSpec, Place, Spawn, Status
from pvz_game.config import PLANT_TYPES, InitialPlant
from pvz_game.types import Event

from pvz_rl.config import load_config, validate_config
from pvz_rl.env import PvZEnv
from pvz_rl.rewards import asset_value, reward_parts
from pvz_rl.spatial_policy import SpatialFeatures


@pytest.mark.parametrize("kind", PLANT_TYPES)
def test_purchase_conserves_assets_and_dig_removes_remaining_value(kind):
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        seed=4, options={"scenario": LevelSpec("value", (Spawn(999, "basic", 0),), initial_sun=900)}
    )
    _, buy, _, _, _ = env.step(env.codec.encode(Place(kind, 2, 3)))
    assert env.public.tick == 0 and asset_value(env.public) == 900 and buy == 0
    _, dig, _, _, _ = env.step(env.codec.encode(Dig(2, 3)))
    cost = env.rules.plants[kind]["cost"]
    assert asset_value(env.public) == 900 - cost
    assert dig == pytest.approx(-cost / 3000)


def test_independent_damage_death_terminal_and_timeout_formula():
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        seed=3,
        options={
            "scenario": LevelSpec(
                "values",
                (Spawn(999, "basic", 0),),
                initial_sun=600,
                plants=(InitialPlant("peashooter", 0, 0),),
            )
        },
    )
    before = env.public
    assert asset_value(before) == 700
    damaged = replace(
        before, plants=(replace(before.plants[0], health=before.plants[0].max_health // 2),)
    )
    assert asset_value(damaged) == 650
    assert reward_parts(before, damaged, cfg)["total"] == pytest.approx(-50 / 3000)
    after = replace(damaged, plants=())
    assert reward_parts(damaged, after, cfg)["total"] == pytest.approx(-50 / 3000)
    assert asset_value(after) == 600
    for status, terminal in ((Status.WON, 1), (Status.LOST, -2)):
        end = replace(after, status=status)
        assert asset_value(end) == 600
        assert reward_parts(damaged, end, cfg)["total"] == pytest.approx(terminal - 50 / 3000)


@pytest.mark.parametrize("sun", [0, 50, 300, 600, 9990])
@pytest.mark.parametrize("kills", [0, 1, 4])
def test_mower_cost_once_independent_of_sun_and_kills(sun, kills):
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(seed=1)
    before = replace(env.public, sun=sun)
    events = [Event("MowerActivated", 1, details=(("row", 0),))]
    for i in range(kills):
        events.extend(
            (Event("DamageApplied", 1, i, (("source", -1),)), Event("ZombieDefeated", 1, i))
        )
    parts = reward_parts(before, before, cfg, events=events)
    assert parts["mower_activation_penalty"] == pytest.approx(-0.2)
    assert parts["mower_kills"] == kills
    assert parts["total"] == pytest.approx(-0.2)
    assert reward_parts(before, before, cfg)["mower_activation_penalty"] == 0


def test_empty_explosion_and_nut_bite_are_only_diagnostics():
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        seed=3,
        options={
            "scenario": LevelSpec(
                "nut", (Spawn(1000, "basic", 0),), plants=(InitialPlant("wall_nut", 0, 0),)
            )
        },
    )
    before = env.public
    events = (
        Event("PlantDamaged", 1, before.plants[0].id, (("damage", 25),)),
        Event("PlantExploded", 1, 99),
    )
    part = reward_parts(before, before, cfg, events=events)
    assert part["wall_nut_damage"] == 25 and part["empty_explosions"] == 1
    assert part["total"] == 0


def test_categorical_boundaries_and_empty_embeddings():
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        seed=3,
        options={
            "scenario": LevelSpec(
                "types",
                (Spawn(1000, "basic", 0),),
                plants=tuple(InitialPlant(kind, 0, i) for i, kind in enumerate(PLANT_TYPES)),
            )
        },
    )
    obs = env.encoder.encode(env.public)
    assert obs.shape == (281,)
    grid = obs[:135].reshape(5, 9, 3)
    np.testing.assert_array_equal(grid[0, :8, 0], np.arange(1, 9))
    assert not grid[1:].any()
    for state, index in env.encoder.plant_states.items():
        public = replace(env.public, plants=(replace(env.public.plants[0], state=state),))
        assert env.encoder.encode(public)[2] == index + 1
    features = SpatialFeatures(env.observation_space, cfg)
    assert torch.count_nonzero(features.plant_types.weight[0]) == 0
    assert torch.count_nonzero(features.plant_states.weight[0]) == 0
    features(torch.tensor(obs).unsqueeze(0)).sum().backward()
    assert torch.count_nonzero(features.plant_types.weight.grad[0]) == 0
    assert torch.count_nonzero(features.plant_states.weight.grad[0]) == 0


def test_profile_is_label_and_numeric_controls_are_flexible():
    cfg = load_config()
    cfg["profile"] = "my-experiment"
    cfg["training"].update(
        learning_rate=1e-5,
        target_kl=0,
        gae_lambda=0.7,
        clip_range=0.1,
        vf_coef=0.8,
        normalize_advantage=False,
    )
    cfg["reward"].update(mower_value=3, progress_weight=0)
    cfg["training"]["gamma"] = 0.9
    cfg["policy"].update(plant_embedding=3, state_embedding=2, channels=[16, 24], scalar_sizes=[24])
    validate_config(cfg)


@pytest.mark.parametrize("discount", [0, 1])
@pytest.mark.parametrize("gae_lambda", [0, 1])
def test_discount_and_gae_endpoints_are_valid(discount, gae_lambda):
    from pvz_rl.training_requirements import require_cuda_training

    cfg = load_config()
    cfg["training"]["gamma"] = discount
    cfg["training"]["gae_lambda"] = gae_lambda
    require_cuda_training(cfg, runtime=False)


@pytest.mark.parametrize("setting", ["shaped", "curriculum", "masked", "fixed"])
def test_retired_training_modes_fail_before_creating_output(tmp_path, setting):
    from pvz_rl.training import train

    cfg = load_config()
    if setting == "fixed":
        cfg["curriculum"]["mode"] = "fixed"
    else:
        cfg["conditions"]["masked"][setting] = False
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="Retired policy|single masked teaching method"):
        train(cfg, "masked", 101, output)
    assert not output.exists()


@pytest.mark.parametrize("version", ["event_v4", "event_v5"])
def test_retired_weights_fail_before_deserialization(tmp_path, version):
    from pvz_rl.provenance import write_json
    from pvz_rl.training import load_policy

    cfg = load_config()
    cfg["encoding"]["version"] = version
    write_json(tmp_path / "metadata.json", {"config": cfg, "condition": "masked"})
    with pytest.raises(ValueError, match="Retired"):
        load_policy(tmp_path / "absent.zip")


@pytest.mark.parametrize("retired", ["network", "observation", "reward", "clock"])
def test_retired_report_rebuild_never_loads_model(tmp_path, monkeypatch, retired):
    from pvz_rl.provenance import write_json
    from pvz_rl.visualization import visualize_run

    cfg = load_config()
    if retired == "network":
        cfg["policy"]["kind"] = "spatial_grouped_v2"
        cfg["encoding"]["version"] = "tactical_v2"
    elif retired == "observation":
        cfg["encoding"]["version"] = "event_v5"
    elif retired == "reward":
        cfg["reward"]["version"] = "potential_mower_v1"
    else:
        cfg["training"].pop("discount_clock")
    # A file exists, but archived reporting must never deserialize its weights.
    (tmp_path / "best.zip").write_bytes(b"not a model")
    write_json(tmp_path / "metadata.json", {"config": cfg, "family": "preset"})
    write_json(tmp_path / "best.json", {"checkpoint_hash": "deleted-model"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Archived report must not deserialize weights")

    monkeypatch.setattr("pvz_rl.training.load_policy", forbidden)
    assert visualize_run(tmp_path, videos=False)["state"] == "complete"
    assert (tmp_path / "visualizations/index.html").is_file()


@pytest.mark.parametrize("target_kl", [0, 0.01])
def test_kl_stops_before_update_and_singleton_forced_states_stay_finite(target_kl):
    from stable_baselines3.common.logger import configure

    from pvz_rl.training import build_model, vector_env

    cfg = load_config()
    cfg["training"].update(
        n_envs=1,
        rollout_steps_per_env=2,
        rollout_size=2,
        batch_size=1,
        n_epochs=1,
        target_kl=target_kl,
    )
    env = vector_env(cfg, "masked", 101)
    try:
        model = build_model(cfg, "masked", env, 101)
        model.set_logger(configure(format_strings=[]))
        observation = env.reset()
        buffer = model.rollout_buffer
        buffer.observations.copy_(observation)
        for key in ("actions", "values", "action_masks"):
            getattr(buffer, key).zero_()
        buffer.action_masks[:, :, 0] = True
        buffer.advantages.fill_(2)
        buffer.returns.fill_(1)
        # Forced wait has log p = 0. An old log p of -1 yields e - 2 > 0.015.
        buffer.log_probs.fill_(-1)
        buffer.full = True
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        model.train()
        metrics = model.logger.name_to_value
        assert np.isfinite(metrics["train/loss"])
        assert metrics["train/optimizer_steps"] == (0 if target_kl else 2)
        assert metrics["train/kl_stopped"] == bool(target_kl)
        if target_kl:
            assert not model.policy.optimizer.state
            for k, v in model.policy.state_dict().items():
                if k.startswith(("features_extractor.", "pi_features_extractor.", "action_net.")):
                    torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            assert model.policy.critic_optimizer.state
            assert metrics["train/critic_optimizer_steps"] == 2
        else:
            assert any(not torch.equal(v, before[k]) for k, v in model.policy.state_dict().items())
    finally:
        env.close()
