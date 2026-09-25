import pytest
import torch

from pvz_rl.config import load_config, validate_config
from pvz_rl.learning.training_requirements import current_model_config, transfer_protocol


def test_periodic_pipeline_is_the_only_training_scheduler():
    cfg = load_config()
    pipeline = cfg["training"]["pipeline"]
    assert pipeline == {"mode": "periodic_on_policy", "depth": 2, "queue_size": 1}
    assert current_model_config(cfg)
    assert transfer_protocol(cfg)["pipeline"] == pipeline

    cfg["training"]["pipeline"]["mode"] = "synchronous"
    with pytest.raises(ValueError, match="periodic_on_policy"):
        validate_config(cfg)


@pytest.mark.parametrize("field,value", [("depth", 1), ("queue_size", 0), ("queue_size", 3)])
def test_periodic_pipeline_depth_and_queue_are_bounded(field, value):
    cfg = load_config()
    cfg["training"]["pipeline"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(cfg)


@pytest.fixture
def pipeline_model(smoke_cfg):
    from stable_baselines3.common.logger import configure

    from pvz_rl.learning.training import build_model, vector_env

    smoke_cfg["training"].update(rollout_size=4, batch_size=4)
    env = vector_env(smoke_cfg, "masked", 101, family="saving")
    try:
        model = build_model(smoke_cfg, "masked", env, 101)
        model.set_logger(configure(format_strings=[]))
        yield model, env
    finally:
        env.close()


def test_measured_overlap_and_interrupt_boundary():
    import signal

    from pvz_rl.learning.periodic import finish_window_on_interrupt, interval_overlap

    assert interval_overlap((1, 5), (3, 8)) == 2
    assert interval_overlap((1, 2), (3, 8)) == 0
    previous = signal.getsignal(signal.SIGINT)
    completed = []
    with pytest.raises(KeyboardInterrupt), finish_window_on_interrupt():
        signal.raise_signal(signal.SIGINT)
        completed.append(True)
    assert completed == [True] and signal.getsignal(signal.SIGINT) == previous


@pytest.mark.parametrize("target_kl", [0, 0.01])
def test_snapshot_slots_frozen_values_overlap_and_window_callbacks(
    pipeline_model, monkeypatch, target_kl
):
    from threading import Event

    from stable_baselines3.common.callbacks import BaseCallback

    from pvz_rl.learning.periodic import WindowState

    model, env = pipeline_model
    model.target_kl = target_kl or None
    collect, train = model.collect_slot, model.train
    records, updates, boundaries, carried = [], [], [], []
    second_started = Event()

    def collect_control(*args):
        policy, buffer, memory, version = args[1], args[2], args[5], args[6]
        digest = model._behavior_hash(policy)
        assert all(not p.requires_grad for p in policy.parameters())
        if len(records) % 2:
            second_started.set()
        if memory is not None:
            torch.testing.assert_close(memory.context().tokens, carried[-1], rtol=0, atol=0)
        result = collect(*args)
        carried.append(result.memory.context().tokens.clone())
        assert model._behavior_hash(policy) == digest
        with torch.no_grad():
            distribution = policy.get_distribution(
                buffer.observations.flatten(0, 1),
                buffer.action_masks.flatten(0, 1),
                context=buffer.context(slice(None)),
            )
            torch.testing.assert_close(
                distribution.log_prob(buffer.actions.flatten().long()),
                buffer.log_probs.flatten(),
                rtol=2e-6,
                atol=2e-6,
            )
            expected = policy.predict_values(
                buffer.observations.flatten(0, 1), context=buffer.context(slice(None))
            ).reshape_as(buffer.values)
        torch.testing.assert_close(buffer.values, expected, rtol=2e-6, atol=2e-6)
        assert torch.isfinite(buffer.advantages).all()
        records.append((version, digest, id(buffer), buffer.memory_archive.data_ptr()))
        return result

    def update_control():
        if not updates:
            assert second_started.wait(10), "second collector did not overlap first update"
        train()
        updates.append(model.num_timesteps)

    class BoundaryControl(BaseCallback):
        def _on_rollout_start(self):
            assert model.pipeline_state == WindowState.IDLE
            boundaries.append(model.num_timesteps)

        def _on_step(self):
            assert len(updates) in (2, 4)
            assert len(records) == len(updates)
            return True

        def _on_rollout_end(self):
            assert model.pipeline_state == WindowState.IDLE
            assert model.pipeline_metrics["depth"] == 2
            if target_kl:
                assert model.logger.name_to_value["train/exact_kl"] <= target_kl

    monkeypatch.setattr(model, "collect_slot", collect_control)
    monkeypatch.setattr(model, "train", update_control)
    model.learn(16, callback=BoundaryControl())
    assert boundaries == [0, 8] and updates == [4, 8, 12, 16]
    assert records[0][:2] == records[1][:2] and records[2][:2] == records[3][:2]
    assert records[0][1] != records[2][1]
    assert records[0][2:] == records[2][2:] and records[1][2:] == records[3][2:]
    assert env.episode_metrics(0)["decisions"] == 16  # No reset at either boundary.
    assert model.pipeline_version == 2 and model.pipeline_metrics["overlap_seconds"] > 0


def test_ctrl_c_drains_window_then_checkpoint_resumes(pipeline_model, monkeypatch, tmp_path):
    import signal

    from pvz_rl.learning.cuda_ppo import CudaMaskablePPO
    from pvz_rl.learning.exploration import set_entropy_factor
    from pvz_rl.learning.periodic import WindowState

    model, env = pipeline_model
    train = model.train
    calls = []

    def interrupted_update():
        if not calls:
            signal.raise_signal(signal.SIGINT)
        train()
        calls.append(True)

    monkeypatch.setattr(model, "train", interrupted_update)
    with pytest.raises(KeyboardInterrupt):
        model.learn(16)
    assert len(calls) == 2 and model.num_timesteps == 8
    assert model.pipeline_state == WindowState.IDLE
    monkeypatch.undo()
    model.__dict__.pop("train", None)
    set_entropy_factor(model, env.cfg, 0.2)
    model.save(tmp_path / "boundary.zip")
    restored = CudaMaskablePPO.load(tmp_path / "boundary.zip", env=env, device="cuda")
    assert restored.entropy_factor == 0.2
    assert restored.policy.exploration_settings == model.policy.exploration_settings
    restored.set_logger(model.logger)
    assert restored.policy.optimizer.state and restored.policy.critic_optimizer.state
    assert restored.pipeline_version == 1 and not hasattr(restored, "_episode_memory")
    restored.learn(8, reset_num_timesteps=False)
    assert restored.num_timesteps == 16 and restored.pipeline_version == 2


@pytest.mark.parametrize("failure", ["collector", "version", "learner"])
def test_failure_drains_worker_and_forbids_partial_checkpoint(
    pipeline_model, monkeypatch, tmp_path, failure
):
    from pvz_rl.learning.cuda_ppo import PeriodicPipelineError
    from pvz_rl.learning.periodic import WindowState

    model, env = pipeline_model
    previous_stream = env.stream
    collect, train = model.collect_slot, model.train

    def broken_collect(*args):
        if failure == "collector":
            raise RuntimeError("collector failure")
        result = collect(*args)
        result.policy_version += 1
        return result

    def broken_update():
        train()
        raise RuntimeError("learner failure")

    if failure == "learner":
        monkeypatch.setattr(model, "train", broken_update)
    else:
        monkeypatch.setattr(model, "collect_slot", broken_collect)
    with pytest.raises((RuntimeError, PeriodicPipelineError)):
        model.learn(8, log_interval=None)
    assert env.stream == previous_stream and model.pipeline_state == WindowState.FAILED
    with pytest.raises(PeriodicPipelineError, match="Cannot save"):
        model.save(tmp_path / "invalid.zip")
    assert not (tmp_path / "invalid.zip").exists()
    with pytest.raises(PeriodicPipelineError, match="Reload"):
        model.learn(8, log_interval=None)


def test_final_partial_window_and_callback_stop_are_complete(pipeline_model):
    from stable_baselines3.common.callbacks import BaseCallback

    model, _ = pipeline_model
    model.learn(12)
    assert model.num_timesteps == 12 and model._n_updates == 3
    assert model.pipeline_metrics["depth"] == 1 and model.pipeline_version == 2

    class Stop(BaseCallback):
        def _on_step(self):
            return False

    model.learn(100, callback=Stop(), reset_num_timesteps=False)
    assert model.num_timesteps == 20 and model._n_updates == 5
    assert model.pipeline_version == 3


@pytest.mark.parametrize("value", [0.02, float("nan"), float("inf")])
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actor_transaction_restores_parameters_and_adam_only(value, initialized, device):
    from copy import deepcopy
    from types import SimpleNamespace

    from pvz_rl.learning.periodic import ActorUpdateState, ActorWindow

    actor, critic = (torch.nn.Parameter(torch.tensor([1.0, 2.0], device=device)) for _ in range(2))
    optimizer = torch.optim.Adam([actor], lr=0.1)
    if initialized:
        actor.square().sum().backward()
        optimizer.step()
    before, saved = actor.detach().clone(), deepcopy(optimizer.state_dict())
    window = ActorWindow(
        SimpleNamespace(actor_parameters=lambda: (actor,), optimizer=optimizer), 0.01
    )
    optimizer.zero_grad()
    actor.square().sum().backward()
    optimizer.step()
    with torch.no_grad():
        critic.add_(5)
    window.stop()
    window.check([value])
    assert window.state == ActorUpdateState.REJECTED
    torch.testing.assert_close(actor, before, rtol=0, atol=0)
    torch.testing.assert_close(critic, torch.tensor([6.0, 7.0], device=device), rtol=0, atol=0)
    assert optimizer.state_dict()["param_groups"] == saved["param_groups"]
    for key, state in saved["state"].items():
        for field, expected in state.items():
            torch.testing.assert_close(
                optimizer.state_dict()["state"][key][field], expected, rtol=0, atol=0
            )
    if not initialized:
        assert not optimizer.state


@pytest.mark.parametrize("reject_slot", [1, 2, "new_context"])
def test_window_rejects_either_slot_without_reverting_critic(
    pipeline_model, monkeypatch, reject_slot
):
    from pvz_rl.learning.periodic import ActorUpdateState

    model, _ = pipeline_model
    model.target_kl = 0.01
    before = [p.detach().clone() for p in model.policy.actor_parameters()]
    original_train, calls, buffers = model.train, [], []

    def train():
        original_train()
        calls.append(True)

    def divergence(buffer):
        if id(buffer) not in buffers:
            buffers.append(id(buffer))
        if model._actor_window.state == ActorUpdateState.REJECTED:
            return 0.0
        if reject_slot == "new_context":
            return 0.02 if len(calls) == 1 and len(buffers) == 2 else 0.0
        return 0.02 if len(calls) >= reject_slot else 0.0

    monkeypatch.setattr(model, "train", train)
    monkeypatch.setattr(model, "_exact_kl", divergence)
    model.learn(8, log_interval=None)
    assert len(calls) == 2 and model.num_timesteps == 8
    for actual, expected in zip(model.policy.actor_parameters(), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not model.policy.optimizer.state
    assert model.policy.critic_optimizer.state
    assert model.logger.name_to_value["train/actor_retained_steps"] == 0
    assert model.logger.name_to_value["train/actor_window_rejected"] == 1
    assert model.logger.name_to_value["train/critic_optimizer_steps"] == 2


def test_exact_window_kl_excludes_forced_wait_and_detects_nonfinite(pipeline_model, monkeypatch):
    import math

    from pvz_rl.policy.grouped_policy import GroupedDistribution

    model, env = pipeline_model
    buffer = model.rollout_buffer
    buffer.observations.copy_(torch.as_tensor(env.reset(), device="cuda"))
    buffer.action_masks.zero_()
    buffer.action_masks[..., 0] = True
    buffer.action_masks[-1, :, 361] = True
    buffer.actions.zero_()
    buffer.log_probs.zero_()
    buffer.behavior_logits = torch.zeros(4, 1, 416, device="cuda")
    buffer.behavior_logits[..., 1] = math.log(1e-6 / (1 - 1e-6))
    buffer.has_behavior_logits, buffer.behavior_epsilon = True, 0.0
    model._checked_policy_metrics = {}
    new_logits = torch.zeros(1, 416, device="cuda")
    new_logits[:, 1] = math.log(0.1 / 0.9)

    def distribution(obs, masks, context=None):
        model.policy.action_dist = GroupedDistribution().proba_distribution(
            new_logits.expand(len(obs), -1), masks
        )
        return model.policy.action_dist

    monkeypatch.setattr(model.policy, "get_distribution", distribution)
    assert model._exact_kl(buffer) == pytest.approx(0.105347897, abs=2e-7)
    new_logits.fill_(float("nan"))
    assert not math.isfinite(model._exact_kl(buffer))


def test_sampled_stop_persists_and_clone_preserves_rng(pipeline_model, monkeypatch):
    from pvz_rl.learning.periodic import ActorUpdateState

    model, _ = pipeline_model
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    model._clone_behavior_policy()
    assert torch.equal(cpu, torch.get_rng_state())
    assert torch.equal(gpu, torch.cuda.get_rng_state())
    train, steps = model.train, []

    def stop_after_first():
        train()
        steps.append(model.logger.name_to_value["train/actor_optimizer_steps"])
        model._actor_window.stop()
        assert model._actor_window.state == ActorUpdateState.STOPPED

    monkeypatch.setattr(model, "train", stop_after_first)
    model.learn(8, log_interval=None)
    assert steps == [1, 0]
    assert model.logger.name_to_value["train/critic_optimizer_steps"] == 2


def test_old_optimizer_protocol_is_inference_and_weights_only(pipeline_model, tmp_path):
    import json

    from pvz_rl.learning.training import initial_weights, load_policy, train

    model, env = pipeline_model
    model.optimizer_protocol = "previous"
    checkpoint = tmp_path / "previous.zip"
    model.save(checkpoint)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "config": env.cfg,
                "condition": "masked",
                "learner_seed": 101,
                "family": "preset",
                "validation_limit": 1,
            }
        )
    )
    loaded, _ = load_policy(checkpoint, "cuda")
    for actual, expected in zip(loaded.policy.parameters(), model.policy.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="retired exploration schedule"):
        initial_weights(checkpoint, env.cfg)
    with pytest.raises(ValueError, match="Optimizer protocol"):
        loaded.learn(8)
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="retired exploration schedule"):
        train(env.cfg, "masked", 101, output, resume=checkpoint, validation_limit=1)
    assert not output.exists()


def test_sparse_action_metrics_use_sums_and_counts():
    from pvz_rl.learning.periodic import aggregate_update_metrics

    slots = []
    for count in (0, 2):
        row = {"train/dig_legal_observations": count, "train/dig_probability_sum": count * 0.3}
        # Two groups whose target means differ; averaging separate explained
        # variances would give a different answer than the combined sample.
        target = torch.tensor([0.0, 2.0]) + count
        residual = torch.tensor([0.0, 1.0])
        row.update(
            {
                "train/value_sample_count": 2,
                "train/return_sum": float(target.sum()),
                "train/return_squared_sum": float(target.square().sum()),
                "train/residual_sum": float(residual.sum()),
                "train/residual_squared_sum": float(residual.square().sum()),
            }
        )
        for action in ("wait", "plant", "dig"):
            row.update(
                {
                    f"train/action_count_{action}": count,
                    f"train/value_target_error_sum_{action}": count * 0.4,
                    f"train/raw_advantage_sum_{action}": -count * 0.2,
                    f"train/positive_advantage_count_{action}": count / 2,
                    f"train/value_target_error_{action}": 0.4 if count else float("nan"),
                }
            )
        slots.append(row)
    merged = aggregate_update_metrics(slots)
    assert merged["train/action_count_dig"] == 2
    assert merged["train/value_target_error_dig"] == pytest.approx(0.4)
    assert merged["train/raw_advantage_mean_dig"] == pytest.approx(-0.2)
    assert merged["train/positive_advantage_fraction_dig"] == 0.5
    assert merged["train/dig_probability_when_legal"] == pytest.approx(0.3)
    # Targets [0,2,2,4] have variance 2; residual variance is 1/4.
    assert merged["train/explained_variance"] == pytest.approx(0.875)
