import pytest
import torch

from pvz_rl.config import load_config, validate_config
from pvz_rl.training_requirements import current_model_config, transfer_protocol


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

    from pvz_rl.training import build_model, vector_env

    smoke_cfg["training"].update(rollout_size=4, batch_size=4)
    env = vector_env(smoke_cfg, "masked", 101, family="placement")
    try:
        model = build_model(smoke_cfg, "masked", env, 101)
        model.set_logger(configure(format_strings=[]))
        yield model, env
    finally:
        env.close()


def test_measured_overlap_and_interrupt_boundary():
    import signal

    from pvz_rl.periodic import finish_window_on_interrupt, interval_overlap

    assert interval_overlap((1, 5), (3, 8)) == 2
    assert interval_overlap((1, 2), (3, 8)) == 0
    previous = signal.getsignal(signal.SIGINT)
    completed = []
    with pytest.raises(KeyboardInterrupt), finish_window_on_interrupt():
        signal.raise_signal(signal.SIGINT)
        completed.append(True)
    assert completed == [True] and signal.getsignal(signal.SIGINT) == previous


def test_snapshot_slots_frozen_values_overlap_and_window_callbacks(pipeline_model, monkeypatch):
    from threading import Event

    from stable_baselines3.common.callbacks import BaseCallback

    from pvz_rl.periodic import WindowState

    model, env = pipeline_model
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
                buffer.observations.flatten(0, 1), buffer.action_masks.flatten(0, 1),
                context=buffer.context(slice(None)),
            )
            torch.testing.assert_close(
                distribution.log_prob(buffer.actions.flatten().long()),
                buffer.log_probs.flatten(), rtol=2e-6, atol=2e-6,
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

    from pvz_rl.cuda_ppo import CudaMaskablePPO
    from pvz_rl.exploration import configure_exploration
    from pvz_rl.periodic import WindowState

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
    model.save(tmp_path / "boundary.zip")
    restored = CudaMaskablePPO.load(tmp_path / "boundary.zip", env=env, device="cuda")
    configure_exploration(restored, env.cfg)
    restored.set_logger(model.logger)
    assert restored.policy.optimizer.state and restored.policy.critic_optimizer.state
    assert restored.pipeline_version == 1 and not hasattr(restored, "_episode_memory")
    restored.learn(8, reset_num_timesteps=False)
    assert restored.num_timesteps == 16 and restored.pipeline_version == 2


@pytest.mark.parametrize("failure", ["collector", "version", "learner"])
def test_failure_drains_worker_and_forbids_partial_checkpoint(
    pipeline_model, monkeypatch, tmp_path, failure
):
    from pvz_rl.cuda_ppo import PeriodicPipelineError
    from pvz_rl.periodic import WindowState

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
        model.learn(8)
    assert env.stream == previous_stream and model.pipeline_state == WindowState.FAILED
    with pytest.raises(PeriodicPipelineError, match="Cannot save"):
        model.save(tmp_path / "invalid.zip")
    assert not (tmp_path / "invalid.zip").exists()
    with pytest.raises(PeriodicPipelineError, match="Reload"):
        model.learn(8)


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
