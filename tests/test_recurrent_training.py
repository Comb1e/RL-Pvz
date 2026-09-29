"""Recurrent default, real handoff, chronological fitting and recovery controls."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from test_demo_initialization import completed_demo as demo_fixture

from pvz_rl.cli import main
from pvz_rl.config import load_config, validate_config
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning import demo_initialization as demo
from pvz_rl.learning.checkpoints import compatible_config, execution_config, inspect_checkpoint
from pvz_rl.learning.cohort import CohortPhase
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.performance import refresh_performance
from pvz_rl.learning.recurrent_q import sequence_batches, sequence_loss
from pvz_rl.learning.training import build_model, initial_weights, load_policy, vector_env
from pvz_rl.learning.training_requirements import resume_protocol
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy


@pytest.fixture
def completed_demo(tmp_path):
    return demo_fixture.__wrapped__(tmp_path)


@pytest.fixture
def recurrent_cfg():
    cfg = load_config()
    cfg["policy"].update(
        entity_width=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_feedforward=16,
        scalar_width=8,
        lstm_hidden=8,
        action_embedding=4,
        outcome_width=4,
        chunk_length=16,
    )
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["training"].update(n_envs=2, total_games=2, n_epochs=2, batch_size=32)
    cfg["training"]["performance"].update(compile_kernels=False, telemetry=False)
    cfg["visualization"].update(enabled=False, live_enabled=False)
    return cfg


def test_removed_time_option_and_recurrent_batch_boundary():
    cfg = load_config()
    with pytest.raises(SystemExit):
        main(["train", "--output", "unused", "--max-minutes", "1"])
    cfg["training"]["batch_size"] = 257
    with pytest.raises(ValueError, match="multiple"):
        validate_config(cfg)


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_actual_cli_keeps_recording_verification_config(
    completed_demo, monkeypatch, tmp_path, device
):
    cfg, archive, replay = completed_demo
    monkeypatch.setattr("pvz_rl.cli.load_demo_config", lambda path: copy.deepcopy(cfg))
    calls = []

    def verify_only(a, r, output, **kwargs):
        assert kwargs["cfg"] == cfg
        assert kwargs["device"] == (device or "cpu")
        calls.append(demo.verify_demo(a, r, kwargs["cfg"]))
        return {"verified": True}

    monkeypatch.setattr(demo, "initialize_demo", verify_only)
    argv = [
        "initialize-demo",
        "--archive",
        str(archive),
        "--replay",
        str(replay),
        "--output",
        str(tmp_path / "fit"),
    ]
    main(argv + (["--device", device] if device else []))
    assert calls[0]["verified"]


def test_missing_manifest_protocol_and_configuration_are_rejected(completed_demo):
    cfg, archive, replay = completed_demo
    path = archive.with_suffix(".jsonl.manifest.json")
    original = json.loads(path.read_text())
    path.unlink()
    with pytest.raises(ValueError, match="manifest is missing"):
        demo.verify_demo(archive, replay, cfg)
    path.write_text(json.dumps({**original, "protocol": "unknown"}))
    with pytest.raises(ValueError, match="archive protocol"):
        demo.verify_demo(archive, replay, cfg)
    path.write_text(json.dumps(original))
    cfg["reward"]["win_reward"] += 1
    with pytest.raises(ValueError, match="configuration digest"):
        demo.verify_demo(archive, replay, cfg)


def test_demo_weights_transfer_exactly_and_autonomous_fit_changes_them(completed_demo, tmp_path):
    cfg, archive, replay = completed_demo
    result = demo.initialize_demo(archive, replay, tmp_path / "init", cfg=cfg, passes=1)
    source, _ = demo.load_demo_checkpoint(result["checkpoint"])
    cfg["training"].update(n_envs=1, total_games=1, n_epochs=1, batch_size=16)
    cfg["visualization"]["live_enabled"] = False
    # The environment cutoff is a protocol parameter and remains unchanged here.
    weights, provenance = initial_weights(result["checkpoint"], cfg)
    env = vector_env(cfg, "masked", 7, "diagnostic")
    model = build_model(cfg, "masked", env, 7)
    try:
        model.policy.load_state_dict(weights)
        assert provenance["type"] == "demonstration" and len(provenance["checkpoint_sha256"]) == 64
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, model.policy.state_dict()[key].cpu(), rtol=0, atol=0)
        obs, _, previous, outcomes, _ = demo._training_tensors(archive, cfg)
        with torch.no_grad():
            expected = source.cuda().forward_sequence(
                collate_observations(obs, "cuda").reshape(1, len(obs)),
                previous_actions=previous[None].cuda(),
                execution_outcomes=outcomes[None].cuda(),
            )[0]
            actual = model.policy.forward_sequence(
                collate_observations(obs, "cuda").reshape(1, len(obs)),
                previous_actions=previous[None].cuda(),
                execution_outcomes=outcomes[None].cuda(),
            )[0]
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert not model.policy.optimizer.state and model.training_games == model.num_timesteps == 0
        # Tiny verified episode as the complete autonomous cohort; no long game is run.
        model._memory = model._new_memory()
        model.trajectory_root = tmp_path
        model._buffer = model._new_buffer()
        rows = np.zeros(len(obs), dtype=model._buffer.dtype)
        rows["active"] = True
        rows["previous"] = previous.numpy()
        rows["previous_outcome"] = outcomes.numpy()
        rows["action"] = demo._training_tensors(archive, cfg)[1].numpy()
        rows["reward"][-1] = 1
        rows["done"][-1] = True
        model._buffer.append(rows, obs)
        model._buffer.finalize()
        from stable_baselines3.common.logger import configure

        model.set_logger(configure(format_strings=[]))
        model._stats = {"q_optimizer_steps": 0}
        model.phase = CohortPhase.FIT
        callback = Stop()
        while model._fit_epoch < 1:
            model._fit_step(callback)
        assert any(
            not torch.equal(weights[k], v.cpu()) for k, v in model.policy.state_dict().items()
        )
        assert model._stats["q_optimizer_steps"] == 1
    finally:
        if model._buffer:
            model._buffer.close()
        env.close()


def test_transfer_rejects_reward_encoding_and_model_changes(recurrent_cfg):
    for group, field, value in [
        ("reward", "win_reward", 3),
        ("encoding", "count_scale", 3),
        ("policy", "lstm_hidden", 16),
    ]:
        changed = copy.deepcopy(recurrent_cfg)
        changed[group][field] = value
        with pytest.raises(ValueError, match="incompatible model transfer"):
            compatible_config(recurrent_cfg, changed)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_step_chunk_equivalence_with_real_outcomes(recurrent_cfg, device):
    from pvz_rl.envs.env import PvZEnv

    env = PvZEnv(recurrent_cfg, family="diagnostic")
    obs, _ = env.reset(seed=8)
    model = TransformerLSTMPolicy(recurrent_cfg).to(device).eval()
    observations = collate_observations([obs] * 8, device).reshape(1, 8)
    actions = torch.tensor([[0, 1, 1, 361, 0, 405, 3, 0]], device=device)
    outcomes = torch.tensor(
        [[[0, 0], [1, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0], [1, 1]]], device=device
    ).float()
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        state, expected = None, []
        for index in range(8):
            out = model.forward_step(
                observations[:, index],
                state,
                previous_action=actions[:, index],
                execution_outcome=outcomes[:, index],
            )
            state = out.state
            expected.append(out.branch_q)
        chunk_state, actual = None, []
        for start in (0, 3, 6):
            q, _, chunk_state = model.forward_sequence(
                observations[:, start : start + 3],
                chunk_state,
                previous_actions=actions[:, start : start + 3],
                execution_outcomes=outcomes[:, start : start + 3],
            )
            actual.append(q)
    torch.testing.assert_close(torch.stack(expected, 1), torch.cat(actual, 1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.hidden, chunk_state.hidden, rtol=1e-5, atol=1e-6)


def test_cli_initialize_train_evaluate_and_in_place_resume(completed_demo, tmp_path, monkeypatch):
    cfg, archive, replay = completed_demo
    # Re-record the tiny fixture with bounded autonomous execution parameters;
    # these do not alter the native game's five recorded decisions.
    cfg["environment"]["cutoff_seconds"] = 1
    cfg["training"].update(n_envs=2, total_games=1, n_epochs=1, batch_size=4)
    cfg["training"]["performance"].update(compile_kernels=False, telemetry=False)
    cfg["visualization"].update(enabled=False, live_enabled=False)
    manifest = archive.with_suffix(".jsonl.manifest.json")
    data = json.loads(manifest.read_text())
    data["config_digest"] = demo.digest(cfg)
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr("pvz_rl.cli.load_demo_config", lambda path: copy.deepcopy(cfg))
    initial, run = tmp_path / "init", tmp_path / "train"
    main(
        [
            "initialize-demo",
            "--archive",
            str(archive),
            "--replay",
            str(replay),
            "--output",
            str(initial),
            "--passes",
            "1",
        ]
    )
    main(["train", "--init-from", str(initial / "initialization.pt"), "--output", str(run)])
    assert json.loads((run / "status.json").read_text())["training_games"] == 1
    run_config = json.loads((run / "config.json").read_text())
    assert run_config["training"]["performance"] == load_config()["training"]["performance"]
    main(
        [
            "evaluate",
            "--checkpoint",
            str(run / "latest.zip"),
            "--count",
            "1",
            "--levels",
            "easy",
            "--output",
            str(tmp_path / "evaluation"),
            "--record",
        ]
    )
    status = json.loads((run / "status.json").read_text())
    status["time_budget"]["elapsed_seconds"] = 60
    (run / "status.json").write_text(json.dumps(status))
    main(["train", "--resume", str(run / "latest.zip"), "--games", "2", "--output", str(run)])
    assert json.loads((run / "status.json").read_text())["training_games"] == 2
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["initialization"]["type"] == "demonstration"
    assert json.loads((run / "status.json").read_text())["time_budget"]["elapsed_seconds"] >= 60
    assert len((run / "training-episodes.jsonl").read_text().splitlines()) == 2


def test_rejected_proposals_zero_ticks_and_isolated_runner_reset(recurrent_cfg):
    from pvz_rl.envs.env import PvZEnv

    env = PvZEnv(recurrent_cfg, family="diagnostic")
    obs, _ = env.reset(seed=3)
    from gymnasium.spaces import Discrete

    from pvz_rl.policy.recurrent_policy import RecurrentQPolicy

    policy = RecurrentQPolicy(
        env.observation_space,
        Discrete(406),
        lambda _: 1e-3,
        features_extractor_kwargs={"layout_cfg": recurrent_cfg},
    )
    runner = PolicyRunner(policy, recurrent_cfg, env.rules, 2, "cpu")
    observations = collate_observations([obs, obs])
    masks = torch.tensor(np.stack([env.action_masks()] * 2))
    runner.decide(observations, masks, None)
    runner.observe_result([120, 361], [False, True], [1, 0])
    assert runner.previous_actions.tolist() == [0, 361]
    assert runner.outcomes.tolist() == [[0, 1], [1, 0]]
    saved = runner.state.clone()
    runner.decide(observations, masks, None, active=torch.tensor([True, False]))
    torch.testing.assert_close(saved.hidden[:, 1], runner.state.hidden[:, 1], rtol=0, atol=0)
    runner.reset([0])
    assert not runner.state.hidden[:, 0].any() and runner.previous_actions.tolist() == [0, 361]
    torch.testing.assert_close(saved.hidden[:, 1], runner.state.hidden[:, 1], rtol=0, atol=0)


def test_returns_group_loss_padding_and_partition_gradients(recurrent_cfg, tmp_path):
    policy = TransformerLSTMPolicy(recurrent_cfg)
    buffer = CompleteGameBuffer(tmp_path / "buffer", 2, block_rows=3)
    rows = np.zeros(8, dtype=buffer.dtype)
    rows["env"] = np.tile([0, 1], 4)
    rows["active"] = [1, 1, 1, 1, 1, 0, 1, 0]
    rows["action"] = [0, 1, 1, 361, 361, 0, 0, 0]
    rows["reward"] = [1, 3, 2, -1, -1, 999, 4, 999]
    rows["done"] = [0, 0, 0, 1, 0, 0, 1, 0]
    rows["previous"][2:] = rows["action"][:-2]
    rows["previous_outcome"][2:, 0] = 1
    buffer.append(rows, [{"entities": [], "globals": [0] * 18}] * len(rows))
    buffer.finalize()
    np.testing.assert_array_equal(buffer.take(np.arange(8))["target"], [6, 2, 5, -1, 3, 0, 4, 0])
    gradients, losses = [], []
    try:
        for budget in [2, 4]:
            policy.zero_grad(set_to_none=True)
            state, total, errors = None, 0, {0: [], 1: [], 2: []}
            for reset, chunk, observations in sequence_batches(buffer, 2, budget):
                if reset:
                    state = None
                loss, state, first, second = sequence_loss(
                    policy, chunk, state, buffer.group_counts, observations
                )
                loss.backward()
                total += float(loss.detach())
                groups = np.where(
                    chunk["action"][chunk["active"]] == 0,
                    0,
                    np.where(chunk["action"][chunk["active"]] < 361, 1, 2),
                )
                tile = iter(second.detach().tolist())
                for group, error in zip(groups, first.detach().tolist()):
                    errors[group].append(error if group == 0 else (error + next(tile)) / 2)
            assert total == pytest.approx(sum(np.mean(v) for v in errors.values()) / 3, rel=1e-6)
            losses.append(total)
            gradients.append([p.grad.clone() for p in policy.parameters()])
        assert losses[0] == pytest.approx(losses[1], rel=1e-6)
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    finally:
        buffer.close()


class Stop(BaseCallback):
    def __init__(self, phase=None):
        super().__init__()
        self.stop_phase = phase
        self.fit_calls = 0

    def _on_step(self):
        return not (self.stop_phase == "collect" and self.model.num_timesteps >= 6)

    def log_progress(self):
        if self.stop_phase == "fit" and self.model._fit_epoch == 1:
            self.fit_calls += 1
            if self.fit_calls == 3:
                raise KeyboardInterrupt


@pytest.mark.parametrize("phase", ["collect", "fit", "idle"])
def test_cuda_recovery_matches_uninterrupted(recurrent_cfg, tmp_path, phase, monkeypatch):
    cfg = recurrent_cfg
    if phase == "idle":
        cfg["training"]["total_games"] = 4
    reference_env = vector_env(cfg, "masked", 17)
    reference = build_model(cfg, "masked", reference_env, 17)

    def inject_once(model):
        if phase != "fit":
            return
        forward = model.policy.forward_sequence
        failed = False

        def fail_first(*args, **kwargs):
            nonlocal failed
            output = forward(*args, **kwargs)
            if not failed:
                failed = True
                return (output[0] * float("nan"), *output[1:])
            return output

        monkeypatch.setattr(model.policy, "forward_sequence", fail_first)

    inject_once(reference)
    reference.trajectory_root = tmp_path
    reference.learn(1, callback=Stop())
    if phase == "idle":
        reference.learn(1, callback=Stop(), reset_num_timesteps=False)
    hashes = [reference_env.batch.state_hash(i) for i in range(2)]
    expected = copy.deepcopy(reference.policy.state_dict())
    expected_optimizer = copy.deepcopy(reference.policy.optimizer.state_dict())
    reference_env.close()
    env = vector_env(cfg, "masked", 17)
    model = build_model(cfg, "masked", env, 17)
    inject_once(model)
    model.trajectory_root = tmp_path
    try:
        if phase == "idle":
            model.learn(1, callback=Stop())
        else:
            with pytest.raises(KeyboardInterrupt):
                model.learn(1, callback=Stop(phase))
        assert model._fit_epoch == {"collect": 0, "fit": 1, "idle": 2}[phase]
        checkpoint = tmp_path / "recovery.zip"
        model.save(checkpoint)
        assert inspect_checkpoint(checkpoint)["config"] == cfg
    finally:
        if model._buffer:
            model._buffer.close()
        env.close()
    restored, _ = load_policy(checkpoint, device="cuda")
    if phase == "fit":
        assert restored.execution_state["precision"] == "fp32"
        assert restored.execution_state["fallbacks"] == ["nonfinite_bf16"]
    env = vector_env(cfg, "masked", 17)
    try:
        restored.set_env(env)
        restored.trajectory_root = tmp_path
        restored.learn(1, callback=Stop(), reset_num_timesteps=False)
        assert restored.training_games == (4 if phase == "idle" else 2)
        assert restored._stats["q_optimizer_steps"] == 2
        assert hashes == [env.batch.state_hash(i) for i in range(2)]
        from test_stage_training import assert_tensor_tree_equal

        assert_tensor_tree_equal(expected, restored.policy.state_dict())
        assert_tensor_tree_equal(expected_optimizer, restored.policy.optimizer.state_dict())
    finally:
        if restored._buffer:
            restored._buffer.close()
        env.close()


def test_final_cohort_size_atomic_save_and_evaluation(recurrent_cfg, tmp_path, monkeypatch):
    from pvz_rl.envs.env import PvZEnv
    from pvz_rl.evaluation.cuda_evaluation import batched_games
    from pvz_rl.evaluation.runner import select_action

    cfg = recurrent_cfg
    cfg["training"]["total_games"] = 1
    env = vector_env(cfg, "masked", 21)
    model = build_model(cfg, "masked", env, 21)
    try:
        model.learn(1, callback=Stop())
        assert model.training_games == 1 and sum(env.task_started.values()) == 1
        checkpoint = tmp_path / "last.zip"
        model.save(checkpoint)
        original = checkpoint.read_bytes()
        with monkeypatch.context() as m:
            m.setattr(Path, "replace", lambda *a: (_ for _ in ()).throw(OSError("blocked")))
            with pytest.raises(OSError, match="blocked"):
                model.save(checkpoint)
        assert checkpoint.read_bytes() == original
        assert len(list(tmp_path.glob("*.zip"))) == 1
        cpu, _ = load_policy(checkpoint)
        reference = PvZEnv(cfg, family="diagnostic")
        traces = []
        for seed in [1, 2, 3]:
            obs, _ = reference.reset(seed=seed)
            trace = []
            while reference.state == "running":
                action = select_action(reference, obs, policy=cpu)
                trace.append(action)
                obs, *_ = reference.step(action)
            traces.append(trace)
        rows = list(
            batched_games(cfg, model, "masked", [1, 2, 3], ["easy"], "diagnostic", record=True)
        )
        assert [row["action_trace"] for row in rows] == traces
        cfg["runtime"]["refill_evaluation"] = False
        assert [
            row["action_trace"]
            for row in batched_games(
                cfg, model, "masked", [1, 2, 3], ["easy"], "diagnostic", record=True
            )
        ] == traces
    finally:
        env.close()


@pytest.mark.learning
def test_default_128_slot_bounded_collection_and_chunk_memory(tmp_path):
    cfg = load_config()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    env = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", env, 101)
    model.trajectory_root = tmp_path

    class ThreeDecisions(BaseCallback):
        def _on_step(self):
            return self.model.num_timesteps < 3 * 128

    try:
        with pytest.raises(KeyboardInterrupt):
            model.learn(1, callback=ThreeDecisions())
        assert model._buffer.size == 384 and model.training_games == 0
        assert model._memory.state.hidden.shape == (1, 128, 256)
        real = model._buffer.take(np.arange(128))
        for i in range(128):
            row = env.action_journal.latest[i]
            assert row is not None and np.isfinite(row["q"]).all()
        # One synthetic default-size fitting chunk, no optimizer update or full games.
        rows = np.repeat(real[:4, None], 256, axis=1)
        rows["action"] = 0
        rows["target"] = 1
        model.policy.train()
        loss, *_ = sequence_loss(
            model.policy, rows, None, [1024, 0, 0], model._buffer.observations(rows, model.device)
        )
        loss.backward()
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated()
        assert allocated < torch.cuda.get_device_properties(0).total_memory
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "slots": 128,
                    "collected_decisions": 384,
                    "synthetic_chunk_decisions": 1024,
                    "completed_games": 0,
                    "optimizer_updates": 0,
                    "peak_torch_allocated_mib": allocated / 2**20,
                }
            )
        )
    finally:
        model.policy.optimizer.zero_grad(set_to_none=True)
        if model._buffer:
            model._buffer.close()
        env.close()


def test_performance_refresh_keeps_learning_settings_and_old_precision():
    current = load_config()
    assert current["training"]["demo"]["passes"] == 5
    old = copy.deepcopy(current)
    old["training"]["performance"].pop("fit_precision")
    old["training"]["performance"].pop("prefetch")
    old["policy"]["encoder_microbatch"] = 16
    old["training"].update(n_epochs=7, batch_size=512)
    old["reward"]["invalid_plant_penalty"] = 0.123
    normalized = execution_config(old)
    assert normalized["training"]["performance"]["fit_precision"] == "fp32"
    new = refresh_performance(normalized, current)
    assert new["policy"]["encoder_microbatch"] == 128
    assert new["policy"]["encoder_token_budget"] == 65536
    assert new["training"]["performance"]["fit_sequence_groups"] == 4
    assert new["training"]["performance"]["fit_precision"] == "features_bf16"
    assert new["training"]["n_epochs"] == 7 and new["training"]["batch_size"] == 512
    assert new["reward"] == old["reward"]
    assert new["training"]["demo"] == old["training"]["demo"]
    assert resume_protocol(new, "masked") == resume_protocol(normalized, "masked")


@pytest.mark.parametrize("fault", ["nonfinite", "memory", "compilation"])
def test_failed_pass_restarts_without_skipping_updates(tmp_path, monkeypatch, fault):
    cfg = load_config()
    cfg["training"].update(n_envs=1, n_epochs=2, batch_size=256)
    cfg["visualization"]["live_enabled"] = False
    env = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", env, 101)
    model.trajectory_root = tmp_path
    model._buffer = model._new_buffer()
    model.set_logger(configure(format_strings=[]))
    raw, _ = observation(cfg, 5)
    rows = np.zeros(3, model._buffer.dtype)
    rows["active"] = True
    rows["reward"] = [0, 0, 1]
    model._buffer.append(rows, [raw] * 3)
    model._buffer.finalize()
    model._stats = {"q_optimizer_steps": 0}
    model.phase = CohortPhase.FIT
    original = model.policy.forward_sequence
    attempts = []

    def failing_once(*args, **kwargs):
        attempts.append(model.policy.fit_precision)
        if len(attempts) == 1 and fault == "memory":
            raise torch.cuda.OutOfMemoryError("injected allocation failure")
        output = original(*args, **kwargs)
        if len(attempts) == 1 and fault == "compilation":
            from pvz_rl.policy.cudagraph_backend import EncoderCompilationError

            def fail_backward(gradient):
                raise EncoderCompilationError("injected backward capture failure")

            output[0].register_hook(fail_backward)
        elif len(attempts) == 1:
            output = (output[0] * float("nan"), *output[1:])
        return output

    monkeypatch.setattr(model.policy, "forward_sequence", failing_once)
    try:
        for _ in range(15):
            model._fit_step(Stop())
            if fault == "compilation" and len(attempts) == 1:
                assert model._fit_epoch == model._stats["q_optimizer_steps"] == 0
                assert model.policy.entity.compilation_status == "fallback"
                assert model.policy.entity._compiled_encode is None
                assert all(p.grad is None for p in model.policy.parameters())
            if model._fit_epoch == 2:
                break
        assert model._fit_epoch == model._stats["q_optimizer_steps"] == 2
        assert len(attempts) == 3
        assert model.execution_state["fallbacks"] == [
            {
                "nonfinite": "nonfinite_bf16",
                "memory": "memory",
                "compilation": "encoder_compilation",
            }[fault]
        ]
        assert attempts[1:] == (["fp32"] * 2 if fault == "nonfinite" else ["features_bf16"] * 2)
        assert model.execution_state["microbatch"] == (64 if fault == "memory" else 128)
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    finally:
        model._drain()
        model._close_sequences()
        model._buffer.close()
        env.close()
