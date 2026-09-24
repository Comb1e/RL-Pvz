import json
import math
import subprocess
import sys
from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest
import torch
from pvz_game import Dig, LevelSpec, Place, Spawn

from pvz_rl.config import load_config
from pvz_rl.curriculum import CurriculumState
from pvz_rl.env import PvZEnv
from pvz_rl.grouped_policy import GroupedDistribution
from pvz_rl.training import ResearchCallback, load_policy, train


def test_event_v6_dimensions_regions_scales_crowds_and_no_leaks():
    cfg = load_config()
    env = PvZEnv(cfg)
    a, _ = env.reset(seed=5, options={"scenario": LevelSpec("hidden", (Spawn(800, "basic", 1),))})
    b, _ = env.reset(
        seed=700, options={"scenario": LevelSpec("other", (Spawn(1800, "buckethead", 4),))}
    )
    assert a.shape == (281,)
    np.testing.assert_array_equal(a, b)
    encoder = env.encoder
    assert a[encoder.slices["globals"]][0] == pytest.approx(50 / 200)
    start, width = env.rules.game["house_x"], encoder.position_scale
    for boundary in (1, 2):
        first = math.ceil(start + width * boundary / 3)
        assert encoder.bin_index(first - 1) == boundary - 1
        assert encoder.bin_index(first) == boundary
    assert encoder.bin_index(start - 9999) == 0
    assert encoder.bin_index(start + width + 9999) == 2
    spawns = tuple(Spawn(1, "buckethead" if i % 2 else "basic", 2) for i in range(120))
    env.reset(seed=4, options={"scenario": LevelSpec("crowd", spawns, initial_sun=500)})
    env.step(env.codec.encode(Place("peashooter", 2, 0)))
    for _ in range(4):
        env.step(0)
    original = encoder.encode(env.public)
    altered = replace(
        env.public,
        level="private",
        plants=tuple(replace(p, id=888) for p in reversed(env.public.plants)),
        zombies=tuple(replace(z, id=999) for z in reversed(env.public.zombies)),
        projectiles=tuple(reversed(env.public.projectiles)),
        mowers=tuple(reversed(env.public.mowers)),
        cards=tuple(reversed(env.public.cards)),
    )
    np.testing.assert_array_equal(original, encoder.encode(altered))
    zombies = original[encoder.slices["zombies"]].reshape(5, 3, 9)
    assert zombies[:, :, :5].sum() * 5 == pytest.approx(120)
    assert np.isfinite(original).all()


def test_grouped_equal_logits_and_wait_exploration():
    env = PvZEnv(load_config())
    env.reset(seed=4)
    mask = env.action_masks()
    assert mask.sum() == 136
    dist = GroupedDistribution().proba_distribution(torch.zeros(1, 416))
    dist.apply_masking(mask)
    assert dist.probs.sum() == pytest.approx(1)
    assert dist.probs[0, 0] == pytest.approx(0.5)
    assert torch.count_nonzero(dist.probs[0, ~torch.tensor(mask)]) == 0
    for group in (0, 2, 4):  # sunflower, wall-nut, potato mine
        assert dist.probs[0, 1 + group * 45 : 1 + (group + 1) * 45].sum() == pytest.approx(1 / 6)
    assert dist.entropy().item() == pytest.approx(math.log(2) + 0.5 * math.log(135))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_distribution_against_independent_numpy_control(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(3, 416))
    mask = rng.random((3, 406)) > 0.5
    mask[:, 0] = True
    mask[1, 46:91] = False
    mask[2, :] = False
    mask[2, 0] = True
    logits = torch.tensor(raw, dtype=torch.float64, device=device, requires_grad=True)
    dist = GroupedDistribution().proba_distribution(logits)
    dist.apply_masking(mask)
    expected = np.zeros((3, 406))
    for row in range(3):
        active = [g for g in range(9) if mask[row, 1 + g * 45 : 1 + (g + 1) * 45].any()]
        species = [g for g in active if g < 8]
        kinds = [0] + ([1] if 8 in active else []) + ([2] if species else [])

        def softmax(values):
            w = np.exp(values - values.max())
            return w / w.sum()

        kp = dict(zip(kinds, softmax(raw[row, kinds])))
        pp = dict(zip(species, softmax(raw[row, 3 + np.array(species)]))) if species else {}
        expected[row, 0] = kp[0]
        for g in active:
            indices = np.arange(1 + g * 45, 1 + (g + 1) * 45)
            indices = indices[mask[row, indices]]
            mass = kp[1] if g == 8 else kp[2] * pp[g]
            expected[row, indices] = mass * softmax(raw[row, indices + 10])
    np.testing.assert_allclose(dist.probs.detach().cpu(), expected, atol=1e-12)
    expected_entropy = -(expected * np.log(np.maximum(expected, 1e-300))).sum(1)
    np.testing.assert_allclose(dist.entropy().detach().cpu(), expected_entropy, atol=1e-12)
    actions = dist.sample()
    joint_log = dist.log_prob(actions)
    np.testing.assert_allclose(
        joint_log.detach().cpu(), np.log(expected[np.arange(3), actions.cpu()]), atol=1e-12
    )
    (joint_log.mean() + dist.entropy().mean()).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[2]) == 0  # forced wait has no trainable choice
    with pytest.raises(ValueError, match="legal action"):
        dist.apply_masking(np.zeros((3, 406), dtype=bool))


def test_deterministic_selection_is_greedy_type_then_tile():
    raw = torch.zeros(1, 416)
    raw[0, 2] = 2
    raw[0, 3] = 0.5
    raw[0, 11 + 7] = 0.1
    dist = GroupedDistribution().proba_distribution(raw)
    # Wait has greater joint mass than any individual sunflower tile, but the
    # sunflower type has greatest type mass. Deterministic evaluation chooses it.
    assert dist.probs.argmax().item() == 0
    assert dist.mode().item() == 8


def test_task_restrictions_dig_cooldown_and_reset_boundaries():
    cfg = load_config()
    env = PvZEnv(cfg, training=True)
    env.reset(seed=3)
    assert env.episode_family == "placement"
    flower = env.codec.encode(Place("sunflower", 0, 0))
    _, _, _, _, info = env.step(flower)
    assert not info["accepted"] and env.public.sun == 100
    shooter = env.codec.encode(Place("peashooter", 0, 0))
    env.step(shooter)
    assert env.public.sun == 0
    assert not env.action_masks()[env.codec.encode(Place("peashooter", 1, 0))]
    assert env.action_masks()[env.codec.encode(Dig(0, 0))]
    before = env.action_masks()
    env.set_curriculum_stage(4)
    np.testing.assert_array_equal(before, env.action_masks())
    assert env.episode_family == "placement"
    env.reset(seed=3)
    assert env.episode_family == "preset"
    assert env.action_masks()[flower]
    cfg["environment"]["cutoff_seconds"] = 1
    env = PvZEnv(cfg, family="saving")
    env.reset(seed=3)
    for _ in range(99):
        assert env.step(0)[2:4] == (False, False)
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


@pytest.mark.learning
@pytest.mark.parametrize("device", ["cuda"])
def test_grouped_learning_save_reload_metrics_and_complete_update_deadline(
    smoke_cfg, tmp_path, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = smoke_cfg
    cfg["training"]["device"] = device
    run = tmp_path / device
    train(cfg, "masked", 101, run, validation_limit=1)
    model, _ = load_policy(run / "final.zip", device)
    assert (
        model.curriculum_state
        == CurriculumState(completed_stage_games=model.training_games).to_dict()
    )
    status = json.loads((run / "status.json").read_text())
    assert status["curriculum_incomplete"]
    metrics = [
        json.loads(line) for line in (run / "training-metrics.jsonl").read_text().splitlines()
    ]
    assert [m["training_steps"] for m in metrics] == [64, 128]
    assert all(m["optimization"]["type_entropy"] >= 0 for m in metrics)
    env = PvZEnv(cfg)
    obs, _ = env.reset(seed=100000)
    loaded, _ = load_policy(run / "final.zip", device)
    assert (
        model.predict(obs, deterministic=True, action_masks=env.action_masks())[0]
        == loaded.predict(obs, deterministic=True, action_masks=env.action_masks())[0]
    )
    timed = tmp_path / "timed"
    train(cfg, "masked", 101, timed, validation_limit=1, deadline=perf_counter() - 1)
    stopped, _ = load_policy(timed / "final.zip", device)
    assert stopped.num_timesteps == 0 and stopped._n_updates == 0


@pytest.mark.learning
def test_curriculum_probe_uses_same_policy_optimizer_and_persists_resume(
    smoke_cfg, tmp_path, monkeypatch, legacy_teaching
):
    cfg = legacy_teaching(smoke_cfg)
    cfg["training"].update(total_steps=384)
    cfg["curriculum"].update(probe_interval=64, minimum_stage_steps=64)
    identities = []
    real_probe = ResearchCallback.probe_curriculum

    def fake_evaluate(cfg, **kwargs):
        from pvz_rl.evaluation import evaluate

        if kwargs.get("split") == "curriculum_validation":
            return [
                {"win": 1, "family": kwargs["family"], "level": level, "scenario_seed": seed}
                for level in kwargs["levels"]
                for seed in kwargs["seeds"]
            ]
        return evaluate(cfg, **kwargs)

    monkeypatch.setattr("pvz_rl.training.evaluate", fake_evaluate)

    def probe(self):
        identities.append((id(self.model.policy), id(self.model.policy.optimizer)))
        real_probe(self)
        if self.model.num_timesteps == 128:
            raise KeyboardInterrupt()

    monkeypatch.setattr(ResearchCallback, "probe_curriculum", probe)
    first = tmp_path / "first"
    with pytest.raises(KeyboardInterrupt):
        train(cfg, "masked", 101, first, validation_limit=1)
    assert len(set(identities)) == 1
    checkpoint, _ = load_policy(first / "interrupted.zip")
    assert checkpoint.curriculum_state["stage"] == 1
    monkeypatch.setattr(ResearchCallback, "probe_curriculum", real_probe)
    second = tmp_path / "second"
    train(cfg, "masked", 101, second, validation_limit=1, resume=first / "interrupted.zip")
    resumed, _ = load_policy(second / "final.zip")
    assert resumed.num_timesteps == 384
    assert resumed.curriculum_state["stage"] == 3
    assert resumed.curriculum_state["consecutive_passes"] == 0
    rows = [
        json.loads(line) for line in (second / "training-episodes.jsonl").read_text().splitlines()
    ]
    assert any(r["family"] == "saving" for r in rows)


@pytest.mark.parametrize("family", ["placement", "saving"])
def test_lesson_checkpoints_cannot_enter_normal_or_final_evaluation(
    cfg, tmp_path, monkeypatch, family
):
    from pvz_rl.cli import main

    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"test")
    monkeypatch.setattr(
        "pvz_rl.training.load_policy",
        lambda p: (
            object(),
            {"config": cfg, "condition": "masked", "learner_seed": 101, "family": family},
        ),
    )
    args = ["evaluate", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="--family"):
        main(args)
    with pytest.raises(ValueError, match="not formal test evidence"):
        main([*args, "--family", family, "--split", "test"])


@pytest.mark.learning
def test_deadline_saves_only_completed_update(smoke_cfg, tmp_path, monkeypatch):
    cfg = smoke_cfg
    original = ResearchCallback.capture_update

    def expire_after_update(self):
        original(self)
        if self.model._n_updates:
            self.deadline = perf_counter() - 1

    monkeypatch.setattr(ResearchCallback, "capture_update", expire_after_update)
    train(cfg, "masked", 101, tmp_path / "stopped", validation_limit=1)
    model, _ = load_policy(tmp_path / "stopped/final.zip")
    assert model.num_timesteps == 64 and model._n_updates == 1
    status = json.loads((tmp_path / "stopped/status.json").read_text())
    assert status["budget_stopped"] and status["curriculum_incomplete"]


@pytest.mark.learning
def test_grouped_spawn_and_same_checkpoint_demos(tmp_path, tiny_cli_config):
    # Original Windows periodic-evaluation/export protocol, still loadable.
    tiny_cli_config.write_text(
        tiny_cli_config.read_text().replace(
            'validation_schedule = "stage_success"', 'validation_schedule = "periodic"'
        )
    )
    output = tmp_path / "grouped-spawn"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pvz_rl",
            "train",
            "--config",
            str(tiny_cli_config),
            "--steps",
            "64",
            "--n-envs",
            "2",
            "--rollout-steps-per-env",
            "32",
            "--batch-size",
            "32",
            "--eval-interval",
            "64",
            "--validation-count",
            "1",
            "--no-videos",
            "--device",
            "cuda",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    demos = json.loads((output / "visualizations/demos.json").read_text())["demos"]
    assert {d["level"] for d in demos} == {"easy", "standard", "hard"}
    assert len({d["checkpoint_hash"] for d in demos}) == 1
    assert all(d["replay"].endswith(".pvzdemo") for d in demos)
