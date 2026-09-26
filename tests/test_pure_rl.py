import numpy as np
import pytest
from pvz_game import Dig, Place

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv


def test_task_restrictions_dig_cooldown_and_reset_boundaries():
    cfg = load_config()
    env = PvZEnv(cfg, training=True)
    env.reset(seed=3)
    assert env.episode_family == "saving"
    flower = env.codec.encode(Place("sunflower", 0, 0))
    assert env.action_masks()[flower] and env.public.sun == 100
    shooter = env.codec.encode(Place("peashooter", 0, 0))
    env.step(shooter)
    assert env.public.sun == 0
    assert env.action_masks()[env.codec.encode(Place("peashooter", 1, 0))]
    assert not env.game.validate_action(Place("peashooter", 1, 0)).accepted
    assert env.action_masks()[env.codec.encode(Dig(0, 0))]
    before = env.action_masks()
    env.set_curriculum_stage(3)
    np.testing.assert_array_equal(before, env.action_masks())
    assert env.episode_family == "saving"
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


@pytest.mark.parametrize("family", ["saving"])
def test_lesson_checkpoints_cannot_enter_normal_or_final_evaluation(
    cfg, tmp_path, monkeypatch, family
):
    from pvz_rl.cli import main

    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"test")
    monkeypatch.setattr(
        "pvz_rl.learning.training.load_policy",
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
