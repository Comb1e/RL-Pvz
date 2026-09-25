"""Live selection, present-state capture, rendering, and training isolation."""

import copy
import random
from queue import Queue

import numpy as np
import pytest
import torch

from pvz_rl.config import load_config, output_settings, validate_config
from pvz_rl.presentation.live_view import (
    Activity,
    LiveSession,
    PanelState,
    Selection,
    draw_view,
    offer_latest,
)


class MemorySession:
    """Exercise the real CUDA capture without an OS window or process timing."""

    def __init__(self, count):
        self.selection = Selection(count, rng=random.Random(13))
        self.fps = 30
        self.enabled = True
        self.stats = {}
        self.published = []
        self.commands = []

    def poll(self):
        changed = False
        for command in self.commands:
            changed = self.selection.switch(*command) or changed
        self.commands.clear()
        return changed

    def publish(self, **kwargs):
        self.published = copy.deepcopy(self.selection.panels)

    def set_activity(self, activity):
        self.activity = activity

    def fail(self, error):
        raise AssertionError(error)

    def close(self):
        self.enabled = False


def test_selection_unique_private_rng_and_generation_rejection():
    random.seed(71)
    before = random.getstate()
    selection = Selection(128)
    original = [p.env for p in selection.panels]
    assert len(set(original)) == 4
    assert selection.switch(1, 0)
    assert selection.panels[1].env not in original
    assert [p.env for i, p in enumerate(selection.panels) if i != 1] == original[:1] + original[2:]
    assert not selection.switch(1, 0)
    assert not selection.accept(1, 0, {"sequence": 10, "outcome": "lost"})
    assert selection.accept(1, 1, {"sequence": 11, "outcome": "running"})
    assert not selection.accept(1, 1, {"sequence": 9, "outcome": "lost"})
    assert random.getstate() == before
    for count in (1, 2, 3, 4):
        small = Selection(count)
        assert len(small.panels) == count
        old = small.panels[0].env
        assert small.switch(0, 0) and small.panels[0].env == old
        assert len({p.env for p in small.panels}) == count


def test_bounded_delivery_replaces_old_frames():
    queue = Queue(maxsize=1)
    for value in range(100):
        assert offer_latest(queue, value)
    assert queue.qsize() == 1 and queue.get_nowait() == 99


@pytest.mark.parametrize(
    "key,value",
    [
        ("live_enabled", 1),
        ("live_fps", 0),
        ("live_fps", 31),
        ("live_window_size", [639, 480]),
        ("live_window_size", [640]),
    ],
)
def test_view_settings_invalid(key, value):
    cfg = load_config()
    cfg["visualization"][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_cli_and_old_checkpoint_output_compatibility(tmp_path, monkeypatch):
    from pvz_rl.cli import main
    from pvz_rl.learning.training_requirements import resume_protocol, transfer_protocol

    cfg = load_config()
    old = copy.deepcopy(cfg)
    for key in ("live_enabled", "live_fps", "live_window_size"):
        del old["visualization"][key]
    assert output_settings(old)["visualization"]["live_enabled"]
    validate_config(old)
    assert resume_protocol(cfg, "masked") == resume_protocol(old, "masked")
    assert transfer_protocol(cfg) == transfer_protocol(old)
    seen = []
    monkeypatch.setattr(
        "pvz_rl.learning.training.train",
        lambda cfg, *a, **kw: seen.append(cfg["visualization"]["live_enabled"]),
    )
    for flags in ([], ["--no-live-view"], ["--live-view"]):
        main(["train", "--output", str(tmp_path), *flags])
    assert seen == [True, False, True]
    with pytest.raises(SystemExit):
        main(["train", "--output", str(tmp_path), "--live-view", "--no-live-view"])


def attach_capture(env):
    from pvz_rl.presentation.live_capture import CudaLiveCapture

    session = MemorySession(env.num_envs)
    env.live_view = CudaLiveCapture(env, output_settings(env.cfg)["visualization"], session=session)
    return session


def test_capture_cpu_reference_terminals_actions_and_switches(per_tick_cfg):
    from pvz_game import Dig, Place

    from pvz_rl.envs.actions import ActionCodec
    from pvz_rl.envs.cuda_env import CudaVecEnv

    cfg = copy.deepcopy(per_tick_cfg)
    cfg["training"]["n_envs"] = 5
    cfg["environment"]["cutoff_seconds"] = 0.03
    env = CudaVecEnv(cfg, "masked", 101, family="saving")
    try:
        env.reset()
        session = attach_capture(env)
        codec = ActionCodec(cfg)
        for command in (Place("sunflower", 0, 0), Dig(0, 0)):
            actions = torch.full((5,), codec.encode(command), device="cuda")
            env.live_view.last_capture = -float("inf")
            env.step_tensors(actions)
            env.live_view.drain(wait=True)
            for panel in session.selection.panels:
                assert panel.frame["observation"] == env.batch.observe(panel.env)
        assert "Plant sunflower at A1" in session.selection.panels[0].frame["actions"][0]
        assert "Dig" in session.selection.panels[0].frame["actions"][1]
        for _ in range(3):
            env.step_tensors(torch.zeros(5, device="cuda", dtype=torch.long))
        env.live_view.drain(wait=True)
        for p in session.selection.panels:
            assert p.state == PanelState.RESULT
            assert p.frame["outcome"] == "truncated" and p.frame["observation"].tick == 3
            assert env.batch.observe(p.env).tick == 0
        old = [p.env for p in session.selection.panels]
        session.commands.append((2, 0))
        env.step_tensors(torch.zeros(5, device="cuda", dtype=torch.long))
        env.live_view.drain(wait=True)
        assert session.selection.panels[2].env not in old
        assert session.selection.panels[2].generation == 1
        assert session.selection.panels[2].frame["outcome"] == "running"
        assert all(
            p.frame["outcome"] == "truncated"
            for i, p in enumerate(session.selection.panels)
            if i != 2
        )
    finally:
        env.close()


def test_paused_completion_can_be_selected_without_restart_or_duplicate_actions(per_tick_cfg):
    from pvz_rl.envs.cuda_env import CudaVecEnv

    cfg = copy.deepcopy(per_tick_cfg)
    cfg["training"]["n_envs"] = 5
    cfg["environment"]["cutoff_seconds"] = 0.01
    env = CudaVecEnv(cfg, "masked", 101, family="saving")
    try:
        env.reset()
        session = attach_capture(env)
        action = torch.zeros(5, device="cuda", dtype=torch.long)
        env.step_tensors(action, autoreset=False)
        env.live_view.drain(wait=True)
        hashes = [env.batch.state_hash(i) for i in range(5)]
        assert not env.enabled_envs.any()
        assert sum(x["active_games"] for x in env.task_counts().values()) == 0
        session.commands.append((2, 0))
        env.step_tensors(action, autoreset=False)
        env.live_view.drain(wait=True)
        p = session.selection.panels[2]
        assert p.generation == 1 and p.state == PanelState.RESULT
        assert p.frame["outcome"] == "truncated" and not p.frame["actions"]
        assert [env.batch.state_hash(i) for i in range(5)] == hashes
        assert sum(x["completed_games"] for x in env.task_counts().values()) == 5
    finally:
        env.close()


@pytest.mark.parametrize("state", ["walking", "carrying_pole", "vaulting", "biting"])
def test_public_decoder_all_entity_fields(state):
    from pvz_game import Game, LevelSpec, Rules, Spawn

    from pvz_rl.envs.cuda_lessons import LessonCudaBatch
    from pvz_rl.presentation.live_capture import public_observation

    game = Game()
    game.reset(LevelSpec("display", (Spawn(100, "pole_vaulting", 1),)), 5)
    raw = game.snapshot()
    raw["tick"] = 100
    raw["spawn_index"] = 1
    raw["wave"] = 1
    raw["plants"] = [
        dict(
            id=100, kind="sunflower", row=1, col=0, health=280, state="ready", due=333, burst_due=0
        )
    ]
    raw["zombies"] = [
        dict(
            id=101,
            kind="pole_vaulting",
            row=1,
            x=2500,
            health=150,
            armor=0,
            state=state,
            slow_until=112,
            has_pole=True,
            vault_until=109,
            landing_x=1500,
            bite_progress=7,
            move_remainder=0,
            target_id=100,
        )
    ]
    raw["projectiles"] = [dict(id=102, row=1, x=1234, damage=20, icy=True, move_remainder=2)]
    raw["mowers"][1].update(state="moving", x=900)
    raw["next_id"] = 103
    # A restore requires extra projectile capacity beyond reachable play.
    from pvz_game.cuda.backend import projectile_bound

    batch = LessonCudaBatch(
        1, zombie_capacity=100, projectile_capacity=projectile_bound(Rules()) + 1
    )
    batch.restore([raw])
    arrays = {k: v[0].get() for k, v in batch.public_state_device().items()}
    assert public_observation(arrays, game.rules, raw["level"]["name"]) == batch.observe(0)


def test_complete_training_cohorts_identical_with_viewer(smoke_cfg, tmp_path):
    from stable_baselines3.common.logger import configure

    from pvz_rl.learning.training import build_model, vector_env
    from pvz_rl.monitoring.benchmark import policy_digest

    cfg = copy.deepcopy(smoke_cfg)
    cfg["training"].update(n_envs=2, batch_size=16)
    cfg["training"]["performance"]["compile_kernels"] = False
    results = []
    for enabled in (False, True):
        env = vector_env(cfg, "masked", 101, "saving")
        try:
            model = build_model(cfg, "masked", env, 101)
            model.set_logger(configure(format_strings=[]))
            if enabled:
                attach_capture(env)
            captured = []
            synchronize = model._synchronize

            def capture(callback):
                captured.append(model._buffer.take(np.arange(model._buffer.size)))
                synchronize(callback)

            model._synchronize = capture
            model.learn(201)  # Waiting cohort, then planting with actor updates.
            assert model.policy.optimizer.state
            del model._synchronize
            buffer = np.concatenate(captured)
            model.save(tmp_path / f"model-{enabled}.zip")
            results.append(
                (
                    policy_digest(model),
                    torch.from_numpy(buffer["action"].astype(np.int64)),
                    torch.from_numpy(buffer["reward"].copy()),
                    torch.from_numpy(buffer["observation"].copy()),
                    model.policy.optimizer.state_dict(),
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(),
                )
            )
        finally:
            env.close()
    a, b = results
    assert a[0] == b[0]
    for index in (1, 2, 3, 5, 6):
        assert torch.equal(a[index], b[index])
    for key, state in a[4]["state"].items():
        for name, value in state.items():
            assert (
                torch.equal(value, b[4]["state"][key][name])
                if torch.is_tensor(value)
                else value == b[4]["state"][key][name]
            )


def test_process_lifecycle_and_offscreen_render(tmp_path, monkeypatch):
    import pygame
    from pvz_game import Game, Place

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings = output_settings(load_config())["visualization"]
    session = LiveSession(128, settings)
    try:
        assert session.ready.wait(12)
        game = Game()
        game.reset("easy", 1)
        game.step(Place("sunflower", 1, 1), ticks=1)
        obs = game.observe()
        for p in session.selection.panels:
            p.frame = dict(
                observation=obs,
                task="saving",
                episode=1,
                outcome="running",
                actions=["Planted sunflower"],
                sequence=1,
            )
        session.publish(force=True)
        session.set_activity(Activity.VALIDATING)
        assert session.activity.value == Activity.VALIDATING
        pygame.font.init()
        surface = pygame.Surface((1600, 1050))
        packets = [
            dict(env=p.env, generation=p.generation, state=p.state, frame=p.frame)
            for p in session.selection.panels
        ]
        buttons = draw_view(surface, packets, Activity.COLLECTING, renderers={}, boards={})
        assert len(buttons) == 4
        pygame.image.save(surface, tmp_path / "four-games.png")
        # Automatic replacement is requested only after the result has been
        # displayed. Commands do not directly touch the collector selection.
        p = session.selection.panels[0]
        p.frame = {**p.frame, "outcome": "won", "sequence": 2}
        p.state = PanelState.RESULT
        session.publish(force=True)
        assert session.commands.get(timeout=5) == (0, p.generation)
        assert p.state == PanelState.RESULT
        # Closing the UI is independent of a training stop.
        session.closed.set()
        session.process.join(timeout=5)
        assert not session.enabled
    finally:
        session.close()
    assert not session.process.is_alive()
    session.close()


def test_display_failure_is_nonfatal_and_reported_once(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "missing-pvz-driver")
    errors = []
    session = LiveSession(4, output_settings(load_config())["visualization"], notify=errors.append)
    try:
        session.process.join(12)
        session.poll()
        session.poll()
        assert not session.enabled and len(errors) == 1
        assert "training continues" in errors[0]
    finally:
        session.close()


def test_mixed_natural_results_and_capture_never_reads_private_templates(per_tick_cfg):
    from pvz_game import LevelSpec, Spawn

    from pvz_rl.envs.cuda_env import CudaVecEnv

    cfg = copy.deepcopy(per_tick_cfg)
    cfg["training"]["n_envs"] = 3
    env = CudaVecEnv(cfg, "masked", 101, family="saving")
    try:
        env.reset()
        from pvz_game import Game

        game = Game()
        game.reset(LevelSpec("win", ()), 1)
        win = game.snapshot()
        game.reset(LevelSpec("loss", (Spawn(1, "basic", 0),), mowers=False), 1)
        game.step(ticks=1)
        loss = game.snapshot()
        loss["zombies"][0]["x"] = env.batch.rules.game["house_x"] + 1
        loss["zombies"][0]["move_remainder"] = 199
        env.batch.restore([win, loss], indices=[0, 1])
        env._level_names[:2] = ["win", "loss"]
        session = attach_capture(env)
        *_, infos = env.step_tensors(torch.zeros(3, device="cuda", dtype=torch.long))
        # Flush after resets: frozen numeric copies must still contain results.
        env.live_view.drain(wait=True)
        results = {p.env: p.frame for p in session.selection.panels}
        assert results[0]["outcome"] == "won"
        assert results[1]["outcome"] == "lost"
        assert results[2]["outcome"] == "running"
        assert infos[0]["episode_metrics"]["win"] == 1
        assert infos[1]["episode_metrics"]["win"] == 0
        assert not any(key.startswith("_") for key in env.live_view.source)
        assert set(env.live_view.source) == {
            "header",
            "plants",
            "zombies",
            "projectiles",
            "mowers",
            "cooldowns",
        }
    finally:
        env.close()
