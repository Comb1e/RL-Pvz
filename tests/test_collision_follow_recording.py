import pytest


def test_single_config_train_and_demo_profiles():
    from pathlib import Path
    from types import SimpleNamespace

    from pvz_rl.cli import configured
    from pvz_rl.config import load_config, load_demo_config, validate_config
    from pvz_rl.learning.exploration import exploration_state
    from pvz_rl.learning.training_requirements import require_supported_policy

    training, demo = load_config(), load_demo_config()
    assert training == load_config("src/pvz_rl/data/train.toml")
    assert demo == load_demo_config("src/pvz_rl/data/train.toml")
    assert demo["policy"] == training["policy"]
    assert demo["training"]["method"] == training["training"]["method"]
    assert demo["reward"]["invalid_plant_penalty"] == training["reward"]["invalid_plant_penalty"]
    assert demo["training"]["max_grad_norm"] == training["training"]["max_grad_norm"] == 5
    assert "gradient_clip" not in demo["training"]["demo"]
    assert training["policy"]["kind"] == "transformer_lstm_q_v3"
    assert tuple(training["curriculum"]["stages"]) == ("easy", "standard", "shared")
    assert "lessons" not in training["curriculum"] and "run_stage" not in training["curriculum"]
    require_supported_policy(training)
    from pvz_rl.learning.training_requirements import require_cuda_training

    require_cuda_training(training, runtime=False)
    require_cuda_training(demo, runtime=False)
    assert {p.stem for p in Path("src/pvz_rl/data").glob("*.toml")} == {"train"}
    assert configured(SimpleNamespace(command="record-demo", config=None)) == demo
    assert configured(SimpleNamespace(command="initialize-demo", config=None)) == demo
    assert demo["encoding"]["version"] == "entity_v1"
    assert exploration_state(demo, 0).tile_epsilon == 0.5
    assert exploration_state(demo, 10000).tile_epsilon == pytest.approx(0.01)
    with pytest.raises(ValueError, match="not available"):
        load_demo_config("src/pvz_rl/data/train.toml", profile="event-memory")
    for invalid in (0, -1, float("inf"), float("nan"), True):
        training["training"]["max_grad_norm"] = invalid
        with pytest.raises(ValueError, match="training.max_grad_norm"):
            validate_config(training)


@pytest.mark.parametrize("demo", [False, True])
def test_profiles_reject_mismatched_optimizer(demo):
    from pvz_rl.config import load_config, load_demo_config, validate_config

    cfg = load_demo_config() if demo else load_config()
    cfg["training"]["method"] = "sequential_q_mc_v2"
    with pytest.raises(ValueError, match="complete-return"):
        validate_config(cfg)


@pytest.mark.parametrize("name", ["coalesce_masks", "cache_rollout_on_device", "allowed_plants"])
def test_removed_settings_are_rejected(name):
    from pvz_rl.config import load_config, validate_config

    cfg = load_config()
    if name == "allowed_plants":
        cfg["curriculum"]["lessons"] = {"obsolete": {name: ["peashooter"]}}
    else:
        cfg["runtime"][name] = True
    with pytest.raises(ValueError, match="Runtime|Retired"):
        validate_config(cfg)


def test_follow_latest_clears_history_but_preserves_horizontal_scroll():
    from pvz_rl.presentation.live_layout import Browse, HistoryMode

    state = Browse(
        start=12,
        horizontal=240,
        selected={"sequence": 4},
        selected_offset=12,
        page={"start": 8, "total": 13, "rows": []},
    )
    state.resume_follow({"start": 0, "total": 13, "rows": []})
    assert state.mode == HistoryMode.FOLLOWING
    assert state.selected is None and state.selected_offset is None
    assert state.start == 13 and state.horizontal == 240
    assert state.request == 1


def test_follow_latest_handles_empty_history_without_page_requests():
    from pvz_rl.presentation.live_layout import Browse, HistoryMode

    state = Browse(horizontal=9, page={"start": 0, "total": 2, "rows": [{"sequence": 1}]})
    state.resume_follow(None)
    assert state.mode == HistoryMode.FOLLOWING
    assert state.page is None and state.start == 0
    assert state.horizontal == 9 and state.request == 1


@pytest.mark.parametrize("focus", [None, 2])
@pytest.mark.parametrize("control", ["key", "button"])
def test_follow_controls_in_real_viewer_loop(tmp_path, monkeypatch, focus, control):
    from queue import Queue
    from threading import Event
    from types import SimpleNamespace

    import pygame
    from pvz_game import Game

    from pvz_rl.presentation import live_layout as layout
    from pvz_rl.presentation.live_view import Activity

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    record = dict(
        sequence=1,
        tick=0,
        action=1,
        greedy_action=1,
        q=[0.0] * 10,
        legal=[True] * 10,
        coins=[False, False],
        accepted=True,
    )
    latest = {**record, "sequence": 9}
    frame = dict(
        observation=Game().reset("easy", 1000),
        task="easy",
        episode=1,
        outcome="running",
        sequence=9,
        decision=latest,
        history=dict(start=8, total=9, rows=[latest]),
    )
    packets = [dict(env=i, generation=0, state="watching", frame=frame) for i in range(4)]
    frames, commands, responses, errors = Queue(), Queue(), Queue(), Queue()
    frames.put(packets)
    original_draw = layout.draw_view
    captured = {}
    renders = 0

    def draw(*args, **kwargs):
        nonlocal renders
        renderers = kwargs["renderers"]
        if not renders:
            renderers["focus"] = focus
            renderers["browse"] = {}
            for i in range(4):
                state = layout.Browse(page=dict(start=0, total=9, rows=[record]), horizontal=50)
                state.choose(record)
                renderers["browse"][(i, 0, 1)] = state
        buttons = original_draw(*args, **kwargs)
        captured.update(renderers=renderers, buttons=buttons)
        renders += 1
        if renders == 1:
            # Outstanding page from before F must not restore historical data.
            responses.put(
                dict(
                    panel=2,
                    env=2,
                    generation=0,
                    episode=1,
                    request=1,
                    page=dict(start=0, total=9, rows=[record]),
                )
            )
        return buttons

    event_calls = 0

    def events():
        nonlocal event_calls
        event_calls += 1
        if event_calls == 1:
            return []
        if event_calls in (2, 3):
            if control == "key":
                return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f)]
            rect = next(
                r
                for r, i, target in captured["buttons"]
                if target[0] == "follow" and i == (focus or 0)
            )
            return [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center)]
        return [pygame.event.Event(pygame.QUIT)]

    monkeypatch.setattr(layout, "draw_view", draw)
    monkeypatch.setattr(pygame.event, "get", events)
    layout.viewer_main(
        frames,
        commands,
        responses,
        errors,
        Event(),
        Event(),
        SimpleNamespace(value=Activity.COLLECTING),
        (1600, 1050),
        4,
    )
    assert errors.empty(), list(errors.queue)
    affected = {focus} if focus is not None else set(range(4)) if control == "key" else {0}
    for (i, _, _), state in captured["renderers"]["browse"].items():
        assert state.follow == (i in affected)
        assert state.horizontal == 50
        if i in affected:
            assert state.selected is None and state.selected_offset is None
            assert state.page["rows"] == [latest] and state.request == 3


def test_follow_skips_unavailable_and_replacement_panels():
    from pvz_rl.presentation.live_layout import Browse, resume_visible_history

    old = Browse(page={"start": 0, "total": 1, "rows": []})
    old.choose({"sequence": 1})
    packets = [
        dict(env=7, generation=2, state="selecting", frame={"episode": 2}),
        dict(env=8, generation=1, state="watching", frame=None),
    ]
    renderers = dict(browse_keys={0: (0, 1, 1)}, browse={(0, 1, 1): old})
    resume_visible_history(packets, renderers)
    assert not old.follow and old.selected["sequence"] == 1
    packets[0].update(state="watching", frame={"episode": 3})
    resume_visible_history(packets, renderers)  # replacement not drawn yet
    assert not old.follow and len(renderers["browse"]) == 1


@pytest.mark.parametrize("artifact", ["replay", "archive", "manifest", "history"])
@pytest.mark.parametrize("contents", [b"", b"incomplete previous attempt\n"])
def test_existing_outputs_rejected_before_window_and_preserved(
    tmp_path, monkeypatch, artifact, contents
):
    import pygame

    from pvz_rl.config import load_demo_config
    from pvz_rl.presentation.demo_recording import DemoRecordingApp

    replay, archive = tmp_path / "human.pvzdemo", tmp_path / "human.jsonl"
    outputs = dict(
        replay=replay,
        archive=archive,
        manifest=archive.with_suffix(".jsonl.manifest.json"),
        history=archive.with_suffix(".jsonl.history.json"),
    )
    outputs[artifact].write_bytes(contents)
    monkeypatch.setattr(
        pygame.display, "init", lambda: pytest.fail("opened window before validation")
    )
    with pytest.raises(FileExistsError) as failure:
        DemoRecordingApp(load_demo_config(), replay_path=replay, archive_path=archive)
    assert str(outputs[artifact]) in str(failure.value)
    assert outputs[artifact].read_bytes() == contents
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    "replay_name", ["human.jsonl.manifest.json", "human.jsonl.history.json", "human.jsonl/child"]
)
def test_overlapping_sidecar_and_parent_paths_rejected(tmp_path, replay_name):
    from pvz_rl.presentation.demo_recording import validate_recording_outputs

    with pytest.raises(ValueError, match="overlap"):
        validate_recording_outputs(tmp_path / replay_name, tmp_path / "human.jsonl")
    assert not list(tmp_path.iterdir())


def test_human_recorder_headless_startup_uses_action_phase_adapter(tmp_path, monkeypatch):
    import pygame
    from pvz_game import Dig, Place

    from pvz_rl.config import load_demo_config
    from pvz_rl.presentation.demo_recording import DemoRecordingApp

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    replay = tmp_path / "human.pvzdemo"
    archive = tmp_path / "human.jsonl"
    app = DemoRecordingApp(load_demo_config(), replay_path=replay, archive_path=archive)
    try:
        app.pending.append(Place("sunflower", 0, 0))
        app.advance()
        assert app.game.observe().tick == 0
        app.pending.append(Place("sunflower", 0, 0))
        app.advance()
        assert app.game.observe().tick == 1
        assert not app.transition_archive.records[-1]["accepted"]
        app.pending.append(Dig(0, 0))
        app.toggle_pause()
        before = app.game.state_hash()
        app.update(1)
        assert app.game.state_hash() == before and len(app.pending) == 1
        assert app.mode.value == "paused"
        app.draw()
        assert "restart" not in app.buttons and "menu" not in app.buttons
        for operation in ("restart", "menu", "level:hard"):
            app.activate(operation)
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        assert app.game.state_hash() == before and app.level == "easy"
        app.toggle_pause()
        app.advance()
        assert app.game.observe().tick == 1 and not app.game.observe().plants
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        app.run()
        assert replay.exists() and archive.exists()
        from pvz_rl.presentation.recordings import verify_replay

        assert verify_replay(replay).state_hash() == app.game.state_hash()
        assert not app.transition_archive.verify()["complete"]
    finally:
        app.transition_archive.close()
        pygame.quit()


def test_easy_recording_natural_completion_is_reusable(tmp_path, monkeypatch):
    import pygame
    from pvz_game import Status

    from pvz_rl.config import load_demo_config
    from pvz_rl.learning.demo_initialization import _load_verified_demo, _training_tensors
    from pvz_rl.presentation.demo_recording import DemoRecordingApp

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    cfg = load_demo_config()
    archive, replay = tmp_path / "complete.jsonl", tmp_path / "complete.pvzdemo"
    app = DemoRecordingApp(cfg, archive_path=archive, replay_path=replay)
    try:
        for _ in range(40000):
            if app.game.observe().status != Status.RUNNING:
                break
            app.advance()
        assert app.game.observe().status == Status.LOST
        assert app._archive_finalized and app.transition_archive.closed
        cfg["reward"]["loss_penalty"] = 3.5
        verified = _load_verified_demo(archive, replay, cfg)
        verification = verified.report
        assert verification["verified"]
        tensors = _training_tensors(verified)
        assert len(tensors[0]) == verification["archive"]["decisions"]
        assert tensors[-1][-1].item() == pytest.approx(-3.5)
        saved = (archive.read_bytes(), replay.read_bytes())
        app.restart()
        app.advance()
        assert saved == (archive.read_bytes(), replay.read_bytes())
    finally:
        app.transition_archive.close()
        pygame.quit()
