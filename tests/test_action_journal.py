"""Independent current-game history, bounded paging and archive controls."""

from zipfile import ZipFile

import numpy as np

from pvz_rl.learning.cuda_buffer import trajectory_dtype
from pvz_rl.presentation.action_journal import ActionJournal


def batch(n=5):
    rows = np.zeros(n, dtype=trajectory_dtype(286, 48))
    rows["active"] = True
    masks = np.zeros((n, 406), dtype=bool)
    masks[:, [0, 1, 361]] = True
    rows["mask"] = np.packbits(masks, axis=1, bitorder="little")
    rows["action"] = 1
    return rows, np.arange(n * 10, dtype=np.float32).reshape(n, 10), np.tile([1, 0], (n, 1))


def test_entire_offscreen_game_spills_pages_and_restores(tmp_path):
    journal = ActionJournal(5, ram_bytes=500, block_rows=2)
    restored = ActionJournal(5, ram_bytes=0, block_rows=2)
    try:
        rows, scores, outcomes = batch()
        for decision in range(150):
            rows["tick"] = decision // 4  # Instantaneous actions share timestamps.
            rows["action"] = 361 if decision % 2 else 1
            rows["greedy_action"] = 1
            outcomes[4] = [0, 1] if decision == 17 else [1, 0]
            journal.record_batch(rows, scores + decision, outcomes, [7] * 5)
        assert journal.ram_used <= 500 and len(list(journal.root.glob("*.npy"))) > 0
        first = journal.page(4, 7, 0)
        assert first["total"] == 150 and len(first["rows"]) == 64
        assert first["rows"][17]["accepted"] is False and first["rows"][17]["reason"] == 1
        assert first["rows"][17]["q"] == (scores[4] + 17).tolist()
        assert first["rows"][17]["legal"] == [True] * 10
        assert first["rows"][0]["tick"] == first["rows"][1]["tick"]
        assert first["rows"][0]["sequence"] != first["rows"][1]["sequence"]
        assert journal.page(4, 6, 0) is None
        with ZipFile(tmp_path / "checkpoint.zip", "w") as archive:
            journal.write_archive(archive)
        with ZipFile(tmp_path / "checkpoint.zip") as archive:
            restored.restore_archive(archive)
        assert restored.page(4, 7, 0) == first
        rows["action"] = 0
        restored.record_batch(rows, scores, outcomes, [7] * 5)
        assert restored.sizes == [150] * 5
        assert restored.page(4, 7)["latest"]["sequence"] == 151
        restored.reset(4, 8)
        assert restored.page(4, 8)["rows"] == []
        assert restored.page(3, 7)["total"] == 150
        assert journal.page(4, 7, 0) == first
    finally:
        journal.close()
        restored.close()


def test_switch_keeps_previous_board_until_atomic_destination_frame():
    import random

    from pvz_rl.presentation.live_view import PanelState, Selection

    selection = Selection(8, rng=random.Random(1))
    old = dict(sequence=1, outcome="running", episode=3, history={"rows": ["old"]})
    selection.accept(0, 0, old)
    selection.switch(0, 0)
    panel = selection.panels[0]
    assert panel.state == PanelState.SELECTING and panel.frame is old
    assert not selection.accept(0, 0, dict(sequence=50, outcome="running"))
    new = dict(sequence=2, outcome="running", episode=9, history={"rows": ["destination"]})
    assert selection.accept(0, 1, new)
    assert panel.frame is new and panel.state == PanelState.WATCHING


def test_pages_reject_other_episodes_generations_and_environments():
    from pvz_rl.presentation.live_view import Selection, history_page

    journal = ActionJournal(5)
    try:
        rows, scores, results = batch()
        journal.record_batch(rows, scores, results, [7] * 5)
        selection = Selection(5)
        env = selection.panels[0].env
        request = dict(panel=0, generation=0, env=env, episode=7, start=0)
        assert history_page(selection, journal, request)["rows"][0]["q"] == scores[env].tolist()
        for key, value in [("generation", 1), ("episode", 6), ("env", (env + 1) % 5)]:
            assert history_page(selection, journal, {**request, key: value}) is None
        selection.switch(0, 0)
        assert history_page(selection, journal, request) is None
    finally:
        journal.close()


def test_diagnostic_spill_failure_is_visible_and_does_not_raise(monkeypatch):
    import pytest

    journal = ActionJournal(5, ram_bytes=0)
    try:

        def failed(_):
            raise OSError("controlled disk failure")

        monkeypatch.setattr(journal, "_allocate", failed)
        rows, scores, results = batch()
        with pytest.warns(RuntimeWarning, match="training continues"):
            journal.record_batch_safe(rows, scores, results, [1] * 5)
        assert journal.disabled
        journal.record_batch_safe(rows, scores, results, [1] * 5)
    finally:
        journal.close()


def test_selecting_a_row_keeps_its_page_position_and_follow_is_explicit():
    from pvz_rl.presentation.live_layout import Browse, HistoryMode

    rows = [dict(sequence=100 + i, tick=20) for i in range(10)]
    state = Browse(page=dict(start=60, total=70, rows=rows))
    state.choose(rows[8])
    assert state.mode == HistoryMode.BROWSING
    assert state.start == 60 and state.selected_offset == 68
    assert state.page["rows"] is rows and len(rows) == 10
    assert state.selected == rows[8] and state.selected is not rows[8]
    assert state.request == 1 and state.sent == float("inf")
    state.follow = True
    assert state.mode == HistoryMode.FOLLOWING


def test_diagnostic_restore_and_cleanup_failures_do_not_abort_training(tmp_path, monkeypatch):
    import pytest

    journal = ActionJournal(5, ram_bytes=0, block_rows=2)
    try:
        rows, scores, results = batch()
        journal.record_batch(rows, scores, results, [1] * 5)
        path_type = type(journal.root)
        original = path_type.unlink

        def denied(path, *args, **kwargs):
            if path.parent == journal.root:
                raise OSError("controlled unlink failure")
            return original(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(path_type, "unlink", denied)
            with pytest.warns(RuntimeWarning, match="training continues"):
                journal.reset(0, 2)
        assert journal.disabled and journal.page(1, 1) is None
        with ZipFile(tmp_path / "optional.zip", "w") as archive:
            journal.write_archive(archive)
            assert archive.namelist() == []
    finally:
        journal.close()

    restored = ActionJournal(5)
    try:
        with ZipFile(tmp_path / "broken.zip", "w") as archive:
            archive.writestr("journal/state.json", "{invalid json")
        with ZipFile(tmp_path / "broken.zip") as archive:
            with pytest.warns(RuntimeWarning, match="training continues"):
                restored.restore_archive_safe(archive)
        assert restored.disabled
    finally:
        restored.close()


def test_pending_switch_cannot_mix_destination_history_with_previous_board():
    from pvz_rl.presentation.live_layout import Browse, accept_history_response

    old = dict(rows=["old game"])
    state = Browse(page=old, request=2)
    state.follow = False
    browse = {(0, 4, 1): state}
    packet = dict(env=7, generation=4, state="selecting", frame=dict(episode=1))
    response = dict(panel=0, env=7, generation=4, episode=1, request=2, page=dict(rows=["new"]))
    assert not accept_history_response([packet], browse, response)
    assert state.page is old
    packet["state"] = "watching"  # Destination board and first page have arrived together.
    for key, value in (("env", 6), ("generation", 3), ("episode", 2), ("request", 1)):
        assert not accept_history_response([packet], browse, {**response, key: value})
    assert accept_history_response([packet], browse, response)
    assert state.page == response["page"]
    state.follow = True
    assert not accept_history_response([packet], browse, response)


def test_128_game_interruption_restores_actual_scores_and_optimizer(tmp_path):
    import copy

    import pytest
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from pvz_rl.config import load_config
    from pvz_rl.learning.cuda_q import CudaSequentialQ
    from pvz_rl.learning.training import build_model, vector_env
    from pvz_rl.monitoring.benchmark import policy_digest

    cfg = load_config()
    cfg["environment"]["cutoff_seconds"] = 1  # Explicit controlled short episode.
    cfg["visualization"]["live_history_ram_mib"] = 0  # Exercise disk and archive.
    cfg["training"]["performance"]["token_cache_gib"] = 0
    outcomes = []

    class Interrupt(BaseCallback):
        def _on_step(self):
            return self.n_calls < 2

    for interrupted in (False, True):
        env = vector_env(copy.deepcopy(cfg), "masked", 101, "saving")
        try:
            model = build_model(cfg, "masked", env, 101)
            model.set_logger(configure(format_strings=[]))
            from pvz_rl.learning.exploration import set_exploration_rate

            set_exploration_rate(model, 0.0)
            with torch.no_grad():
                model.policy.branch_head[-1].bias[1] = 0.1
                model.policy.branch_head[-1].bias[-1] = -0.2
            if interrupted:
                with pytest.raises(KeyboardInterrupt):
                    model.learn(1, callback=Interrupt())
                page = env.action_journal.page(127, env._episode_serial[127])
                assert [r["action"] for r in page["rows"]] == [1, 2]
                assert [r["sequence"] for r in page["rows"]] == [1, 2]
                assert [r["tick"] for r in page["rows"]] == [0, 0]
                assert page["rows"][0]["q"][1] == pytest.approx(0.1)
                assert page["rows"][1]["q"] == page["rows"][0]["q"]
                model.save(tmp_path / "interrupted.zip")
                model = CudaSequentialQ.load(tmp_path / "interrupted.zip", env=env)
                model.set_logger(configure(format_strings=[]))
                model.learn(1, reset_num_timesteps=False)
            else:
                model.learn(1)
            assert model.training_games == 128
            assert model.cohort_metrics["q_optimizer_steps"] == 52
            assert model.cohort_metrics["planting_samples"] == 128 * 101
            outcomes.append(
                (
                    policy_digest(model),
                    model.policy.optimizer.state_dict(),
                    env.action_journal.page(127, env._episode_serial[127]),
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(),
                )
            )
        finally:
            env.close()
    a, b = outcomes
    assert a[0] == b[0] and a[2] == b[2]
    for key, state in a[1]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, b[1]["state"][key][name])
    assert torch.equal(a[3], b[3]) and torch.equal(a[4], b[4])


def test_page_selection_survives_scroll_and_stale_requests():
    from pvz_rl.presentation.live_layout import Browse, accept_history_response

    rows = [dict(sequence=5 + i, q=[float(i)] * 10) for i in range(64)]
    page = dict(start=64, total=200, rows=rows)
    state = Browse(page=page)
    state.choose(rows[63])
    original = list(state.selected["q"])
    assert state.selected_offset == 127 and page["start"] == 64 and len(page["rows"]) == 64
    rows[63]["q"][0] = -999  # Selection owns the recorded vector independently.
    state.start = 128
    state.request += 1
    packets = [dict(env=4, generation=2, state="watching", frame=dict(episode=3))]
    browse = {(0, 2, 3): state}
    response = dict(
        panel=0,
        env=4,
        generation=2,
        episode=3,
        request=state.request,
        page=dict(start=128, total=201, rows=[dict(sequence=500)]),
    )
    assert accept_history_response(packets, browse, response)
    assert state.selected["q"] == original and state.selected["sequence"] == 68
    assert not accept_history_response(packets, browse, {**response, "request": state.request - 1})
    assert state.page["start"] == 128 and state.selected_offset == 127
    state.follow = True
    assert not accept_history_response(packets, browse, response)


def test_journal_penalties_and_exact_q_survive_archive(tmp_path):
    from pvz_rl.config import load_config
    from pvz_rl.presentation.live_layout import result_label

    settings = load_config()["reward"]
    a = ActionJournal(5, ram_bytes=0, block_rows=2, reward_settings=settings)
    b = ActionJournal(5, ram_bytes=0, block_rows=2, reward_settings=settings)
    try:
        rows, scores, results = batch()
        rows["action"] = [1, 2, 361, 362, 0]
        results[:] = [[0, 3], [0, 2], [0, 4], [1, 0], [1, 0]]
        a.record_batch(rows, scores, results, [7] * 5)
        assert a.latest[0]["penalty"] == -settings["invalid_plant_penalty"]
        assert a.latest[2]["penalty"] == -settings["empty_dig_penalty"]
        assert a.latest[3]["penalty"] == 0
        assert "automatic wait: insufficient_sun" in result_label(a.page(0, 7)["latest"])
        assert "penalty -0.000333333" in result_label(a.page(2, 7)["latest"])
        with ZipFile(tmp_path / "history.zip", "w") as archive:
            a.write_archive(archive)
        with ZipFile(tmp_path / "history.zip") as archive:
            b.restore_archive(archive)
        for env in range(5):
            assert a.page(env, 7) == b.page(env, 7)
            assert b.page(env, 7)["latest"]["q"] == scores[env].tolist()
            assert b.page(env, 7)["latest"]["legal"] == [True] * 10
    finally:
        a.close()
        b.close()
