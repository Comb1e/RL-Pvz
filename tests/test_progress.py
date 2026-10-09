"""Independent terminal cadence, output modes and failure events."""

import io
from types import SimpleNamespace

import pytest

from pvz_rl.monitoring.progress import CollectionProgress, Phase, ProgressReporter


@pytest.mark.parametrize("interactive", [False, True])
def test_progress_events_cadence_duplicates_and_log(tmp_path, interactive):
    now, stream = [0.0], io.StringIO()
    with ProgressReporter(
        tmp_path / "train.log", clock=lambda: now[0], stream=stream, interactive=interactive
    ) as reporter:
        assert reporter.due()
        assert reporter.emit("games 1", change=1)
        now[0] = 15
        assert reporter.due()
        assert not reporter.emit("games 2", change=2)
        now[0] = 60
        assert reporter.emit("games 2", change=2)
        now[0] = 120
        assert not reporter.emit("elapsed changed | games 2", change=2)
        reporter.phase(Phase.UPDATING, "fit starts")
        assert not reporter.emit("fit starts", force=True)
        assert reporter.emit("Checkpoint saved: latest.zip", force=True)
        reporter.warn("\x1b[31mforced failure\x1b[0m")
        reporter.phase(Phase.FAILED, "stopped")
        assert reporter.emit("Report refreshed", force=True, key="report")
        assert reporter.emit("different event", force=True)
        assert not reporter.emit("Report refreshed", force=True, key="report")
    terminal, log = stream.getvalue(), (tmp_path / "train.log").read_text("utf-8")
    assert "\x1b" not in terminal + log and "\r" not in log
    assert ("\r" in terminal) == interactive
    assert len(log.splitlines()) == 8
    if not interactive:
        assert terminal == log
    assert "[failed] stopped" in log and "Warning: forced failure" in log


@pytest.mark.parametrize("width", [80, 120])
def test_narrow_routine_preserves_cohort_rewards(tmp_path, monkeypatch, width):
    stream = io.StringIO()
    monkeypatch.setattr(
        "pvz_rl.monitoring.progress.shutil.get_terminal_size",
        lambda *args: SimpleNamespace(columns=width),
    )
    text = (
        "cohort 21 done 128/128 wins 63/128 Rraw=-0.776 Rtrain=-0.9~ | throughput 800/s | GPU 40%"
    )
    with ProgressReporter(tmp_path / "train.log", stream=stream, interactive=True) as reporter:
        reporter.emit(text)
    terminal = stream.getvalue()
    assert "wins 63/128" in terminal and "Rraw=-0.776 Rtrain=-0.9~" in terminal
    assert "throughput 800/s | GPU 40%" in (tmp_path / "train.log").read_text("utf-8")


@pytest.mark.parametrize(
    "width,interactive,force",
    [
        (80, True, False),
        (120, True, False),
        (200, True, False),
        (120, False, False),
        (120, True, True),
    ],
)
def test_routine_width_prioritizes_rewards_and_rates_without_shortening_logs(
    tmp_path, monkeypatch, width, interactive, force
):
    stream = io.StringIO()
    monkeypatch.setattr(
        "pvz_rl.monitoring.progress.shutil.get_terminal_size",
        lambda *args: SimpleNamespace(columns=width),
    )
    cohort = "cohort 21 done 64/128 wins 31/128 Rraw=-0.776 Rtrain=-0.9~"
    rates = CollectionProgress.text(
        dict(
            active_games=64,
            interval_decisions_per_game_per_second=12.34,
            interval_active_transitions_per_second=790,
        ),
        128,
    )
    text = f"{cohort} | {rates} | 64/128 games | GPU 40%"
    with ProgressReporter(
        tmp_path / "train.log", stream=stream, interactive=interactive
    ) as reporter:
        assert reporter.emit(text, force=force)
    terminal = stream.getvalue()
    log = (tmp_path / "train.log").read_text("utf-8")
    assert cohort in terminal
    assert log.endswith(text + "\n")
    if interactive and not force and len(text) > width - 1:
        assert "decisions/game/s" not in terminal and "active transitions/s" not in terminal
        if width == 120:
            assert "active 64/128" in terminal
            assert "12.34 dec/g/s" in terminal and "790 active t/s" in terminal
    else:
        assert text in terminal
    if not interactive:
        assert terminal == log


@pytest.mark.parametrize("games", [128, 64, 32, 16, 4, 1])
def test_collection_interval_constant_independent_control(games):
    now = [0.0]
    progress = CollectionProgress(clock=lambda: now[0])
    progress.observe(transitions=1000, active_games=games, cohort=1, phase="collect")
    assert progress.snapshot()["interval_active_transitions_per_second"] is None
    now[0] = 10
    progress.observe(transitions=1000 + games * 20, active_games=games, cohort=1, phase="collect")
    row = progress.snapshot()
    assert row["interval_active_transitions"] == games * 20
    assert row["interval_active_game_seconds"] == games * 10
    assert row["interval_active_transitions_per_second"] == games * 2
    assert row["interval_decisions_per_game_per_second"] == 2
    assert progress.snapshot() == row


def test_collection_shrinking_workers_phase_and_cohort_boundaries():
    now = [0.0]
    progress = CollectionProgress(clock=lambda: now[0])
    progress.observe(transitions=100, active_games=4, cohort=1, phase="collect")
    now[0] = 2
    progress.observe(transitions=108, active_games=2, cohort=1, phase="collect")
    now[0] = 5
    progress.observe(transitions=114, active_games=1, cohort=1, phase="collect")
    row = progress.snapshot()
    assert row["active_games"] == 1
    assert row["interval_mean_active_games"] == pytest.approx(2.8)
    assert row["interval_decisions_per_game_per_second"] == 1
    assert row["interval_active_transitions_per_second"] == pytest.approx(2.8)
    progress.advance()
    now[0] = 6
    progress.observe(transitions=115, active_games=0, cohort=1, phase="finalize_rewards")
    assert progress.snapshot()["lifetime_active_transitions"] == 15
    assert progress.snapshot()["interval_decisions_per_game_per_second"] is None
    now[0] = 90
    progress.observe(transitions=115, active_games=0, cohort=1, phase="fit")
    now[0] = 100
    progress.observe(transitions=115, active_games=2, cohort=2, phase="collect")
    now[0] = 102
    progress.observe(transitions=119, active_games=2, cohort=2, phase="collect")
    row = progress.snapshot()
    assert row["interval_active_transitions"] == 4
    assert row["interval_decisions_per_game_per_second"] == 1
    assert row["lifetime_active_transitions"] == 19
    assert row["lifetime_collection_seconds"] == 8
    assert row["lifetime_active_game_seconds"] == 19
    assert CollectionProgress.text(row, 2) == (
        "active 2/2 | 1.00 decisions/game/s | 2 active transitions/s"
    )


def test_collection_unmeasured_idle_and_counter_recovery_boundaries():
    now = [10.0]
    progress = CollectionProgress(clock=lambda: now[0])
    progress.observe(transitions=50, active_games=0, cohort=1, phase="collect")
    assert CollectionProgress.text(progress.snapshot(), 4) == (
        "active 0/4 | n/a decisions/game/s | n/a active transitions/s"
    )
    now[0] = 15
    progress.observe(transitions=50, active_games=0, cohort=1, phase="collect")
    assert progress.snapshot()["interval_active_transitions_per_second"] == 0
    assert progress.snapshot()["interval_decisions_per_game_per_second"] is None
    now[0] = 8
    progress.observe(transitions=2, active_games=1, cohort=1, phase="collect")
    assert progress.snapshot()["interval_seconds"] == 0
    now[0] = 9
    progress.observe(transitions=3, active_games=1, cohort=1, phase="collect")
    assert progress.snapshot()["interval_decisions_per_game_per_second"] == 1


@pytest.mark.parametrize(
    "field,value",
    [("transitions", -1), ("active_games", -1), ("transitions", 1.0), ("active_games", True)],
)
def test_collection_rejects_non_host_counters(field, value):
    context = dict(transitions=0, active_games=0, cohort=1, phase="collect")
    context[field] = value
    with pytest.raises(ValueError, match="host integer"):
        CollectionProgress().observe(**context)
