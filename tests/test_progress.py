"""Independent terminal cadence, output modes and failure events."""

import io

import pytest

from pvz_rl.monitoring.progress import Phase, ProgressReporter


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
