"""CPU browser viewer protocol controls."""

import socket
from multiprocessing import Queue, Value
from threading import Thread

import pytest

from pvz_rl.presentation.web_viewer import HTML, _Server


def server():
    return _Server(("127.0.0.1", 0), Queue(8), Queue(4), Value("i", 1), (("Learning", ("lr=1",)),))


def test_history_query_requires_all_nonnegative_fields():
    with pytest.raises(ValueError):
        _Server.parse_history({"panel": ["0"]})
    with pytest.raises(ValueError):
        _Server.parse_history(
            {name: ["0"] for name in ("panel", "generation", "env", "episode", "start", "request")}
            | {"start": ["-1"]}
        )


def test_history_forwarding_keeps_request_identity():
    s = server()
    try:
        request = dict(panel=0, generation=0, env=1, episode=2, start=0, request=3)

        def respond():
            queued = s.commands.get()
            s.history_responses.put({**queued, "page": {"start": 0, "total": 1, "rows": []}})

        Thread(target=respond, daemon=True).start()
        result = s.history(request)
        assert result["request"] == 3 and result["page"]["total"] == 1
        assert s.state()["learning_settings"] == [["Learning", ["lr=1"]]]
    finally:
        s.server_close()


def test_web_viewer_binds_requested_loopback_port():
    from pvz_rl.config import load_config, output_settings
    from pvz_rl.presentation.live_view import LiveSession

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = output_settings(load_config())["visualization"]
    settings.update(viewer="web", viewer_port=port)
    session = LiveSession(1, settings)
    try:
        assert session.ready.wait(12)
        assert session.web_port.value == port
        assert session.proxy_url == f"./proxy/{port}/"
    finally:
        session.close()


def test_browser_renders_public_snapshots_with_canvas_and_diagnostics():
    for token in (
        '<canvas class="board"',
        "normalizeObservation",
        "o.plants",
        "o.zombies",
        "o.projectiles",
        "o.mowers",
        "devicePixelRatio",
        "historyTable",
        "Raw frame JSON",
        "Learning settings",
    ):
        assert token in HTML
    assert "fetch('./api/state'" in HTML
    assert "fetch('./api/history?" in HTML
    assert "fetch('./api/command'" in HTML
    assert "fetch('/api/" not in HTML


def test_browser_snapshot_normalization_covers_empty_and_terminal_shapes():
    # These names are the public Observation fields serialized by _jsonable.
    # The browser normalizer must supply empty arrays/default counters when a
    # switching or terminal frame has no complete observation yet.
    for field in ("cards", "plants", "zombies", "projectiles", "mowers", "counts"):
        assert field in HTML
    assert "Math.max(1,n(o.rows,5))" in HTML
    assert "frame.outcome" in HTML
    assert "accepted===false" in HTML
