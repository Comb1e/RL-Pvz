"""CPU browser viewer protocol controls."""

import socket
from multiprocessing import Queue, Value
from threading import Thread

import pytest

from pvz_rl.presentation.web_viewer import _Server


def server():
    return _Server(("127.0.0.1", 0), Queue(8), Queue(4), Value("i", 1), (("Learning", ("lr=1",)),))


def test_history_query_requires_all_nonnegative_fields():
    with pytest.raises(ValueError):
        _Server.parse_history({"panel": ["0"]})
    with pytest.raises(ValueError):
        _Server.parse_history({name: ["0"] for name in ("panel", "generation", "env", "episode", "start", "request")} | {"start": ["-1"]})


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
