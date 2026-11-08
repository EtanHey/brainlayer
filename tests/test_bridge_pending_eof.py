import json
import os

import pytest

from brainlayer import mcp_stdio_bridge as bridge


@pytest.mark.parametrize("backend", ["absent", "blocked"])
def test_eof_deadline_includes_unsent_requests(monkeypatch, backend):
    clock = [0.0]
    real_select = bridge.select.select

    def monotonic():
        clock[0] += 0.05
        assert clock[0] < 5, "bridge exceeded overall EOF deadline"
        return clock[0]

    def select(readers, writers, errors, timeout):
        return real_select(readers, writers, errors, 0)

    class Unavailable:
        def setblocking(self, _flag):
            pass

        def connect(self, _path):
            raise FileNotFoundError("backend never returns")

        def close(self):
            pass

    monkeypatch.setattr(bridge.time, "monotonic", monotonic)
    monkeypatch.setattr(bridge.select, "select", select)
    if backend == "absent":
        monkeypatch.setattr(bridge.socket, "socket", lambda *_a, **_kw: Unavailable())
    else:
        real_pair = bridge.socket.socketpair()

        class Blocked(Unavailable):
            def connect(self, _path):
                pass

            def fileno(self):
                return real_pair[0].fileno()

            def send(self, _data):
                raise BlockingIOError()

        monkeypatch.setattr(bridge.socket, "socket", lambda *_a, **_kw: Blocked())
    input_r, input_w = os.pipe()
    output_r, output_w = os.pipe()
    os.write(input_w, b'{"jsonrpc":"2.0","id":7,"method":"ping"}\n')
    os.close(input_w)
    try:
        with os.fdopen(input_r, "rb") as stdin, os.fdopen(output_w, "wb") as stdout:
            assert (
                bridge.run_bridge(
                    bridge.BridgeConfig(
                        socket_path="unused-test-socket", connect_timeout_ms=100, stdin_eof_drain_ms=20
                    ),
                    stdin=stdin,
                    stdout=stdout,
                )
                == 0
            )
        assert clock[0] < 2
        reply = json.loads(os.read(output_r, 4096))
        assert reply["id"] == 7
        assert "outcome is unknown" in reply["error"]["message"]
    finally:
        os.close(output_r)
        if backend == "blocked":
            for sock in real_pair:
                sock.close()
