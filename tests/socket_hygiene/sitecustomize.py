"""Refuse production BrainBar connections in pytest and inherited Python subprocesses."""

import os
import socket

FORBID_BRAINBAR_SOCKET_ENV = "BRAINLAYER_FORBID_BRAINBAR_SOCKET"
_PRODUCTION_SOCKET = os.path.realpath("/tmp/brainbar.sock")


def _refuse_production_brainbar(sock, address):
    if (
        os.environ.get(FORBID_BRAINBAR_SOCKET_ENV) == "1"
        and sock.family == socket.AF_UNIX
        and isinstance(address, (str, bytes, os.PathLike))
        and os.path.realpath(os.fsdecode(address)) == _PRODUCTION_SOCKET
    ):
        raise RuntimeError(
            "suite hygiene: this test connected to the production BrainBar socket; "
            "mark it `integration` or `live` for a deliberate live check"
        )


def install_brainbar_socket_guard():
    if getattr(socket.socket.connect, "_brainlayer_hygiene_guard", False):
        return
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address):
        _refuse_production_brainbar(self, address)
        return original_connect(self, address)

    def guarded_connect_ex(self, address):
        _refuse_production_brainbar(self, address)
        return original_connect_ex(self, address)

    guarded_connect._brainlayer_hygiene_guard = True
    guarded_connect_ex._brainlayer_hygiene_guard = True
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


install_brainbar_socket_guard()
