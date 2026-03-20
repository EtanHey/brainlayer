#!/usr/bin/env python3

import argparse
import os
import selectors
import signal
import socket
import sys


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        view = view[written:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge stdio to BrainBar's Unix socket.")
    parser.add_argument(
        "--socket",
        default=os.environ.get("BRAINBAR_SOCKET_PATH", "/tmp/brainbar.sock"),
        help="Unix socket path for the running BrainBar daemon",
    )
    args = parser.parse_args()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(args.socket)
    except OSError as exc:
        sock.close()
        sys.stderr.write(f"brainbar-stdio-adapter: failed to connect to {args.socket}: {exc}\n")
        return 1

    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()
    sock_fd = sock.fileno()

    os.set_blocking(stdin_fd, False)

    selector = selectors.DefaultSelector()
    selector.register(stdin_fd, selectors.EVENT_READ, "stdin")
    selector.register(sock_fd, selectors.EVENT_READ, "socket")
    stdin_open = True

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    try:
        while True:
            events = selector.select()
            for key, _ in events:
                if key.data == "stdin":
                    try:
                        data = os.read(stdin_fd, 65536)
                    except BlockingIOError:
                        continue
                    except InterruptedError:
                        continue
                    if not data:
                        if stdin_open:
                            stdin_open = False
                            selector.unregister(stdin_fd)
                            try:
                                sock.shutdown(socket.SHUT_WR)
                            except OSError:
                                return 0
                        continue
                    sock.sendall(data)
                else:
                    try:
                        data = sock.recv(65536)
                    except InterruptedError:
                        continue
                    if not data:
                        return 0
                    write_all(stdout_fd, data)
    finally:
        selector.close()
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
