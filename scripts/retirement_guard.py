"""Private sitecustomize fixture; refuses attempts before candidate imports."""

import builtins
import importlib
import importlib.util
import json
import os
import socket
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ARMED = False
FORBIDDEN = (
    "google.genai",
    "google.generativeai",
    "google.cloud.aiplatform",
    "vertexai",
    "groq",
    "openai",
    "anthropic",
    "cohere",
    "mistralai",
    "litellm",
    "xai_sdk",
    "together",
    "brainlayer.enrichment_controller",
    "brainlayer.pipeline.enrichment",
    "brainlayer.pipeline.groq",
)


class Attempt(RuntimeError):
    pass


def refuse(kind):
    fd = os.open(os.environ["RETIREMENT_EVENTS"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(
            fd, (json.dumps({"kind": kind, "phase": os.environ.get("RETIREMENT_PHASE", "candidate")}) + "\n").encode()
        )
    finally:
        os.close(fd)
    raise Attempt("retirement fixture refused " + kind)


def check_import(name):
    if any(name == item or name.startswith(item + ".") for item in FORBIDDEN):
        refuse("model_or_retired_import")


def check_db(database):
    name = os.fspath(database)
    if name == ":memory:" or name.startswith("file::memory:"):
        return
    if name.startswith("file:"):
        name = unquote(urlsplit(name).path)
    if not Path(name).expanduser().resolve().is_relative_to(Path(os.environ["HOME"]).resolve()):
        refuse("outside_fixture_db")


def arm():
    global ARMED
    original_import, original_dynamic = builtins.__import__, importlib.import_module

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        resolved = importlib.util.resolve_name("." * level + name, globals["__package__"]) if level else name
        check_import(resolved)
        for item in fromlist or ():
            check_import(resolved + "." + item)
        return original_import(name, globals, locals, fromlist, level)

    def guarded_dynamic(name, package=None):
        check_import(importlib.util.resolve_name(name, package))
        return original_dynamic(name, package)

    builtins.__import__, importlib.import_module = guarded_import, guarded_dynamic
    for method in ("connect", "connect_ex", "send", "sendall", "sendto"):
        original = getattr(socket.socket, method)

        def guarded(self, *args, _original=original, _method=method, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
                refuse("socket." + _method)
            return _original(self, *args, **kwargs)

        setattr(socket.socket, method, guarded)
    socket.getaddrinfo = lambda *a, **k: refuse("dns")
    original_process = subprocess.Popen

    class FixtureProcess(original_process):
        def __init__(self, *args, **kwargs):
            command = args[0] if args else kwargs.get("args")
            # Actual writer PID provenance needs this local read. No shell,
            # alternate executable, env replacement or arbitrary child is admitted.
            if (
                command == ["ps", "-o", "lstart=", "-p", str(os.getpid())]
                and not kwargs.get("shell")
                and kwargs.get("executable") is None
                and kwargs.get("env") is None
                and os.environ.get("RETIREMENT_OS_BOUNDARY") == "1"
            ):
                return super().__init__(["/bin/ps", *command[1:]], *args[1:], **kwargs)
            refuse("subprocess")

    subprocess.Popen = FixtureProcess
    original_sqlite = sqlite3.connect

    def sqlite_connect(database, *args, **kwargs):
        check_db(database)
        return original_sqlite(database, *args, **kwargs)

    sqlite3.connect = sqlite_connect
    apsw = original_dynamic("apsw")
    connection = apsw.Connection

    class FixtureConnection(connection):
        def __init__(self, database, *args, **kwargs):
            check_db(database)
            super().__init__(database, *args, **kwargs)

    apsw.Connection = FixtureConnection
    # Patches at the send layer also catch pre-bound transports before DNS/connect.
    for module_name, classes in (
        ("requests", (("Session", "send"),)),
        ("httpx", (("Client", "send"), ("HTTPTransport", "handle_request"))),
    ):
        module = original_dynamic(module_name)
        for class_name, method in classes:
            setattr(getattr(module, class_name), method, lambda *a, **k: refuse("http_send"))
    httpx = original_dynamic("httpx")

    async def async_send(*args, **kwargs):
        refuse("async_http_send")

    httpx.AsyncClient.send = async_send
    httpx.AsyncHTTPTransport.handle_async_request = async_send
    ARMED = True


if os.environ.get("BRAINLAYER_RETIREMENT_GUARD") == "1":
    arm()
