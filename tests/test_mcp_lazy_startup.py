"""Lazy-connect / non-blocking MCP startup contract (fix/mcp-lazy-connect).

Root cause of the Codex boot-hang: brainlayer-mcp's initialize+tools/list
handshake was delayed 4-9s because serve() -> validate_config() eagerly imported
`torch` (via embeddings.py module-load), routinely exceeding Codex's
startup_timeout_sec=5. The DB itself was never the blocker (the store is already
lazy; the handshake never touches it).

These tests pin the contract: importing the MCP/embeddings modules and running
startup validation must NOT pull in torch, and the live handshake must return
quickly even when the DB is held under an exclusive write lock.

Each "no torch" check runs in a fresh interpreter subprocess so a torch import
elsewhere in the test session can't mask a regression.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import apsw
import pytest

from brainlayer.paths import get_db_path

# Run subprocesses against THIS worktree's src (not whatever editable install is
# active), so the test exercises the code under review pre-merge.
_REPO_SRC = str(Path(__file__).resolve().parent.parent / "src")


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_snippet(snippet: str) -> str:
    """Run a Python snippet in a fresh interpreter; return its stdout (stripped)."""
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=60,
        env=_child_env(),
    )
    assert proc.returncode == 0, f"snippet failed:\n{proc.stderr}"
    return proc.stdout.strip()


def test_importing_embeddings_does_not_load_torch():
    """`import brainlayer.embeddings` must not eagerly import torch.

    torch import is ~3.7s cold; keeping it lazy is the core of the fix.
    """
    out = _run_snippet(
        "import sys; import brainlayer.embeddings; "
        "print('torch' in sys.modules or 'sentence_transformers' in sys.modules)"
    )
    assert out == "False", "importing brainlayer.embeddings eagerly loaded torch/sentence_transformers"


def test_validate_config_does_not_load_torch():
    """serve()'s startup validation must not pull torch onto the critical path."""
    out = _run_snippet(
        "import sys, brainlayer.mcp as m; m.validate_config(); "
        "print('torch' in sys.modules)"
    )
    assert out == "False", "validate_config() eagerly imported torch (blocks the MCP handshake)"


def test_get_embedding_model_is_cheap_wrapper():
    """get_embedding_model() returns a wrapper without loading torch or the model."""
    out = _run_snippet(
        "import sys; from brainlayer.embeddings import get_embedding_model; "
        "m = get_embedding_model(); "
        "print('torch' in sys.modules, m.__class__.__name__)"
    )
    loaded, cls = out.split()
    assert loaded == "False", "get_embedding_model() eagerly imported torch"
    assert cls == "EmbeddingModel"


# --- Live handshake under DB write-lock (the brief's requested integration test) ---

HANDSHAKE_BUDGET_SEC = 3.0  # must beat Codex's startup_timeout_sec=5 with margin


def _hold_exclusive_lock(db_path, stop_event, ready_event):
    conn = apsw.Connection(str(db_path))
    conn.cursor().execute("PRAGMA busy_timeout=200")  # fail fast, don't block on a real writer
    try:
        conn.cursor().execute("BEGIN IMMEDIATE")  # reserve the write lock (WAL-friendly)
        conn.cursor().execute("CREATE TABLE IF NOT EXISTS _repro_lock(x)")
    except apsw.Error:
        # A real writer (the enrichment pipeline) already holds the lock — that is
        # itself the contended condition we want to test against, so proceed.
        pass
    ready_event.set()
    while not stop_event.is_set():
        time.sleep(0.05)
    try:
        conn.cursor().execute("ROLLBACK")
    except apsw.Error:
        pass
    conn.close()


def _read_json_line(proc, timeout):
    box = {}

    def _r():
        box["line"] = proc.stdout.readline()

    t = threading.Thread(target=_r, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or not box.get("line"):
        return None
    return json.loads(box["line"])


@pytest.mark.live
def test_handshake_fast_under_db_write_lock():
    """brainlayer-mcp must answer initialize+tools/list within budget even when the
    10GB DB is held under an exclusive write lock."""
    db_path = get_db_path()
    if not db_path.exists():
        pytest.skip("real DB not present")

    stop_event = threading.Event()
    ready_event = threading.Event()
    lock_thread = threading.Thread(
        target=_hold_exclusive_lock, args=(db_path, stop_event, ready_event), daemon=True
    )
    lock_thread.start()
    assert ready_event.wait(timeout=10), "could not acquire exclusive DB lock"

    proc = subprocess.Popen(
        [sys.executable, "-c", "from brainlayer.mcp import serve; serve()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_child_env(),
    )
    try:
        proc.stdin.write(
            (json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "test", "version": "0"}},
            }) + "\n").encode()
        )
        proc.stdin.flush()
        t0 = time.time()
        init_resp = _read_json_line(proc, timeout=HANDSHAKE_BUDGET_SEC + 5)
        elapsed = time.time() - t0
        assert init_resp is not None, "initialize never responded"
        assert elapsed < HANDSHAKE_BUDGET_SEC, (
            f"initialize took {elapsed:.2f}s (budget {HANDSHAKE_BUDGET_SEC}s) — would trip "
            f"Codex startup_timeout_sec=5"
        )

        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n").encode())
        proc.stdin.flush()
        tools_resp = _read_json_line(proc, timeout=HANDSHAKE_BUDGET_SEC)
        assert tools_resp is not None, "tools/list never responded"
        assert len(tools_resp["result"]["tools"]) > 0
    finally:
        proc.kill()
        stop_event.set()
        lock_thread.join(timeout=5)
