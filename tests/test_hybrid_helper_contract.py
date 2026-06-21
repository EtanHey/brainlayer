from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _wait_for_socket(socket_path: Path, process: subprocess.Popen, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"helper exited early with {process.returncode}: {process.stderr.read()}")
        if socket_path.exists():
            return
        time.sleep(0.05)
    stdout = process.stdout.read() if process.poll() is not None and process.stdout else ""
    stderr = process.stderr.read() if process.poll() is not None and process.stderr else ""
    raise AssertionError(f"helper socket did not appear within {timeout}s; stdout={stdout!r} stderr={stderr!r}")


def test_brainbar_hybrid_helper_ndjson_contract_accepts_documented_brain_search_keys(tmp_path):
    socket_path = Path(f"/tmp/brainbar-contract-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")
    db_path = tmp_path / "fixture.db"
    sitecustomize_dir = tmp_path / "sitecustomize"
    sitecustomize_dir.mkdir()
    (sitecustomize_dir / "sitecustomize.py").write_text(
        """
import brainlayer.mcp._shared as shared
import brainlayer.mcp.search_handler as search_handler
from mcp.types import TextContent


class FakeModel:
    def embed_query(self, query):
        return [0.01] * 1024


class FakeCursor:
    def execute(self, _sql):
        return self

    def fetchone(self):
        return (0,)


class FakeStore:
    _binary_index_available = False

    def _read_cursor(self):
        return FakeCursor()

    def hybrid_search(self, **_kwargs):
        return []

    def get_chunk(self, chunk_id):
        if chunk_id != "contract-fixture-001":
            return None
        return {
            "id": "contract-fixture-001",
            "content": "Contract fixture chunk returned through the helper.",
            "project": "brainlayer",
            "content_type": "assistant_text",
            "source_file": "contract-fixture.jsonl",
            "created_at": "2026-05-29T00:00:00Z",
            "importance": 8,
            "summary": "Contract fixture summary",
            "tags": '["contract"]',
            "status": "active",
            "chunk_origin": "unknown",
        }


shared._get_embedding_model = lambda: FakeModel()
shared._get_search_vector_store = lambda: FakeStore()


async def fake_brain_search(**kwargs):
    return (
        [TextContent(type="text", text="contract helper result")],
        {"query": kwargs["query"], "accepted_keys": sorted(kwargs.keys())},
    )


search_handler._brain_search = fake_brain_search
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["BRAINLAYER_REPO_ROOT"] = str(REPO_ROOT)
    env["BRAINBAR_PYTHON"] = sys.executable
    env["PYTHONPATH"] = os.pathsep.join([str(sitecustomize_dir), str(SRC_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "brainlayer.brainbar_hybrid_helper",
            "--socket-path",
            str(socket_path),
            "--db-path",
            str(db_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_socket(socket_path, process)
        request = {
            "method": "brain_search",
            "arguments": {
                "query": "contract-fixture-001",
                "project": "brainlayer",
                "source": "all",
                "tag": "contract",
                "importance_min": 7,
                "agent_id": "contract-test-agent",
                "num_results": 3,
                "max_results": 9,
                "detail": "compact",
                "_profile_query_id": "contract-profile-id",
            },
        }

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(str(socket_path))
            client.sendall(json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n")
            payload = b""
            while not payload.endswith(b"\n"):
                chunk = client.recv(65536)
                if not chunk:
                    raise AssertionError("helper closed socket before sending a full NDJSON line")
                payload += chunk

        response = json.loads(payload.decode("utf-8"))
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        socket_path.unlink(missing_ok=True)

    assert response["ok"] is True
    assert response["text"] == "contract helper result"
    assert set(response) == {"ok", "text", "metadata"}
    structured = response["metadata"]["structuredContent"]
    assert structured["query"] == "contract-fixture-001"
    assert structured["accepted_keys"] == [
        "agent_id",
        "allow_helper_route",
        "brainbar_helper_fast_profile",
        "detail",
        "importance_min",
        "max_results",
        "num_results",
        "profile_query_id",
        "profile_scope",
        "project",
        "query",
        "source",
        "tag",
    ]
