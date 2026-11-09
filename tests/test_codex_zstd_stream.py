"""Real zstd fixtures; these never open the production database or socket."""

import json
import os
import subprocess

import pytest

from brainlayer.ingest.codex import parse_codex_session
from brainlayer.zstd_stream import resolve_zstd


@pytest.fixture
def compress(tmp_path):
    binary = resolve_zstd()
    if binary is None:
        if os.environ.get("CI"):
            pytest.fail("CI must install zstd for compressed Codex fixtures")
        pytest.skip("zstd CLI is not installed")

    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            subprocess.run([binary, "-q", "-c", "--check"], input=data, stdout=output, check=True)
        return path

    return write


def rollout(text="Please remember this multiline\nUTF-8 message: שלום 🌱"):
    return b"".join(
        (json.dumps(item, ensure_ascii=False) + "\n").encode()
        for item in [
            {"type": "session_meta", "payload": {"id": "native-id", "cwd": "/fixture/repo"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            },
        ]
    )


def test_parser_keeps_plain_identity_and_native_id(compress):
    path = compress("rollout.jsonl.zst", rollout())
    entries = list(parse_codex_session(path))
    assert len(entries) == 1
    assert entries[0]["session_id"] == "native-id"
    assert entries[0]["metadata"]["source_file"] == str(path.with_suffix(""))
    assert "\nUTF-8 message: שלום 🌱" in entries[0]["content"]
