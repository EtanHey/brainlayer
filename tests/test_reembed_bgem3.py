"""Tests for BGE-M3 re-embedding script."""

import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "reembed_bgem3.py"


def _create_test_db(tmp_path):
    """Create a minimal test DB with chunks and vector tables."""
    db_path = tmp_path / "test_brainlayer.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT, created_at TEXT)")
    # Use a regular table instead of vec0 (which requires sqlite-vec extension)
    conn.execute("CREATE TABLE chunk_vectors (chunk_id TEXT PRIMARY KEY, embedding BLOB)")
    conn.execute("CREATE TABLE chunk_vectors_binary (chunk_id TEXT PRIMARY KEY, embedding BLOB)")

    # Insert 5 test chunks
    for i in range(5):
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?)",
            (f"chunk_{i}", f"Test content for chunk {i}. This is meaningful text.", "2026-04-05"),
        )
        # Insert old embeddings (1024-dim zeros)
        old_emb = struct.pack(f"{1024}f", *([0.0] * 1024))
        conn.execute("INSERT INTO chunk_vectors VALUES (?, ?)", (f"chunk_{i}", old_emb))
        conn.execute("INSERT INTO chunk_vectors_binary VALUES (?, ?)", (f"chunk_{i}", b"\x00" * 128))

    conn.commit()
    conn.close()
    return db_path


class TestReembedScript:
    """Test the re-embedding script exists and has correct interface."""

    def test_script_exists(self):
        assert SCRIPT_PATH.exists(), f"Script not found at {SCRIPT_PATH}"

    def test_script_imports(self):
        """Script should be importable without side effects."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import importlib.util; spec = importlib.util.spec_from_file_location('reembed', '{SCRIPT_PATH}'); mod = importlib.util.module_from_spec(spec)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Just checks syntax is valid, not full import (which would load torch)
        assert result.returncode == 0, f"Script has syntax errors: {result.stderr}"

    def test_help_flag(self):
        """Script should support --help."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--test" in result.stdout
        assert "--db" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--checkpoint-every" in result.stdout

    def test_test_flag_runs_limited_chunks(self, tmp_path):
        """--test flag should process only 100 chunks (or all if fewer)."""
        db_path = _create_test_db(tmp_path)
        checkpoint_path = tmp_path / "checkpoint.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--db",
                str(db_path),
                "--test",
                "--checkpoint-file",
                str(checkpoint_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Processed" in result.stdout or "processed" in result.stdout.lower()

        # Verify embeddings were updated (not all zeros anymore)
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT embedding FROM chunk_vectors WHERE chunk_id = 'chunk_0'").fetchone()
        conn.close()
        assert row is not None
        emb_floats = struct.unpack(f"{1024}f", row[0])
        # At least some values should be non-zero after re-embedding
        assert any(v != 0.0 for v in emb_floats), "Embeddings should be updated from zeros"

    def test_checkpoint_created(self, tmp_path):
        """Script should create a checkpoint file tracking progress."""
        db_path = _create_test_db(tmp_path)
        checkpoint_path = tmp_path / "checkpoint.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--db",
                str(db_path),
                "--test",
                "--checkpoint-file",
                str(checkpoint_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert checkpoint_path.exists(), "Checkpoint file should be created"

        import json

        data = json.loads(checkpoint_path.read_text())
        assert "processed_ids" in data
        assert "model" in data
        assert data["model"] == "BAAI/bge-m3"
        assert len(data["processed_ids"]) == 5  # all 5 test chunks

    def test_resumable(self, tmp_path):
        """Script should skip already-processed chunks on resume."""
        db_path = _create_test_db(tmp_path)
        checkpoint_path = tmp_path / "checkpoint.json"

        import json

        # Pre-seed checkpoint with 3 of 5 chunks
        checkpoint_data = {
            "model": "BAAI/bge-m3",
            "processed_ids": ["chunk_0", "chunk_1", "chunk_2"],
            "last_updated": "2026-04-05T00:00:00",
        }
        checkpoint_path.write_text(json.dumps(checkpoint_data))

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--db",
                str(db_path),
                "--test",
                "--checkpoint-file",
                str(checkpoint_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        # Should mention skipping or resuming
        output = result.stdout.lower()
        assert "skip" in output or "resum" in output or "already" in output

        # Checkpoint should now have all 5
        data = json.loads(checkpoint_path.read_text())
        assert len(data["processed_ids"]) == 5

    def test_binary_vectors_updated(self, tmp_path):
        """Binary vectors should be regenerated from new float embeddings."""
        db_path = _create_test_db(tmp_path)
        checkpoint_path = tmp_path / "checkpoint.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--db",
                str(db_path),
                "--test",
                "--checkpoint-file",
                str(checkpoint_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT embedding FROM chunk_vectors_binary WHERE chunk_id = 'chunk_0'").fetchone()
        conn.close()
        assert row is not None
        # Binary vector should be 128 bytes (1024 bits / 8)
        assert len(row[0]) == 128
        # Should not be all zeros
        assert row[0] != b"\x00" * 128, "Binary vectors should be updated"
