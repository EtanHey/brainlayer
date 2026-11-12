import hashlib
import json
import os
import random

import pytest

from brainlayer import jsonl_backup as backup


def test_default_local_retention_enabled(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", raising=False)
    assert backup._local_retention_enabled()


def test_random_retention_and_crashes(tmp_path, monkeypatch):
    rng = random.Random(1597)
    now = 2000000000
    monkeypatch.delenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", raising=False)
    for case in range(120):
        root = tmp_path / str(case)
        root.mkdir()
        receipts, originals = {}, {}
        for i in range(rng.randint(2, 8)):
            path = root / f"claude-jsonl-2026-01-{i + 1:02}.tar.gz"
            path.write_bytes(bytes([i]) * 5)
            age = rng.choice([0, 13.999, 14, 15, 40]) * 86400
            os.utime(path, (now - age, now - age))
            verified = rng.choice([True, False])
            size = rng.choice([5, 6])
            receipts[path.name] = dict(
                name=path.name,
                size=size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                verified=verified,
                archive_id=f"id-{i}",
                archive_md5=hashlib.md5(path.read_bytes()).hexdigest(),
                verified_at=now - age,
            )
            hash_matches = rng.choice([True, False])
            if not hash_matches:
                receipts[path.name]["sha256"] = "wrong"
            md5_matches = rng.choice([True, False])
            if not md5_matches:
                receipts[path.name]["archive_md5"] = "wrong"
            originals[path] = (age, verified and size == 5 and hash_matches and md5_matches)
        state = root / "state.json"
        state.write_text(json.dumps({"drive_archives": receipts}))
        live = {r["archive_id"]: r["archive_md5"] for r in receipts.values()}
        for path in originals:
            if rng.choice([True, False]):
                live.pop(receipts[path.name]["archive_id"])
                originals[path] = (originals[path][0], False)
        newest = max(originals, key=lambda p: (p.stat().st_mtime, p.name))
        original_unlink = backup.Path.unlink
        cut = rng.randrange(4)
        calls = 0

        def crash_unlink(path, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == cut:
                raise OSError("simulated crash before unlink")
            original_unlink(path, *args, **kwargs)
            if calls == cut + 1:
                raise OSError("simulated crash after unlink")

        with monkeypatch.context() as patch:
            patch.setattr(backup.Path, "unlink", crash_unlink)
            try:
                backup.prune_local_archives(
                    root, state_path=state, log_path=root / "log", now=now, surviving_archives=live
                )
            except OSError:
                pass
        backup.prune_local_archives(root, state_path=state, log_path=root / "log", now=now, surviving_archives=live)
        assert newest.exists(), "deleted newest"
        for path, (age, verified) in originals.items():
            if not path.exists():
                assert verified, "deleted without verified exact copy"
                assert age >= 14 * 86400, "deleted younger than 14 days"
            elif path != newest and age >= 14 * 86400 and verified:
                pytest.fail("eligible archive was not swept")
        assert json.loads(state.read_text())["drive_archives"] == receipts


def test_disable_and_invalid_days_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", "0")
    assert not backup._local_retention_enabled()
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", "1")
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION_DAYS", "-1")
    with pytest.raises(ValueError):
        backup.prune_local_archives(
            tmp_path, state_path=tmp_path / "state", log_path=tmp_path / "log", now=0, surviving_archives={}
        )


def test_disabled_sweep_preserves_eligible_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", "0")
    old = tmp_path / "claude-jsonl-2026-01-01.tar.gz"
    old.write_bytes(b"old")
    os.utime(old, (0, 0))
    (tmp_path / "claude-jsonl-2026-02-01.tar.gz").write_bytes(b"new")
    state = tmp_path / "state"
    state.write_text(
        json.dumps(
            {
                "drive_archives": {
                    old.name: {
                        "name": old.name,
                        "size": 3,
                        "sha256": hashlib.sha256(b"old").hexdigest(),
                        "verified": True,
                        "archive_id": "verified-old",
                    }
                }
            }
        )
    )
    assert (
        backup.prune_local_archives(
            tmp_path, state_path=state, log_path=tmp_path / "log", now=2000000000, surviving_archives={}
        )
        == []
    )
    assert old.exists()


def test_real_rule_mutations_fail_randomized_oracle():
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [sys.executable, "formal/jsonl-retention/check_mutations.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verified mutation rejected" in result.stdout
    assert "age mutation rejected" in result.stdout
