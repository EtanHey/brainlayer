"""Refresh persistence uses fake Google replies, never live credentials."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("entry", ["loader", "status"])
@pytest.mark.parametrize(
    "changed, expiry_shift, should_write",
    [
        ("none", 0, False),
        ("access", 0, True),
        ("lifetime", 0, True),
        ("none", -61, True),
        ("none", -60, False),
        ("none", 60, False),
        ("none", 61, True),
    ],
)
def test_refresh_replaces_file_only_for_changed_access_facts(
    tmp_path, monkeypatch, changed, expiry_shift, should_write, entry
):
    import google.auth._helpers as helpers

    from brainlayer.drive_credentials import credential_status, load_credentials

    now = datetime(2026, 11, 10, tzinfo=UTC)
    monkeypatch.setattr(helpers, "utcnow", lambda: now.replace(tzinfo=None))
    token, client = tmp_path / "token.json", tmp_path / "client.json"
    token.write_text(
        json.dumps(
            {
                "access_token": "fake-access",
                "refresh_token": "fake-refresh",
                "expiry": (now + timedelta(hours=12)).replace(tzinfo=None).isoformat(),
            }
        )
    )
    token.chmod(0o600)
    client.write_text(
        json.dumps(
            {
                "installed": {
                    "token_uri": "https://example.invalid/token",
                    "client_id": "fake-id",
                    "client_secret": "fake-secret",
                }
            }
        )
    )
    before = token.read_bytes(), token.stat().st_ino, token.stat().st_mtime_ns
    calls = []

    def endpoint(**kwargs):
        assert kwargs["url"] == "https://example.invalid/token"
        calls.append(True)
        return SimpleNamespace(
            status=200,
            data=json.dumps(
                {
                    "access_token": "fake-new" if changed == "access" else "fake-access",
                    "expires_in": 43200 + expiry_shift,
                    **({"refresh_token_expires_in": 86400} if changed == "lifetime" else {}),
                }
            ).encode(),
        )

    first = None
    for _ in range(2):
        kwargs = dict(
            token_path=token,
            client_path=client,
            request_factory=lambda: endpoint,
            alert_reporter=lambda *_: None,
            now=now,
        )
        if entry == "loader":
            load_credentials(**kwargs, force_refresh=True, persist_refresh=True)
        else:
            assert credential_status(**kwargs)["state"] in {"valid", "expiring"}
        stamp = token.read_bytes(), token.stat().st_ino, token.stat().st_mtime_ns
        if first is None:
            first = stamp
        else:
            assert stamp == first
    assert len(calls) == 2
    after = token.read_bytes(), token.stat().st_ino, token.stat().st_mtime_ns
    if should_write:
        if changed == "access":
            assert json.loads(token.read_text())["access_token"] == "fake-new"
        elif changed == "lifetime":
            assert json.loads(token.read_text())["refresh_expires_at"] == (now + timedelta(days=1)).isoformat()
        else:
            assert (
                json.loads(token.read_text())["expiry"]
                == (now.replace(tzinfo=None) + timedelta(seconds=43200 + expiry_shift)).isoformat()
            )
        assert after != before
    else:
        assert after == before


def test_live_clock_same_token_shrinking_lifetime_never_rewrites(tmp_path, monkeypatch):
    import time

    from brainlayer import drive_credentials

    deadline = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=12)
    token, client = tmp_path / "token.json", tmp_path / "client.json"
    token.write_text(
        json.dumps({"access_token": "fake-access", "refresh_token": "fake-refresh", "expiry": deadline.isoformat()})
    )
    token.chmod(0o600)
    client.write_text(
        json.dumps(
            {
                "installed": {
                    "token_uri": "https://example.invalid/token",
                    "client_id": "fake-id",
                    "client_secret": "fake-secret",
                }
            }
        )
    )
    before = token.read_bytes(), token.stat().st_ino, token.stat().st_mtime_ns
    replies, writes = [], []
    atomic_write = drive_credentials._atomic_write

    def record_write(*args):
        writes.append(True)
        return atomic_write(*args)

    monkeypatch.setattr(drive_credentials, "_atomic_write", record_write)

    def endpoint(**kwargs):
        assert kwargs["url"] == "https://example.invalid/token"
        remaining = int((deadline - datetime.now(UTC).replace(tzinfo=None)).total_seconds())
        replies.append(remaining)
        return SimpleNamespace(
            status=200, data=json.dumps({"access_token": "fake-access", "expires_in": remaining}).encode()
        )

    for index in range(3):
        if index:
            time.sleep(1.1)
        status = drive_credentials.credential_status(
            token_path=token,
            client_path=client,
            request_factory=lambda: endpoint,
            alert_reporter=lambda *_: None,
        )
        assert status["state"] == "valid"
    assert len(replies) == 3 and replies[0] > replies[1] > replies[2]
    assert writes == []
    assert (token.read_bytes(), token.stat().st_ino, token.stat().st_mtime_ns) == before
