"""Refresh persistence uses deterministic fake Google replies, never live credentials."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("entry", ["loader", "status"])
@pytest.mark.parametrize("changed", ["none", "access", "lifetime"])
def test_refresh_replaces_file_only_for_changed_access_facts(tmp_path, monkeypatch, changed, entry):
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
                    "expires_in": 43200,
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
    if changed != "none":
        if changed == "access":
            assert json.loads(token.read_text())["access_token"] == "fake-new"
        else:
            assert json.loads(token.read_text())["refresh_expires_at"] == (now + timedelta(days=1)).isoformat()
        assert after != before
    else:
        assert after == before
