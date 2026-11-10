"""Status probes use fake Google endpoints and owned scratch credentials only."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "failure, expected", [("invalid_grant", "invalid"), ("offline", "unknown"), ("retryable", "unknown")]
)
def test_status_probes_fresh_access_token(tmp_path, failure, expected, monkeypatch):
    from google.auth.exceptions import TransportError
    from brainlayer.drive_credentials import credential_status
    import google.auth._exponential_backoff as backoff

    monkeypatch.setattr(backoff, "time", SimpleNamespace(sleep=lambda _: None))
    token, client = tmp_path / "token.json", tmp_path / "client.json"
    token.write_text(
        json.dumps(
            {
                "access_token": "fake-access",
                "refresh_token": "fake-refresh",
                "expiry": (datetime.now(UTC) + timedelta(hours=12)).replace(tzinfo=None).isoformat(),
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
    before = token.read_bytes()
    calls, alerts = [], []

    def endpoint(**kwargs):
        calls.append(True)
        if failure == "offline":
            raise TransportError("private payload")
        return SimpleNamespace(
            status=503 if failure == "retryable" else 400,
            data=json.dumps({"error": "backend_error" if failure == "retryable" else failure}).encode(),
        )

    result = credential_status(
        token_path=token,
        client_path=client,
        request_factory=lambda: endpoint,
        alert_reporter=lambda *args: alerts.append(args),
    )
    assert result["state"] == expected
    assert calls
    assert token.read_bytes() == before
    assert "private payload" not in json.dumps(result)
    if expected == "invalid":
        assert alerts and alerts[-1][0] == "drive-consent" and alerts[-1][1]
    else:
        assert alerts == []
