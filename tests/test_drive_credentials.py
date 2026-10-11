"""Isolated Drive credentials tests: never use the real config or network."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


def _client(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"installed": {"token_uri": "https://example.invalid/token", "client_id": "id", "client_secret": "secret"}}
        )
    )


def _token(path: Path, *, value: str = "old") -> None:
    path.write_text(
        json.dumps(
            {"access_token": value, "refresh_token": "refresh", "scope": "https://www.googleapis.com/auth/drive"}
        )
    )
    path.chmod(0o600)


def test_paths_prefer_explicit_override_and_do_not_use_mcp_token(tmp_path):
    from brainlayer import drive_credentials

    env = {
        "BRAINLAYER_DRIVE_TOKEN_PATH": str(tmp_path / "override.json"),
        "BRAINLAYER_DRIVE_CLIENT_PATH": str(tmp_path / "client.json"),
    }
    token, client, legacy = drive_credentials.resolve_paths(environ=env, home=tmp_path)
    assert token == tmp_path / "override.json"
    assert client == tmp_path / "client.json"
    assert legacy == tmp_path / ".config/google-drive-mcp/tokens.json"
    token, _, _ = drive_credentials.resolve_paths(environ={}, home=tmp_path)
    assert token == tmp_path / ".config/brainlayer/drive-tokens.json"


def test_missing_owned_token_copies_legacy_once_without_removing_it(tmp_path):
    from brainlayer import drive_credentials

    legacy = tmp_path / "legacy.json"
    owned = tmp_path / "owned/drive-tokens.json"
    client = tmp_path / "client.json"
    _token(legacy)
    _client(client)
    drive_credentials.load_credentials(
        token_path=owned,
        client_path=client,
        legacy_token_path=legacy,
        credentials_class=FakeCredentials,
        request_factory=object,
    )
    assert json.loads(owned.read_text())["access_token"] == "old"
    assert owned.stat().st_mode & 0o777 == 0o600
    assert legacy.exists()
    _token(legacy, value="new")
    drive_credentials.load_credentials(
        token_path=owned,
        client_path=client,
        legacy_token_path=legacy,
        credentials_class=FakeCredentials,
        request_factory=object,
    )
    assert json.loads(owned.read_text())["access_token"] == "old"


class FakeCredentials:
    def __init__(self, *, token, refresh_token, token_uri, client_id, client_secret, scopes, expiry):
        self.token = token
        self.refresh_token = refresh_token
        self.expiry = expiry
        self.valid = True
        self.expired = False
        self.fail = False

    def refresh(self, request):
        if self.fail:
            raise RuntimeError("invalid_grant secret-payload")
        self.token = "refreshed"


def test_invalid_grant_never_deletes_or_overwrites_token(tmp_path):
    from brainlayer import drive_credentials

    token = tmp_path / "token.json"
    client = tmp_path / "client.json"
    _token(token)
    _client(client)
    before = token.read_bytes()

    class InvalidCredentials(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True
            self.fail = True

    with pytest.raises(drive_credentials.DriveCredentialError, match="Drive access expired") as exc:
        drive_credentials.load_credentials(
            token_path=token,
            client_path=client,
            legacy_token_path=tmp_path / "none",
            credentials_class=InvalidCredentials,
            request_factory=object,
        )
    assert "secret-payload" not in str(exc.value)
    assert token.read_bytes() == before


def test_refresh_replaces_token_atomically_and_private(tmp_path, monkeypatch):
    from brainlayer import drive_credentials

    token = tmp_path / "token.json"
    client = tmp_path / "client.json"
    _token(token)
    _client(client)
    replacements = []
    real_replace = os.replace

    def observed_replace(source, destination):
        replacements.append((Path(source), Path(destination), Path(source).stat().st_mode & 0o777))
        real_replace(source, destination)

    monkeypatch.setattr(drive_credentials.os, "replace", observed_replace)

    class ExpiredCredentials(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True

    drive_credentials.load_credentials(
        token_path=token,
        client_path=client,
        legacy_token_path=tmp_path / "none",
        credentials_class=ExpiredCredentials,
        request_factory=object,
    )
    assert json.loads(token.read_text())["access_token"] == "refreshed"
    assert replacements == [(replacements[0][0], token, 0o600)]
    assert token.stat().st_mode & 0o777 == 0o600


def test_status_is_nonmutating_and_does_not_migrate(tmp_path):
    from brainlayer import drive_credentials

    legacy = tmp_path / "legacy.json"
    token = tmp_path / "token.json"
    client = tmp_path / "client.json"
    _token(legacy)
    _client(client)
    result = drive_credentials.credential_status(
        token_path=token, client_path=client, credentials_class=FakeCredentials, request_factory=object
    )
    assert result["state"] == "missing"
    assert legacy.exists() and not token.exists()


def test_status_json_stays_safe_on_refresh_transport_failure(tmp_path):
    from google.auth.exceptions import TransportError

    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)

    class Offline(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True

        def refresh(self, request):
            raise TransportError("private-payload")

    status = drive_credentials.credential_status(token_path=token, client_path=client, credentials_class=Offline)
    assert status["state"] == "valid" and "private-payload" not in json.dumps(status)


@pytest.mark.parametrize("elapsed_days, expected", [(2, "valid"), (6.5, "expiring")])
def test_offline_status_keeps_local_consent_clock(tmp_path, elapsed_days, expected):
    from google.auth.exceptions import TransportError

    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    consented = datetime(2026, 9, 30, tzinfo=UTC)
    data = json.loads(token.read_text())
    data["consented_at"] = consented.isoformat()
    token.write_text(json.dumps(data))

    class Offline(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True

        def refresh(self, request):
            raise TransportError("private-payload")

    status = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        credentials_class=Offline,
        now=consented + timedelta(days=elapsed_days),
    )
    assert status["state"] == expected
    assert status["expires_at"] == (consented + timedelta(days=7)).isoformat()
    assert status["days_left"] == pytest.approx(7 - elapsed_days)
    assert "offline" in status["reason"]
    assert "private-payload" not in json.dumps(status)


def test_authorize_writes_only_owned_token_with_drive_scope(tmp_path):
    from brainlayer import drive_credentials

    token = tmp_path / "owned/drive-tokens.json"
    legacy = tmp_path / "legacy.json"
    client = tmp_path / "client.json"
    _token(legacy)
    _client(client)

    class Flow:
        @classmethod
        def from_client_secrets_file(cls, path, *, scopes):
            assert path == str(client)
            assert scopes == drive_credentials.DRIVE_SCOPES
            return cls()

        def run_local_server(self, **kwargs):
            assert kwargs["port"] == 0 and kwargs["timeout_seconds"] == 180
            assert kwargs["open_browser"] is True
            return SimpleNamespace(token="authorized", refresh_token="refresh", expiry=None)

    consented = datetime(2026, 9, 30, tzinfo=UTC)
    assert (
        drive_credentials.authorize(
            token_path=token, client_path=client, flow_class=Flow, now=consented, alert_reporter=lambda *_: None
        )["status"]
        == "ok"
    )
    assert json.loads(token.read_text())["access_token"] == "authorized"
    assert json.loads(token.read_text())["consented_at"] == consented.isoformat()
    assert token.stat().st_mode & 0o777 == 0o600
    assert json.loads(legacy.read_text())["access_token"] == "old"


def test_browser_denial_reports_cancelled_without_writing_token(tmp_path):
    from oauthlib.oauth2.rfc6749.errors import AccessDeniedError

    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _client(client)

    class Flow:
        @classmethod
        def from_client_secrets_file(cls, path, *, scopes):
            return cls()

        def run_local_server(self, **kwargs):
            raise AccessDeniedError()

    result = drive_credentials.authorize(token_path=token, client_path=client, flow_class=Flow)
    assert result == {"status": "cancelled", "reason": "Google authorization cancelled"}
    assert not token.exists()


def test_testing_mode_consent_expires_after_seven_days(tmp_path):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    consented = datetime(2026, 9, 30, tzinfo=UTC)
    data = json.loads(token.read_text())
    data["consented_at"] = consented.isoformat()
    token.write_text(json.dumps(data))
    kwargs = dict(
        token_path=token,
        client_path=client,
        credentials_class=FakeCredentials,
        request_factory=object,
        alert_reporter=lambda *_: None,
    )
    assert drive_credentials.credential_status(now=consented + timedelta(days=5), **kwargs)["state"] == "valid"
    expiring = drive_credentials.credential_status(now=consented + timedelta(days=6, minutes=1), **kwargs)
    assert expiring["state"] == "expiring"
    assert expiring["expires_at"] == (consented + timedelta(days=7)).isoformat()
    assert 0 < expiring["days_left"] < 1
    assert drive_credentials.credential_status(now=consented + timedelta(days=7), **kwargs)["state"] == "invalid"


def test_cli_status_json_and_auth_json_are_single_payloads(monkeypatch):
    from typer.testing import CliRunner

    from brainlayer import drive_credentials
    from brainlayer.cli import app

    monkeypatch.setattr(drive_credentials, "credential_status", lambda: {"state": "missing", "reason": "Authorize"})
    monkeypatch.setattr(drive_credentials, "authorize", lambda: {"status": "ok", "reason": "Saved"})
    runner = CliRunner()
    status = runner.invoke(app, ["backup", "auth", "--status", "--json"])
    auth = runner.invoke(app, ["backup", "auth", "--json"])
    assert status.exit_code == 0 and json.loads(status.output) == {"state": "missing", "reason": "Authorize"}
    assert auth.exit_code == 0 and json.loads(auth.output) == {"status": "ok", "reason": "Saved"}


def test_backup_modules_do_not_delete_token_paths():
    from brainlayer import backup_daily, drive_credentials, jsonl_backup

    for module in (backup_daily, jsonl_backup, drive_credentials):
        source = Path(module.__file__).read_text()
        assert not any(f"{name}.unlink(" in source for name in ("token_path", "token", "legacy", "client"))


def test_refresh_network_error_stays_retryable_without_token_payload(tmp_path):
    from google.auth.exceptions import TransportError

    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    before = token.read_bytes()

    class NetworkFailure(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True

        def refresh(self, request):
            raise TransportError("private-payload")

    with pytest.raises(TransportError, match="network failure") as caught:
        drive_credentials.load_credentials(token_path=token, client_path=client, credentials_class=NetworkFailure)
    assert "private-payload" not in str(caught.value)
    assert token.read_bytes() == before
