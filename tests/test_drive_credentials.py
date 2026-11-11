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


class UnchangedCredentials(FakeCredentials):
    def refresh(self, request):
        pass


@pytest.mark.parametrize("hours_left", [48, 24, 0.5])
@pytest.mark.parametrize("prior_consent_alert", [None, "prior consent failure"])
def test_valid_upcoming_expiry_is_advisory_and_recovers_only_consent_alert(
    tmp_path, hours_left, prior_consent_alert, monkeypatch
):
    from brainlayer import drive_credentials, job_alerts

    token, client, alerts = tmp_path / "token.json", tmp_path / "client.json", tmp_path / "alerts.json"
    _token(token)
    _client(client)
    expires = datetime(2026, 10, 17, 18, 50, tzinfo=UTC)
    data = json.loads(token.read_text())
    data["refresh_expires_at"] = expires.isoformat()
    token.write_text(json.dumps(data))
    before = token.read_bytes()
    failed_job = "Drive access expires at fixture: Reconnect in BrainBar"
    initial = {"backup-daily": failed_job}
    if prior_consent_alert is not None:
        initial["drive-consent"] = prior_consent_alert
    alerts.write_text(json.dumps(initial))

    def report(key, reason):
        return job_alerts.report(key, reason, path=alerts, notify=lambda _: None)

    status = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        credentials_class=UnchangedCredentials,
        request_factory=object,
        alert_reporter=report,
        now=expires - timedelta(hours=hours_left),
    )

    assert status["state"] == "expiring"
    assert status["expires_at"] == expires.isoformat()
    assert status["days_left"] == pytest.approx(hours_left / 24)
    assert job_alerts.active_alerts(alerts) == {"backup-daily": failed_job}
    assert token.read_bytes() == before


@pytest.mark.parametrize("failure", ["expired", "invalid_grant", "network", "refresh"])
def test_upcoming_expiry_does_not_erase_alerts_when_credentials_fail(tmp_path, failure, monkeypatch):
    from google.auth.exceptions import TransportError

    from brainlayer import drive_credentials, job_alerts

    token, client, alerts = tmp_path / "token.json", tmp_path / "client.json", tmp_path / "alerts.json"
    _token(token)
    _client(client)
    expires = datetime(2026, 10, 17, tzinfo=UTC)
    data = json.loads(token.read_text())
    data["refresh_expires_at"] = expires.isoformat()
    token.write_text(json.dumps(data))
    alerts.write_text(json.dumps({"drive-consent": "prior failure", "backup-daily": "upload failed"}))

    class FailingCredentials(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True

        def refresh(self, request):
            if failure == "network":
                raise TransportError("offline")
            raise RuntimeError("invalid_grant" if failure == "invalid_grant" else "refresh failed")

    def report(key, reason):
        return job_alerts.report(key, reason, path=alerts, notify=lambda _: None)

    status = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        credentials_class=FailingCredentials,
        request_factory=object,
        alert_reporter=report,
        now=expires if failure == "expired" else expires - timedelta(days=1),
    )

    remaining = job_alerts.active_alerts(alerts)
    assert status["state"] == ("unknown" if failure == "network" else "invalid")
    assert remaining["backup-daily"] == "upload failed"
    assert "drive-consent" in remaining
    if failure in {"network", "refresh"}:
        assert remaining["drive-consent"] == "prior failure"


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
    assert status["state"] == "unknown" and "private-payload" not in json.dumps(status)


OFFLINE_REASON = "status checked offline; last known state"


@pytest.mark.parametrize(
    "days_left, expected",
    [
        (5, "valid"),
        (2 + 1 / 86400, "valid"),  # one second outside the two-day window
        (2, "expiring"),  # days_left == 2 exactly
        (1.5, "expiring"),
        (0.5, "expiring"),
        (1 / 86400, "expiring"),  # last second before the local deadline
    ],
)
def test_offline_status_keeps_local_consent_clock(tmp_path, days_left, expected, monkeypatch):
    from google.auth.exceptions import TransportError

    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    expires = datetime(2026, 9, 30, tzinfo=UTC)
    data = json.loads(token.read_text())
    data["refresh_expires_at"] = expires.isoformat()
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
        now=expires - timedelta(days=days_left),
    )
    deadline = expires.isoformat()
    assert status["state"] == "unknown"
    assert status["expires_at"] == deadline
    assert status["days_left"] == pytest.approx(days_left)
    assert "offline" in status["reason"]
    assert "private-payload" not in json.dumps(status)
    if expected == "expiring":
        # #1364: text --status prints only the reason, so the local deadline must be in it.
        assert status["reason"] == f"{OFFLINE_REASON}: Drive access expires at {deadline}: Reconnect in BrainBar"
    else:
        assert status["reason"] == OFFLINE_REASON


def test_cli_text_status_offline_expiring_prints_local_deadline(tmp_path, monkeypatch):
    """Real CLI text formatter + real credential_status; only the token endpoint is offline."""
    from google.auth.exceptions import TransportError
    from google.oauth2.credentials import Credentials
    from typer.testing import CliRunner

    from brainlayer.cli import app

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    expires = datetime.now(UTC) + timedelta(hours=36)
    data = json.loads(token.read_text())
    data["refresh_expires_at"] = expires.isoformat()
    data["expiry"] = "2000-01-01T00:00:00"  # stale access token forces the refresh attempt
    token.write_text(json.dumps(data))
    before = token.read_bytes()
    monkeypatch.setenv("BRAINLAYER_DRIVE_TOKEN_PATH", str(token))
    monkeypatch.setenv("BRAINLAYER_DRIVE_CLIENT_PATH", str(client))

    def offline_refresh(self, request):
        raise TransportError("private-payload")

    monkeypatch.setattr(Credentials, "refresh", offline_refresh)
    runner = CliRunner()
    text = runner.invoke(app, ["backup", "auth", "--status"])
    as_json = runner.invoke(app, ["backup", "auth", "--status", "--json"])

    deadline = expires.isoformat()
    assert text.exit_code == 0 and as_json.exit_code == 0
    assert text.output == f"{OFFLINE_REASON}: Drive access expires at {deadline}: Reconnect in BrainBar\n"
    payload = json.loads(as_json.output)
    assert payload["state"] == "unknown" and payload["expires_at"] == deadline
    assert 1 < payload["days_left"] < 2
    assert payload["reason"] == text.output.rstrip("\n")
    assert "private-payload" not in text.output + as_json.output
    assert token.read_bytes() == before


def test_authorize_writes_only_owned_token_with_drive_scope(tmp_path):
    from brainlayer import drive_credentials

    token = tmp_path / "owned/drive-tokens.json"
    legacy = tmp_path / "legacy.json"
    client = tmp_path / "client.json"
    _token(legacy)
    _client(client)

    class Flow:
        oauth2session = SimpleNamespace(token={})

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
        oauth2session = SimpleNamespace(token={})

        @classmethod
        def from_client_secrets_file(cls, path, *, scopes):
            return cls()

        def run_local_server(self, **kwargs):
            raise AccessDeniedError()

    result = drive_credentials.authorize(token_path=token, client_path=client, flow_class=Flow)
    assert result == {"status": "cancelled", "reason": "Google authorization cancelled"}
    assert not token.exists()


def test_reported_deadline_controls_status(tmp_path, monkeypatch):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    expires = datetime(2026, 9, 30, tzinfo=UTC)
    data = json.loads(token.read_text())
    data["refresh_expires_at"] = expires.isoformat()
    token.write_text(json.dumps(data))
    kwargs = dict(
        token_path=token,
        client_path=client,
        credentials_class=FakeCredentials,
        request_factory=object,
        alert_reporter=lambda *_: None,
    )
    assert drive_credentials.credential_status(now=expires - timedelta(days=2, seconds=1), **kwargs)["state"] == "valid"
    expiring = drive_credentials.credential_status(now=expires - timedelta(hours=23, minutes=59), **kwargs)
    assert expiring["state"] == "expiring"
    assert expiring["expires_at"] == expires.isoformat()
    assert 0 < expiring["days_left"] < 1
    assert drive_credentials.credential_status(now=expires, **kwargs)["state"] == "invalid"


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


@pytest.mark.parametrize("hours_left", [48, 36, 0, -1])
def test_consent_warning_names_exact_expiry_without_changing_gate(tmp_path, hours_left, caplog, monkeypatch):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    expires = datetime(2026, 10, 17, 18, 50, tzinfo=UTC)
    data = json.loads(token.read_text())
    data["refresh_expires_at"] = expires.isoformat()
    token.write_text(json.dumps(data))
    before = token.read_bytes()
    alerts = []
    kwargs = dict(
        token_path=token,
        client_path=client,
        credentials_class=UnchangedCredentials,
        request_factory=object,
        now=expires - timedelta(hours=hours_left),
        alert_reporter=lambda *args: alerts.append(args),
    )
    if hours_left <= 0:
        with pytest.raises(drive_credentials.DriveCredentialError) as caught:
            drive_credentials.load_credentials(**kwargs)
        assert expires.isoformat() in str(caught.value)
    else:
        drive_credentials.load_credentials(**kwargs)
        assert f"Drive access expires at {expires.isoformat()}" in caplog.text
    if hours_left <= 0:
        assert alerts and expires.isoformat() in alerts[-1][1]
    else:
        assert alerts == [("drive-consent", None)]
    status = drive_credentials.credential_status(**kwargs)
    assert status["state"] == ("invalid" if hours_left <= 0 else "expiring")
    assert status["expires_at"] == expires.isoformat()
    assert expires.isoformat() in status["reason"]
    assert token.read_bytes() == before


def _deadline_token(tmp_path: Path, expires: datetime) -> tuple[Path, Path]:
    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    data = json.loads(token.read_text())
    data["refresh_expires_at"] = expires.isoformat()
    token.write_text(json.dumps(data))
    return token, client


def test_invalid_grant_names_known_local_deadline_without_claiming_revocation_time(tmp_path, monkeypatch):
    """#1343: a server revocation before the Google-reported deadline keeps the local deadline."""
    from brainlayer import drive_credentials

    expires = datetime(2026, 10, 17, 18, 50, tzinfo=UTC)
    token, client = _deadline_token(tmp_path, expires)
    before = token.read_bytes()
    alerts = []

    class Revoked(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True
            self.fail = True

    kwargs = dict(
        token_path=token,
        client_path=client,
        credentials_class=Revoked,
        request_factory=object,
        now=expires - timedelta(days=3),
        alert_reporter=lambda *args: alerts.append(args),
    )
    with pytest.raises(drive_credentials.DriveCredentialError) as caught:
        drive_credentials.load_credentials(**kwargs)
    status = drive_credentials.credential_status(**kwargs)

    for text in (str(caught.value), alerts[-1][1], status["reason"]):
        assert text.startswith("Drive access expired: Reconnect in BrainBar")
        assert f"local consent deadline {expires.isoformat()}" in text
        assert "not Google's revocation time" in text
        assert "secret-payload" not in text
    assert status["state"] == "invalid"
    assert token.read_bytes() == before


def test_invalid_grant_without_consent_metadata_keeps_legacy_text(tmp_path):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)

    class Revoked(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True
            self.fail = True

    with pytest.raises(drive_credentials.DriveCredentialError) as caught:
        drive_credentials.load_credentials(
            token_path=token, client_path=client, credentials_class=Revoked, alert_reporter=lambda *_: None
        )
    assert str(caught.value) == "Drive access expired: Reconnect in BrainBar"


def test_cli_text_status_shows_local_deadline_on_real_google_refresh_error(tmp_path, monkeypatch):
    """CLI boundary: real google-auth Credentials and RefreshError; only the HTTP refresh is stubbed."""
    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials
    from typer.testing import CliRunner

    from brainlayer.cli import app

    expires = datetime.now(UTC) + timedelta(days=4)
    token, client = _deadline_token(tmp_path, expires)
    data = json.loads(token.read_text())
    data["expiry"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()  # forces a real refresh attempt
    token.write_text(json.dumps(data))

    def revoked(self, request):
        raise RefreshError("invalid_grant: Token has been expired or revoked.", {"error": "invalid_grant"})

    monkeypatch.setattr(Credentials, "refresh", revoked)
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_DRIVE_TOKEN_PATH", str(token))
    monkeypatch.setenv("BRAINLAYER_DRIVE_CLIENT_PATH", str(client))
    result = CliRunner().invoke(app, ["backup", "auth", "--status"])
    assert f"local consent deadline {expires.isoformat()}" in result.output
    assert "not Google's revocation time" in result.output
    assert "Token has been expired or revoked" not in result.output


# --- #1032: a retryable token-endpoint failure is temporary, not lost access -----------------

TEMPORARY_REASON = "status checked while Google's token endpoint was temporarily unavailable; last known state"


class _TokenEndpoint:
    """The only stub: the HTTP transport that real google-auth refresh (and its retries) calls."""

    def __init__(self, status: int, body: dict):
        self.status, self.body, self.calls = status, body, 0

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(status=self.status, data=json.dumps(self.body).encode())


def _stale_deadline_token(tmp_path: Path, expires: datetime) -> tuple[Path, Path]:
    token, client = _deadline_token(tmp_path, expires)
    data = json.loads(token.read_text())
    data["expiry"] = "2000-01-01T00:00:00"  # stale access token forces a real refresh attempt
    token.write_text(json.dumps(data))
    return token, client


@pytest.fixture
def no_backoff_sleep(monkeypatch):
    import google.auth._exponential_backoff as backoff

    sleeps = []
    monkeypatch.setattr(backoff, "time", SimpleNamespace(sleep=sleeps.append))
    return sleeps


@pytest.mark.parametrize(
    "status, body, days_left, expected",
    [
        (503, {"error": "backend_error", "error_description": "private-payload"}, 5, "valid"),
        (429, {"error": "rate_limited", "error_description": "private-payload"}, 1.5, "expiring"),
        (500, {"error": "internal_failure", "error_description": "private-payload"}, 0.5, "expiring"),
        (503, {"error": "backend_error", "error_description": "private-payload"}, -0.5, "invalid"),
    ],
)
def test_status_retryable_refresh_error_keeps_local_consent_clock(
    tmp_path, no_backoff_sleep, status, body, days_left, expected, monkeypatch
):
    from brainlayer import drive_credentials

    expires = datetime.now(UTC) + timedelta(days=days_left)
    deadline = expires.isoformat()
    token, client = _stale_deadline_token(tmp_path, expires)
    before = token.read_bytes()
    endpoint, alerts = _TokenEndpoint(status, body), []
    result = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        request_factory=lambda: endpoint,
        now=expires - timedelta(days=days_left),
        alert_reporter=lambda *args: alerts.append(args),
    )

    assert result["state"] == ("invalid" if expected == "invalid" else "unknown") and result["expires_at"] == deadline
    assert "private-payload" not in json.dumps(result)
    assert token.read_bytes() == before
    if expected == "invalid":  # the local clock already ran out; no refresh is attempted
        assert endpoint.calls == 0
        assert result["reason"] == f"Drive access expired at {deadline}: Reconnect in BrainBar"
        return
    assert endpoint.calls == 3 and len(no_backoff_sleep) == 2  # google-auth's own retries ran first
    assert alerts == []  # a temporary failure neither raises nor clears a consent alert
    assert result["days_left"] == pytest.approx(days_left, abs=1e-3)
    if expected == "expiring":
        assert result["reason"] == f"{TEMPORARY_REASON}: Drive access expires at {deadline}: Reconnect in BrainBar"
    else:
        assert result["reason"] == TEMPORARY_REASON


@pytest.mark.parametrize(
    "status, error, expected_reason_start",
    [
        (400, "invalid_grant", "Drive access expired: Reconnect in BrainBar (Google rejected refresh;"),
        (401, "unauthorized_client", "Drive token refresh failed; run brainlayer backup auth"),
    ],
)
def test_status_permanent_refresh_rejection_stays_invalid(
    tmp_path, no_backoff_sleep, status, error, expected_reason_start, monkeypatch
):
    from brainlayer import drive_credentials

    expires = datetime.now(UTC) + timedelta(days=4)
    token, client = _stale_deadline_token(tmp_path, expires)
    endpoint = _TokenEndpoint(status, {"error": error, "error_description": "private-payload"})
    result = drive_credentials.credential_status(
        token_path=token, client_path=client, request_factory=lambda: endpoint, alert_reporter=lambda *_: None
    )

    assert endpoint.calls == 1  # not retryable: google-auth gives up at once
    assert result["state"] == "invalid" and result["reason"].startswith(expected_reason_start)
    assert "private-payload" not in json.dumps(result)


def test_backup_job_retryable_refresh_error_is_refresh_failed_not_consent(tmp_path, no_backoff_sleep):
    from brainlayer import drive_credentials, health_verdict

    token, client = _stale_deadline_token(tmp_path, datetime.now(UTC) + timedelta(days=4))
    alerts = []
    with pytest.raises(drive_credentials.DriveCredentialError) as caught:
        drive_credentials.load_credentials(
            token_path=token,
            client_path=client,
            request_factory=lambda: _TokenEndpoint(503, {"error": "backend_error", "error_description": "private"}),
            alert_reporter=lambda *args: alerts.append(args),
        )

    assert isinstance(caught.value, drive_credentials.DriveRefreshUnavailableError)
    assert "run brainlayer backup auth" not in str(caught.value) and "private" not in str(caught.value)
    assert alerts == []
    receipt = {"error_type": type(caught.value).__name__, "error": str(caught.value)}
    assert health_verdict.drive_event(receipt) == "refresh_failed"  # amber retrying, never Reconnect


def test_cli_text_status_temporary_token_endpoint_failure_prints_local_deadline(
    tmp_path, monkeypatch, no_backoff_sleep
):
    """Real CLI + real google-auth Credentials and retry loop; only the token endpoint answers 503."""
    import google.auth.transport.requests as transport
    from typer.testing import CliRunner

    from brainlayer.cli import app

    expires = datetime.now(UTC) + timedelta(hours=36)
    deadline = expires.isoformat()
    token, client = _stale_deadline_token(tmp_path, expires)
    before = token.read_bytes()
    endpoint = _TokenEndpoint(503, {"error": "backend_error", "error_description": "private-payload"})
    monkeypatch.setattr(transport, "Request", lambda: endpoint)
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_DRIVE_TOKEN_PATH", str(token))
    monkeypatch.setenv("BRAINLAYER_DRIVE_CLIENT_PATH", str(client))
    runner = CliRunner()
    text = runner.invoke(app, ["backup", "auth", "--status"])
    as_json = runner.invoke(app, ["backup", "auth", "--status", "--json"])

    assert text.exit_code == 0 and as_json.exit_code == 0
    assert text.output == f"{TEMPORARY_REASON}: Drive access expires at {deadline}: Reconnect in BrainBar\n"
    payload = json.loads(as_json.output)
    assert payload["state"] == "unknown" and payload["expires_at"] == deadline
    assert payload["reason"] == text.output.rstrip("\n")
    assert "private-payload" not in text.output + as_json.output
    assert endpoint.calls == 6 and token.read_bytes() == before
    assert not (tmp_path / "alerts.json").exists()


@pytest.mark.parametrize("days_left", [0.5, -1])
def test_reported_deadline_requires_no_mode(tmp_path, monkeypatch, days_left):
    from brainlayer import drive_credentials

    expires = datetime(2026, 10, 17, tzinfo=UTC)
    token, client = _deadline_token(tmp_path, expires)
    alerts = []
    status = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        credentials_class=FakeCredentials,
        request_factory=object,
        now=expires - timedelta(days=days_left),
        alert_reporter=lambda *args: alerts.append(args),
    )
    assert status["state"] == ("expiring" if days_left > 0 else "invalid")
    assert status["expires_at"] == expires.isoformat()


def test_production_invalid_grant_still_reports_consent_failure(tmp_path, monkeypatch):
    from brainlayer import drive_credentials

    expires = datetime(2026, 10, 17, tzinfo=UTC)
    token, client = _deadline_token(tmp_path, expires)
    data = json.loads(token.read_text())
    data.pop("refresh_expires_at", None)
    token.write_text(json.dumps(data))
    alerts = []

    class Revoked(FakeCredentials):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.expired = True
            self.fail = True

    status = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        credentials_class=Revoked,
        request_factory=object,
        now=expires + timedelta(days=1),
        alert_reporter=lambda *args: alerts.append(args),
    )
    assert status["state"] == "invalid"
    assert status["expires_at"] is None
    assert status["reason"] == "Drive access expired: Reconnect in BrainBar"
    assert alerts == [("drive-consent", status["reason"])]


def test_naive_refresh_deadline_agrees_between_loader_and_status(tmp_path, monkeypatch):
    from brainlayer import drive_credentials

    expires = datetime(2026, 10, 17, 12)
    token, client = _deadline_token(tmp_path, expires)
    before = token.read_bytes()
    kwargs = dict(
        token_path=token,
        client_path=client,
        credentials_class=FakeCredentials,
        request_factory=object,
        now=datetime(2026, 10, 18, 12, tzinfo=UTC),
        alert_reporter=lambda *_: None,
    )
    status = drive_credentials.credential_status(**kwargs)
    assert status["state"] == "invalid"
    with pytest.raises(drive_credentials.DriveCredentialError) as caught:
        drive_credentials.load_credentials(**kwargs)
    assert str(caught.value) == status["reason"] == "BrainLayer Drive consent metadata is invalid"
    assert token.read_bytes() == before


@pytest.mark.parametrize("seconds", [None, 3600])
def test_google_response_controls_lifetime_through_real_refresh(tmp_path, seconds):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    now = datetime.now(UTC)
    data = json.loads(token.read_text())
    data["expiry"] = (now - timedelta(hours=1)).isoformat()
    token.write_text(json.dumps(data))
    response = {"access_token": "new", "expires_in": 86400, "token_type": "Bearer"}
    if seconds is not None:
        response["refresh_token_expires_in"] = seconds

    def endpoint(**kwargs):
        assert kwargs["url"] == "https://example.invalid/token"
        return SimpleNamespace(status=200, data=json.dumps(response).encode())

    kwargs = dict(
        token_path=token, client_path=client, request_factory=lambda: endpoint, alert_reporter=lambda *_: None, now=now
    )
    drive_credentials.credential_status(**kwargs)
    stored = json.loads(token.read_text())
    assert stored.get("refresh_expires_at") == (
        None if seconds is None else (now + timedelta(seconds=seconds)).isoformat()
    )
    status = drive_credentials.credential_status(**kwargs)
    assert status["state"] == ("valid" if seconds is None else "expiring")
    assert status["days_left"] == (None if seconds is None else seconds / 86400)


def test_legacy_consent_date_is_unknown(tmp_path, monkeypatch):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    data = json.loads(token.read_text())
    data["consented_at"] = "2000-01-01T00:00:00+00:00"
    token.write_text(json.dumps(data))
    status = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        credentials_class=FakeCredentials,
        request_factory=object,
        alert_reporter=lambda *_: None,
    )
    assert status["state"] == "valid"
    assert status["expires_at"] is None


def test_token_response_observer_never_fails_the_refresh_on_an_unreadable_body():
    from brainlayer.drive_credentials import _TokenResponseRequest

    response = SimpleNamespace(status=200, data=b"<html>not json</html>")
    observed = _TokenResponseRequest(lambda *args, **kwargs: response)

    assert observed(method="POST", url="https://oauth2.googleapis.com/token") is response
    assert observed.token_response is None


@pytest.mark.parametrize("reported", ["private-invalid-lifetime", [], {}, 1e300, float("nan"), float("inf"), True])
def test_bad_lifetime_metadata_does_not_break_or_reclassify_real_refresh(tmp_path, caplog, reported):
    status, stored, deadline = _refresh_with_known_grant(tmp_path, {"refresh_token_expires_in": reported})
    assert stored["access_token"] == "new"
    assert stored["refresh_expires_at"] == deadline
    assert status["state"] == "valid" and status["expires_at"] == deadline
    assert "private-invalid-lifetime" not in caplog.text


@pytest.mark.parametrize("response", [{}, {"refresh_token_expires_in": None}])
def test_refresh_without_reported_lifetime_preserves_google_deadline(tmp_path, response):
    status, stored, deadline = _refresh_with_known_grant(tmp_path, response)
    assert stored["access_token"] == "new"
    assert stored["refresh_expires_at"] == deadline
    assert status["state"] == "valid" and status["expires_at"] == deadline


def test_explicit_refresh_lifetime_still_replaces_the_known_deadline(tmp_path):
    status, stored, deadline = _refresh_with_known_grant(tmp_path, {"refresh_token_expires_in": 3600})
    assert stored["access_token"] == "new"
    assert stored["refresh_expires_at"] != deadline
    assert status["state"] == "expiring" and status["days_left"] == 3600 / 86400


def _refresh_with_known_grant(tmp_path, extra):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    now = datetime.now(UTC)
    deadline = (now + timedelta(days=5)).isoformat()
    data = json.loads(token.read_text())
    data.update(expiry=(now - timedelta(hours=1)).isoformat(), refresh_expires_at=deadline)
    token.write_text(json.dumps(data))
    response = {"access_token": "new", "expires_in": 86400, "token_type": "Bearer", **extra}

    def endpoint(**kwargs):
        assert kwargs["url"] == "https://example.invalid/token"
        return SimpleNamespace(status=200, data=json.dumps(response).encode())

    status = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        request_factory=lambda: endpoint,
        alert_reporter=lambda *_: None,
        now=now,
    )
    return status, json.loads(token.read_text()), deadline


@pytest.mark.parametrize("reported", ["private-invalid-lifetime", 1e300])
def test_bad_lifetime_metadata_does_not_discard_successful_consent(tmp_path, reported):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _client(client)

    class Flow:
        oauth2session = SimpleNamespace(token={"refresh_token_expires_in": reported})

        @classmethod
        def from_client_secrets_file(cls, path, *, scopes):
            return cls()

        def run_local_server(self, **kwargs):
            return SimpleNamespace(token="authorized", refresh_token="refresh", expiry=None)

    result = drive_credentials.authorize(
        token_path=token,
        client_path=client,
        flow_class=Flow,
        alert_reporter=lambda *_: None,
    )
    assert result["status"] == "ok"
    data = json.loads(token.read_text())
    assert data["access_token"] == "authorized" and data["refresh_token"] == "refresh"
    assert data["refresh_expires_at"] is None
    assert token.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("known_deadline", [False, True])
@pytest.mark.parametrize("reply", [{}, {"refresh_token_expires_in": None}])
def test_refresh_omission_never_infers_permanent(tmp_path, known_deadline, reply):
    from brainlayer import drive_credentials

    token, client = tmp_path / "token.json", tmp_path / "client.json"
    _token(token)
    _client(client)
    now = datetime.now(UTC)
    deadline = (now + timedelta(days=5)).isoformat()
    data = json.loads(token.read_text())
    data["expiry"] = (now - timedelta(hours=1)).isoformat()
    if known_deadline:
        data["refresh_expires_at"] = deadline
    token.write_text(json.dumps(data))

    def endpoint(**kwargs):
        assert kwargs["url"] == "https://example.invalid/token"
        return SimpleNamespace(
            status=200, data=json.dumps({"access_token": "new", "expires_in": 86400, **reply}).encode()
        )

    status = drive_credentials.credential_status(
        token_path=token,
        client_path=client,
        request_factory=lambda: endpoint,
        alert_reporter=lambda *_: None,
        now=now,
    )
    stored = json.loads(token.read_text())
    assert stored.get("refresh_expires_at") == (deadline if known_deadline else None)
    assert status["expires_at"] == (deadline if known_deadline else None)
