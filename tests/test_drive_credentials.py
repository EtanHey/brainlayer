"""Isolated Drive credentials tests: never use the real config or network."""

import json
import os
from pathlib import Path

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
