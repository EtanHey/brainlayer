"""BrainLayer-owned Google Drive credentials shared by both backup jobs.

The MCP's token is a migration source only. Backup refreshes and interactive auth
write solely to BrainLayer's private token path.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import logging
import os
import socket
import stat
import tempfile
from pathlib import Path
from typing import Any

from . import job_alerts

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_ENV = "BRAINLAYER_DRIVE_TOKEN_PATH"
CLIENT_ENV = "BRAINLAYER_DRIVE_CLIENT_PATH"
DEFAULT_TOKEN_PATH = Path.home() / ".config/brainlayer/drive-tokens.json"
DEFAULT_CLIENT_PATH = Path.home() / ".config/google-drive-mcp/gcp-oauth.keys.json"
LEGACY_TOKEN_PATH = Path.home() / ".config/google-drive-mcp/tokens.json"
LOGGER = logging.getLogger(__name__)


def _report_consent(reporter: Any, reason: str | None) -> None:
    try:
        reporter("drive-consent", reason)
    except OSError:
        LOGGER.warning("Drive consent alert could not be recorded")


class DriveCredentialError(RuntimeError):
    """A payload-free credential failure safe for logs and notifications."""


def resolve_paths(*, environ: dict[str, str] | None = None, home: Path | None = None) -> tuple[Path, Path, Path]:
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    token = Path(environ.get(TOKEN_ENV) or home / ".config/brainlayer/drive-tokens.json").expanduser()
    client = Path(environ.get(CLIENT_ENV) or home / ".config/google-drive-mcp/gcp-oauth.keys.json").expanduser()
    legacy = home / ".config/google-drive-mcp/tokens.json"
    return token, client, legacy


def _paths(
    token_path: Path | None, client_path: Path | None, legacy_token_path: Path | None
) -> tuple[Path, Path, Path]:
    defaults = resolve_paths()
    return tuple(
        Path(path).expanduser() if path is not None else default
        for path, default in zip((token_path, client_path, legacy_token_path), defaults, strict=True)
    )


def _read_object(path: Path, *, private: bool) -> dict[str, Any]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise DriveCredentialError("BrainLayer Drive token missing; run brainlayer backup auth") from None
    except OSError:
        raise DriveCredentialError("Drive credential file cannot be read") from None
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise DriveCredentialError("Drive credential file must be a regular file owned by this user")
        if private and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise DriveCredentialError("BrainLayer Drive token must have mode 0600")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("not an object")
        return payload
    except (UnicodeError, ValueError, OSError):
        raise DriveCredentialError("Drive credential file is invalid") from None
    finally:
        if fd >= 0:
            os.close(fd)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        # Only an uncommitted temporary file is removable; a token is never deleted.
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def _token_lock(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _migrate_if_missing(token: Path, legacy: Path) -> None:
    if token.exists() or not legacy.exists():
        return
    payload = _read_object(legacy, private=False)
    if not payload.get("access_token") or not payload.get("refresh_token"):
        raise DriveCredentialError("Legacy Drive token is invalid; run brainlayer backup auth")
    _atomic_write(token, payload)
    LOGGER.warning("Copied legacy Google Drive token into BrainLayer-owned storage; legacy token retained")


def _expiry(data: dict[str, Any]) -> dt.datetime | None:
    raw = data.get("expiry")
    if not raw and data.get("expiry_date"):
        raw = dt.datetime.fromtimestamp(int(data["expiry_date"]) / 1000, tz=dt.UTC).isoformat()
    if not raw:
        return None
    parsed = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return parsed.astimezone(dt.UTC).replace(tzinfo=None) if parsed.tzinfo else parsed


def load_credentials(
    *,
    token_path: Path | None = None,
    client_path: Path | None = None,
    legacy_token_path: Path | None = None,
    credentials_class: Any = None,
    request_factory: Any = None,
    migrate: bool = True,
    persist_refresh: bool = True,
    now: dt.datetime | None = None,
    alert_reporter: Any = job_alerts.report,
):
    """Read/refresh under one lock; never erase the prior token on any failure."""
    if credentials_class is None:
        from google.oauth2.credentials import Credentials

        credentials_class = Credentials
    if request_factory is None:
        from google.auth.transport.requests import Request

        request_factory = Request
    token, client, legacy = _paths(token_path, client_path, legacy_token_path)
    with _token_lock(token):
        if migrate:
            _migrate_if_missing(token, legacy)
        token_data = _read_object(token, private=True)
        consented = token_data.get("consented_at")
        if consented:
            try:
                consent_time = dt.datetime.fromisoformat(str(consented).replace("Z", "+00:00")).astimezone(dt.UTC)
            except ValueError:
                raise DriveCredentialError("BrainLayer Drive consent metadata is invalid") from None
            remaining = consent_time + dt.timedelta(days=7) - (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
            if remaining <= dt.timedelta(0):
                _report_consent(alert_reporter, "Drive access expired: Reconnect in BrainBar")
                raise DriveCredentialError("Drive access expired: Reconnect in BrainBar")
            if remaining < dt.timedelta(days=1):
                _report_consent(
                    alert_reporter,
                    "Google Drive access for BrainLayer backups expires tomorrow: click Reconnect in BrainBar",
                )
        if not token_data.get("refresh_token"):
            raise DriveCredentialError("BrainLayer Drive token lacks refresh authorization; run brainlayer backup auth")
        try:
            client_data = _read_object(client, private=False)["installed"]
            creds = credentials_class(
                token=token_data.get("access_token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=client_data["token_uri"],
                client_id=client_data["client_id"],
                client_secret=client_data["client_secret"],
                scopes=str(token_data.get("scope") or " ".join(DRIVE_SCOPES)).split(),
                expiry=_expiry(token_data),
            )
        except (KeyError, TypeError, ValueError, DriveCredentialError):
            raise DriveCredentialError("Google OAuth client or BrainLayer token is invalid") from None
        refresh_before = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=2)
        if creds.expired or not creds.valid or (creds.expiry and creds.expiry < refresh_before):
            try:
                creds.refresh(request_factory())
            except Exception as exc:
                from google.auth.exceptions import TransportError

                if isinstance(exc, TransportError):
                    raise TransportError("Drive token refresh network failure") from None
                if "invalid_grant" in str(exc).lower():
                    _report_consent(alert_reporter, "Drive access expired: Reconnect in BrainBar")
                    raise DriveCredentialError("Drive access expired: Reconnect in BrainBar") from None
                raise DriveCredentialError("Drive token refresh failed; run brainlayer backup auth") from None
            if persist_refresh:
                token_data["access_token"] = creds.token
                token_data["expiry"] = creds.expiry.isoformat() if creds.expiry else None
                _atomic_write(token, token_data)
        return creds


def credential_status(*, now: dt.datetime | None = None, **kwargs: Any) -> dict[str, Any]:
    """Check the owned token only; do not migrate or write a refreshed token."""
    from google.auth.exceptions import TransportError

    token, _, _ = _paths(kwargs.get("token_path"), kwargs.get("client_path"), None)
    if not token.exists():
        return {
            "state": "missing",
            "reason": "BrainLayer Drive authorization is missing",
            "expires_at": None,
            "days_left": None,
        }
    try:
        data = _read_object(token, private=True)
        consented = data.get("consented_at")
        try:
            expires = (
                dt.datetime.fromisoformat(str(consented).replace("Z", "+00:00")) + dt.timedelta(days=7)
                if consented
                else None
            )
            if expires and expires.tzinfo is None:
                raise ValueError("consent time has no timezone")
        except ValueError:
            raise DriveCredentialError("BrainLayer Drive consent metadata is invalid") from None
        current = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        days_left = max(0.0, (expires - current).total_seconds() / 86400) if expires else None
        timing = {"expires_at": expires.isoformat() if expires else None, "days_left": days_left}
        if expires and current >= expires:
            _report_consent(
                kwargs.get("alert_reporter", job_alerts.report), "Drive access expired: Reconnect in BrainBar"
            )
            return {"state": "invalid", "reason": "Drive access expired: Reconnect in BrainBar", **timing}
        load_credentials(**kwargs, migrate=False, persist_refresh=False, now=now)
    except TransportError:
        return {
            "state": "expiring" if days_left is not None and days_left < 1 else "valid",
            "reason": "status checked offline; last known state",
            **timing,
        }
    except DriveCredentialError as exc:
        return {"state": "invalid", "reason": str(exc), "expires_at": None, "days_left": None}
    if days_left is not None and days_left < 1:
        return {
            "state": "expiring",
            "reason": "Google Drive access for BrainLayer backups expires tomorrow: click Reconnect in BrainBar",
            **timing,
        }
    return {"state": "valid", "reason": "BrainLayer Drive authorization is ready", **timing}


def authorize(
    *,
    token_path: Path | None = None,
    client_path: Path | None = None,
    flow_class: Any = None,
    now: dt.datetime | None = None,
    alert_reporter: Any = job_alerts.report,
) -> dict[str, str]:
    """Open Google consent in the system browser and save only BrainLayer's token."""
    from oauthlib.oauth2.rfc6749.errors import AccessDeniedError

    token, client, _ = _paths(token_path, client_path, None)
    if flow_class is None:
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow_class = InstalledAppFlow
    try:
        _read_object(client, private=False)
        flow = flow_class.from_client_secrets_file(str(client), scopes=DRIVE_SCOPES)
        # Suppress the consent URL on stdout; BrainBar expects one JSON result.
        creds = flow.run_local_server(
            host="localhost",
            bind_addr="127.0.0.1",
            port=0,
            timeout_seconds=180,
            open_browser=True,
            authorization_prompt_message=None,
            access_type="offline",
            prompt="consent",
        )
        if not creds.refresh_token:
            raise DriveCredentialError("Google did not grant refresh authorization")
        payload = {
            "access_token": creds.token,
            "refresh_token": creds.refresh_token,
            "expiry": creds.expiry.isoformat() if creds.expiry else None,
            "scope": " ".join(DRIVE_SCOPES),
            "consented_at": (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).isoformat(),
        }
        with _token_lock(token):
            _atomic_write(token, payload)
        _report_consent(alert_reporter, None)
        return {"status": "ok", "reason": "BrainLayer Drive authorization saved"}
    except (KeyboardInterrupt, AccessDeniedError):
        return {"status": "cancelled", "reason": "Google authorization cancelled"}
    except Exception as exc:
        if isinstance(exc, (socket.timeout, TimeoutError)) or type(exc).__name__ == "WSGITimeoutError":
            return {"status": "timeout", "reason": "Google authorization timed out"}
        return {"status": "error", "reason": "Google authorization failed; retry brainlayer backup auth"}
