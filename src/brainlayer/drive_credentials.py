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


class DriveRefreshUnavailableError(DriveCredentialError):
    """Google's token endpoint failed retryably (5xx/429) after google-auth's own retries.

    Not lost access: the grant was never rejected. ``credential_status`` keeps the local
    consent clock; backup receipts still read as a refresh failure, not a consent loss.
    """


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


def _consent_deadline(data: dict[str, Any]) -> dt.datetime | None:
    """Only a deadline reported by Google's token endpoint is authoritative."""
    raw = data.get("refresh_expires_at")
    if raw is None or raw == "permanent":
        return None
    try:
        deadline = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            raise ValueError("deadline has no timezone")
        return deadline.astimezone(dt.UTC)
    except ValueError:
        raise DriveCredentialError("BrainLayer Drive consent metadata is invalid") from None


def _grant_metadata(response: dict[str, Any], now: dt.datetime) -> str | None:
    seconds = response.get("refresh_token_expires_in")
    if seconds is None:
        return "permanent"
    try:
        if isinstance(seconds, bool):
            raise ValueError("boolean lifetime")
        return (now.astimezone(dt.UTC) + dt.timedelta(seconds=float(seconds))).isoformat()
    except (TypeError, ValueError, OverflowError):
        LOGGER.warning("Ignoring invalid refresh-token lifetime metadata; retaining known grant lifetime")
        return None


class _TokenResponseRequest:
    """Observe successful token responses without changing google-auth's retry behavior."""

    def __init__(self, request: Any):
        self.request = request
        self.token_response = None

    def __call__(self, *args: Any, **kwargs: Any):
        response = self.request(*args, **kwargs)
        try:
            if response.status == 200:
                payload = json.loads(response.data)
                if isinstance(payload, dict) and payload.get("access_token"):
                    self.token_response = payload
        except (AttributeError, TypeError, ValueError):
            pass  # Observation only: an unreadable body must never fail google-auth's own refresh.
        return response


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
        expires = _consent_deadline(token_data)
        if expires is not None:
            remaining = expires - (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
            if remaining <= dt.timedelta(0):
                reason = f"Drive access expired at {expires.isoformat()}: Reconnect in BrainBar"
                _report_consent(alert_reporter, reason)
                raise DriveCredentialError(reason)
            if remaining <= dt.timedelta(hours=48):
                reason = f"Drive access expires at {expires.isoformat()}: Reconnect in BrainBar"
                LOGGER.warning(reason)
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
                observed = _TokenResponseRequest(request_factory())
                creds.refresh(observed)
            except Exception as exc:
                from google.auth.exceptions import TransportError

                if isinstance(exc, TransportError):
                    raise TransportError("Drive token refresh network failure") from None
                if "invalid_grant" in str(exc).lower():
                    reason = "Drive access expired: Reconnect in BrainBar"
                    if expires is not None:
                        # Google revoked before our local clock ran out; name our deadline, not theirs.
                        reason += (
                            f" (Google rejected refresh; local consent deadline {expires.isoformat()}"
                            " is not Google's revocation time)"
                        )
                    _report_consent(alert_reporter, reason)
                    raise DriveCredentialError(reason) from None
                if getattr(exc, "retryable", False) is True:
                    # A permanent rejection (checked above) always wins over a retryable flag.
                    raise DriveRefreshUnavailableError(
                        "Drive token refresh temporarily unavailable: Google's token endpoint failed retryably"
                    ) from None
                raise DriveCredentialError("Drive token refresh failed; run brainlayer backup auth") from None
            if observed.token_response is not None and (
                observed.token_response.get("refresh_token_expires_in") is not None
                or token_data.get("refresh_expires_at") is None
            ):
                lifetime = _grant_metadata(observed.token_response, now or dt.datetime.now(dt.UTC))
                if lifetime is not None:
                    token_data["refresh_expires_at"] = lifetime
            if persist_refresh:
                token_data["access_token"] = creds.token
                token_data["expiry"] = creds.expiry.isoformat() if creds.expiry else None
                _atomic_write(token, token_data)
        creds.brainlayer_refresh_expires_at = token_data.get("refresh_expires_at")
        # Upcoming expiry is credential_status's advisory, not a failed job.
        _report_consent(alert_reporter, None)
        return creds


def credential_status(*, now: dt.datetime | None = None, **kwargs: Any) -> dict[str, Any]:
    """Check the owned token without migration; persist successful refresh facts."""
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
        expires = _consent_deadline(data)
        current = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        days_left = max(0.0, (expires - current).total_seconds() / 86400) if expires else None
        timing = {"expires_at": expires.isoformat() if expires else None, "days_left": days_left}
        if expires and current >= expires:
            reason = f"Drive access expired at {expires.isoformat()}: Reconnect in BrainBar"
            _report_consent(kwargs.get("alert_reporter", job_alerts.report), reason)
            return {"state": "invalid", "reason": reason, **timing}
        creds = load_credentials(**kwargs, migrate=False, persist_refresh=True, now=now)
        expires = _consent_deadline({"refresh_expires_at": creds.brainlayer_refresh_expires_at})
        days_left = max(0.0, (expires - current).total_seconds() / 86400) if expires else None
        timing = {"expires_at": expires.isoformat() if expires else None, "days_left": days_left}
    except (TransportError, DriveRefreshUnavailableError) as exc:
        reason = "status checked offline; last known state"
        if isinstance(exc, DriveRefreshUnavailableError):
            reason = "status checked while Google's token endpoint was temporarily unavailable; last known state"
        if days_left is not None and days_left <= 2:
            # Text --status prints only the reason, so name the local consent deadline (never a server time).
            reason = f"{reason}: Drive access expires at {expires.isoformat()}: Reconnect in BrainBar"
            return {"state": "expiring", "reason": reason, **timing}
        return {"state": "valid", "reason": reason, **timing}
    except DriveCredentialError as exc:
        return {"state": "invalid", "reason": str(exc), "expires_at": None, "days_left": None}
    if days_left is not None and days_left <= 2:
        return {
            "state": "expiring",
            "reason": f"Drive access expires at {expires.isoformat()}: Reconnect in BrainBar",
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
        payload["refresh_expires_at"] = _grant_metadata(flow.oauth2session.token, now or dt.datetime.now(dt.UTC))
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
