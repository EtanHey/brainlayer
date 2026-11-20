"""Alert once when a transcript being backed up holds a key that looks current.

The scan reads only the bundle's own per-member snapshot, the exact bytes the
JSONL backup archives, so it never opens a source file and never changes a byte
of the backup. Detection reuses the provider patterns in
``pipeline.secret_scrub`` (never the generic entropy rule) and never calls any
provider to test a key.

No secret value is ever kept: a finding is a provider name, a path relative to
its source root, a line number and the sha256 of the value. Pending findings
are persisted before alert delivery and retried until the alert is recorded.

Cost. A nightly re-bundles whole transcripts that mostly only grew, and a full
pass of the provider patterns is several times slower than gzip. So when the
backup state proves a file's first N bytes are exactly a snapshot an earlier
scanning run already bundled (same size prefix, same sha256), only the bytes
from the last line boundary before N are scanned. Anything else (a new file, a
shrink, a rewrite, an entry written before scanning existed) is scanned whole.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Mapping

from .pipeline.secret_scrub import provider_secret_spans

ALERT_KEY = "backup-credential-seen"
STATE_FILENAME = "jsonl-backup-credential-scan.json"
CURRENT_WINDOW_SECONDS = 7 * 24 * 60 * 60
# Transcript text only; media, databases and protobufs in the same roots are bundled unscanned.
SCANNED_SUFFIXES = frozenset({".jsonl", ".json"})
# Scan in blocks of about this size, cut at the last newline so a key never straddles a cut.
_BLOCK_BYTES = 1024 * 1024
# A block with no newline at all is cut here instead; the tail is rescanned with the next block.
_LONG_LINE_OVERLAP_CHARS = 1024
_LONG_LINE_CARRY_BYTES = 4 * _LONG_LINE_OVERLAP_CHARS
# JSON escapes glue the next word to the letter n/r/t, which defeats the patterns' \b.
# Two spaces keep every offset, so line numbers stay exact.
_JSON_ESCAPES = ("\\n", "\\r", "\\t")
_MAX_FILES_IN_ALERT = 3
# Standard base64 contains "/", so a provider shape like google_oauth_refresh's
# "1//..." turns up inside pasted images: 493 of 496 findings in the #1825 live
# check, every one inside a base64 run of 100+ chars. A real key stands between
# quotes, spaces or a short "label=", so its base64-alphabet neighbourhood is
# small; a match whose run reaches this far past it is data, not a credential.
_BASE64_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
_EMBEDDED_MIN_EXTENSION = 32
# A value whose tail repeats a short unit ("Fa1KeFa1Ke...", "0000...", "xxxx...") is
# a placeholder: a real key never has 16 chars of period <= 8.
_PLACEHOLDER_MAX_PERIOD = 8
_PLACEHOLDER_MIN_RUN = 16
# Public OpenSSH security-key algorithm identifiers, never credentials. Match
# the whole identifier so a credential sharing a prefix is still detected.
_SSH_ALGORITHMS = frozenset(
    f"sk-{kind}{cert}@openssh.com" for kind in ("ssh-ed25519", "ecdsa-sha2-nistp256") for cert in ("", "-cert-v01")
)


def is_embedded_in_base64(text: str, start: int, end: int) -> bool:
    """True when the base64-alphabet run around [start, end) extends 32+ chars beyond it."""
    left = start
    left_limit = max(0, start - _EMBEDDED_MIN_EXTENSION)
    while left > left_limit and text[left - 1] in _BASE64_CHARS:
        left -= 1
    right = end
    right_limit = min(len(text), end + _EMBEDDED_MIN_EXTENSION)
    while right < right_limit and text[right] in _BASE64_CHARS:
        right += 1
    return (start - left) + (right - end) >= _EMBEDDED_MIN_EXTENSION


def is_placeholder(value: str) -> bool:
    """True when a tail of 16+ chars, at least half the value, repeats a unit of <= 8 chars."""
    length = len(value)
    needed = max(_PLACEHOLDER_MIN_RUN, length // 2)
    for period in range(1, _PLACEHOLDER_MAX_PERIOD + 1):
        index = length - 1
        while index - period >= 0 and value[index] == value[index - period]:
            index -= 1
        if length - 1 - index + period >= needed:
            return True
    return False


@dataclass(frozen=True)
class CredentialFinding:
    provider: str
    path: str
    line: int
    fingerprint: str
    current: bool


@dataclass(frozen=True)
class PriorSnapshot:
    """A snapshot of this file that an earlier scanning run bundled and scanned."""

    size: int
    sha256: str


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FileScan:
    """Scan of one file's bytes, fed in any chunking; see ``scan_snapshot`` for the bundle path."""

    def __init__(self, relative_path: str, prior: PriorSnapshot | None = None) -> None:
        self.relative_path = relative_path
        self.scanned_bytes = 0
        self.proven_prefix_bytes = 0
        self.ignored_embedded = 0
        self.ignored_placeholder = 0
        self._prior = prior
        self._observed = 0
        self._prefix_hash = hashlib.sha256() if prior is not None else None
        self._prefix_newlines = 0
        self._prefix_last_newline = -1
        self._pending: list[bytes] = []
        self._pending_bytes = 0
        # After a long-line carry, the first chars were judged by the previous block
        # with full context; here their left context is cut, so they are skipped.
        self._skip_head = 0
        self._line = 1
        self._found: dict[tuple[str, str], int] = {}

    def observe(self, chunk: bytes) -> None:
        """Watch the bundler's read, in order, to prove (or not) the prior prefix."""
        if self._prefix_hash is not None and self._observed < self._prior.size:
            part = chunk[: self._prior.size - self._observed]
            self._prefix_hash.update(part)
            newlines = part.count(b"\n")
            if newlines:
                self._prefix_newlines += newlines
                self._prefix_last_newline = self._observed + part.rfind(b"\n")
        self._observed += len(chunk)

    def scan_snapshot(self, snapshot: IO[bytes]) -> None:
        """Scan the member snapshot, skipping a proven prefix up to its last line boundary."""
        start = 0
        if (
            self._prefix_hash is not None
            and self._observed >= self._prior.size
            and self._prefix_hash.hexdigest() == self._prior.sha256
        ):
            start = self._prefix_last_newline + 1
            self._line = self._prefix_newlines + 1
            self.proven_prefix_bytes = start
        snapshot.seek(start)
        while chunk := snapshot.read(_BLOCK_BYTES):
            self.feed(chunk)

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.scanned_bytes += len(chunk)
        self._pending.append(chunk)
        self._pending_bytes += len(chunk)
        if self._pending_bytes >= _BLOCK_BYTES:
            self._scan_pending(final=False)

    def finish(self) -> list[tuple[str, int, str]]:
        """Scan what is left; return (provider, line, fingerprint), first line per value."""
        self._scan_pending(final=True)
        return [(provider, line, digest) for (provider, digest), line in self._found.items()]

    def _scan_pending(self, *, final: bool) -> None:
        data = b"".join(self._pending)
        self._pending = []
        self._pending_bytes = 0
        if not data:
            return
        skip_head, self._skip_head = self._skip_head, 0
        if final:
            self._scan(data, accept_before=None, skip_head=skip_head)
            return
        cut = data.rfind(b"\n") + 1
        if cut:
            self._scan(data[:cut], accept_before=None, skip_head=skip_head)
            self._line += data.count(b"\n", 0, cut)
            carry = data[cut:]
        else:
            # One line longer than a block: keep memory bounded, accept only matches
            # clear of the tail, and rescan that tail with the next block.
            self._scan(data, accept_before=_LONG_LINE_OVERLAP_CHARS, skip_head=skip_head)
            carry = data[-_LONG_LINE_CARRY_BYTES:]
            self._skip_head = _EMBEDDED_MIN_EXTENSION
        if carry:
            self._pending = [carry]
            self._pending_bytes = len(carry)

    def _scan(self, data: bytes, *, accept_before: int | None, skip_head: int = 0) -> None:
        text = data.decode("utf-8", errors="replace")
        for escape in _JSON_ESCAPES:
            text = text.replace(escape, "  ")
        limit = None if accept_before is None else len(text) - accept_before
        for span in provider_secret_spans(text):
            if (limit is not None and span.end > limit) or span.start < skip_head:
                continue
            if span.provider == "openai" and any(
                text.startswith(name, span.start)
                and span.end < span.start + len(name)
                and (
                    span.start + len(name) == len(text)
                    or text[span.start + len(name)]
                    not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.@+-"
                )
                for name in _SSH_ALGORITHMS
            ):
                continue
            if is_embedded_in_base64(text, span.start, span.end):
                self.ignored_embedded += 1
                continue
            if is_placeholder(span.original):
                self.ignored_placeholder += 1
                continue
            key = (span.provider, fingerprint(span.original))
            if key not in self._found:
                self._found[key] = self._line + text.count("\n", 0, span.start)


@dataclass
class CredentialScan:
    """Collects findings across one bundling run.

    ``priors`` is the backup state's ``files`` map. An entry is trusted as already
    scanned only when its recorded mtime is strictly after ``scanning_since`` (the
    last state write before the first scanning run; see ``_last_unscanned_write``):
    an entry written by a run that did not scan records an mtime at or before
    that, so it can never vouch for bytes nobody scanned.
    """

    now: float
    priors: Mapping[str, Any] = field(default_factory=dict)
    scanning_since: float | None = None
    findings: list[CredentialFinding] = field(default_factory=list)
    scanned_files: int = 0
    scanned_bytes: int = 0
    proven_prefix_bytes: int = 0
    ignored_embedded: int = 0
    ignored_placeholder: int = 0

    def open_file(self, path: Path, root: Path) -> FileScan | None:
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            return None
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        return FileScan(relative, self._prior(path))

    def _prior(self, path: Path) -> PriorSnapshot | None:
        entry = self.priors.get(path.as_posix())
        if self.scanning_since is None or not isinstance(entry, dict) or entry.get("credential_scan_failed"):
            return None
        size, digest, mtime = entry.get("size"), entry.get("sha256"), entry.get("mtime")
        if not (isinstance(size, int) and size > 0 and isinstance(digest, str) and digest):
            return None
        if not isinstance(mtime, (int, float)) or mtime <= self.scanning_since:
            return None
        return PriorSnapshot(size=size, sha256=digest)

    def commit(self, scan: FileScan, snapshot: IO[bytes], *, mtime: float) -> None:
        """Scan a member's snapshot once it is safely in the bundle, and record its findings."""
        scan.scan_snapshot(snapshot)
        current = self.now - mtime < CURRENT_WINDOW_SECONDS
        for provider, line, digest in scan.finish():
            self.findings.append(
                CredentialFinding(
                    provider=provider, path=scan.relative_path, line=line, fingerprint=digest, current=current
                )
            )
        self.scanned_files += 1
        self.scanned_bytes += scan.scanned_bytes
        self.proven_prefix_bytes += scan.proven_prefix_bytes
        self.ignored_embedded += scan.ignored_embedded
        self.ignored_placeholder += scan.ignored_placeholder


def state_path_for(backup_state_path: Path) -> Path:
    return Path(backup_state_path).expanduser().parent / STATE_FILENAME


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    alerted = data.get("alerted")
    data["alerted"] = alerted if isinstance(alerted, dict) else {}
    for key in ("last_new_at", "scanning_since"):
        if not isinstance(data.get(key), (int, float)):
            data[key] = None
    return data


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temp_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _last_unscanned_write(backup_state: Mapping[str, Any], now: float) -> float:
    """The newest moment a run that did not scan can have written a state entry.

    Every entry records its file's mtime as seen before the run wrote state, so an
    entry from a run that did not scan has mtime <= that run's ``updated_at``. The
    first scanning run uses the state's ``updated_at`` (the last write before it)
    as the cutoff, so its own entries vouch on the very next night. Using its own
    start time instead made night 2 re-scan everything (live check, Defect 2: an
    entry's mtime is always 600+ s older than the run that writes it).
    """
    files = backup_state.get("files")
    if not files:
        return 0.0  # No entry exists that a run without the scan could have written.
    updated_at = backup_state.get("updated_at")
    if isinstance(updated_at, str):
        try:
            return dt.datetime.fromisoformat(updated_at).timestamp()
        except ValueError:
            pass
    return now  # Entries of unknown age: trust none of them.


def start(backup_state: Mapping[str, Any], *, state_path: Path, now: float) -> CredentialScan:
    """Begin a run's scan; the first scanning run records the trust cutoff once."""
    state = _load(state_path)
    if state["scanning_since"] is None:
        state["scanning_since"] = _last_unscanned_write(backup_state, now)
        _write(state_path, state)
    return CredentialScan(now=now, priors=backup_state.get("files") or {}, scanning_since=state["scanning_since"])


def alert_message(new: list[CredentialFinding]) -> str:
    providers = ", ".join(sorted({finding.provider for finding in new}))
    names = list(dict.fromkeys(Path(finding.path).name for finding in new))
    shown = ", ".join(names[:_MAX_FILES_IN_ALERT])
    if len(names) > _MAX_FILES_IN_ALERT:
        shown += f" and {len(names) - _MAX_FILES_IN_ALERT} more"
    return f"A key that looks current ({providers}) is in a transcript: rotate it ({shown})"


def apply_findings(scan: CredentialScan, *, state_path: Path, alert_path: Path) -> dict:
    """Alert once per new current-looking fingerprint; clear after a quiet window.

    Returns counts only, so the backup log can carry them.
    """
    from .job_alerts import active_alerts, report

    now = scan.now
    state = _load(state_path)
    alerted: dict[str, Any] = state["alerted"]
    new: list[CredentialFinding] = []
    new_fingerprints: set[str] = set()
    pending = [CredentialFinding(**row) for row in state.get("pending", [])]
    for finding in pending + scan.findings:
        if finding.current and finding.fingerprint not in alerted and finding.fingerprint not in new_fingerprints:
            new_fingerprints.add(finding.fingerprint)
            new.append(finding)
    if new:
        # Persist only safe finding metadata before alert I/O. An unchanged file
        # may never be bundled again, so retry delivery even on a no-op backup.
        state["pending"] = [asdict(finding) for finding in new]
        _write(state_path, state)
        message = alert_message(new)
        # job_alerts keeps the first reason of an open episode; a new key must still be seen.
        if active_alerts(alert_path).get(ALERT_KEY) not in (None, message):
            report(ALERT_KEY, None, path=alert_path)
        report(ALERT_KEY, message, path=alert_path)
        seen_at = dt.datetime.fromtimestamp(now, dt.UTC).isoformat()
        for finding in new:
            alerted[finding.fingerprint] = {
                "provider": finding.provider,
                "path": finding.path,
                "line": finding.line,
                "alerted_at": seen_at,
            }
        state["last_new_at"] = now
        state["pending"] = []
        _write(state_path, state)
    elif state["last_new_at"] is None or now - state["last_new_at"] >= CURRENT_WINDOW_SECONDS:
        report(ALERT_KEY, None, path=alert_path)
    return {
        "findings": len(scan.findings),
        "current_findings": sum(1 for finding in scan.findings if finding.current),
        "new_alerts": len(new),
        "scanned_files": scan.scanned_files,
        "scanned_bytes": scan.scanned_bytes,
        "proven_prefix_bytes": scan.proven_prefix_bytes,
        "ignored_embedded_in_base64": scan.ignored_embedded,
        "ignored_placeholders": scan.ignored_placeholder,
    }
