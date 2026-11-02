"""Build tests/fixtures/verdict_replay/oracle.json from the fixture logs.

Deliberately separate code from brainlayer.health_verdict: plain dict walking, no imports
from brainlayer. Each event names its evidence (file:line).

Completion bound: receipts record when an invocation started, not when it wrote the row, and the
fixtures carry no completion field. So an event is known_from=attempted_at and
known_by=attempted_at+8h. Eight hours is the longest default run limit of the two writers
(backup-daily: DEFAULT_BACKUP_TIMEOUT_SECONDS = 8 h; jsonl-backup: 2 h wrapper default). An
invocation that ran past its limit was killed before it could write a verified row, so 8 h is an
upper bound, not an estimate; the oracle calls every hour inside the bound unknown.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

U = timedelta(hours=8)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso(t):
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def created(r):
    if r.get("snapshot_created_at"):
        return ts(r["snapshot_created_at"])
    name = Path(r.get("snapshot") or "").name[:10]
    return datetime.fromisoformat(name + "T00:00:00+00:00")  # legacy: artifact date, conservative


def event(check, value, at, line, f, why, value_at=None):
    a = ts(at)
    return {
        "check": check,
        "value": value,
        "value_at": iso(value_at) if value_at else None,
        "known_from": iso(a),
        "known_by": iso(a + U),
        "evidence": f"{f}:{line} {why}",
    }


def events_for(d: Path, mac: str) -> list[dict]:
    ev: list[dict] = []

    def add(*args, **kw):
        ev.append(event(*args, **kw))

    snapshots = {}
    drive_snapshots = {}
    creation_times = {}
    f = f"{mac}-backup-daily.log"
    # Terminal receipts can retain an older invocation timestamp. Fold each invocation
    # before later ones, preserving journal order within an invocation.
    rows = [(n, json.loads(line)) for n, line in enumerate((d / f).read_text().splitlines(), 1)]
    for n, r in sorted(rows, key=lambda row: row[1].get("attempted_at", "")):
        if r.get("backup_log_provenance") != "real" or not r.get("attempted_at"):
            continue
        err = f"{r.get('error_type') or ''} {r.get('error') or ''}"
        if "DriveCredentialError" in err:
            add(
                "drive_auth",
                "consent_needed" if "expired" in err.lower() else "refresh_failed",
                r["attempted_at"],
                n,
                f,
                err[:80],
            )
        up = (
            r.get("uploaded") is True
            and r.get("verified") is True
            and r.get("drive_md5_match") is not False
            and not r.get("verification_error")
        )
        if up and r.get("snapshot"):
            add(
                "backup_offsite",
                "verified",
                r["attempted_at"],
                n,
                f,
                "uploaded+verified " + Path(r["snapshot"]).name,
                created(r),
            )
            add("drive_auth", "ok", r["attempted_at"], n, f, "verified Drive upload")
        key = r.get("snapshot")
        if key:
            stamp = created(r) if r.get("snapshot_created_at") else (creation_times.get(key) or created(r))
            creation_times[key] = stamp
            before = max([*snapshots.values(), *drive_snapshots.values()], default=None)
            revoked = bool(r.get("verification_error")) or r.get("drive_md5_match") is False
            if revoked:
                snapshots.pop(key, None)
                drive_snapshots.pop(key, None)
            else:
                if up:
                    drive_snapshots[key] = stamp
                    snapshots[key] = stamp
                elif r.get("local_verified") is True and r.get("local_integrity") == "ok":
                    snapshots[key] = stamp
            if (
                r.get("local_verified") is False
                or r.get("upload_state") in {"abandoned", "superseded"}
                or r.get("local_removed") is True
            ):
                snapshots.pop(key, None)
            newest = max([*snapshots.values(), *drive_snapshots.values()], default=None)
            if newest != before or up or (r.get("local_verified") is True and not revoked):
                why = "newest verified snapshot " + Path(key).name
                if r.get("local_removed") is True and key in drive_snapshots:
                    why += "; local copy removed after Drive verification"
                add("backup_local", "verified" if newest else "none", r["attempted_at"], n, f, why, newest)
        if isinstance(r.get("local_integrity"), str) and r["local_integrity"] not in ("ok", "skipped", ""):
            add("integrity", "corrupt?", r["attempted_at"], n, f, r["local_integrity"][:80])
        elif r.get("local_integrity") == "ok":
            add("integrity", "ok", r["attempted_at"], n, f, f"{r.get('local_integrity_pragma')} ok")
    f = f"{mac}-jsonl-backup.log"
    for n, line in enumerate((d / f).read_text().splitlines(), 1):
        r = json.loads(line)
        if r.get("verified") is True and r.get("status") in ("uploaded", "no-op", "created"):
            add("transcripts", "verified", r["attempted_at"], n, f, f"status={r['status']}", ts(r["attempted_at"]))
            if r.get("status") == "uploaded":
                add("drive_auth", "ok", r["attempted_at"], n, f, "verified Drive upload")
        if "Drive access expired" in str(r.get("error")):
            add("drive_auth", "consent_needed", r["attempted_at"], n, f, str(r["error"])[:80])
    # The final receipt at an invocation timestamp determines its surviving clock.
    clocks = {(e["check"], e["known_from"]): e for e in ev if e["check"] == "backup_local"}
    ev = [
        e
        for e in ev
        if e["check"] != "backup_local" or e["value_at"] == clocks[(e["check"], e["known_from"])]["value_at"]
    ]
    ev.sort(key=lambda e: (e["check"], e["known_from"]))
    return ev


def main() -> None:
    d = Path(sys.argv[1])
    out = {mac: {"tz": "Asia/Jerusalem", "events": events_for(d, mac)} for mac in ("m4", "m1")}
    (d / "oracle.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for mac in out:
        from collections import Counter

        print(mac, Counter((e["check"], e["value"]) for e in out[mac]["events"]))


if __name__ == "__main__":
    main()
