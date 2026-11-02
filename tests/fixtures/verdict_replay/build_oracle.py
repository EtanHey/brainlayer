"""Build tests/fixtures/verdict_replay/oracle.json from the fixture logs.

Deliberately separate code from brainlayer.health_verdict: plain dict walking, no imports
from brainlayer. Each event names its evidence (file:line). Completion time of an invocation
is not recorded, so an event is known_from=attempted_at and known_by=attempted_at+3h.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> None:
    d = Path(sys.argv[1])
    U = timedelta(hours=3)

    def ts(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    def iso(t):
        return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def created(r):
        if r.get("snapshot_created_at"):
            return ts(r["snapshot_created_at"])
        name = Path(r.get("snapshot") or "").name[:10]
        return datetime.fromisoformat(name + "T00:00:00+00:00")  # legacy: artifact date, conservative

    out = {}
    for mac in ("m4", "m1"):
        ev = []

        def add(check, value, at, line, f, why, value_at=None):
            a = ts(at)
            ev.append(
                {
                    "check": check,
                    "value": value,
                    "value_at": iso(value_at) if value_at else None,
                    "known_from": iso(a),
                    "known_by": iso(a + U),
                    "evidence": f"{f}:{line} {why}",
                }
            )

        f = f"{mac}-backup-daily.log"
        for n, line in enumerate((d / f).read_text().splitlines(), 1):
            r = json.loads(line)
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
            if r.get("snapshot") and (up or (r.get("local_verified") is True and r.get("local_integrity") == "ok")):
                add(
                    "backup_local",
                    "verified",
                    r["attempted_at"],
                    n,
                    f,
                    "verified snapshot " + Path(r["snapshot"]).name,
                    created(r),
                )
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
        ev.sort(key=lambda e: (e["check"], e["known_from"]))
        out[mac] = {"tz": "Asia/Jerusalem", "events": ev}
    (d / "oracle.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for mac in out:
        from collections import Counter

        print(mac, Counter((e["check"], e["value"]) for e in out[mac]["events"]))


if __name__ == "__main__":
    main()
