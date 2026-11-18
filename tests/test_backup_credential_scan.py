"""The transcript backup's credential scan: alert once on a key that looks current, store no values."""

import base64
import datetime as dt
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

from brainlayer import backup_credential_scan as scan
from brainlayer import jsonl_backup as backup
from brainlayer.job_alerts import active_alerts, report

DAY = 24 * 60 * 60


def _fake_key(seed: str, provider: str = "anthropic") -> str:
    # Built at runtime from a hash so no real-looking secret is ever committed.
    body = hashlib.sha256(seed.encode()).hexdigest()[:40]
    prefix = {"anthropic": "sk-" + "ant-" + "api03-", "github": "gh" + "p_"}[provider]
    return prefix + body


def _transcript(path: Path, value: str, *, mtime: float, lines_before: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"type": "message", "n": index}) for index in range(lines_before)]
    # The key follows a JSON-escaped newline, as pasted text does inside a transcript.
    rows.append(json.dumps({"type": "message", "text": f"here it is:\n{value} thanks"}))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _run(tmp_path: Path, source: Path, *, now: float) -> dict:
    return backup.run_backup(
        source_roots=[source],
        state_path=tmp_path / "state" / "state.json",
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "state" / "backup.log",
        queue_dir=tmp_path / "state" / "queue",
        date_stamp="2026-11-18",
        now=now,
        upload=False,
        active_skip_seconds=60,
    )


def _counts(result: dict) -> dict:
    keys = ("findings", "current_findings", "new_alerts")
    return {key: result["credential_scan"][key] for key in keys}


def _alerts(tmp_path: Path) -> dict[str, str]:
    return active_alerts(tmp_path / "state" / "job-alerts.json")


def _assert_value_nowhere(tmp_path: Path, value: str, result: dict, captured: str) -> None:
    """Alerts, logs, scan state, queue and the result carry neither the value nor a prefix of it."""
    probes = (value, value[:16], value[-16:])
    for path in (tmp_path / "state").rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            assert not any(probe in text for probe in probes), path
    serialized = json.dumps(result) + captured
    assert not any(probe in serialized for probe in probes)


def test_current_provider_key_alerts_once_naming_provider_and_file_never_the_value(tmp_path, capsys, caplog):
    now = time.time()
    value = _fake_key("current")
    source = tmp_path / "sources"
    _transcript(source / "proj" / "session.jsonl", value, mtime=now - 3600, lines_before=4)

    result = _run(tmp_path, source, now=now)

    assert _counts(result) == {"findings": 1, "current_findings": 1, "new_alerts": 1}
    reason = _alerts(tmp_path)[scan.ALERT_KEY]
    assert reason == "A key that looks current (anthropic) is in a transcript: rotate it (session.jsonl)"
    state = json.loads(scan.state_path_for(tmp_path / "state" / "state.json").read_text())
    assert state["alerted"] == {
        hashlib.sha256(value.encode()).hexdigest(): {
            "provider": "anthropic",
            "path": "proj/session.jsonl",
            "line": 5,
            "alerted_at": state["alerted"][hashlib.sha256(value.encode()).hexdigest()]["alerted_at"],
        }
    }
    out = capsys.readouterr()
    _assert_value_nowhere(tmp_path, value, result, out.out + out.err + caplog.text)


def test_same_key_again_even_in_another_file_never_alerts_twice(tmp_path):
    now = time.time()
    value = _fake_key("repeat")
    source = tmp_path / "sources"
    _transcript(source / "a.jsonl", value, mtime=now - 3600)
    assert _run(tmp_path, source, now=now)["credential_scan"]["new_alerts"] == 1
    report(scan.ALERT_KEY, None, path=tmp_path / "state" / "job-alerts.json")  # owner dismissed it

    _transcript(source / "b.jsonl", value, mtime=now - 1800)
    result = _run(tmp_path, source, now=now + 60)

    assert _counts(result) == {"findings": 2, "current_findings": 2, "new_alerts": 0}
    assert scan.ALERT_KEY not in _alerts(tmp_path)


def test_key_in_a_file_older_than_seven_days_does_not_alert(tmp_path):
    now = time.time()
    source = tmp_path / "sources"
    _transcript(source / "old.jsonl", _fake_key("old"), mtime=now - 8 * DAY)

    result = _run(tmp_path, source, now=now)

    assert _counts(result) == {"findings": 1, "current_findings": 0, "new_alerts": 0}
    assert scan.ALERT_KEY not in _alerts(tmp_path)
    assert json.loads(scan.state_path_for(tmp_path / "state" / "state.json").read_text())["alerted"] == {}


def test_generic_high_entropy_string_does_not_alert(tmp_path):
    now = time.time()
    source = tmp_path / "sources"
    generic = hashlib.sha256(b"generic").hexdigest()[:20] + "Zq9XwP7LmN2kJ4vB8"
    path = source / "generic.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"api_key": generic, "token": "Ab3" + generic[::-1]}) + "\n", encoding="utf-8")
    os.utime(path, (now - 3600, now - 3600))

    result = _run(tmp_path, source, now=now)

    assert _counts(result) == {"findings": 0, "current_findings": 0, "new_alerts": 0}
    assert scan.ALERT_KEY not in _alerts(tmp_path)


def test_bundle_bytes_are_identical_with_and_without_the_scan(tmp_path):
    now = time.time()
    source = tmp_path / "sources"
    _transcript(source / "a.jsonl", _fake_key("bytes"), mtime=now - 3600)
    (source / "b.jsonl").write_bytes(b"x" * (3 * 1024 * 1024) + b"\n" + _fake_key("tail").encode() + b"\n")
    candidates = backup._discover_jsonl_candidates([source])
    collector = scan.CredentialScan(now=now)

    plain, plain_digests, plain_sizes = backup.create_jsonl_bundle_with_digests(
        candidates, tmp_path / "plain", date_stamp="2026-11-18"
    )
    scanned, scanned_digests, scanned_sizes = backup.create_jsonl_bundle_with_digests(
        candidates, tmp_path / "scanned", date_stamp="2026-11-18", credential_scan=collector
    )

    assert gzip.decompress(plain.read_bytes()) == gzip.decompress(scanned.read_bytes())
    assert (plain_digests, plain_sizes) == (scanned_digests, scanned_sizes)
    assert sorted((f.path, f.line) for f in collector.findings) == [("a.jsonl", 3), ("b.jsonl", 2)]


def test_alert_clears_only_after_seven_days_without_a_new_key(tmp_path):
    now = time.time()
    source = tmp_path / "sources"
    _transcript(source / "a.jsonl", _fake_key("clear"), mtime=now - 3600)
    _run(tmp_path, source, now=now)

    _run(tmp_path, source, now=now + 6 * DAY)
    assert scan.ALERT_KEY in _alerts(tmp_path)

    result = _run(tmp_path, source, now=now + 7 * DAY + 60)
    assert result["credential_scan"]["new_alerts"] == 0
    assert scan.ALERT_KEY not in _alerts(tmp_path)


def test_a_new_key_while_an_alert_is_open_replaces_the_message(tmp_path):
    now = time.time()
    source = tmp_path / "sources"
    _transcript(source / "a.jsonl", _fake_key("first"), mtime=now - 3600)
    _run(tmp_path, source, now=now)

    _transcript(source / "b.jsonl", _fake_key("second", "github"), mtime=now - 60)
    result = _run(tmp_path, source, now=now + 600)

    assert result["credential_scan"]["new_alerts"] == 1
    assert _alerts(tmp_path)[scan.ALERT_KEY] == (
        "A key that looks current (github) is in a transcript: rotate it (b.jsonl)"
    )


def test_file_scan_finds_a_key_across_block_cuts_with_the_right_line():
    value = _fake_key("block")
    # Line 1 fills most of a block; line 2 holds the key across the reader's chunk edges.
    data = b"a" * (scan._BLOCK_BYTES - 10) + b"\n" + b'{"t":"' + value.encode() + b'"}\n'
    file_scan = scan.FileScan("x.jsonl")
    for start in range(0, len(data), 16 * 1024):
        file_scan.feed(data[start : start + 16 * 1024])
    assert file_scan.finish() == [("anthropic", 2, hashlib.sha256(value.encode()).hexdigest())]


def test_file_scan_handles_a_single_line_longer_than_a_block():
    value = _fake_key("longline")
    data = b"a " * scan._BLOCK_BYTES + value.encode() + b" " + b"b " * scan._BLOCK_BYTES
    file_scan = scan.FileScan("x.json")
    for start in range(0, len(data), 7919):
        file_scan.feed(data[start : start + 7919])
    assert file_scan.finish() == [("anthropic", 1, hashlib.sha256(value.encode()).hexdigest())]


def test_binary_sources_are_bundled_but_never_scanned(tmp_path):
    collector = scan.CredentialScan(now=time.time())
    assert collector.open_file(tmp_path / "conversation.db", tmp_path) is None
    assert collector.open_file(tmp_path / "s.jsonl", tmp_path) is not None


def _grown_file(tmp_path: Path, *, old_key: str, new_key: str) -> tuple[Path, bytes]:
    """A transcript whose first bundled snapshot ends mid-line, then grows."""
    path = tmp_path / "sources" / "grown.jsonl"
    path.parent.mkdir(parents=True)
    prefix = b'{"n":1}\n{"t":"' + old_key.encode() + b'"}\n{"partial":'
    path.write_bytes(prefix + b'"rest"}\n{"t":"' + new_key.encode() + b'"}\n')
    return path, prefix


def _bundle_with(tmp_path: Path, path: Path, priors: dict, *, scanning_since: float) -> scan.CredentialScan:
    collector = scan.CredentialScan(now=time.time(), priors=priors, scanning_since=scanning_since)
    candidates = backup._discover_jsonl_candidates([path.parent])
    backup.create_jsonl_bundle_with_digests(
        candidates, tmp_path / f"staging-{len(list(tmp_path.iterdir()))}", date_stamp="d", credential_scan=collector
    )
    return collector


def test_proven_prefix_is_skipped_up_to_its_last_line_boundary(tmp_path):
    old_key, new_key = _fake_key("in-prefix"), _fake_key("appended")
    path, prefix = _grown_file(tmp_path, old_key=old_key, new_key=new_key)
    prior = {"size": len(prefix), "sha256": hashlib.sha256(prefix).hexdigest(), "mtime": 2000.0}

    collector = _bundle_with(tmp_path, path, {path.as_posix(): prior}, scanning_since=1000.0)

    # Only the appended key: the prefix was scanned when it was bundled.
    assert [(f.fingerprint, f.line) for f in collector.findings] == [(scan.fingerprint(new_key), 4)]
    assert collector.proven_prefix_bytes == prefix.rfind(b"\n") + 1
    assert collector.scanned_bytes == path.stat().st_size - collector.proven_prefix_bytes


def test_rewritten_prefix_or_untrusted_entry_is_scanned_whole(tmp_path):
    old_key, new_key = _fake_key("in-prefix"), _fake_key("appended")
    path, prefix = _grown_file(tmp_path, old_key=old_key, new_key=new_key)
    expected = {(scan.fingerprint(old_key), 2), (scan.fingerprint(new_key), 4)}
    good = {"size": len(prefix), "sha256": hashlib.sha256(prefix).hexdigest(), "mtime": 2000.0}
    cases = {
        "rewritten": {**good, "sha256": hashlib.sha256(b"x" + prefix[1:]).hexdigest()},
        "written-before-scanning": {**good, "mtime": 999.0},
        "shrunk": {**good, "size": path.stat().st_size + 10},
    }
    for name, prior in cases.items():
        collector = _bundle_with(tmp_path, path, {path.as_posix(): prior}, scanning_since=1000.0)
        assert {(f.fingerprint, f.line) for f in collector.findings} == expected, name
        assert collector.proven_prefix_bytes == 0, name


def test_first_night_entries_vouch_on_night_two(tmp_path, monkeypatch):
    """End to end: the full first-night scan happens once (live check, Defect 2)."""
    from tests.test_jsonl_backup import _mock_drive_success

    _mock_drive_success(backup, monkeypatch)
    monkeypatch.setattr(backup.backup_daily, "resolve_machine_id", lambda: "fixture")
    monkeypatch.setattr(
        backup.backup_daily,
        "upload_file_to_drive_raw",
        lambda path, *a, **k: {"id": f"id-{Path(path).name}", "md5Checksum": backup._md5_file(path)},
    )
    t0 = time.time()
    source = tmp_path / "sources"
    path = source / "s.jsonl"
    path.parent.mkdir(parents=True)

    def grow(line: str, mtime: float) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        os.utime(path, (mtime, mtime))

    def run(now: float, stamp: str) -> dict:
        return backup.run_backup(
            source_roots=[source],
            state_path=tmp_path / "state" / "state.json",
            staging_dir=tmp_path / "staging",
            log_path=tmp_path / "state" / "backup.log",
            queue_dir=tmp_path / "state" / "queue",
            date_stamp=stamp,
            now=now,
            active_skip_seconds=60,
        )

    grow('{"n":1}', t0 - 3600)
    first = run(t0, "2026-11-01")["credential_scan"]
    grow('{"n":2}', t0 + 100)
    second = run(t0 + 700, "2026-11-02")["credential_scan"]
    grow('{"n":3}', t0 + 800)
    third = run(t0 + 1500, "2026-11-03")["credential_scan"]

    assert first["proven_prefix_bytes"] == 0
    # Night 1's entry records an mtime older than night 1 itself; it must still vouch.
    assert second["proven_prefix_bytes"] == len('{"n":1}\n')
    assert second["scanned_bytes"] == len('{"n":2}\n')
    assert third["proven_prefix_bytes"] == len('{"n":1}\n{"n":2}\n')
    assert third["scanned_bytes"] == len('{"n":3}\n')


def test_entries_written_before_scanning_existed_never_vouch(tmp_path):
    """The cutoff is the last state write before the first scanning run."""
    path, prefix = _grown_file(tmp_path, old_key=_fake_key("pre-scan"), new_key=_fake_key("later"))
    last_write = 5000.0
    entry = {"size": len(prefix), "sha256": hashlib.sha256(prefix).hexdigest()}
    backup_state = {
        "files": {path.as_posix(): {**entry, "mtime": last_write}},
        "updated_at": dt.datetime.fromtimestamp(last_write, dt.UTC).isoformat(),
    }
    state_path = tmp_path / "scan-state.json"

    collector = scan.start(backup_state, state_path=state_path, now=last_write + 900)
    assert collector.scanning_since == last_write
    assert collector._prior(path) is None  # mtime == last write: may predate scanning
    later = {**backup_state, "files": {path.as_posix(): {**entry, "mtime": last_write + 1}}}
    assert scan.start(later, state_path=state_path, now=last_write + 2000)._prior(path) is not None
    # With no prior state at all, nothing exists that could predate scanning.
    assert scan.start({"files": {}}, state_path=tmp_path / "fresh.json", now=9e9).scanning_since == 0.0


def test_a_provider_shape_inside_a_base64_run_is_not_a_key(tmp_path):
    """Live check Defect 1: 493 of 496 findings were google_oauth_refresh inside base64."""
    blob = base64.b64encode(b"".join(hashlib.sha256(bytes([i])).digest() for i in range(40))).decode()
    refresh_shape = "1//" + "0" + hashlib.sha256(b"refresh").hexdigest()[:30]
    embedded = blob[:300] + refresh_shape + "+" + blob[300:600]
    standalone = "1//0" + hashlib.sha256(b"real refresh").hexdigest()[:40]
    file_scan = scan.FileScan("x.jsonl")
    lines = [
        json.dumps({"type": "image", "data": embedded}),
        json.dumps({"refresh_token": standalone}),
        "token=" + standalone + " end",
    ]
    file_scan.feed(("\n".join(lines) + "\n").encode())

    assert file_scan.finish() == [("google_oauth_refresh", 2, scan.fingerprint(standalone))]
    assert file_scan.ignored_embedded == 1


def test_repeated_filler_values_are_placeholders_and_never_alert(tmp_path):
    """The audio-dashboard transcript's line 401: every value was "Fa1Ke" repeated."""
    filler = "Fa1Ke" * 8
    now = time.time()
    source = tmp_path / "sources"
    path = source / "fixture-output.jsonl"
    path.parent.mkdir(parents=True)
    values = ["sk_live_" + filler, "whsec_" + filler, "sk-proj-" + filler, "gh" + "p_" + filler, "AKIA" + "0" * 16]
    path.write_text(json.dumps({"type": "user", "text": " ".join(values)}) + "\n", encoding="utf-8")
    os.utime(path, (now - 3600, now - 3600))

    result = _run(tmp_path, source, now=now)

    assert _counts(result) == {"findings": 0, "current_findings": 0, "new_alerts": 0}
    assert result["credential_scan"]["ignored_placeholders"] == len(values)
    assert scan.ALERT_KEY not in _alerts(tmp_path)
    assert not any(scan.is_placeholder(_fake_key(seed)) for seed in ("a", "b", "c", "d"))
