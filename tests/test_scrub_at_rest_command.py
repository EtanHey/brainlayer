"""Copy-only maintenance proofs with synthetic credentials and the real schema."""

import datetime as dt
import hashlib
import json
import logging
import subprocess
import sys
from types import SimpleNamespace

import apsw
import pytest
from typer.testing import CliRunner

from brainlayer.cli import app
from brainlayer.vector_store import VectorStore

TOKENS = {
    "google_oauth_access": "ya29." + "0" * 24,
    "google_oauth_refresh": "1//0" + "0" * 24,
    "google_client_secret": "GOCSPX-" + "0" * 24,
}
CHUNK_COLUMNS = ["content", "summary", "key_facts", "preview_text", "context_summary", "resolved_query"]
SESSION_COLUMNS = [
    "file_path",
    "enrichment_version",
    "enrichment_model",
    "enrichment_timestamp",
    "session_start_time",
    "session_end_time",
    "session_summary",
    "primary_intent",
    "decisions_made",
    "corrections",
    "learnings",
    "mistakes",
    "patterns",
    "topic_tags",
    "tool_usage_stats",
    "what_worked",
    "what_failed",
]


@pytest.fixture
def db(tmp_path):
    store = VectorStore(tmp_path / "copy.db")
    yield store
    store.close()


def run(db, dry=False):
    from brainlayer.scrub_at_rest import scrub_at_rest

    return scrub_at_rest(db.db_path, dry_run=dry, batch_size=2)


@pytest.mark.parametrize("provider", TOKENS)
@pytest.mark.parametrize("column", CHUNK_COLUMNS + SESSION_COLUMNS)
def test_every_provider_column_fts_and_idempotency(db, provider, column):
    token = TOKENS[provider]
    table = "chunks" if column in CHUNK_COLUMNS else "session_enrichments"
    if table == "chunks":
        db.conn.execute("INSERT INTO chunks(id,content,metadata,source_file) VALUES('c','ordinary','{}','fixture')")
        db.conn.execute(f"UPDATE chunks SET {column}=? WHERE id='c'", (token,))
    else:
        db.conn.execute("INSERT INTO session_enrichments(session_id) VALUES('s')")
        db.conn.execute(f"UPDATE session_enrichments SET {column}=? WHERE session_id='s'", (token,))
        if column in {"session_summary", "what_worked", "what_failed"}:
            db.conn.execute(f"INSERT INTO session_enrichments_fts({column},session_id) VALUES(?,'s')", (token,))
    report = run(db)
    assert report["tables"][table]["columns"][column][provider] == 1
    assert db.conn.execute(f"SELECT {column} FROM {table}").fetchone()[0] == f"[REDACTED:{provider}]"
    for fts in ["chunks_fts", "chunks_fts_operational", "chunks_fts_trigram", "session_enrichments_fts"]:
        assert (
            db.conn.execute(f"SELECT count(*) FROM {fts} WHERE {fts} MATCH ?", ('"' + token + '"',)).fetchone()[0] == 0
        )
    assert all(t["rows"] == 0 for t in run(db)["tables"].values())


def test_provider_scope_and_hashes_preserve_enriched_summary(db):
    from brainlayer.chunk_write import canonical_content_hash
    from brainlayer.dedupe import compute_dedupe_fields
    from brainlayer.pipeline.secret_scrub import scrub_secrets

    token = TOKENS["google_oauth_refresh"]
    other = "ghp_" + "0" * 36
    entropy = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = f"note {token} {other} token={entropy} ordinary ya29 prose 1// URL fragment"
    selected = scrub_secrets(text, providers=frozenset(TOKENS))
    assert other in selected.text and entropy in selected.text and not selected.quarantine
    db.conn.execute(
        "INSERT INTO chunks(id,content,metadata,source_file,summary,created_at) VALUES('c',?,'{}','fixture','enriched summary','2026-03-01')",
        (text,),
    )
    run(db)
    row = db.conn.execute(
        "SELECT content,content_hash,dedupe_hash,simhash,simhash_band_0,simhash_band_1,simhash_band_2,simhash_band_3,char_count,summary FROM chunks"
    ).fetchone()
    expected = compute_dedupe_fields(selected.text, "2026-03-01")
    assert row == (
        selected.text,
        canonical_content_hash(selected.text),
        expected.dedupe_hash,
        expected.simhash,
        *expected.bands,
        len(selected.text),
        "enriched summary",
    )


def test_dry_run_cli_writes_nothing_and_emits_counts_only(db):
    token = TOKENS["google_client_secret"]
    db.conn.execute("INSERT INTO chunks(id,content,metadata,source_file) VALUES('c',?,'{}','fixture')", (token,))
    db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = hashlib.sha256(db.db_path.read_bytes()).digest()
    result = CliRunner().invoke(
        app, ["scrub-at-rest", "--providers", "google_oauth", "--dry-run", "--db", str(db.db_path)]
    )
    assert result.exit_code == 0, result.output
    assert token not in result.output
    assert json.loads(result.output)["tables"]["chunks"]["rows"] == 1
    assert hashlib.sha256(db.db_path.read_bytes()).digest() == before
    assert db.conn.execute("SELECT content FROM chunks").fetchone()[0] == token


def test_history_preimages_orphan_fts_and_near_misses(db):
    from brainlayer.bitemporal import apply_bitemporal_migration

    token = TOKENS["google_oauth_access"]
    apply_bitemporal_migration(db.conn)
    db.conn.execute("INSERT INTO chunks(id,content,metadata,source_file) VALUES('c',?,'{}','fixture')", (token,))
    db.conn.execute("UPDATE chunks SET summary='old summary' WHERE id='c'")
    db.conn.execute("CREATE TABLE chunks_fts_preimage(payload TEXT)")
    db.conn.execute("INSERT INTO chunks_fts_preimage VALUES(?)", (json.dumps({"nested": token}),))
    db.conn.execute("INSERT INTO chunks_fts(content,chunk_id) VALUES(?,'orphan')", (token,))
    db.conn.execute("INSERT INTO chunks_fts_preimage VALUES('ya29 prose; GOCSPX-short; 1// URL fragment')")
    history_before = db.conn.execute("SELECT count(*) FROM _chunks_history").fetchone()[0]
    run(db)
    assert db.conn.execute("SELECT count(*) FROM _chunks_history").fetchone()[0] == history_before
    assert token not in db.conn.execute("SELECT content FROM _chunks_history").fetchone()[0]
    assert token not in db.conn.execute("SELECT payload FROM chunks_fts_preimage ORDER BY rowid").fetchone()[0]
    assert (
        db.conn.execute("SELECT payload FROM chunks_fts_preimage ORDER BY rowid DESC").fetchone()[0]
        == "ya29 prose; GOCSPX-short; 1// URL fragment"
    )
    assert all(t["rows"] == 0 for t in run(db)["tables"].values())


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_refuses_live_alias_before_open(db, monkeypatch, tmp_path, alias):
    from brainlayer import chunk_origin_wipe, scrub_at_rest

    monkeypatch.setattr(chunk_origin_wipe, "_live_db_candidates", lambda: [db.db_path])
    target = db.db_path
    if alias != "direct":
        target = tmp_path / "alias.db"
        if alias == "symlink":
            target.symlink_to(db.db_path)
        else:
            target.hardlink_to(db.db_path)
    monkeypatch.setattr(scrub_at_rest, "WriterRuntimeStore", lambda *args: pytest.fail("opened protected DB"))
    with pytest.raises(scrub_at_rest.ScrubAtRestError):
        scrub_at_rest.scrub_at_rest(target)


def test_failure_is_value_free_and_rolls_back_batch(db):
    from brainlayer.scrub_at_rest import ScrubAtRestError

    token = TOKENS["google_oauth_refresh"]
    db.conn.execute("INSERT INTO chunks(id,content,metadata,source_file) VALUES('c',?,'{}','fixture')", (token,))
    db.conn.execute(
        "CREATE TRIGGER refusal BEFORE UPDATE ON chunks BEGIN SELECT RAISE(ABORT,'synthetic private token'); END"
    )
    with pytest.raises(ScrubAtRestError) as error:
        run(db)
    assert "synthetic private token" not in str(error.value)
    assert db.conn.execute("SELECT content FROM chunks").fetchone()[0] == token


def test_fresh_cli_sqlite_trigger_cannot_log_credentials(db, tmp_path):
    token = TOKENS["google_oauth_refresh"]
    db.conn.execute("INSERT INTO chunks(id,content,metadata,source_file) VALUES('c',?,'{}','fixture')", (token,))
    db.conn.execute("CREATE TRIGGER refusal BEFORE UPDATE ON chunks BEGIN SELECT RAISE(ABORT,old.content); END")
    path = db.db_path
    db.close()
    logfile = tmp_path / "command.log"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging,sys; logging.getLogger().addHandler(logging.FileHandler(sys.argv.pop(1))); logging.getLogger().addHandler(logging.StreamHandler()); from brainlayer.cli import app; app()",
            str(logfile),
            "scrub-at-rest",
            "--db",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 1
    assert token not in child.stdout, "credential in stdout"
    assert token not in child.stderr, "credential in stderr"
    assert token not in logfile.read_text(), "credential in command log"
    assert json.loads(child.stdout)["error_type"] == "ScrubAtRestError"


@pytest.mark.parametrize("phase", ["open", "survey", "close"])
def test_private_sqlite_logging_covers_whole_lifecycle_and_restores(db, monkeypatch, caplog, phase):
    from brainlayer import scrub_at_rest as module
    from brainlayer import vector_store

    token = TOKENS["google_client_secret"]
    # APSW's normal logger may lower its level after schema/notice diagnostics.
    caplog.set_level(logging.DEBUG)
    previous = vector_store._SQLITE_LOG_HANDLER
    real_store = module.ReadonlyStore

    class Store:
        def __enter__(self):
            self.store = real_store(db.db_path)
            if phase == "open":
                apsw.log(apsw.SQLITE_ERROR, token)
            return self.store

        def __exit__(self, *args):
            self.store.close()
            if phase == "close":
                apsw.log(apsw.SQLITE_ERROR, token)

    monkeypatch.setattr(module, "ReadonlyStore", lambda path: Store())
    if phase == "survey":

        def fail_survey(*args):
            apsw.log(apsw.SQLITE_ERROR, token)
            raise ValueError(token)

        monkeypatch.setattr(module, "_run", fail_survey)
        with pytest.raises(module.ScrubAtRestError):
            module.scrub_at_rest(db.db_path, dry_run=True)
    else:
        module.scrub_at_rest(db.db_path, dry_run=True)
    assert vector_store._SQLITE_LOG_HANDLER is previous
    assert all(token not in repr(vars(record)) for record in caplog.records)
    apsw.log(apsw.SQLITE_ERROR, "normal logging restored")
    assert caplog.records[-1].sqlite_message == "normal logging restored"


def test_nested_private_sqlite_logging_restores_prior_callback():
    from brainlayer import vector_store

    previous = vector_store._SQLITE_LOG_HANDLER
    with vector_store.value_free_sqlite_logging():
        with vector_store.value_free_sqlite_logging():
            assert vector_store._SQLITE_LOG_HANDLER is vector_store._value_free_sqlite_log
        assert vector_store._SQLITE_LOG_HANDLER is vector_store._value_free_sqlite_log
    assert vector_store._SQLITE_LOG_HANDLER is previous


@pytest.fixture
def live_guard(db, tmp_path, monkeypatch):
    from brainlayer import chunk_origin_wipe, maintenance, scrub_at_rest

    now = dt.datetime(2026, 10, 1, 4, 30, tzinfo=dt.UTC)
    pause = tmp_path / "pause.sentinel"
    pause.write_text(json.dumps({"labels": ["com.brainlayer.enrichment"]}))
    backup = tmp_path / "backup.jsonl"
    receipt = {
        "db": str(db.db_path),
        "attempted_at": now.isoformat(),
        "uploaded": True,
        "verified": True,
        "drive_file": {"id": "fixture"},
    }
    backup.write_text(json.dumps(receipt) + "\n")
    config_type = maintenance.MaintenanceConfig
    monkeypatch.setattr(
        maintenance,
        "MaintenanceConfig",
        lambda **kwargs: config_type(**kwargs, now_fn=lambda: now, backup_log_path=backup),
    )
    monkeypatch.setattr(maintenance, "PAUSE_SENTINEL_PATH", pause)
    monkeypatch.setattr(chunk_origin_wipe, "_live_db_candidates", lambda: [db.db_path])
    events, loaded = [], {}
    monkeypatch.setattr(maintenance, "_run_gates", lambda config: events.append("gates"))
    monkeypatch.setattr(maintenance, "_check_lsof_clean", lambda config: events.append("no-writers"))
    monkeypatch.setattr(scrub_at_rest, "_check_no_brainbar_processes", lambda: events.append("no-brainbar"))
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda service: loaded.get(service, True))

    def bootout(service):
        events.append(("stop", service))
        loaded[service] = False
        return True

    def resume(root, service):
        events.append(("resume", service))
        loaded[service] = True

    monkeypatch.setattr(maintenance, "_bootout_service", bootout)
    monkeypatch.setattr(maintenance, "_resume_service", resume)
    db.conn.execute(
        "INSERT INTO chunks(id,content,metadata,source_file) VALUES('c',?,'{}','fixture')",
        (TOKENS["google_oauth_access"],),
    )
    from brainlayer.scrub_at_rest import _run

    total = sum(table["rows"] for table in _run(db, True, 100)["tables"].values())
    return SimpleNamespace(
        events=events, loaded=loaded, pause=pause, backup=backup, receipt=receipt, now=now, total=total
    )


def test_live_dry_run_needs_no_flag_or_service_actions(db, live_guard):
    result = CliRunner().invoke(app, ["scrub-at-rest", "--db", str(db.db_path), "--dry-run"])
    assert result.exit_code == 0
    assert json.loads(result.output)["tables"]["chunks"]["rows"] == 1
    assert live_guard.events == []


@pytest.mark.parametrize("backup_state", ["missing", "stale", "unverified", "different-db"])
def test_live_apply_refuses_unqualified_backup_before_services_or_writes(db, live_guard, backup_state):
    from brainlayer.scrub_at_rest import ScrubAtRestError, scrub_at_rest

    receipt = live_guard.receipt.copy()
    if backup_state == "missing":
        live_guard.backup.unlink()
    else:
        if backup_state == "stale":
            receipt["attempted_at"] = (live_guard.now - dt.timedelta(hours=24, seconds=1)).isoformat()
        elif backup_state == "unverified":
            receipt["verified"] = False
        else:
            receipt["db"] = "other.db"
        live_guard.backup.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(ScrubAtRestError):
        scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert not any(isinstance(e, tuple) for e in live_guard.events)
    assert db.conn.execute("SELECT content FROM chunks").fetchone()[0] == TOKENS["google_oauth_access"]


@pytest.mark.parametrize("expect", [None, 0, 999])
def test_live_apply_requires_current_row_total(db, live_guard, expect):
    from brainlayer.scrub_at_rest import ScrubAtRestError, scrub_at_rest

    with pytest.raises(ScrubAtRestError):
        scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=expect)
    assert db.conn.execute("SELECT content FROM chunks").fetchone()[0] == TOKENS["google_oauth_access"]
    assert not any(e[0] == "stop" for e in live_guard.events if isinstance(e, tuple)) or any(
        e[0] == "resume" for e in live_guard.events if isinstance(e, tuple)
    )


def test_live_apply_quiesces_all_writers_and_preserves_enrichment_pause(db, live_guard):
    from brainlayer.scrub_at_rest import scrub_at_rest

    result = scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert result["tables"]["chunks"]["rows"] == 1
    stopped = {e[1] for e in live_guard.events if isinstance(e, tuple) and e[0] == "stop"}
    resumed = {e[1] for e in live_guard.events if isinstance(e, tuple) and e[0] == "resume"}
    assert {
        "fleet-watchdog",
        "throughput-watchdog",
        "decay",
        "tier0-watchdog",
        "health-check",
        "brainbar",
        "brainbar-daemon",
        "hotlane-brainbar",
        "watch",
        "drain",
    } <= stopped
    assert resumed == stopped - {"enrichment"}
    assert "no-writers" in live_guard.events


@pytest.mark.parametrize("fail_apply", [False, True])
def test_guarded_restoration_preserves_real_fleet_pause(db, live_guard, monkeypatch, fail_apply):
    from brainlayer import scrub_at_rest as scrub_module

    live_guard.pause.write_text(
        json.dumps({"labels": ["com.brainlayer.enrichment", "com.etanhey.brainlayer-fleet-watchdog"]})
    )
    if fail_apply:

        def fail(*args, **kwargs):
            raise scrub_module.ScrubAtRestError("fixture failure")

        monkeypatch.setattr(scrub_module, "_apply", fail)
        with pytest.raises(scrub_module.ScrubAtRestError):
            scrub_module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    else:
        scrub_module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)

    stopped = {e[1] for e in live_guard.events if isinstance(e, tuple) and e[0] == "stop"}
    resumed = {e[1] for e in live_guard.events if isinstance(e, tuple) and e[0] == "resume"}
    assert stopped == set(scrub_module.LIVE_SERVICES)
    assert resumed == stopped - {"enrichment", "fleet-watchdog"}
    assert live_guard.loaded["fleet-watchdog"] is False


def test_live_failure_resumes_services_and_quiesce_failure_never_writes(db, live_guard, monkeypatch):
    from brainlayer import maintenance
    from brainlayer.scrub_at_rest import ScrubAtRestError, scrub_at_rest

    def fail_stop(service):
        live_guard.events.append(("stop", service))
        if service == "drain":
            return False
        live_guard.loaded[service] = False
        return True

    monkeypatch.setattr(maintenance, "_bootout_service", fail_stop)
    with pytest.raises(ScrubAtRestError):
        scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert db.conn.execute("SELECT content FROM chunks").fetchone()[0] == TOKENS["google_oauth_access"]
    assert any(e == ("resume", "brainbar-daemon") for e in live_guard.events)


def test_live_apply_error_still_resumes(db, live_guard):
    from brainlayer.scrub_at_rest import ScrubAtRestError, scrub_at_rest

    db.conn.execute("CREATE TRIGGER refusal BEFORE UPDATE ON chunks BEGIN SELECT RAISE(ABORT,old.content); END")
    with pytest.raises(ScrubAtRestError):
        scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert ("resume", "brainbar-daemon") in live_guard.events
    assert ("resume", "watch") in live_guard.events


def test_count_changes_during_quiesce_abort_before_writer_open(db, live_guard, monkeypatch):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    bootout = maintenance._bootout_service

    def changed(service):
        if service == "watch":
            db.conn.execute(
                "INSERT INTO chunks(id,content,metadata,source_file) VALUES('new',?,'{}','fixture')",
                (TOKENS["google_client_secret"],),
            )
        return bootout(service)

    monkeypatch.setattr(maintenance, "_bootout_service", changed)
    monkeypatch.setattr(module, "WriterRuntimeStore", lambda path: pytest.fail("writer opened before count gate"))
    with pytest.raises(module.ScrubAtRestError) as error:
        module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert error.value.reason == "expected-row-count-mismatch"
    assert ("resume", "watch") in live_guard.events


def test_writer_revived_during_survey_blocks_apply(db, live_guard, monkeypatch):
    from brainlayer import scrub_at_rest as module

    run = module._run

    def surveyed(*args):
        result = run(*args)
        live_guard.loaded["watch"] = True
        return result

    monkeypatch.setattr(module, "_run", surveyed)
    monkeypatch.setattr(module, "WriterRuntimeStore", lambda path: pytest.fail("writer opened after failed quiescence"))
    with pytest.raises(module.ScrubAtRestError) as error:
        module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert error.value.reason == "quiesce-failed"
    assert ("resume", "brainbar-daemon") in live_guard.events


def test_live_cli_accepts_verified_backup_older_than_six_hours(db, live_guard):
    receipt = live_guard.receipt.copy()
    receipt["attempted_at"] = (live_guard.now - dt.timedelta(hours=23, minutes=59)).isoformat()
    live_guard.backup.write_text(json.dumps(receipt) + "\n")
    result = CliRunner().invoke(
        app, ["scrub-at-rest", "--db", str(db.db_path), "--allow-live-db", "--expect-rows", str(live_guard.total)]
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["tables"]["chunks"]["rows"] == 1
    assert all(token not in result.output for token in TOKENS.values())


def test_resume_failure_is_value_free_and_other_services_still_resume(db, live_guard, monkeypatch):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    resume = maintenance._resume_service

    def fail_one(root, service):
        if service == "drain":
            raise OSError(TOKENS["google_client_secret"])
        resume(root, service)

    monkeypatch.setattr(maintenance, "_resume_service", fail_one)
    with pytest.raises(module.ScrubAtRestError) as error:
        module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert error.value.reason == "resume-failed"
    assert TOKENS["google_client_secret"] not in str(error.value)
    assert ("resume", "brainbar-daemon") in live_guard.events


@pytest.mark.parametrize(
    "service,label",
    [
        ("brainbar", "com.brainlayer.brainbar"),
        ("brainbar-daemon", "com.brainlayer.brainbar-daemon"),
        ("fleet-watchdog", "com.etanhey.brainlayer-fleet-watchdog"),
    ],
)
def test_resident_service_helpers_restore_existing_launchagent(tmp_path, monkeypatch, service, label):
    from pathlib import Path

    from brainlayer import maintenance

    calls = []
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        maintenance, "run_command", lambda args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(returncode=0)
    )
    assert maintenance._bootout_service(service)
    maintenance._resume_service(tmp_path, service)
    assert calls[0][0][-1].endswith("/" + label)
    assert calls[1][0][:2] == ["launchctl", "bootstrap"]
    assert calls[1][0][-1] == str(tmp_path / "Library" / "LaunchAgents" / (label + ".plist"))


def test_unregistered_brainbar_process_blocks_apply(monkeypatch):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    monkeypatch.setattr(
        maintenance,
        "run_command",
        lambda args, **kwargs: SimpleNamespace(stdout="/Applications/BrainBar.app/Contents/MacOS/BrainBar\n"),
    )
    with pytest.raises(module.ScrubAtRestError, match="remains running"):
        module._check_no_brainbar_processes()


@pytest.mark.parametrize("pause_state", ["missing", "expired", "wrong-label"])
def test_live_apply_requires_active_enrichment_pause(db, live_guard, pause_state):
    from brainlayer.scrub_at_rest import ScrubAtRestError, scrub_at_rest

    if pause_state == "missing":
        live_guard.pause.unlink()
    else:
        payload = {"labels": ["com.brainlayer.enrichment"]}
        if pause_state == "expired":
            payload["expires_at"] = (live_guard.now - dt.timedelta(seconds=1)).isoformat()
        else:
            payload["labels"] = ["com.brainlayer.watch"]
        live_guard.pause.write_text(json.dumps(payload))
    with pytest.raises(ScrubAtRestError):
        scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert not any(isinstance(e, tuple) for e in live_guard.events)


def test_checkpoint_failure_preserves_value_free_apply_cause(db, monkeypatch):
    from brainlayer import scrub_at_rest as module

    calls = []

    def checkpoint(store):
        calls.append(None)
        if len(calls) == 2:
            raise OSError("private checkpoint diagnostic")

    def fail_run(*args):
        raise apsw.ConstraintError(TOKENS["google_client_secret"])

    monkeypatch.setattr(module, "_checkpoint", checkpoint)
    monkeypatch.setattr(module, "_run", fail_run)
    with pytest.raises(module.ScrubAtRestError) as error:
        module.scrub_at_rest(db.db_path)
    assert isinstance(error.value.__cause__, apsw.ConstraintError)
    assert TOKENS["google_client_secret"] not in str(error.value.__cause__)
    assert "OSError" in " ".join(getattr(error.value, "__notes__", []))


def test_rollback_failure_preserves_apply_error_and_never_retries(db, monkeypatch):
    from brainlayer import scrub_at_rest as module

    db.conn.execute("CREATE TABLE diagnostic(payload TEXT)")

    # An independent stub exercises failure of the rollback operation itself.
    class Connection:
        def getautocommit(self):
            return False

        def execute(self, sql, *args):
            if sql == "ROLLBACK":
                raise OSError("private rollback diagnostic")
            return []

    monkeypatch.setattr(module, "_tables", lambda conn: [("diagnostic", ["payload"], "table", ["_rowid_"])])
    attempts = []

    def fail_batch(*args):
        attempts.append(None)
        raise apsw.BusyError(TOKENS["google_oauth_access"])

    monkeypatch.setattr(module, "_batch", fail_batch)
    with pytest.raises(module.ScrubAtRestError) as error:
        module._run(SimpleNamespace(conn=Connection()), False, 2)
    assert isinstance(error.value.__cause__, apsw.BusyError)
    assert TOKENS["google_oauth_access"] not in str(error.value.__cause__)
    assert len(attempts) == 1


def test_composite_without_rowid_keys_are_supported(db):
    db.conn.execute("CREATE TABLE saved_labels(owner TEXT,tag TEXT,PRIMARY KEY(owner,tag)) WITHOUT ROWID")
    for token in TOKENS.values():
        db.conn.execute("INSERT INTO saved_labels VALUES('fixture',?)", (token,))
    assert run(db)["tables"]["saved_labels"]["rows"] == 3
    assert db.conn.execute("SELECT count(*) FROM saved_labels").fetchone()[0] == 3
    assert run(db)["tables"]["saved_labels"]["rows"] == 0


def test_busy_retry_batches_and_checkpoints_use_writer_connection(db, monkeypatch):
    from brainlayer import scrub_at_rest as module

    token = TOKENS["google_oauth_refresh"]
    for i in range(5):
        db.conn.execute(
            "INSERT INTO chunks(id,content,metadata,source_file) VALUES(?,?,'{}','fixture')", (str(i), token)
        )
    original_batch, original_checkpoint = module._batch, module._checkpoint
    attempts, checkpoints, delays = [], [], []

    def batch(*args):
        attempts.append(True)
        if len(attempts) == 1:
            raise apsw.BusyError("synthetic contention")
        return original_batch(*args)

    def checkpoint(store):
        checkpoints.append(store.conn.execute("SELECT count(*) FROM chunks WHERE instr(content,'1//')>0").fetchone()[0])
        original_checkpoint(store)

    monkeypatch.setattr(module, "_batch", batch)
    monkeypatch.setattr(module, "_checkpoint", checkpoint)
    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=delays.append))
    assert run(db)["tables"]["chunks"]["rows"] == 5
    assert delays == [module._busy_retry_delay(0)] and checkpoints == [5, 0]


def test_held_maintenance_lock_refuses_without_changes(db, monkeypatch):
    import fcntl

    from brainlayer import maintenance
    from brainlayer.scrub_at_rest import ScrubAtRestError

    monkeypatch.setattr(maintenance, "MAINTENANCE_LOCK_TIMEOUT_SECONDS", 0)
    with (db.db_path.parent / ".maintenance.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ScrubAtRestError):
            run(db)
    assert db.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_manifest_text_in_numeric_affinity_is_scrubbed(db):
    db.conn.execute("CREATE TABLE typed_notes(value INTEGER)")
    db.conn.execute("INSERT INTO typed_notes VALUES(?)", (TOKENS["google_client_secret"],))
    assert run(db)["tables"]["typed_notes"]["rows"] == 1
    assert db.conn.execute("SELECT value FROM typed_notes").fetchone()[0] == "[REDACTED:google_client_secret]"


def test_fts_schema_whitespace_and_identity_dry_counts(db):
    from brainlayer.scrub_at_rest import ScrubAtRestError

    token = TOKENS["google_client_secret"]
    db.conn.execute("CREATE VIRTUAL TABLE extra_search USING\nfts5(text)")
    db.conn.execute("INSERT INTO extra_search VALUES(?)", (token,))
    assert run(db)["tables"]["extra_search"]["rows"] == 1
    db.conn.execute("CREATE TABLE references_only(owner_id TEXT)")
    db.conn.execute("INSERT INTO references_only VALUES(?)", (token,))
    assert run(db, dry=True)["tables"]["references_only"]["rows"] == 1
    with pytest.raises(ScrubAtRestError):
        run(db)


def test_preview_regeneration_trigger_does_not_replace_independent_preview(db):
    token = TOKENS["google_client_secret"]
    db.conn.execute(
        "INSERT INTO chunks(id,content,metadata,source_file,summary,preview_text) VALUES('c',?,'{}','fixture',?,?)",
        (token, "enriched " + token, "independent preview " + token),
    )
    sql = """CREATE TRIGGER chunks_preview_text_update AFTER UPDATE OF content, summary ON chunks
        BEGIN UPDATE chunks SET preview_text=coalesce(new.summary,new.content) WHERE rowid=new.rowid; END"""
    db.conn.execute(sql)
    original_trigger = db.conn.execute(
        "SELECT sql FROM sqlite_schema WHERE name='chunks_preview_text_update'"
    ).fetchone()[0]
    run(db)
    assert db.conn.execute("SELECT preview_text FROM chunks").fetchone()[0] == (
        "independent preview [REDACTED:google_client_secret]"
    )
    assert db.conn.execute("SELECT sql FROM sqlite_schema WHERE name='chunks_preview_text_update'").fetchone()[0] == (
        original_trigger
    )
    db.conn.execute("UPDATE chunks SET summary='later summary'")
    assert db.conn.execute("SELECT preview_text FROM chunks").fetchone()[0] == "later summary"


@pytest.mark.parametrize("provider", TOKENS)
def test_provider_span_longer_than_scan_window_is_removed_completely(db, provider):
    from brainlayer.pipeline.secret_scrub import MAX_SCAN_BYTES

    token = TOKENS[provider] + "0" * (MAX_SCAN_BYTES + 600)
    context = "ordinary " * (MAX_SCAN_BYTES // 9)
    db.conn.execute(
        "INSERT INTO chunks(id,content,metadata,source_file) VALUES('c',?,'{}','fixture')",
        (context + token + " suffix prose",),
    )
    run(db)
    assert db.conn.execute("SELECT content FROM chunks").fetchone()[0] == (
        context + f"[REDACTED:{provider}] suffix prose"
    )
    assert run(db)["tables"]["chunks"]["rows"] == 0


def test_schema_identifiers_and_content_cannot_inject_sql(db):
    db.conn.execute("INSERT INTO chunks(id,content,metadata,source_file) VALUES('c','ordinary','{}','fixture')")
    table = 'saved"; DROP TABLE chunks; --'
    ddl_name = '"saved""; DROP TABLE chunks; --"'
    column = '"payload""; DROP TABLE chunks; --"'
    db.conn.execute(f"CREATE TABLE {ddl_name} ({column} TEXT)")
    db.conn.execute(f"INSERT INTO {ddl_name} VALUES(?)", ("'; DROP TABLE chunks; -- " + TOKENS["google_oauth_access"],))
    assert run(db)["tables"][table]["rows"] == 1
    assert db.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    assert db.conn.execute(f"SELECT {column} FROM {ddl_name}").fetchone()[0] == (
        "'; DROP TABLE chunks; -- [REDACTED:google_oauth_access]"
    )
