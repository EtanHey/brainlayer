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

from brainlayer import maintenance, scrub_at_rest
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


def _fixture_next(values):
    try:
        return next(values)
    except StopIteration:
        pytest.fail("quiesce fixture exceeded its sample budget")


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
    monkeypatch.setattr(maintenance, "run_command", lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    loaded_probe = maintenance._service_is_loaded
    events, loaded = [], {}
    monkeypatch.setattr(maintenance, "_run_gates", lambda config: events.append("gates"))
    monkeypatch.setattr(maintenance, "_check_lsof_clean", lambda config: events.append("no-writers"))
    monkeypatch.setattr(scrub_at_rest, "_check_no_brainbar_processes", lambda: events.append("no-brainbar"))
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda service, **kwargs: loaded.get(service, True))

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
        events=events,
        loaded=loaded,
        loaded_probe=loaded_probe,
        pause=pause,
        backup=backup,
        receipt=receipt,
        now=now,
        total=total,
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
@pytest.mark.parametrize("operator_disabled", [False, True])
def test_guarded_restoration_preserves_real_fleet_pause(db, live_guard, monkeypatch, fail_apply, operator_disabled):
    from brainlayer import scrub_at_rest as scrub_module

    labels = ["com.brainlayer.enrichment"] + ([] if operator_disabled else ["com.etanhey.brainlayer-fleet-watchdog"])
    live_guard.pause.write_text(json.dumps({"labels": labels}))
    if operator_disabled:
        monkeypatch.setattr(maintenance, "is_launchd_label_disabled", lambda *args, **kwargs: True)
        monkeypatch.setattr(maintenance, "run_command", lambda *a, **k: pytest.fail("operator-disable modified"))
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


@pytest.mark.parametrize(
    "mode,shape",
    [
        ("context7", "{}"),
        ("exa_labeled", "EXA_API_KEY={}"),
        ("exa_labeled", '"EXA_API_KEY": "{}"'),
        ("exa_labeled", "exa_api_key: {}"),
    ],
)
def test_new_key_modes_survey_apply_hashes_and_fts(db, mode, shape):
    from brainlayer.chunk_write import canonical_content_hash
    from brainlayer.dedupe import compute_dedupe_fields
    from brainlayer.scrub_at_rest import scrub_at_rest

    uuid = "00000000-0000-0000-0000-000000000000"
    token = "ctx7sk-" + "0" * 24 if mode == "context7" else shape.format(uuid)
    other = "ya29." + "0" * 24
    text = f"ordinary {token} bare {uuid} other {other}"
    db.conn.execute(
        "INSERT INTO chunks(id,content,summary,key_facts,metadata,source_file,created_at) "
        "VALUES('c',?,?,?,'{}','fixture','2026-03-01')",
        (text, token, token),
    )
    # Include standalone derived copies even when a classification trigger omits them.
    for fts in ["chunks_fts_operational", "chunks_fts_trigram"]:
        if not db.conn.execute(f"SELECT count(*) FROM {fts}").fetchone()[0]:
            db.conn.execute(f"INSERT INTO {fts}(content,chunk_id) VALUES(?,'c')", (text,))
    result = CliRunner().invoke(app, ["scrub-at-rest", "--providers", mode, "--dry-run", "--db", str(db.db_path)])
    assert result.exit_code == 0, result.output
    survey = json.loads(result.output)
    assert survey["tables"]["chunks"]["rows"] == 1
    assert survey["tables"]["chunks"]["columns"]["content"] == {mode: 1}
    assert survey["tables"]["chunks_fts"]["rows"] == 1
    assert db.conn.execute("SELECT content FROM chunks").fetchone()[0] == text
    assert token not in result.output
    scrub_at_rest(db.db_path, providers=mode, batch_size=1)
    clean = text.replace("ctx7sk-" + "0" * 24 if mode == "context7" else uuid, f"[REDACTED:{mode}]", 1)
    row = db.conn.execute("SELECT content,content_hash,dedupe_hash,simhash,char_count FROM chunks").fetchone()
    fields = compute_dedupe_fields(clean, "2026-03-01")
    assert row == (clean, canonical_content_hash(clean), fields.dedupe_hash, fields.simhash, len(clean))
    for fts in ["chunks_fts", "chunks_fts_operational", "chunks_fts_trigram"]:
        assert db.conn.execute(f"SELECT content FROM {fts}").fetchone()[0] == clean
    assert all(t["rows"] == 0 for t in scrub_at_rest(db.db_path, providers=mode, dry_run=True)["tables"].values())
    assert uuid in clean and other in clean


@pytest.mark.parametrize("mode", ["context7", "exa_labeled"])
def test_new_modes_use_guarded_survey_and_apply(db, live_guard, mode):
    from brainlayer.scrub_at_rest import PROVIDER_MODES, _run, scrub_at_rest

    token = "ctx7sk-" + "0" * 24 if mode == "context7" else "EXA_API_KEY=" + "00000000-0000-0000-0000-000000000000"
    db.conn.execute("UPDATE chunks SET content=? WHERE id='c'", (token,))
    total = sum(t["rows"] for t in _run(db, True, 100, PROVIDER_MODES[mode])["tables"].values())
    result = scrub_at_rest(db.db_path, providers=mode, allow_live_db=True, expect_rows=total)
    assert result["tables"]["chunks"]["rows"] == 1
    assert token not in db.conn.execute("SELECT content FROM chunks WHERE id='c'").fetchone()[0]
    assert "no-writers" in live_guard.events
    stopped = {e[1] for e in live_guard.events if isinstance(e, tuple) and e[0] == "stop"}
    resumed = {e[1] for e in live_guard.events if isinstance(e, tuple) and e[0] == "resume"}
    assert stopped and resumed == stopped - {"enrichment"}


def test_exa_cli_survey_and_apply_preserve_example_join_key(db):
    text = "example_key=00000000-0000-0000-0000-000000000000"
    db.conn.execute(
        "INSERT INTO chunks(id,content,summary,metadata,source_file) VALUES('join-key',?,?,'{}','fixture')",
        (text, text),
    )
    before = db.conn.execute("SELECT * FROM chunks WHERE id='join-key'").fetchone()
    fts_before = db.conn.execute("SELECT * FROM chunks_fts WHERE chunk_id='join-key'").fetchone()
    runner = CliRunner()
    for flags in (["--dry-run"], []):
        result = runner.invoke(app, ["scrub-at-rest", "--db", str(db.db_path), "--providers", "exa_labeled", *flags])
        assert result.exit_code == 0, result.output
        survey = json.loads(result.output)
        assert all(table["rows"] == 0 for table in survey["tables"].values())
        assert text not in result.output
        assert db.conn.execute("SELECT * FROM chunks WHERE id='join-key'").fetchone() == before
        assert db.conn.execute("SELECT * FROM chunks_fts WHERE chunk_id='join-key'").fetchone() == fts_before


@pytest.mark.parametrize("mode", ["google_oauth", "context7", "exa_labeled"])
@pytest.mark.parametrize("failure", ["quiesce", "apply"])
def test_cli_refusal_and_failure_exit_nonzero_with_value_free_detail(db, live_guard, monkeypatch, mode, failure):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    if failure == "quiesce":

        def fail_stop(service):
            if service == "drain":
                raise OSError("private-diagnostic")
            live_guard.loaded[service] = False
            return True

        monkeypatch.setattr(maintenance, "_bootout_service", fail_stop)
        expected_detail = "bootout:com.brainlayer.drain"
    else:

        def fail_apply(*args):
            raise OSError("private-diagnostic")

        monkeypatch.setattr(module, "_apply", fail_apply)
        expected_detail = None
    total = sum(t["rows"] for t in module._run(db, True, 100, module.PROVIDER_MODES[mode])["tables"].values())
    result = CliRunner().invoke(
        app,
        ["scrub-at-rest", "--db", str(db.db_path), "--providers", mode, "--allow-live-db", "--expect-rows", str(total)],
    )
    assert result.exit_code == 1
    assert "private-diagnostic" not in result.output
    payload = json.loads(result.stdout)
    assert payload["detail"] == expected_detail
    assert payload["reason"] == ("quiesce-failed" if failure == "quiesce" else "scrub-failed")


@pytest.mark.parametrize("service", ["fleet-watchdog", "brainbar", "drain"])
def test_quiesced_loaded_job_names_exact_label(db, live_guard, monkeypatch, service):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda name: name == service)
    with pytest.raises(module.ScrubAtRestError) as error:
        module._quiesced_gates(maintenance.MaintenanceConfig(db_path=db.db_path))
    assert error.value.detail == "loaded:" + maintenance._launchd_label(service)


def test_quiesced_lsof_failure_has_gate_detail_without_values(db, live_guard, monkeypatch):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _, **kwargs: False)

    def fail_lsof(config):
        raise maintenance.MaintenanceAbort("private-command-value")

    monkeypatch.setattr(maintenance, "_check_lsof_clean", fail_lsof)
    with pytest.raises(module.ScrubAtRestError) as error:
        module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert error.value.reason == "quiesce-failed"
    assert error.value.detail == "lsof-writers"
    assert "private-command-value" not in str(error.value)
    assert "private-command-value" not in str(error.value.__cause__)


@pytest.mark.parametrize("name", ["BrainBar", "BrainBarDaemon"])
def test_remaining_brainbar_process_names_gate(monkeypatch, name):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    monkeypatch.setattr(
        maintenance,
        "run_command",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"/Applications/BrainBar.app/Contents/MacOS/{name}\n"),
    )
    with pytest.raises(module.ScrubAtRestError) as error:
        module._check_no_brainbar_processes()
    assert error.value.detail == "process:" + name


def test_unknown_launchd_state_names_service(db, live_guard, monkeypatch):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    monkeypatch.setattr(maintenance, "is_launchd_label_loaded", lambda label, **kwargs: None)
    monkeypatch.setattr(maintenance, "_service_is_loaded", live_guard.loaded_probe)
    monkeypatch.setattr(maintenance, "_bootout_service", lambda _, **kwargs: False)
    with pytest.raises(module.ScrubAtRestError) as error:
        module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert error.value.detail == "state:com.etanhey.brainlayer-fleet-watchdog"


def test_async_brainbar_exit_settles_before_gate_without_weakening_gate(monkeypatch):
    from brainlayer import scrub_at_rest as module

    pending = iter([True, True, False])
    checks = []

    def check():
        checks.append(True)
        if _fixture_next(pending):
            raise module.ScrubAtRestError("still exiting", reason="quiesce-failed", detail="process:BrainBar")

    monkeypatch.setattr(module, "_check_no_brainbar_processes", check)
    monkeypatch.setattr(module.time, "monotonic", lambda: 0)
    waits = []
    monkeypatch.setattr(module.time, "sleep", waits.append)
    module._wait_for_brainbar_exit()
    assert len(checks) == 3
    assert len(waits) == 2


def test_brainbar_exit_timeout_refuses_and_retains_process_detail(monkeypatch):
    from brainlayer import scrub_at_rest as module

    def check():
        raise module.ScrubAtRestError("still running", reason="quiesce-failed", detail="process:BrainBarDaemon")

    monkeypatch.setattr(module, "_check_no_brainbar_processes", check)
    ticks = iter([0, 31])
    monkeypatch.setattr(module.time, "monotonic", lambda: _fixture_next(ticks))
    monkeypatch.setattr(module.time, "sleep", lambda _: pytest.fail("slept beyond deadline"))
    with pytest.raises(module.ScrubAtRestError) as error:
        module._wait_for_brainbar_exit()
    assert error.value.detail == "process:BrainBarDaemon"


def test_daemon_cannot_revive_ui_during_bootout(db, live_guard, monkeypatch):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    def bootout(service):
        live_guard.events.append(("stop", service))
        live_guard.loaded[service] = False
        if service == "brainbar" and live_guard.loaded.get("brainbar-daemon", True):
            # Daemon UI-watchdog's kickstart fails for an unloaded UI, then openBundle revives it.
            live_guard.loaded["brainbar"] = True
        return True

    monkeypatch.setattr(maintenance, "_bootout_service", bootout)
    result = module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert result["tables"]["chunks"]["rows"] == 1
    stops = [event[1] for event in live_guard.events if isinstance(event, tuple) and event[0] == "stop"]
    assert stops.index("brainbar-daemon") < stops.index("brainbar")


@pytest.mark.parametrize("code", [0, 1, 75])
def test_oneoff_wrapper_preserves_cli_exit_before_timestamp(tmp_path, code):
    from pathlib import Path

    interpreter = tmp_path / "fake-python"
    interpreter.write_text(f"#!/bin/sh\necho '{{\"synthetic\":true}}'\nexit {code}\n")
    interpreter.chmod(0o700)
    log = tmp_path / "oneoff.log"
    result = subprocess.run(
        [
            "/bin/sh",
            str(Path(__file__).resolve().parents[1] / "scripts/scrub-at-rest-oneoff.sh"),
            str(log),
            str(interpreter),
            "--db",
            str(tmp_path / "never-opened.db"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == code
    assert log.read_text().splitlines()[-1].endswith(f"exit={code}")
    assert not (tmp_path / "never-opened.db").exists()


def test_unexpected_quiesce_detail_is_not_retained():
    from brainlayer import scrub_at_rest as module

    error = module.ScrubAtRestError("safe", reason="quiesce-failed", detail="private-value")
    assert module._failure(error).detail is None


@pytest.mark.parametrize("mode", ["google_oauth", "context7", "exa_labeled"])
@pytest.mark.parametrize("entry", [["-m", "brainlayer"], ["-c", "from brainlayer.cli import app; app()"]])
def test_real_cli_process_refusal_is_nonzero(tmp_path, mode, entry):
    import os
    from pathlib import Path

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run(
        [
            sys.executable,
            *entry,
            "scrub-at-rest",
            "--db",
            str(tmp_path / "missing.db"),
            "--dry-run",
            "--providers",
            mode,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["reason"] == "scrub-failed"
    assert payload["detail"] is None


def test_process_probe_failure_refuses_without_waiting_or_values(monkeypatch):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    def fail(*args, **kwargs):
        raise OSError("private-probe-value")

    monkeypatch.setattr(maintenance, "run_command", fail)
    monkeypatch.setattr(module.time, "sleep", lambda _: pytest.fail("probe failure retried"))
    with pytest.raises(module.ScrubAtRestError) as error:
        module._wait_for_brainbar_exit()
    assert error.value.detail == "brainbar-process-probe"
    assert "private-probe-value" not in str(error.value.__cause__)


def test_launchd_probe_exception_names_service_and_discards_value(db, live_guard, monkeypatch):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    def fail(*args, **kwargs):
        raise OSError("private-probe-value")

    monkeypatch.setattr(maintenance, "is_launchd_label_loaded", fail)
    monkeypatch.setattr(maintenance, "_service_is_loaded", live_guard.loaded_probe)
    monkeypatch.setattr(maintenance, "_bootout_service", lambda _, **kwargs: False)
    with pytest.raises(module.ScrubAtRestError) as error:
        module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert error.value.detail == "state:com.etanhey.brainlayer-fleet-watchdog"
    assert "private-probe-value" not in str(error.value.__cause__)


@pytest.mark.parametrize("mode", ["google_oauth", "context7", "exa_labeled"])
@pytest.mark.parametrize(
    "outcome", ["appears", "final-poll", "oversleep", "timeout", "window", "zero", "dry-run", "missing-count"]
)
def test_backup_wait_before_lock_and_quiesce(db, live_guard, monkeypatch, mode, outcome):
    from contextlib import contextmanager

    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    live_guard.backup.unlink()
    monkeypatch.setenv("BRAINLAYER_MCP_SOCKET", str(db.db_path.parent / "absent.sock"))
    monkeypatch.setenv("BRAINLAYER_FORBID_BRAINBAR_SOCKET", "1")
    elapsed, sleeps, locks = [0.0], [], []
    factory = maintenance.MaintenanceConfig

    def config(**kwargs):
        result = factory(**kwargs)
        start = {"window": live_guard.now.replace(hour=5, minute=39, second=59)}.get(outcome, live_guard.now)
        result.now_fn = lambda: start + dt.timedelta(seconds=elapsed[0])
        return result

    monkeypatch.setattr(maintenance, "MaintenanceConfig", config)

    def sleep(seconds):
        assert not locks
        assert not any(isinstance(e, tuple) for e in live_guard.events)
        elapsed[0] += seconds + {"final-poll": 0.01, "window": 1.0}.get(outcome, 0)
        elapsed[0] = {("oversleep", 2): 91}.get((outcome, len(sleeps)), elapsed[0])
        sleeps.append(seconds)
        arrival = (
            elapsed[0] >= 85
            if outcome == "final-poll"
            else len(sleeps) == {"appears": 2, "oversleep": 3, "window": 1}.get(outcome)
        )
        if arrival:
            live_guard.backup.write_text(json.dumps(live_guard.receipt) + "\n")

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep))
    lock = module._maintenance_lock

    @contextmanager
    def tracked_lock(path):
        locks.append(path)
        with lock(path):
            yield

    monkeypatch.setattr(module, "_maintenance_lock", tracked_lock)
    total = sum(t["rows"] for t in module._run(db, True, 100, module.PROVIDER_MODES[mode])["tables"].values())
    flags = ["--dry-run"] if outcome == "dry-run" else ["--allow-live-db", "--expect-rows", str(total)]
    if outcome == "missing-count":
        flags = ["--allow-live-db"]
    result = CliRunner().invoke(
        app,
        [
            "scrub-at-rest",
            "--db",
            str(db.db_path),
            "--providers",
            mode,
            "--wait-for-backup-seconds",
            "0" if outcome == "zero" else "90",
            *flags,
        ],
    )
    if outcome in {"appears", "final-poll", "dry-run"}:
        assert result.exit_code == 0, result.output
        assert len(sleeps) == {"appears": 2, "final-poll": 5, "dry-run": 0}[outcome]
        assert len(locks) == (0 if outcome == "dry-run" else 1)
    else:
        assert result.exit_code == 1, result.output
        expected = "verified-backup-required" if outcome == "zero" else "verified-backup-timeout"
        if outcome == "missing-count":
            expected = "expected-row-count-required"
        assert json.loads(result.stdout)["reason"] == expected
        if outcome != "zero":
            assert not locks
        assert not any(isinstance(e, tuple) for e in live_guard.events)
        assert elapsed[0] == {"timeout": 90, "oversleep": 91, "window": 1.5, "zero": 0, "missing-count": 0}[outcome]


@pytest.mark.parametrize("guarded", [False, True])
def test_backup_wait_does_not_sleep_for_offline_apply_or_existing_receipt(db, live_guard, monkeypatch, guarded):
    from brainlayer import chunk_origin_wipe
    from brainlayer import scrub_at_rest as module

    if not guarded:
        live_guard.backup.unlink()
        monkeypatch.setattr(chunk_origin_wipe, "_live_db_candidates", lambda: [])
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(monotonic=lambda: 0, sleep=lambda _: pytest.fail("unexpected backup wait")),
    )
    result = module.scrub_at_rest(
        db.db_path,
        allow_live_db=guarded,
        expect_rows=live_guard.total,
        wait_for_backup_seconds=90,
    )
    assert result["tables"]["chunks"]["rows"] == 1


@pytest.mark.parametrize("state", ["before-window", "after-window", "within-reserve", "missing-pause"])
def test_backup_wait_refuses_without_sleep_for_unavailable_window_or_other_gate(db, live_guard, monkeypatch, state):
    from brainlayer import maintenance
    from brainlayer import scrub_at_rest as module

    live_guard.backup.unlink()
    if state == "missing-pause":
        live_guard.pause.unlink()
    monkeypatch.setenv("BRAINLAYER_MCP_SOCKET", str(db.db_path.parent / "absent.sock"))
    monkeypatch.setenv("BRAINLAYER_FORBID_BRAINBAR_SOCKET", "1")
    start = {
        "before-window": live_guard.now.replace(hour=3, minute=50),
        "after-window": live_guard.now.replace(hour=7, minute=50),
        "within-reserve": live_guard.now.replace(hour=5, minute=50),
    }.get(state, live_guard.now)
    factory = maintenance.MaintenanceConfig
    elapsed, sleeps = [0.0], []

    def config(**kwargs):
        result = factory(**kwargs)
        result.now_fn = lambda: start + dt.timedelta(seconds=elapsed[0])
        return result

    def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] += 1801

    monkeypatch.setattr(maintenance, "MaintenanceConfig", config)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep))
    monkeypatch.setattr(module, "_maintenance_lock", lambda _: pytest.fail("lock taken before refusal"))
    reason = "enrichment-pause-required" if state == "missing-pause" else "verified-backup-required"
    with pytest.raises(module.ScrubAtRestError) as error:
        module.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total, wait_for_backup_seconds=1800)
    assert error.value.reason == reason
    assert sleeps == []
    assert not any(isinstance(e, tuple) for e in live_guard.events)


@pytest.mark.parametrize("slow_service", ["fleet-watchdog", "hotlane-brainbar"])
@pytest.mark.parametrize("timeout", [False, True])
def test_watchdog_hold_waits_and_restores_last(db, live_guard, monkeypatch, slow_service, timeout):
    clock, held = [0.0], [False]

    def launchctl(args, **kwargs):
        if args[1] in {"disable", "enable"}:
            held[0] = args[1] == "disable"
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(maintenance, "run_command", launchctl)
    monkeypatch.setattr(maintenance, "monotonic", lambda: clock[0])
    monkeypatch.setattr(maintenance, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(
        maintenance,
        "_service_is_loaded",
        lambda service, **kwargs: (
            clock[0] < (100 if timeout else 30) if service == slow_service else live_guard.loaded.get(service, True)
        ),
    )
    apply = scrub_at_rest._apply

    def guarded_apply(*args):
        assert held[0]
        return apply(*args)

    monkeypatch.setattr(scrub_at_rest, "_apply", guarded_apply)
    if timeout:
        monkeypatch.setattr(scrub_at_rest, "WriterRuntimeStore", lambda _: pytest.fail("writer opened after timeout"))
        with pytest.raises(scrub_at_rest.ScrubAtRestError) as error:
            scrub_at_rest.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
        assert error.value.reason == "quiesce-failed"
        assert error.value.detail == "loaded:" + maintenance._launchd_label(slow_service)
        assert clock[0] == 45
    else:
        scrub_at_rest.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
        assert clock[0] == pytest.approx(30)
    assert not held[0]
    assert live_guard.events[-1] == ("resume", "fleet-watchdog")


def test_hung_unload_probe_is_bounded_and_refuses(db, live_guard, monkeypatch):
    timeouts = []

    def launchctl(args, **kwargs):
        if args[1] == "print":
            timeouts.append(kwargs.get("timeout"))
            raise subprocess.TimeoutExpired(args, 45)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(maintenance, "run_command", launchctl)
    monkeypatch.setattr(maintenance, "_service_is_loaded", live_guard.loaded_probe)
    with pytest.raises(scrub_at_rest.ScrubAtRestError) as error:
        scrub_at_rest.scrub_at_rest(db.db_path, allow_live_db=True, expect_rows=live_guard.total)
    assert error.value.detail == "state:com.etanhey.brainlayer-fleet-watchdog"
    assert 0 < timeouts[0] <= 45
