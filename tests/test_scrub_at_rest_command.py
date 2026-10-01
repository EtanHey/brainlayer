"""Copy-only maintenance proofs with synthetic credentials and the real schema."""

import hashlib
import json

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
    monkeypatch.setattr(module.time, "sleep", delays.append)
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
