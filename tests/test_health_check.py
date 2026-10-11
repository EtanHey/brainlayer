"""Health SQL uses archived_at + lineage. Full coverage: test_stability_health_check.py."""

import pytest

import brainlayer.health_check as health_check


def test_missing_embeddings_sql_uses_archived_at_not_flag():
    assert "archived_at IS NULL" in health_check.MISSING_EMBEDDINGS_SQL
    assert "COALESCE(c.archived, 0) = 0" not in health_check.MISSING_EMBEDDINGS_SQL
    assert "COALESCE(c.status, 'active') = 'active'" not in health_check.MISSING_EMBEDDINGS_SQL


@pytest.mark.parametrize("mode", ["default", "custom-db", "env-health", "explicit-health"])
def test_t3_producer_consumer_share_resolver_and_public_cli(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace

    from typer.testing import CliRunner

    import brainlayer.ingest.t3 as t3
    from brainlayer import paths
    from brainlayer.cli import app

    canonical = tmp_path / "canonical" / "brainlayer.db"
    custom = tmp_path / "custom" / "brainlayer.db"
    override = tmp_path / "override" / "health.json"
    explicit = tmp_path / "explicit" / "health.json"
    monkeypatch.setattr(paths, "_CANONICAL_DB_PATH", canonical)
    monkeypatch.delenv("BRAINLAYER_DB", raising=False)
    monkeypatch.delenv("BRAINLAYER_T3_INGEST_HEALTH_PATH", raising=False)
    destination = canonical
    if mode != "default":
        destination = custom
        monkeypatch.setenv("BRAINLAYER_DB", str(custom))
    if mode in {"env-health", "explicit-health"}:
        monkeypatch.setenv("BRAINLAYER_T3_INGEST_HEALTH_PATH", str(override))
    cli_explicit = explicit if mode == "explicit-health" else None
    expected = cli_explicit or (override if mode == "env-health" else destination.parent / "t3-health.json")
    assert paths.resolve_t3_health_path(destination, cli_explicit) == expected
    assert health_check.HealthCheckConfig(db_path=destination, t3_health_path=cli_explicit).t3_health_path == expected
    captured = {}

    def consume(config):
        captured["consumer"] = config.t3_health_path
        return SimpleNamespace(ok=True, to_dict=lambda: {"ok": True})

    def produce(_source, **kwargs):
        captured["producer"] = kwargs["health_path"]
        return SimpleNamespace(
            **dict.fromkeys(
                [
                    "threads_seen",
                    "threads_ingested",
                    "messages_seen",
                    "messages_ingested",
                    "chunks_planned",
                    "chunks_indexed",
                    "duplicates_accepted",
                ],
                0,
            )
        )

    monkeypatch.setattr(health_check, "run_health_check", consume)
    monkeypatch.setattr(t3, "ingest_t3", produce)
    producer_args = ["ingest-t3", "--db", str(destination), "--dry-run"]
    consumer_args = ["health-check", "--db", str(destination), "--json"]
    if cli_explicit:
        producer_args += ["--health-path", str(cli_explicit)]
        consumer_args += ["--t3-health-path", str(cli_explicit)]
    runner = CliRunner()
    producer = runner.invoke(app, producer_args)
    consumer = runner.invoke(app, consumer_args)
    assert producer.exit_code == 0, producer.output
    assert consumer.exit_code == 0, consumer.output
    assert captured == {"producer": expected, "consumer": expected}
    assert not destination.exists() and not expected.exists()
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text('{"alerting": true, "alert_reasons": ["reader_failure"]}')
    issue = health_check._t3_health_issue(health_check._load_json(captured["consumer"]))
    assert issue is not None and issue.code == "t3_ingest_unhealthy"
