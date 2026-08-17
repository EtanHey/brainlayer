"""Health SQL uses archived_at + lineage. Full coverage: test_stability_health_check.py."""

import brainlayer.health_check as health_check


def test_missing_embeddings_sql_uses_archived_at_not_flag():
    assert "archived_at IS NULL" in health_check.MISSING_EMBEDDINGS_SQL
    assert "COALESCE(c.archived, 0) = 0" not in health_check.MISSING_EMBEDDINGS_SQL
    assert "COALESCE(c.status, 'active') = 'active'" not in health_check.MISSING_EMBEDDINGS_SQL
