"""Additive covering-index migration for the observability reader."""

MIGRATION = "2026_11_11_observability_indexes"
INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_observability_time ON chunks (datetime(created_at) DESC, id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_observability_source ON chunks (source_class, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_observability_author ON chunks "
    "(archived_at, provenance_class, source_class, source_file, created_at)",
)


def migrate_observability_indexes(cursor):
    """Run in the schema owner's transaction; never from the reader."""
    if cursor.execute("SELECT 1 FROM schema_migrations WHERE name = ?", (MIGRATION,)).fetchone():
        return
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
    if "source_class" not in columns:
        return
    for sql in INDEXES:
        cursor.execute(sql)
    cursor.execute("INSERT INTO schema_migrations(name, applied_at) VALUES (?, datetime('now'))", (MIGRATION,))
