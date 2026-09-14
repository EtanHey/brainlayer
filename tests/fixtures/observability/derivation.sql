-- Review aid for scripts/derive_observability_goldens.py.
-- Run against a built case DB with: sqlite3 "file:db/<case>.sqlite?immutable=1" < derivation.sql
SELECT COALESCE(content_class, 'knowledge') AS content_class, COUNT(*) AS count
FROM chunks GROUP BY COALESCE(content_class, 'knowledge') ORDER BY content_class;
SELECT source_class, COUNT(*) AS count
FROM chunks GROUP BY source_class ORDER BY source_class IS NOT NULL DESC, source_class;
SELECT COUNT(*) AS total_chunks FROM chunks;
SELECT COUNT(*) AS never_classified FROM chunks
WHERE (provenance_class IS NULL OR source_class IS NULL) AND archived_at IS NULL;
SELECT COUNT(*) AS classified_unknown FROM chunks
WHERE provenance_class = 'unknown' AND archived_at IS NULL;
SELECT source_file, COUNT(*) AS count FROM chunks
WHERE archived_at IS NULL
  AND (provenance_class IS NULL OR source_class IS NULL OR provenance_class = 'unknown')
GROUP BY source_file ORDER BY count DESC, source_file LIMIT 2;

-- author_unknown.trend_7d: live-only seven-day counts, grouped by UTC day.
SELECT substr(created_at, 1, 10) AS day,
       SUM(CASE WHEN provenance_class = 'unknown' AND archived_at IS NULL THEN 1 ELSE 0 END) AS classified_unknown,
       SUM(CASE WHEN (provenance_class IS NULL OR source_class IS NULL) AND archived_at IS NULL THEN 1 ELSE 0 END) AS never_classified
FROM chunks
GROUP BY day
ORDER BY day;
