-- Standalone SQL migration for pre-004 databases. Python and Swift startup
-- migrators add this column through their own column-exists checks.
ALTER TABLE chunks ADD COLUMN enrich_status TEXT;

UPDATE chunks
SET enriched_at = NULL
WHERE enriched_at IS NOT NULL
  AND TRIM(enriched_at) = '';

UPDATE chunks
SET enrich_status = 'success'
WHERE enriched_at IS NOT NULL
  AND TRIM(enriched_at) != ''
  AND enriched_at NOT LIKE 'skipped:%'
  AND enrich_status IS NULL;

UPDATE chunks
SET enrich_status = NULLIF(TRIM(SUBSTR(enriched_at, LENGTH('skipped:') + 1)), ''),
    enriched_at = NULL
WHERE enriched_at LIKE 'skipped:%';

CREATE INDEX IF NOT EXISTS idx_chunks_enrich_status ON chunks(enrich_status);
