# Enrichment Gold Fixtures

`enrichment_gold.py` writes `enrichment-gold.jsonl` here by default when run with
`--snapshot-path` against a read-only SQLite snapshot.

The initial PR intentionally does not include a live sampled fixture because the production
BrainLayer DB is under load. See `# TODO(PR1-followup)` in the sampler for the real
60-chunk snapshot pull.
