# ABCDE Experiment Isolation

PR-ISO keeps ABCDE experiment data physically separate from the live BrainLayer DB.

## Paths

- Live DB, forbidden: `~/.local/share/brainlayer/brainlayer.db`
- Experiment DB: `~/.local/share/brainlayer/experiments/abcde-experiment.db`
- Read-only snapshot DB: `~/.local/share/brainlayer/experiments/abcde-snapshot.db`
- Verified backup source: `Brain Drive/06_ARCHIVE/backups/brainlayer-db`

## Snapshot Materialization

Materialization is an explicit operator action. Do not run this in CI.

```bash
PYTHONPATH=src python3 -m brainlayer.eval.experiment_store materialize-snapshot \
  --backup-gzip-path "/path/to/Brain Drive/06_ARCHIVE/backups/brainlayer-db/<backup>.db.gz"
```

The loader refuses to write to the live DB path and writes the decompressed snapshot to
`~/.local/share/brainlayer/experiments/abcde-snapshot.db` by default. Pass
`--snapshot-db-path` to use a different experiment snapshot path.

## CI Contract

CI tests use synthetic SQLite fixtures only. They must not open the live DB and must not
download or decompress the 4GB backup.
