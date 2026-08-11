# Spotlight Exclusion Setup Design

## Goal

BrainLayer setup must exclude every high-churn runtime tree from Spotlight before it creates
runtime children. Doctor must warn when the active database directory is not excluded. Migration
of the existing production tree is deliberately deferred to a coordinated writer-stop window.

## Mechanism

Use a `.metadata_never_index` marker in each high-churn root. `mdimport -t` is importer evidence
only: its manual says test imports do not write the Spotlight index, so a live verdict must compare
`mdls`/`mdfind` after a real `mdimport -i` against an unmarked positive control on the same volume.

The mandatory pair review at exact head `e3659579` invalidated the earlier uncontrolled mechanism
claim. Its macOS 26.5.1 (25F80) transcript recorded indexing enabled, a normal RichText importer
returning 29 attributes for the test import, and then null `mdls` plus zero `mdfind` results after a
real import for *all* cases—including the unmarked control in both `$HOME` and `~/Documents`.
Therefore a missing excluded-file result alone is vacuous on that workstation. The runbook now
fails closed unless an identical unmarked control, imported in the same command, receives non-null
metadata and is returned at its exact path by `mdfind`. The source transcript and verdict are under
`DONE_SPOTLIGHT_REVIEW` in the Wave 25 collaboration record.

The marker approach preserves existing canonical paths and needs no environment rewiring on fresh
installs. [Apple documents Search Privacy](https://support.apple.com/guide/mac-help/mchl1bb43b84/mac)
as the supported UI fallback, but that is manual and therefore cannot satisfy unattended setup.

## Excluded layout

Setup creates each root first, writes its marker atomically, then creates its children:

- `~/.local/share/brainlayer`: canonical SQLite DB, `-wal`/`-shm`, sqlite-vec tables and any vector
  sidecars, legacy Chroma/vector and experiment directories, enrichment checkpoints/scratch, prompt cache,
  watcher/drain state, backup staging, and data-root logs.
- `~/.brainlayer`: durable drain queue, queue quarantine, and drain runtime logs.
- `~/Library/Logs/brainlayer`: launchd and BrainBar runtime logs.
- `~/.brainlayer-p0-counter`: longitudinal counter logs.

Configuration and user-facing exports remain searchable: `~/.config/brainlayer` and
`~/.brainlayer-brain` are intentionally not excluded.

## Components

- `brainlayer.paths` owns the marker name plus pure exclusion detection by walking a path and its
  ancestors.
- `brainlayer.setup` owns idempotent layout creation. Defaults cover all four roots; injected roots
  make tests hermetic. It preflights every root and refuses a legacy nonempty, unmarked tree before
  making any marker, directing that machine to the coordinated migration runbook.
- `brainlayer setup` calls layout creation before writing configuration or installing launchd.
- `brainlayer doctor` adds a warning-only `spotlight_indexing_enabled` issue for an unexcluded active
  DB directory on Darwin. The warning never changes the existing fatal/exit-code contract.

## Migration boundary

This PR does not touch `~/.local/share/brainlayer` or any live writer. The runbook requires one
coordinated stop window: checkpoint WAL, stop all writers, move the data/runtime trees to staging,
create marker-bearing targets, move data back without crossing filesystems, restore configured
paths, restart, then prove exclusion with a same-volume positive control plus `mdimport`, `mdls`,
`mdfind`, and `mdutil` volume status. An unindexed positive control makes the result inconclusive
and stops the window.
Rollback keeps the stopped trees intact and reverses the move before writers restart.

## Tests

- Layout creation makes every root and marker, enumerates expected high-churn children, is
  idempotent, and fails without partial mutation when any legacy root needs migration.
- Exclusion detection accepts a marker on the directory or an ancestor and rejects an unmarked
  path.
- Doctor emits exactly the warning for an unexcluded DB directory and omits it for a marked one.
- CLI setup invokes layout creation before env-file creation.
