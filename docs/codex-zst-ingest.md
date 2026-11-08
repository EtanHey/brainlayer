# Compressed Codex recovery

Compression is a physical representation of the same logical `.jsonl` FILE. Source paths,
chunk keys and decoded offsets retain that identity. The first native `session_meta` id is
metadata only: forks can share it and still have different tails. Never deduplicate files by it.

Archive discovery stays **off** unless `BRAINLAYER_CODEX_ZST_ADMISSION=bounded` is explicitly set.
The bounded watcher defaults to two archive files and 1 MiB of consumed decoded input per poll
(`BRAINLAYER_CODEX_ZST_FILES_PER_CYCLE`, `BRAINLAYER_CODEX_ZST_BYTES_PER_CYCLE`). It observes the
existing durable queue's backlog threshold (25 files), and retains active decoders across polls.
An unchanged completed archive is not decoded again. A cold logical path with a positive saved
offset is held without decoding; that is not a completeness claim. Active plain/compressed
transitions preserve decoded progress. Turning discovery on is production backfill.

Use **manifest mode** for recovery. It does not discover other files, needs no discovery opt-in,
and verifies `physical_path`, `logical_path`, `size`, `mtime_ns` and `inode` before reading and
after validation. Missing expected stats fail closed. Native ids never skip files. Explicit
manifest replay re-evaluates offset-only files from byte zero without resetting/deleting their
saved offsets. Manifest mode leaves offsets.json byte-identical; queue events carry decoded source_end_offset. Source archives remain unchanged; no expanded source files are written to disk.

The same classification/chunking/scrubbing/provenance path produces dry-run queue candidates
and real queue events. `would_insert` counts candidates **before database content deduplication**,
not promised new database rows. Real admission skips already-committed per-file chunk ids and exact-content matches
using the drain's normalized hash, project/content-type scope and active-row rules,
observes queue backpressure before each publication, and never opens another database writer.
A partial publication, error, or budget stop is reported and does not checkpoint a file complete.
Receipts contain paths/counts/status only; they distinguish `queued` from `committed`.
`live_chunks_at_start` is the observed durable row count under this logical `source_file`, not a
promise that queued events have committed. It is cumulative across bounded windows; compare
it with the pre-replay source count for a net insertion count. `duplicate_content` counts
candidates resolved under an existing different id/source, with at most five distinct
`duplicate_matches` (`chunk_id` and `source_file` only). `filtered` counts classifier/scrub
pipeline drops. Dry runs report these DB-dependent counts as zero and still report
`would_insert`; they make no dedup measurement. Drain between publication windows, then
repeat until an accepted receipt has `queued=0` to reconcile durable counts. A row with
only existing content matches has `state` and `reason` `indexed_elsewhere_dedup`, in class
`accepted`. Resolved candidates are never re-queued and do not bump seen counters.
With `--manifest-machine`, valid M1/M4 rows for the other machine receive
`filtered/skipped_other_machine`; missing or unknown machine values receive
`error/invalid_machine`. Neither consumes a selected-machine resume ordinal.
Every selected row has exactly one `class`: `readable` (validated dry-run candidates),
`accepted` (queued or already committed), `filtered`, `empty`, `error`, or `deferred`,
and a nonempty `reason`. `state` retains publication detail, including partial queueing.
JSON failures report `truncated_final_record` or `malformed_record@<decoded offset>`;
decoder failures retain their zstd error. Unresolved provenance is `deferred`.
A malformed manifest row gets its own error receipt; later rows are still evaluated.

## Commands for the approved per-machine manifest

Transfer the stat-complete, approved manifest to `docs.local/codex-recovery/manifest.jsonl`
on each machine. These are examples for the first bounded window, **not commands to run during
installation**. Replace `M4` with `M1` on the other Mac; use the matching machine's installed CLI.

```sh
brainlayer ingest-codex --from-manifest docs.local/codex-recovery/manifest.jsonl --manifest-machine M4 --manifest-start 0 --max-files 2 --max-decoded-bytes 16777216 --dry-run
brainlayer ingest-codex --from-manifest docs.local/codex-recovery/manifest.jsonl --manifest-machine M4 --manifest-start 0 --max-files 2 --max-decoded-bytes 16777216 --receipts-out docs.local/codex-recovery/window-M4-001.jsonl
brainlayer ingest-codex --from-manifest docs.local/codex-recovery/manifest.jsonl --manifest-machine M1 --manifest-start 0 --max-files 2 --max-decoded-bytes 16777216 --dry-run
brainlayer ingest-codex --from-manifest docs.local/codex-recovery/manifest.jsonl --manifest-machine M1 --manifest-start 0 --max-files 2 --max-decoded-bytes 16777216 --receipts-out docs.local/codex-recovery/window-M1-001.jsonl
```

`--manifest-start` is zero-based within the selected machine. Advance the window only after
reconciling its receipts with the existing drain; retry a partial window before advancing.
Exit 2 means some rows were refused/held, including rows beyond the requested window. Increase
a byte budget only after reviewing the dry run; a budget stop is never proof a source is empty.

`--max-decoded-bytes` is a global budget over DECODED input, including replay from byte zero.
It must exceed the decoded size of each selected file (with headroom to observe decoder EOF),
not merely its compressed size. Successful full dry runs report each row's `decoded_bytes_total`;
it is null when the budget or an error prevents measuring EOF. The example 16 MiB is a small
initial window, not a budget known to fit M1 files. Increase it under the same custody before
claiming those files readable. There is no per-file resume in manifest mode. Repeated real windows
replay a growing prefix while the queue caps each publication window: O(n²) cumulative decode
work for n equally sized queue windows. Drain/reconcile between windows; never infer completion
from a positive checkpoint. `watch-backfill` exits 2 for every unfinished archive, including
checkpoint-held and never-admitted paths, and prints their breakdown.

The decoder uses `BRAINLAYER_INGEST_ZSTD_PATH` when set, falling back to
`BRAINLAYER_BACKUP_ZSTD_PATH` and the normal executable resolver.

## First-activation cost

With discovery off: zero archive decodes/queue admissions. With it bounded: directory metadata
walk still sees the corpus, but at most two archives consume 1 MiB per poll in aggregate,
including decoded transition prefixes. CLI/pipe read-ahead is separate bounded buffering.
Manifest mode only stats/decodes its listed window: the commands above touch at most two sources
and consume at most 16 MiB, stopping at observed queue depth 25 (concurrent producers can race).
Queue content growth is bounded by the admitted text plus event metadata, not the compressed
source size. The existing drain owns database/WAL growth; no reserve/maintenance gate is bypassed.

The frozen recovery baseline has 456 files: M4 362 / 29,639,924 compressed bytes; M1 94 /
1,365,905,278 compressed bytes. These are archive-read costs, not decoded sizes or database-growth
estimates. Streaming adds no expanded-source disk cost. Cumulative decoded/DB cost remains unknown
until per-file dry-run receipts; keep queue windows bounded while all 456 are reconciled. The approved
manifest v2 supplies expected inodes for all 456 files; missing or changed stats still fail closed.

## Resume proofs and complete receipts

Bounded watcher admission persists `archive` only at confirmed decoded offsets. Marked
archives resume after restarts; historical represented checkpoints without this marker remain
held and appear as `held_archives` in watcher health / `represented_cold` in backfill output.
Backfill exits 0 when only represented-cold archives remain, and 2 for unadmitted, incomplete
or failed work. This is completion of admitted work, not acceptance of represented archives.

Plain/compressed transitions preserve #1656's fingerprint: the decoded offset, a hash of
up to the first 4 KiB, and the last 256 decoded bytes before the offset. The compressed
reader computes it during bounded decoded-prefix replay. A missing fingerprint or mismatch
replays from decoded byte zero; archive mtime alone never admits a checkpoint. Known stored
chunks from acknowledged archive bytes are checked read-only with the existing writer's
duplicate function before publication, so reconciliation is not another sighting. Plain
files otherwise retain #1656's existing cold-resume behavior. This is a sampled fingerprint;
it does not establish a full consumed-prefix digest or detect every middle-only rewrite.
Header identity and cwd are recovered during replay. Quarantined final bytes participate
in the same head/tail fingerprint as plain quarantine. Completion records the confirmed
decoded offset and exact physical size/mtime_ns/inode; unchanged completed archives skip
decoding across restarts. Stat changes invalidate completion and terminal-file reuse.

Every non-dry manifest run requires a fresh `--receipts-out <path>` (the Python API requires
`receipts_out` too). Existing paths are refused before replay; use a new path for each window.
For example, append `--receipts-out docs.local/codex-recovery/window-001.jsonl` to the replay
command above. Dry runs can optionally name a receipt path. Each ledger row is flushed and
fsynced before its stdout summary. It contains `all_duplicate_matches`: every candidate ID,
matched chunk ID, matched source_file and resolution mechanism (`exact_hash`, `writer_alias`, `writer_sha256` or `writer_simhash`),
while stdout keeps five examples. Writer aliases preserve its prior merge decisions even when
the merged canonical's normalized hash changed. For an archived same-file own ID, the reader uses the writer's read-only `find_duplicate`
helper before deciding whether reactivation is needed; its exact or simhash hit is resolved
and recorded without re-queueing. Other candidates retain exact-hash or writer-alias resolution.
`live_chunks_at_start` and `live_chunk_ids_at_start` are the live durable rows observed for this logical source at
row start; `queued_chunk_ids` names newly queued candidates, whose persistence is decided by
the existing single writer. Same-file archived watcher rows without a live duplicate are delivered for reactivation;
superseded/aggregated own IDs remain resolved. A clean decoder EOF with a partial final JSON
record keeps all complete records and reports `quarantined_tail` plus `decoded_bytes_total`.
The ledger records the discarded-tail outcome without retaining its private text.
