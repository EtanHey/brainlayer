# Claude cloud JSONL ingest contract audit (2026-11-05)

Task marker: cloud-credit-20261105-brainlayer-last-ingest-contract. Base: origin/main `d6b82f04`.
Proof: `tests/test_claude_cloud_ingest_contract_audit.py`. It uses synthetic fixtures and a scratch DB under `tmp_path`, and
drives the real `create_flush_callback` on both paths: direct writer (`arbitrated=False`) and
arbitrated queue + `drain.burn_drain_once`. The assertions pin what main does **today**, not what it should do.

## Code traced
- Project: `watcher_bridge._extract_project_from_source` (L260). If the path holds a `projects/<encoded>`
  segment, that segment is used. Entry `cwd`/`workspace`/`workdir` is only a fallback.
- Chunk ID: `rt-{Path(source_file).stem[:8]}-{normalized_exact_hash(clean_content)[:16]}` (L513-515).
- conversation_id = `chunk.metadata["session_id"]` (the entry's `sessionId`), or the file stem if that is missing (L527).
  `claude_conversation_id` is added only when the path yields a UUID (`_extract_claude_conversation_id`).
- Dedupe/upsert (same on both paths): `dedupe.find_duplicate` checks the exact `dedupe_hash`,
  **scoped by project + content_type**, and `merge_duplicate_chunk` handles a hit. If the generated
  chunk_id already exists, `merge_existing_chunk_seen` runs. That function is **not project-scoped** and updates only
  `seen_count`, `last_seen_at`, `tags`, `importance`, `half_life_days`, `source_class`.

## Observed (RUN, both paths identical)
1. **The repo-encoded path wins.** Path `projects/-home-user-Gits-repoA/...` with entry `cwd=/home/user/brainlayer` → stored `project = "repoA"`.
2. **The 8-character prefix collides for `cse_` files.** Three repoA files plus one repoB file, all `cse_01SYNTH*`, with identical
   text → **one** row `rt-cse_01SY-<hash>`, `seen_count=4`. It keeps only the first file's
   `source_file`, `conversation_id` (`cse_01SYNTH0000`), `metadata.session_id` and `project` (repoA).
   The other three sessions and the **repoB project assignment are lost**: no sibling row and no
   per-source record. Only the counter shows they existed. The repoB file reached the cross-project
   `merge_existing_chunk_seen` path because its stem prefix and content hash matched.
   *Contrast:* UUID-named files with identical text in repoA and repoB → two rows
   (`rt-3679128a-…`, `rt-9c1d2e3f-…`), one per project. Within the *same* project, identical text from different
   UUID files would still collapse through the project-scoped `find_duplicate`. That aggregation is intended
   (`seen_count`), but it also keeps only the canonical row's source/session (a code reading; this test does not assert it).
3. **Session identity.** The cloud `sessionId` is stored in both `conversation_id` and `metadata.session_id`.
   The per-entry `uuid` is not stored anywhere in the row. For `cse_` filenames, `claude_conversation_id` is
   absent (it is set only for UUID-path files).
4. **Linkage and short output.** An assistant text entry → `content_type = assistant_text`, `sender = assistant`.
   A `tool_result` entry with content `"OK"` is dropped at Layer 1 (`should_skip_entry` → `too_short`).
   `classify_content` would also return `None`. No row is written, and `tool_use_id` linkage is not
   stored for it. Linkage for long tool_results was **UNRUN**.

## Verdict
Dedup intentionally aggregates repeats, but it **loses source-session (and cross-project)
context**: the only trace of the merged sources is `seen_count`. The `cse_` prefix makes
`file_stem[:8]` a near-constant (`cse_01SY` is shared by many cloud sessions), so chunk_id equality acts as
a content-only key across projects. UUID filenames do not have this property.

## Safe repair separation (not implemented here)
- **Structural IDs we own** (e.g. deriving the chunk_id prefix from the full stem or a session hash, or
  re-keying collided `rt-cse_*` rows): changes neither body nor path. It still rewrites a primary key that
  `kg_entity_chunks`, `superseded_by` and vectors reference, so it needs a migration tested against a DB copy.
- **Source/project metadata** (re-attributing a lost repoB project or a lost session to rows, or splitting
  aggregated rows): this changes stored provenance, so it falls under the repo writer-maintenance law
  (quiesce writers, checkpoint WAL, batch, pre-merge live check on a DB copy). The current rows cannot
  recover the lost sessions. That requires a re-read of the source JSONLs, which still exist (they are never deleted).

## UNRUN
Full pytest suite; real JSONL archives or the real DB (forbidden); long tool_result / tool_use linkage; the simhash
near-dup path; the subagent and brain-worker paths; any repair.
