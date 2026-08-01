# D4-T3 ingestion iterate report

Status: iteration implemented on `feat/d4-t3-thread-ingest`; production entrypoint
executed; no push or PR.

## Review blockers resolved

- `brainlayer ingest-t3` is now a first-class CLI entrypoint. The dedicated
  `scripts/launchd/com.brainlayer.t3-ingest.plist` runs it at load and daily at
  03:45; `scripts/launchd/install.sh all` installs it.
- Provider runtime linkage now reaches every emitted chunk as
  `t3_provider_name`, `t3_provider_session_id`, and `t3_mirrored` metadata.
- The fatal schema contract is limited to columns actually consumed:
  `projection_threads(thread_id, project_id, title, created_at)`,
  `projection_thread_messages(message_id, thread_id, role, text, created_at)`,
  and `provider_session_runtime(thread_id, provider_name, resume_cursor_json)`.
  `projection_thread_sessions` and all `updated_at` columns are no longer
  required.
- The dead `read_t3_threads` export was removed.
- `health-check` accepts `--t3-health-path`, includes the JSON snapshot in its
  result, and raises a critical `t3_ingest_unhealthy` issue when `alerting` is
  true.
- SQLite/DB sources no longer pass a 704 MB binary file through the JSONL
  timestamp sniffer; T3 chunk metadata supplies its timestamps.

Relevant implementation locations are `src/brainlayer/cli/__init__.py:3681-3715`,
`scripts/launchd/com.brainlayer.t3-ingest.plist`,
`src/brainlayer/ingest/t3.py:28-32,156-235,295-317`, and
`src/brainlayer/health_check.py:779-795,961-964` (iteration changes; final
branch commit is recorded below). The original reader’s read-only and alarm
primitives remain at `src/brainlayer/ingest/t3.py:156-180`.

## Re-verified live source facts

The source is opened with SQLite URI `mode=ro&immutable=0`, autocommit,
`PRAGMA query_only=ON`, and a 1-second busy timeout. The production source
remained read-only during the live ingest.

| Fact | Observed |
|---|---:|
| `~/.t3/userdata/state.sqlite` size | 704,430,080 bytes |
| `projection_threads` | 45 |
| `projection_thread_messages` | 2,349 |
| `projection_thread_sessions` | 44 (not required by the reader) |
| `projection_thread_activities` | 38,381 |
| `provider_session_runtime` | 34 |

## Schema map and provider linkage

| Table/column | Meaning and relationship |
|---|---|
| `projection_threads.thread_id` | Thread identity and primary join key. |
| `projection_threads.project_id`, `title`, `created_at` | Thread metadata retained on emitted chunks. |
| `projection_thread_messages.message_id` | Message identity. |
| `projection_thread_messages.thread_id` | Message-to-thread link to `projection_threads.thread_id`. |
| `projection_thread_messages.role`, `text`, `created_at` | Message payload and ordering fields consumed. |
| `provider_session_runtime.thread_id` | Runtime-mirror-to-thread link; a matching row marks the thread mirrored. |
| `provider_session_runtime.provider_name` | Provider name persisted to chunk metadata. |
| `provider_session_runtime.resume_cursor_json.threadId` | Provider session ID persisted to `t3_provider_session_id` when present. |

The reader validates this consumed contract before snapshot queries. Missing
tables/columns raise the existing `t3_schema_drift` alarm and write an alerting
health snapshot before raising. The 34 runtime rows are accepted as duplicate
content by design; no mirror exclusion or source-specific dedup heuristic was
added.

## The 11 unmirrored threads

“Unmirrored” means no matching `provider_session_runtime` row. It is not an
exclusion rule.

| Thread ID | Created | Messages |
|---|---|---:|
| `a7b35b2a-50f2-4e8c-83a5-747e7a29757c` | 2026-03-06T17:45:31.449Z | 1 |
| `2b5cad3c-eefb-4766-949b-01ffcdfcfbf5` | 2026-03-07T18:21:10.185Z | 1 |
| `e9cf5dd4-a039-4dec-bfe8-d717dd2e9c23` | 2026-03-07T18:23:16.780Z | 1 |
| `44dd2387-fbf5-4b92-b778-b1f5487a15f0` | 2026-03-07T18:25:01.591Z | 1 |
| `263650ba-2190-43f7-a32a-2a87975a7d5e` | 2026-03-07T18:28:00.738Z | 1 |
| `7eb8d353-7eb2-4420-b7ad-e31926128aef` | 2026-03-07T18:28:53.755Z | 0 |
| `de99ab14-0595-42da-b3f1-cf863e8d5835` | 2026-03-07T18:29:26.869Z | 1 |
| `da804581-3b59-49e3-bef1-96e9f851a1f1` | 2026-03-07T19:08:49.225Z | 1 |
| `1babbcbf-0e37-40a4-bb14-14b1b336d542` | 2026-03-07T19:13:08.735Z | 1 |
| `d762ec6e-cb56-41a2-aea6-46dd547f2e75` | 2026-03-07T19:16:05.780Z | 1 |
| `95eee65c-59b3-49c7-9f3f-5051ead264ca` | 2026-03-07T19:17:18.220Z | 1 |

## Real production invocation

The installed console entrypoint was run with the checked-out source:

```text
PYTHONPATH=src brainlayer ingest-t3 \
  --state-db /Users/etanheyman/.t3/userdata/state.sqlite \
  --db /Users/etanheyman/.local/share/brainlayer/brainlayer.db \
  --health-path /Users/etanheyman/.local/share/brainlayer/t3-health.json
```

| Metric | Count |
|---|---:|
| Threads seen / ingested | 45 / 45 |
| Messages seen / ingested | 2,349 / 2,349 |
| Chunks planned | 2,506 |
| Chunks indexed | 2,505 |
| Mirrored threads accepted as duplicates | 34 |
| T3 messages skipped by role/empty-text policy | 0 |
| Chunks filtered by shared system-prompt guard | 1 |

The one filtered chunk is message
`c65c95ed-2d60-4f6d-92a6-8a2ff6ad0b40` in thread
`f2a72bc0-1abc-450d-ae50-ab6b3e9812dd`; its user text begins with agent
instruction scaffolding and matched the existing global guard. This is why the
indexed count is one below the planned count; it was not a T3 reader failure.

Read-only post-run verification of the canonical BrainLayer DB found:

- `2,505` rows with `provenance_class = 't3-thread'` and `source = 't3'`;
- `44` nonempty T3 conversations;
- `2,495` mirrored rows and `2,487` rows carrying a non-null provider session ID;
- `t3-health.json` with `alerting=false`, `threads_seen=45`,
  `messages_seen=2349`, `chunks_planned=2506`, and `chunks_indexed=2505`.

## Verification

Passed:

- Focused and cross-cutting suite: **81 passed**.
- Isolated baseline phase-2 queue file: **11 passed**.
- Live production invocation and read-only destination/source verification.

The full `pytest -q` run reached the repository’s existing phase-2 queue
failure at 68%, then cascaded into file-descriptor errors (`OSError: [Errno
24] Too many open files`) during temporary-directory cleanup. The isolated
phase-2 file passes, and no phase-2 queue file changed in this iteration.

## Commit

One iteration commit will contain the implementation, launchd wiring, health
consumer, behavioral tests, and this report. No push or PR was performed.

TASK_DONE
