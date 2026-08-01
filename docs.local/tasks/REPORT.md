# D4-T3 ingestion report

Status: implementation complete on `feat/d4-t3-thread-ingest`; no push or PR.

## Decision and estimate

Implemented the already-settled Option C: ingest all 45 T3 threads as a first-class
source with `source = "t3"` and `provenance_class = "t3-thread"`. The estimate was
90–120 minutes because this is a new live SQLite reader and ingestion path rather
than another watched JSONL root.

The parent branch confirms that the existing watcher roots are Claude, Codex,
Cursor, and Gemini only (`src/brainlayer/watcher.py:86-98 @ cc1d21f6`). T3 is
therefore not covered by the watcher’s normal ingestion path.

## Re-verified source facts

The live source was opened with SQLite URI `mode=ro&immutable=0`,
`PRAGMA query_only=ON`, and a 1-second busy timeout. No production BrainLayer DB
write was performed.

| Fact | Observed |
|---|---:|
| `~/.t3/userdata/state.sqlite` size | 704,430,080 bytes |
| `projection_threads` | 45 |
| `projection_thread_messages` | 2,349 |
| `projection_thread_sessions` | 44 |
| `projection_thread_activities` | 38,381 |
| `provider_session_runtime` | 34 |
| message roles | 2,083 assistant; 266 user |

## Schema map

| Table/column | Meaning and relationship |
|---|---|
| `projection_threads.thread_id` | Thread primary key; the reader’s thread identity. |
| `projection_threads.project_id`, `title`, `created_at`, `updated_at` | Thread metadata retained on every emitted chunk. |
| `projection_thread_messages.message_id` | Message primary key. |
| `projection_thread_messages.thread_id` | Message-to-thread link: joins to `projection_threads.thread_id`. |
| `projection_thread_messages.role`, `text`, `created_at`, `updated_at` | Message payload and ordering fields consumed by the reader. |
| `projection_thread_sessions.thread_id` | Session-projection-to-thread link. `provider_session_id` and `provider_thread_id` are present but all 44 live values are NULL. |
| `provider_session_runtime.thread_id` | Runtime-mirror-to-thread link. Presence of a row marks a thread as mirrored; 34 rows matched 34 threads. |
| `provider_session_runtime.resume_cursor_json.threadId` | Provider session identifier used when present; 33 of 34 runtime rows contain it. The live provider counts are Codex 29, Claude 4, Cursor 1. |

The schema was validated at open time for every column consumed by the reader.
Missing or renamed tables/columns raise a `t3_schema_drift` alarm and write a
health failure before raising. The alarm route is the existing fatal primitive
(`src/brainlayer/alarm.py:105-115 @ cc1d21f6`); the new adapter also writes an
atomic JSON health snapshot at `~/.local/share/brainlayer/t3-health.json` by
default.

## The 11 unmirrored threads

“Unmirrored” means no matching row in `provider_session_runtime`; it does not
mean the other threads are safe to exclude. The following list is the complete
read-only query result, ordered by creation time.

| Thread ID | Created | Updated | Messages |
|---|---|---|---:|
| `a7b35b2a-50f2-4e8c-83a5-747e7a29757c` | 2026-03-06T17:45:31.449Z | 2026-03-06T21:59:37.337Z | 1 |
| `2b5cad3c-eefb-4766-949b-01ffcdfcfbf5` | 2026-03-07T18:21:10.185Z | 2026-03-07T18:23:15.704Z | 1 |
| `e9cf5dd4-a039-4dec-bfe8-d717dd2e9c23` | 2026-03-07T18:23:16.780Z | 2026-03-07T18:23:39.985Z | 1 |
| `44dd2387-fbf5-4b92-b778-b1f5487a15f0` | 2026-03-07T18:25:01.591Z | 2026-03-07T18:27:58.686Z | 1 |
| `263650ba-2190-43f7-a32a-2a87975a7d5e` | 2026-03-07T18:28:00.738Z | 2026-03-07T18:29:24.243Z | 1 |
| `7eb8d353-7eb2-4420-b7ad-e31926128aef` | 2026-03-07T18:28:53.755Z | 2026-07-29T23:55:56.741Z | 0 |
| `de99ab14-0595-42da-b3f1-cf863e8d5835` | 2026-03-07T18:29:26.869Z | 2026-03-07T19:08:47.583Z | 1 |
| `da804581-3b59-49e3-bef1-96e9f851a1f1` | 2026-03-07T19:08:49.225Z | 2026-03-07T19:13:05.827Z | 1 |
| `1babbcbf-0e37-40a4-bb14-14b1b336d542` | 2026-03-07T19:13:08.735Z | 2026-03-07T19:17:01.910Z | 1 |
| `d762ec6e-cb56-41a2-aea6-46dd547f2e75` | 2026-03-07T19:16:05.780Z | 2026-03-07T19:16:41.025Z | 1 |
| `95eee65c-59b3-49c7-9f3f-5051ead264ca` | 2026-03-07T19:17:18.220Z | 2026-03-11T14:04:50.782Z | 1 |

## Ingestion behavior and counts

The reader is `src/brainlayer/ingest/t3.py`. It emits stable IDs of the form
`t3:<thread_id>:<message_id>:<chunk_index>`, preserves source timestamps and
project IDs, and records thread/message URI metadata. Short non-empty messages
are retained through a direct single-chunk fallback; there is no Codex-style
minimum-length drop policy.

The live dry-run result was:

| Metric | Count |
|---|---:|
| Threads seen | 45 |
| Threads ingested | 45 |
| Messages seen | 2,349 |
| Messages ingested | 2,349 |
| Chunks planned | 2,506 |
| Chunks indexed | 0 (dry-run) |
| Mirrored threads deliberately accepted as duplicates | 34 |
| Messages skipped | 0 |

The 34 mirrored threads are an accepted duplication cost, not a defect to
optimize. The vector-store path honors an explicit per-chunk duplicate opt-out
and does not add a source-specific dedup heuristic. Re-running the same stable
T3 IDs remains upsert-idempotent while distinct T3 messages with equal content
remain distinct rows.

## Verification

Passed:

- `pytest -q tests/test_ingest_t3.py tests/test_vector_store_upsert_transactions.py tests/test_ingest_codex.py tests/test_watcher_provenance_ingest.py tests/test_phase2_plugin_queue.py::test_append_queue_event_is_safe_under_concurrent_process_writers`
  → **57 passed**, 101 warnings.
- Focused Ruff checks on all changed Python files.
- Live T3 read-only dry-run with the counts above.

The full `pytest -x -vv` run collected 3,770 tests and stopped at
`tests/test_phase2_plugin_queue.py::test_append_queue_event_is_safe_under_concurrent_process_writers`
at 68%. Pytest then exhausted file descriptors during temporary-directory
cleanup (`OSError: [Errno 24] Too many open files`). The failing process test
passes in isolation, and no file in that phase-2 queue surface was changed by
this task. This remains an environment/baseline verification caveat rather
than a claimed full-suite pass.

## Commit and review handoff

The deliverable is one branch commit containing the adapter, indexing/upsert
plumbing, behavioral tests, and this report. No push or PR was performed.

TASK_DONE
