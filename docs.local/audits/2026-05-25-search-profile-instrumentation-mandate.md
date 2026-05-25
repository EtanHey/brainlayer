# Phase 2.4-G Search Profile Instrumentation Mandate

Status: instrumentation-only PR.

Etan reported search latency around 15 seconds after Phase 2.4-F/PR #320 was expected to make warm hybrid search fast. This PR deliberately does not optimize the search path. It adds opt-in timing logs so the next pass can identify the slow segment from data instead of speculation.

## Enablement

Set `BRAINLAYER_SEARCH_PROFILE=1` to emit verbose timing events. With the flag unset, the instrumentation is silent.

Each event is a single JSON object with:

- `ts`: UTC ISO timestamp
- `scope`: `search.brainbar`, `search.helper`, `search.mcp`, or the supplied search scope
- `step`: measured step name
- `query_id`: best-effort correlation ID
- `dur_ms`: duration for completed timed steps, when applicable

## Instrumented Points

1. BrainBar command bar keystroke debounce to `submitSearch`: `step=keystroke_submit`.
2. BrainBar MCP router dispatch into `handleBrainSearch`: `step=router_dispatch`.
3. BrainBar helper RPC start and completion: `step=helper_rpc_start`, `step=helper_rpc_done`.
4. Python helper/MCP embedding call: `step=embed`.
5. Python hybrid search leg: `step=hybrid_search`.
6. Hybrid internals: `step=binary_knn`, `step=float_rerank`, and `step=fts_merge`.
7. Helper startup warm state: `step=startup_warm_state`, including `warm_called`, `binary_index_available`, and `binary_knn_mmap_size`.
8. BrainBar result rendering completion: `step=render_done`.

## Reading Guidance

Run a real query with the flag enabled and group by `query_id`. Compare `dur_ms` across `embed`, `binary_knn`, `float_rerank`, `fts_merge`, `hybrid_search`, `helper_rpc_done`, and `render_done`. The largest duration is the first candidate for the follow-up performance fix.

Do not infer a performance fix from this PR alone. Use the captured production log lines first.
