# BrainLayer Core-4 Palette Design

## Goal

Reduce the default BrainLayer MCP boot surface to the four tools that cover 98.62% of observed calls while preserving every existing handler and allowing a session to expose the full palette without reconnecting.

## Context

The live BrainLayer surface is BrainBar's 17-tool MCP router. The Python MCP server is a secondary transport with 13 tools, including the Python-only `brain_resume`. The measured live schema is 4,038 bytes / 1,027 `o200k_base` tokens; the four canonical schemas (`brain_search`, `brain_store`, `brain_recall`, `brain_expand`) are 1,263 bytes / 326 tokens before adding the small expansion control.

Both transports therefore need the same profile semantics, but each keeps its existing canonical inventory and dispatch implementation.

## Considered approaches

### 1. Stateful per-session palette with one-shot expansion (chosen)

Each server process resolves `BRAINLAYER_MCP_PROFILE` once at construction/import time. `core` lists the Core 4 plus `expand_palette`; `full` and `operator` list the transport's complete canonical inventory. BrainBar tracks expansion per socket client, while the Python stdio server has one module session. Calling `expand_palette` changes only that MCP session to full and makes its subsequent `tools/list` calls return the full inventory.

This directly mirrors cmuxlayer's proven seam, meets the boot budget, and supports a mid-session upgrade. The trade-off is a small amount of instance state and a `tools/list_changed` capability declaration even though BrainBar does not currently push that notification.

### 2. Reconnect-only profile switching

Keep `tools/list` stateless and require a reconnect with `BRAINLAYER_MCP_PROFILE=full`. This is simpler but fails the explicit self-upgrade requirement.

### 3. Advertise all tools with deferred metadata

Return all schemas and mark non-core tools with `defer_loading`. This preserves direct calls but does not reduce the measured `tools/list` bytes, so it fails the primary acceptance gate.

## Profile contract

- `core` is the default when the environment variable is missing or blank.
- `full` exposes the complete transport inventory.
- `operator` is a documented alias of `full` for operator sessions.
- Unknown profile values fail closed to `core` and log one warning; they never silently expose operator tools.
- Profile resolution is per server session. Changing the process environment after construction does not mutate an existing session.

The resident core is exactly:

1. `brain_search`
2. `brain_store`
3. `brain_recall`
4. `brain_expand`

`expand_palette` is a control tool and is not counted as a BrainLayer domain tool. It appears only while the server is in an unexpanded core profile.

## BrainBar design

`MCPRouter` receives an optional profile override for deterministic tests and otherwise reads `BRAINLAYER_MCP_PROFILE` during initialization. It stores a boolean expanded state behind the router instance.

`tools/list` derives compact core declarations from the immutable canonical `toolDefinitions` array and appends a minimal `expand_palette` definition. The projection keeps names and validation-relevant input-schema fields while removing verbose nested descriptions/annotations and shortening top-level descriptions. This was required because the current in-repo Core 4 serialized to 3,722 bytes even though the older live-front audit payload was 1,263 bytes. Full/operator selection returns all 17 canonical definitions byte-for-byte and omits the redundant control.

`tools/call` validates against the currently exposed definitions. A deferred tool called before expansion receives the existing unknown-tool JSON-RPC error, including BrainBar's server-handled subscription tools. `expand_palette` is handled without touching the database, flips the client session once, and returns an idempotent structured receipt containing `expanded`, `already_expanded`, and the newly exposed names. Subsequent calls are successful no-ops. BrainBar keeps the expansion state in each `ClientState`; direct router users receive one router-local default session.

`initialize` advertises `tools.listChanged = true` because the list can change during a session. BrainBar's request/response router has no server-push channel at this seam, so clients discover the new list on the next `tools/list`; the expansion response makes that contract explicit.

## Python design

The Python MCP module keeps one immutable full-tool builder and adds a small palette controller resolved from `BRAINLAYER_MCP_PROFILE`. The registered `list_tools` callback returns Core 4 plus `expand_palette` by default, or the existing 13-tool inventory for full/operator/expanded sessions.

The existing `call_tool` branches remain intact. A profile exposure guard rejects deferred names before dispatch, while `expand_palette` updates only the module server session and returns the same idempotent receipt shape as BrainBar. No aliases are renamed or removed; Python-only `brain_resume` remains available in full/operator mode.

## Compatibility and safety

- No database, schema, migration, search, or write-path code changes.
- No canonical tool definition or handler is deleted.
- Core schemas are mechanically projected from the existing definitions, keeping validation-relevant fields tied to the canonical source while dropping only boot-cost metadata.
- Full/operator output is contract-compatible with today's complete inventory.
- `brain_expand` remains resident despite its Python deprecation notice because the live corpus and handoff explicitly require Core 4 for this release.
- Unknown profiles fail closed to core.

## Verification

Tests will prove, on both transports:

- missing/blank/`core` profiles list exactly Core 4 plus the control tool;
- `full` and `operator` list the complete pre-change inventory;
- invalid profiles fail closed;
- deferred calls fail before expansion and dispatch after expansion;
- expansion is idempotent and round-trips from core to full;
- canonical definitions remain present and unchanged;
- the serialized default BrainBar `tools` array stays below 1,500 bytes;
- measured full and core bytes/tokens are reported with the same compact JSON and `o200k_base` method used by the audit.

Focused Python and Swift tests run first, followed by the full non-live Python suite and the full BrainBar Swift suite. The final receipt records measured before/after bytes and token counts.

## Non-goals

- No Core-3 migration or folding `brain_expand` into `brain_search`.
- No launcher, daemon, database, schema, or writer changes.
- No deletion or renaming of compatibility tools.
- No attempt to make profile changes affect other sessions in the same process.
