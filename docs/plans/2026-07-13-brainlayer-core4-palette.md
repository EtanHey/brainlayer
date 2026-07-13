# BrainLayer Core-4 Palette Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Core 4 the default BrainLayer MCP palette on BrainBar and Python, with full/operator overrides and an idempotent mid-session `expand_palette` upgrade.

**Architecture:** Keep each transport's canonical tool definitions and dispatch handlers unchanged. Add a small stateful palette controller at each `tools/list`/`tools/call` seam; core sessions expose four domain tools plus a minimal expansion control, while full/operator sessions expose the pre-change complete inventory.

**Tech Stack:** Swift 6/XCTest, Python 3.13/pytest, MCP JSON-RPC, compact JSON measurement, `tiktoken` `o200k_base`.

---

### Task 1: Pin BrainBar profile and budget behavior

**Files:**
- Modify: `brain-bar/Tests/BrainBarTests/MCPRouterTests.swift`
- Modify: `brain-bar/Sources/BrainBar/MCPRouter.swift`

**Step 1: Write the failing profile tests**

Add tests that construct `MCPRouter(profile: ...)`, call `tools/list`, and assert:

```swift
let coreNames = ["brain_search", "brain_store", "brain_recall", "brain_expand", "expand_palette"]
XCTAssertEqual(toolNames(router: MCPRouter(profile: nil)), coreNames)
XCTAssertEqual(toolNames(router: MCPRouter(profile: "core")), coreNames)

for profile in ["full", "operator"] {
    XCTAssertEqual(toolNames(router: MCPRouter(profile: profile)).count, 17)
}
```

Also cover blank and invalid profiles, asserting both fail closed to the same core list. Move existing canonical schema/annotation assertions to `MCPRouter.toolDefinitions` or a full-profile router so they continue to pin all 17 definitions.

Serialize the core `tools` array with sorted compact JSON and assert `data.count <= 1_500`.

**Step 2: Run the focused Swift tests and verify RED**

Run:

```bash
cd brain-bar
swift test --filter MCPRouterTests
```

Expected: compilation/test failures because `MCPRouter(profile:)` and palette filtering do not exist.

**Step 3: Implement the minimal BrainBar profile controller**

In `MCPRouter.swift`:

- add `BRAINLAYER_MCP_PROFILE` resolution in `init`, with an optional explicit profile for tests;
- represent `core`, `full`, and `operator` without modifying `toolDefinitions`;
- add a lock-safe `PaletteSession`, use one router-local default for direct callers, and attach a distinct session to each BrainBar socket client;
- compute compact core projections from the canonical Core 4 (preserving validation fields while dropping verbose descriptive metadata) plus a minimal control definition, or all 17 canonical definitions unchanged;
- make `handleToolsList` return `exposedToolDefinitions`;
- make the existing call guard validate against exposed definitions;
- set `tools.listChanged` to true.

Unknown and blank values resolve to core. Log a warning only for a nonblank unknown environment value.

**Step 4: Run focused tests and verify GREEN**

Run `cd brain-bar && swift test --filter MCPRouterTests`.

Expected: all `MCPRouterTests` pass with zero failures.

**Step 5: Commit the BrainBar profile seam**

```bash
git add brain-bar/Sources/BrainBar/MCPRouter.swift brain-bar/Tests/BrainBarTests/MCPRouterTests.swift
git commit -m "feat: add BrainBar core4 MCP profile"
```

### Task 2: Add BrainBar expansion round-trip

**Files:**
- Modify: `brain-bar/Tests/BrainBarTests/MCPRouterTests.swift`
- Modify: `brain-bar/Sources/BrainBar/MCPRouter.swift`

**Step 1: Write failing expansion tests**

Exercise one core router instance:

```swift
XCTAssertEqual(listedNames(router).count, 5)
let first = call(router, name: "expand_palette")
XCTAssertEqual(first["expanded"] as? Bool, true)
XCTAssertEqual(listedNames(router).count, 17)
let second = call(router, name: "expand_palette")
XCTAssertEqual(second["already_expanded"] as? Bool, true)
XCTAssertEqual(listedNames(router).count, 17)
```

Also prove a deferred no-database tool such as `brain_tags` receives `-32601` before expansion and reaches its existing handler after expansion (returning a normal tool result, even if that result reports missing database).

**Step 2: Run focused tests and verify RED**

Run `cd brain-bar && swift test --filter MCPRouterTests`.

Expected: failure because the expansion control is listed but not dispatched and does not mutate the palette.

**Step 3: Implement idempotent expansion**

Handle `expand_palette` before canonical schema validation. On the first call, set `paletteExpanded = true` and return content plus top-level metadata:

```swift
[
    "expanded": true,
    "already_expanded": false,
    "registered_tools": deferredNames,
]
```

Later calls return `expanded: false`, `already_expanded: true`, and an empty registered list. The canonical switch and all 17 handlers remain unchanged.

**Step 4: Run focused tests and verify GREEN**

Run `cd brain-bar && swift test --filter MCPRouterTests`.

Expected: all focused tests pass.

**Step 5: Commit the expansion path**

```bash
git add brain-bar/Sources/BrainBar/MCPRouter.swift brain-bar/Tests/BrainBarTests/MCPRouterTests.swift
git commit -m "feat: expand BrainBar palette in session"
```

### Task 3: Pin the Python palette contract

**Files:**
- Create: `tests/test_mcp_palette.py`
- Create: `src/brainlayer/mcp/palette.py`
- Modify: `src/brainlayer/mcp/__init__.py`

**Step 1: Write failing controller and transport tests**

Test a new `ToolPalette` directly and the MCP callbacks through an injected/reset palette fixture:

```python
CORE = {"brain_search", "brain_store", "brain_recall", "brain_expand", "expand_palette"}

@pytest.mark.parametrize("profile", [None, "", "core", "bogus"])
def test_core_profiles_fail_closed(profile):
    palette = ToolPalette(profile)
    assert {tool.name for tool in palette.expose(full_tools())} == CORE

@pytest.mark.parametrize("profile", ["full", "operator"])
def test_full_profiles_preserve_all_python_tools(profile):
    assert len(ToolPalette(profile).expose(full_tools())) == 13
```

Add an async round-trip test proving the first expansion exposes all 13 tools, the second expansion is a no-op, and a deferred call is rejected before expansion.

**Step 2: Run the focused Python tests and verify RED**

Run:

```bash
pytest tests/test_mcp_palette.py -q
```

Expected: import/attribute failures because the palette controller does not exist.

**Step 3: Implement the Python palette controller**

Create `palette.py` with:

- `PROFILE_ENV = "BRAINLAYER_MCP_PROFILE"`;
- immutable Core 4 names;
- normalized `core`/`full`/`operator` resolution with invalid fail-closed behavior;
- `expose(full_tools)` that selects existing `Tool` objects and appends a minimal control in unexpanded core mode;
- `is_exposed(name)` and idempotent `expand()` returning a receipt dictionary.

Refactor the current literal `list_tools` body into `_full_tool_definitions()`. Keep the decorated callback thin:

```python
@server.list_tools()
async def list_tools() -> list[Tool]:
    return _tool_palette.expose(_full_tool_definitions())
```

At the start of `call_tool`, handle `expand_palette`, then reject other currently deferred names before entering the existing branch chain. Do not modify any existing handler body.

**Step 4: Run focused tests and verify GREEN**

Run `pytest tests/test_mcp_palette.py -q`.

Expected: all palette tests pass.

**Step 5: Run nearby Python contract tests**

Run:

```bash
pytest tests/test_mcp_input_schema_limits.py tests/test_mcp_labeled_field_output.py tests/test_mcp_digest_modes.py tests/test_smart_search_entity_dedup.py -q
```

Expected: zero failures. Update tests that intentionally inspect the complete inventory to use the full profile fixture; do not weaken canonical schema assertions.

**Step 6: Commit the Python seam**

```bash
git add src/brainlayer/mcp/__init__.py src/brainlayer/mcp/palette.py tests/test_mcp_palette.py tests/test_mcp_input_schema_limits.py tests/test_mcp_labeled_field_output.py tests/test_mcp_digest_modes.py tests/test_smart_search_entity_dedup.py
git commit -m "feat: add Python core4 MCP profile"
```

### Task 4: Document parity and measure the palette

**Files:**
- Modify: `contracts/engine-ui-contract.md`
- Create: `scripts/measure_mcp_palette.py`
- Create: `tests/test_measure_mcp_palette.py`

**Step 1: Write a failing measurement test**

Add a pure helper test with a small fixed schema fixture. Assert compact JSON uses UTF-8 byte length and `o200k_base` tokens, and assert the live BrainBar core payload is at most 1,500 bytes.

**Step 2: Run the measurement tests and verify RED**

Run `pytest tests/test_measure_mcp_palette.py -q`.

Expected: failure because the script/helper does not exist.

**Step 3: Implement the measurement script**

The script must accept JSON from a file/stdin, extract the `tools` array when given a JSON-RPC envelope, serialize with compact separators and sorted keys, and print exact bytes/tokens. Use `tiktoken.get_encoding("o200k_base")` and exit nonzero when `--max-bytes` is exceeded.

**Step 4: Update the transport contract**

Document:

- `BRAINLAYER_MCP_PROFILE=core|full|operator`;
- core default and its four domain names;
- `expand_palette` semantics;
- BrainBar's 17-tool and Python's 13-tool full inventories remain canonical;
- profile state is per session and invalid values fail closed.

**Step 5: Run tests and record measurements**

Run:

```bash
pytest tests/test_measure_mcp_palette.py tests/test_mcp_palette.py -q
cd brain-bar && swift test --filter MCPRouterTests
```

Generate compact full/core `tools` JSON for BrainBar and run the measurement script on both. Record exact before/after bytes and tokens for the receipt.

**Step 6: Commit docs and measurement tooling**

```bash
git add contracts/engine-ui-contract.md scripts/measure_mcp_palette.py tests/test_measure_mcp_palette.py
git commit -m "test: enforce BrainLayer palette budget"
```

### Task 5: Full verification and PR loop

**Files:**
- Modify: `~/Gits/orchestrator/docs.local/collab/driver-buddy-2026-07-12.md`

**Step 1: Run complete local verification**

Run fresh:

```bash
pytest tests/ -m "not integration and not live" --tb=short
cd brain-bar && swift test
git diff --check origin/main...HEAD
```

Read the complete summaries and report exact pass/fail/skip counts. Do not claim green from exit codes alone.

**Step 2: Re-read the handoff and design acceptance gates**

Verify line by line: Core 4 default, 13 deferred on BrainBar, full/operator override, mid-session expansion, no removal/rename, no DB/schema change, per-profile tests, <=1.5 KB core budget, exact measurements.

**Step 3: Push and open a ready PR**

```bash
git push -u origin feat/core4-palette
gh pr create --title "feat: add BrainLayer Core-4 MCP palette" --body-file <prepared-body>
```

The PR body includes the design, exact before/after bytes/tokens, test receipts, and risk note that only MCP exposure changes.

**Step 4: Request required reviews**

Post `@codex review`, `@cursor`, and `@bugbot` on the ready PR. Inspect CI with `gh pr checks --watch` and inspect all review threads. Fix or explicitly answer every actionable/high/critical item, with no more than three review rounds.

**Step 5: Write completion receipts**

Append to `~/Gits/orchestrator/docs.local/collab/driver-buddy-2026-07-12.md` under `@orc-driver-v2` with:

- PR URL and head SHA;
- exact full/core bytes and tokens;
- local and CI test counts;
- review status;
- explicit note that DB/schema were untouched.

Store the same milestone in BrainLayer with WHAT + WHY, then search it back to verify persistence. Stop at the handoff's worker endpoint (ready PR + reviews addressed); do not merge unless separately instructed.

TASK_DONE
