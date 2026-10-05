# #1060 part 1: hybrid compact release size gate

Status: implementation verified against the #1058 synthetic hybrid fixture; review pending.

The default compact renderer labels metadata columns once and uses S/P for summary/preview.
Canonical IDs remain verbatim, including `rt-rollout--` identifiers; score precision remains
four decimals and dates remain YYYY-MM-DD. A scoped result leaves its project column empty
only when that exact project is implied by the request. A different or unknown project stays.
Summaries wholly contained in the visible preview are omitted; distinct summaries retain
the existing 100-character limit, and previews retain the existing 200-character limit.
Display fields escape pipes, backslashes and line breaks; canonical IDs stay verbatim.
The operation-receipt count reader recognizes the compact header and its shared legend.
The structured helper payload and full-detail rendering remain unchanged.

## Wire bytes for ten queries

| Stage | Scoped | Entity-matched | Delta from previous stage (scoped / entity) |
|---|---:|---:|---:|
| Pre-#1058 ceiling, original receipt | 27,440 | 29,110 | — |
| Current main 784e8833, reproduced | 30,190 | 31,200 | — |
| 1. Omit implied project label/value | 29,240 | 31,200 | -950 / 0 |
| 2. Short IDs: skipped, canonical IDs retained | 29,240 | 31,200 | 0 / 0 |
| 3. Suppress fully visible summary | 29,240 | 31,200 | 0 / 0 |
| 4. Collapse labels and framing | 27,310 | 28,670 | -1,930 / -2,530 |
| Margin below release ceiling | 130 | 440 | — |

Stage 4 uses an empty implied-project column rather than deleting a labeled segment, so its
delta includes that framing change. IDs in this fixture are already short. Shortening them
would not cover the gap and would add a database-wide resolver contract; no resolver changes
were needed. Both content fields are deliberately distinct in this fixture, so stage 3 saves
zero bytes. Score/date reductions and lossy options were unnecessary.

## Reproduce

```sh
BRAINLAYER_MCP_SOCKET=/tmp/size-gate-unused.sock \
BRAINLAYER_FORBID_BRAINBAR_SOCKET=1 \
MCP_DIET_RECEIPT=/tmp/size-gate-hybrid.json \
swift test --package-path brain-bar --filter MCPDietTests/testFixtureWireMeasurements
```

The default fixture now enables the injected hybrid helper, making both ceiling assertions run
in ordinary Swift CI. Set `MCP_DIET_HYBRID=0` explicitly for the separate local FTS measurement.
The fixture creates its own temporary database/socket, overrides both client subprocesses'
socket environment, and asserts 30 helper searches. The shipped stdio bridge drives all ten
original queries and measures complete UTF-8 JSON-RPC response bodies, including unchanged
structuredContent. It also round-trips all 100 compact row IDs through brain_expand.
No canonical DB, production socket, real embedding model, provider call, install, or deployment
is involved. The original pre-#1058 ceilings come from PR #1058's recorded hybrid receipt;
the current-main baseline and each implemented stage were measured locally on this lane.

RED evidence: scoped omission failed on both local and hybrid paths; summary de-duplication
failed on contained content; the original wire ceilings failed at 29,240 / 31,200 bytes before
framing changes. Existing tests retain canonical rt-rollout-- ID round-trip coverage.

## Validation and scope

- Final full Swift suite: 1,078 BrainBar tests (3 skips), plus 10 daemon tests; zero failures.
- Python formatter and stdio-bridge pytest checks: 78 passed.
- Local CodeRabbit: one minor delimiter-escaping finding, fixed after a failing regression test.
- The first full Swift run caught the changed header's operation-receipt count consumer;
  its parser was updated with positive and unknown-shape regressions before the final green run.
- Swift CI explicitly supplies Python 3.12 for the now-default hybrid helper fixture; the
  system Python 3.9 helper import was measured failing on the formatter's union annotations.
- Entry points: checked compact/default and full brain_search, brain_expand and receipt counting.
- Clients: checked the shipped Python stdio bridge over a guarded scratch BrainBar socket;
  installed Claude/Codex sessions and the production app were outside this lane.
- Providers: checked injected hybrid success and local FTS/fallback tests; no real provider.
- Contracts: checked canonical rt-rollout-- IDs, both content fields, unknown/different/scoped
  projects, source de-duplication and delimiter escaping; structuredContent is preserved.
- Reverse states: checked empty results, helper fallback and unknown receipt header handling.
- Connection modes: checked scratch sockets and both Swift targets, which share formatter/receipt
  source through symlinks; production socket/DB, installation and deployment were not touched.
- Docs: this receipt contains the table, reproducible command and scope; the tool schema/default
  is unchanged. This is a source/PR receipt, with lead-routed Opus review still required.
