# Phase 1 Findings

## Formatting Deltas

- Search result cards now have a typed `SearchResult` model with a separate compact metadata row (`project • date • imp • score`) for SwiftUI rendering. The terminal formatter still preserves Python `_format.py` structure for MCP text output.
- `brain_entity` in `MCPRouter` now follows Python's simple entity lookup structure rather than the richer card formatter. This matches the Python handler path that formats lookup results with relations, associated memories, and metadata.
- `brain_recall` stats required a small DB payload expansion. `BrainDatabase.recallStats()` now returns `projects` and `content_types` arrays so the Swift formatter can match Python's stats lines instead of showing count-only output.
- Quick Capture search results now render through the new shared `SearchResultsList`/`SearchResultCard` components by bridging existing `QuickCaptureSearchRow` data into `SearchResult`.

## Branch Repairs

- The worktree had existing compile blockers unrelated to formatting: `HotkeyRouteStatus` and `BrainBarURLAction` were referenced but missing. Minimal compatibility implementations were restored so phase-1 formatter tests could run.

## Residual Notes

- `brain-bar/build-app.sh` succeeds, but the release build still emits two pre-existing warnings in `BrainDatabase.swift` about unused bindings in `listTags(query:)` and `digest(content:)`.
