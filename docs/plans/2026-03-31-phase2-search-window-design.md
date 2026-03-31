# Phase 2 Search Window Design

## Goal

Ship a Spotlight-style BrainBar search window on `Cmd+K` with sub-100ms typeahead, keyboard-first navigation, filter chips, FTS5-first retrieval, vector reranking, and Phase 1 result-card reuse.

## Current Context

- The current branch already has a quick-capture search mode inside `QuickCapturePanel.swift`.
- That flow does immediate synchronous `BrainDatabase.search()` calls on keystroke and renders a custom `LazyVStack` result list.
- The branch baseline is currently broken because `BrainBarApp.swift` references `BrainBarURLAction` and `HotkeyRouteStatus`, which exist in the Phase 1 formatting foundation but are not present here yet.
- Phase 1 also introduced `SearchResult`, `SearchResultCard`, and `SearchResultsList`, which this phase should reuse instead of inventing another result card.

## Recommended Approach

Build the search window as a separate `NSPanel(.nonactivatingPanel)` flow rather than evolving the quick-capture panel in place.

Why:

- It keeps `Cmd+K` search behavior isolated from quick-capture behavior.
- It allows a dedicated keyboard/focus model with `canBecomeKey` override and search-specific routing.
- It cleanly separates the phase-2 retrieval pipeline (`FTS5 -> candidate set -> vector rerank`) from the simpler quick-capture search path.

## Architecture

### 1. Shared prerequisites from Phase 1

Cherry-pick the Phase 1 formatting foundation commit that adds:

- `BrainBarSupport.swift`
- `SearchResult.swift`
- `SearchResultCard.swift`
- `SearchResultsList.swift`

This is both a dependency for the phase-2 UI and the fix for the current branch’s missing symbols.

### 2. New search-window flow

Add a dedicated search panel stack:

- `SearchPanelController.swift`
- `SearchViewModel.swift`
- `SearchFilters.swift`
- `SearchQueryActor.swift`

The panel controller owns window lifecycle, focus, centering, and dismissal. The view model owns query text, chip state, selection, and visible results. The actor owns cancellable query execution and serial ordering.

### 3. Retrieval pipeline

The retrieval path should be:

1. FTS5 prefix search against a tuned index
2. limit to roughly `50-100` lexical candidates
3. rerank only those candidates with vector similarity
4. return a small visible result set to the UI

The design explicitly avoids:

- `snippet()` during live typing
- vector full scans over the full embedding table
- synchronous DB work on the main actor

### 4. Database/index strategy

Phase 2 needs a new optimized search surface in `BrainDatabase`:

- FTS5 with `prefix='2 3 4'`
- tokenizer `unicode61 remove_diacritics 2`
- `ANALYZE` run after index creation/rebuild
- precomputed preview text stored in the indexed/searchable record path

The search window should operate on preview text already materialized in SQLite rows, not generated per keypress.

### 5. Result presentation

Reuse `SearchResultCard` from Phase 1 for each row.

The search window should not use `LazyVStack`; use a regular `VStack`/eager list container for the expected small result count because this is a measured performance constraint in the spec.

### 6. Keyboard behavior

Required behavior:

- `Cmd+K` opens the search panel
- search field gets keyboard focus immediately
- up/down changes selection
- return activates selected result
- escape dismisses
- filter chips are keyboard reachable but should not steal the initial search focus

### 7. Concurrency model

`SearchQueryActor` should be the single query coordinator with serial execution semantics:

- every new query invalidates the prior in-flight result
- stale responses never overwrite newer UI state
- DB and rerank work stay off the main actor

## Error Handling

- empty query returns an idle/empty state immediately
- DB/index errors surface through a non-crashing error state in the view model
- if vector reranking is unavailable, the lexical candidate set should still render rather than failing the panel

## Testing Strategy

### Baseline prerequisite

First restore the missing Phase 1 dependency set so the existing BrainBar package builds again.

### New tests

- panel controller behavior: non-activating panel config and keyboard eligibility
- filter state transitions
- query actor cancellation and stale-result suppression
- view model: selection, filters, query updates, activation
- DB lexical candidate limiting / preview behavior / index configuration where testable

### Verification

- `swift test --package-path brain-bar`
- `bash brain-bar/build-app.sh`

## Risks

- Cherry-picking the whole Phase 1 formatting foundation may introduce adjacent formatting/model changes beyond the minimum card reuse surface.
- sqlite-vec integration details may already live elsewhere in the repo; if not, the rerank implementation needs to degrade gracefully without blocking lexical search.
- The current quick-capture search path and the new search window may drift if they share too little presentation/model code; result shaping should converge on the shared `SearchResult` model.
