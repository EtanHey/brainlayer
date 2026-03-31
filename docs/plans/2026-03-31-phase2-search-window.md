# Phase 2 Search Window Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a dedicated BrainBar search window with fast typeahead, FTS-first candidate retrieval, vector reranking, keyboard navigation, and shared Phase 1 result cards.

**Architecture:** First restore the missing Phase 1 formatting foundation so this branch builds and the shared search-card models exist. Then add an isolated `NSPanel(.nonactivatingPanel)` search flow driven by a `SearchQueryActor`, with the UI layered on top of optimized lexical candidate retrieval and optional vector reranking over only a bounded candidate set.

**Tech Stack:** Swift 5.9, AppKit, SwiftUI, SQLite3/FTS5, sqlite-vec integration hooks in BrainLayer DB access, XCTest.

---

### Task 1: Restore Phase 1 shared search foundation

**Files:**
- Cherry-pick/update: `brain-bar/Sources/BrainBar/BrainBarSupport.swift`
- Cherry-pick/update: `brain-bar/Sources/BrainBar/Models/SearchResult.swift`
- Cherry-pick/update: `brain-bar/Sources/BrainBar/Views/Components/SearchResultCard.swift`
- Cherry-pick/update: `brain-bar/Sources/BrainBar/Views/Components/SearchResultsList.swift`
- Test: existing BrainBar package build/tests

**Step 1: Verify the branch fails before the prerequisite**

Run: `swift test --package-path brain-bar`
Expected: FAIL because `HotkeyRouteStatus` and `BrainBarURLAction` are missing from the current branch.

**Step 2: Cherry-pick the approved Phase 1 foundation**

Apply the shared search/result-card foundation from commit `4f9fc991b0b6c0f26bfb0bc431f6bfc42117e13b`.

**Step 3: Re-run package tests**

Run: `swift test --package-path brain-bar`
Expected: package compiles again, or failures move past the missing-symbol baseline blocker.

### Task 2: Add search query actor red-green cycle

**Files:**
- Create: `brain-bar/Sources/BrainBar/SearchQueryActor.swift`
- Create: `brain-bar/Tests/BrainBarTests/SearchQueryActorTests.swift`

**Step 1: Write the failing tests**

Cover:
- serial execution ordering
- stale query suppression
- cancellation behavior when a newer query arrives
- bounded candidate rerank input

**Step 2: Run test to verify it fails**

Run: `swift test --package-path brain-bar --filter SearchQueryActorTests`
Expected: FAIL because the actor does not exist yet.

**Step 3: Write minimal implementation**

Implement the actor with a query token/generation model and injected search backend closures.

**Step 4: Run test to verify it passes**

Run: `swift test --package-path brain-bar --filter SearchQueryActorTests`
Expected: PASS

### Task 3: Add search filters model

**Files:**
- Create: `brain-bar/Sources/BrainBar/SearchFilters.swift`
- Create: `brain-bar/Tests/BrainBarTests/SearchFiltersTests.swift`

**Step 1: Write the failing tests**

Cover:
- default chip state
- toggling chips
- converting chip state into DB search parameters
- keyboard-safe deterministic ordering of chips

**Step 2: Run test to verify it fails**

Run: `swift test --package-path brain-bar --filter SearchFiltersTests`
Expected: FAIL because the filter model does not exist yet.

**Step 3: Write minimal implementation**

Implement a small immutable or value-semantic filter surface that the view model and actor can share.

**Step 4: Run test to verify it passes**

Run: `swift test --package-path brain-bar --filter SearchFiltersTests`
Expected: PASS

### Task 4: Add search view model

**Files:**
- Create: `brain-bar/Sources/BrainBar/SearchViewModel.swift`
- Create: `brain-bar/Tests/BrainBarTests/SearchViewModelTests.swift`

**Step 1: Write the failing tests**

Cover:
- empty-query idle state
- typeahead requests flowing into `SearchQueryActor`
- filter changes re-querying
- selection and keyboard movement
- activating a result

**Step 2: Run test to verify it fails**

Run: `swift test --package-path brain-bar --filter SearchViewModelTests`
Expected: FAIL because the view model does not exist yet.

**Step 3: Write minimal implementation**

Implement the observable state holder and keep all expensive work delegated to the actor.

**Step 4: Run test to verify it passes**

Run: `swift test --package-path brain-bar --filter SearchViewModelTests`
Expected: PASS

### Task 5: Add non-activating search panel controller

**Files:**
- Create: `brain-bar/Sources/BrainBar/SearchPanelController.swift`
- Create: `brain-bar/Tests/BrainBarTests/SearchPanelControllerTests.swift`
- Modify: `brain-bar/Sources/BrainBar/BrainBarApp.swift`

**Step 1: Write the failing tests**

Cover:
- panel uses `.nonactivatingPanel`
- panel overrides `canBecomeKey`
- show/dismiss lifecycle
- initial focus request behavior
- app wiring from `Cmd+K` / URL route into the search panel

**Step 2: Run test to verify it fails**

Run: `swift test --package-path brain-bar --filter SearchPanelControllerTests`
Expected: FAIL because the controller does not exist yet.

**Step 3: Write minimal implementation**

Build the isolated panel controller and wire it into `BrainBarApp` without disturbing quick capture.

**Step 4: Run test to verify it passes**

Run: `swift test --package-path brain-bar --filter SearchPanelControllerTests`
Expected: PASS

### Task 6: Extend DB search path for phase-2 candidate retrieval

**Files:**
- Modify: `brain-bar/Sources/BrainBar/BrainDatabase.swift`
- Create: `brain-bar/Tests/BrainBarTests/SearchDatabaseTests.swift`

**Step 1: Write the failing tests**

Cover:
- FTS prefix matching behavior
- candidate limiting before rerank
- precomputed preview text usage
- no live `snippet()` dependency
- lexical search still works when reranking is unavailable

**Step 2: Run test to verify it fails**

Run: `swift test --package-path brain-bar --filter SearchDatabaseTests`
Expected: FAIL because the phase-2 search API/config does not exist yet.

**Step 3: Write minimal implementation**

Add a dedicated phase-2 search API with:
- tuned FTS5 setup
- bounded candidate retrieval
- rerank hook over only the candidate list
- preview-text output compatible with `SearchResult`

**Step 4: Run test to verify it passes**

Run: `swift test --package-path brain-bar --filter SearchDatabaseTests`
Expected: PASS

### Task 7: Integrate shared result cards and search UI

**Files:**
- Modify: `brain-bar/Sources/BrainBar/Views/Components/SearchResultsList.swift`
- Modify: `brain-bar/Sources/BrainBar/Views/Components/SearchResultCard.swift`
- Modify: `brain-bar/Sources/BrainBar/SearchPanelController.swift`
- Test: `brain-bar/Tests/BrainBarTests/SearchPanelControllerTests.swift`

**Step 1: Write the failing tests**

Cover:
- search panel renders shared `SearchResultCard`
- results list does not use `LazyVStack`
- selected/copy/activation state renders through the shared result components

**Step 2: Run test to verify it fails**

Run: `swift test --package-path brain-bar --filter SearchPanelControllerTests`
Expected: FAIL for the new shared-card list behavior.

**Step 3: Write minimal implementation**

Update the search UI to reuse the shared component set while honoring the performance constraint against `LazyVStack`.

**Step 4: Run test to verify it passes**

Run: `swift test --package-path brain-bar --filter SearchPanelControllerTests`
Expected: PASS

### Final verification

Run:
- `swift test --package-path brain-bar`
- `bash brain-bar/build-app.sh`

Expected:
- all BrainBar tests pass
- BrainBar app bundle builds successfully
