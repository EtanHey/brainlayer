// StabilityFixTests.swift — RED tests for BrainBar stability fixes.
//
// TDD: Written before implementation.
// Covers: async store, search result dates, Enter key behavior, popover sizing.

import AppKit
import XCTest
@testable import BrainBar

// MARK: - (1) Store should be async (non-blocking)

final class AsyncStoreTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-async-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    func testStoreAsyncReturnsChunkID() async throws {
        let stored = try await db.storeAsync(
            content: "Test content",
            tags: ["test"],
            importance: 5,
            source: "unit-test"
        )
        XCTAssertFalse(stored.chunkID.isEmpty)
    }

    func testStoreAsyncContentRetrievable() async throws {
        let stored = try await db.storeAsync(
            content: "Async stored content for retrieval test",
            tags: ["async-test"],
            importance: 7,
            source: "unit-test"
        )
        XCTAssertFalse(stored.chunkID.isEmpty)
        // Verify the stored content is searchable
        let results = try db.search(query: "retrieval test", limit: 5)
        XCTAssertFalse(results.isEmpty)
    }
}

// MARK: - (2) Search results must include dates

final class SearchResultDateTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-date-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    func testSearchCandidateIncludesDate() throws {
        try db.insertChunk(
            id: "dated-chunk", content: "React Server Components architecture",
            sessionId: "s1", project: "brainlayer", contentType: "ai_code", importance: 7
        )
        let candidates = try db.searchCandidates(query: "React", limit: 5)
        XCTAssertFalse(candidates.isEmpty)
        XCTAssertFalse(candidates.first!.date.isEmpty, "Candidate must include a date")
    }

    func testSearchCandidateIncludesProject() throws {
        try db.insertChunk(
            id: "proj-chunk", content: "BrainLayer memory pipeline",
            sessionId: "s1", project: "brainlayer", contentType: "ai_code", importance: 5
        )
        let candidates = try db.searchCandidates(query: "memory pipeline", limit: 5)
        XCTAssertFalse(candidates.isEmpty)
        XCTAssertEqual(candidates.first!.project, "brainlayer")
    }

    func testSearchCandidateIncludesImportance() throws {
        try db.insertChunk(
            id: "imp-chunk", content: "Important decision about database",
            sessionId: "s1", project: "test", contentType: "ai_code", importance: 8
        )
        let candidates = try db.searchCandidates(query: "decision database", limit: 5)
        XCTAssertFalse(candidates.isEmpty)
        XCTAssertEqual(candidates.first!.importance, 8)
    }
}

// MARK: - (3) Enter in search should select result, not switch to capture

@MainActor
final class EnterKeySearchTests: XCTestCase {

    func testSubmitInSearchModeWithResultsSelectsFirst() throws {
        let tempDBPath = NSTemporaryDirectory() + "brainbar-enter-test-\(UUID().uuidString).db"
        let db = BrainDatabase(path: tempDBPath)
        defer {
            db.close()
            try? FileManager.default.removeItem(atPath: tempDBPath)
            try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
            try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        }

        try db.insertChunk(
            id: "c1", content: "React Server Components",
            sessionId: "s1", project: "test", contentType: "ai_code", importance: 7
        )

        let panelState = QuickCapturePanelState()
        let vm = QuickCaptureViewModel(db: db, panelState: panelState)

        // Switch to search mode and populate results
        vm.setMode(.search)
        vm.inputText = "React"
        vm.handleInputChange("React")

        // Verify we have results and no selection yet is OK — submit should select first
        guard !vm.results.isEmpty else {
            XCTFail("Search should have returned results")
            return
        }

        // Now press Enter — should copy result, NOT switch to capture mode
        vm.submit()

        // After submit in search mode with results: result should be activated
        // The mode should remain search OR the content should be copied to clipboard
        // It should NOT just switch to capture mode with the title in the input
        XCTAssertNotNil(vm.copiedResultID, "Enter on search result should copy it")
    }
}

// MARK: - (4) Popover size should be stable

final class PopoverSizeTests: XCTestCase {
    private var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-popover-size-\(UUID().uuidString).db"
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    @MainActor
    func testStatusPopoverViewFrameMatchesStableUtilityPanel() {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let popoverView = StatusPopoverView(collector: collector)
        _ = popoverView.view
        popoverView.view.layoutSubtreeIfNeeded()

        XCTAssertEqual(popoverView.view.frame.width, 560)
        XCTAssertEqual(popoverView.view.frame.height, 520)
    }
}
