// QuickCaptureTests.swift — TDD tests for BrainBar Quick Capture.
//
// Tests the capture→store and search→results flows,
// hotkey manager configuration, and panel state management.

import XCTest
import SQLite3
@testable import BrainBar

final class QuickCaptureTests: XCTestCase {

    // MARK: - Capture → Store flow

    func testCaptureStoresChunkWithTags() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-capture-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        let result = try QuickCaptureController.capture(
            db: db,
            content: "Decision: Use SwiftUI for the quick capture panel",
            tags: ["decision", "ui"],
            importance: 8
        )

        XCTAssertFalse(result.chunkID.isEmpty, "Should return a chunk ID")
        XCTAssertTrue(result.formatted.contains("\u{2714}"), "Should contain checkmark in formatted output")
        XCTAssertTrue(result.formatted.contains(result.chunkID), "Formatted output should include chunk ID")
    }

    func testCaptureWithEmptyContentFails() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-capture-empty-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        XCTAssertThrowsError(try QuickCaptureController.capture(
            db: db,
            content: "",
            tags: [],
            importance: 5
        ), "Empty content should throw")
    }

    func testCaptureDefaultImportanceIs5() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-capture-default-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        let result = try QuickCaptureController.capture(
            db: db,
            content: "A simple note",
            tags: []
        )

        XCTAssertFalse(result.chunkID.isEmpty)
        // Verify stored with default importance by searching
        let found = try db.search(query: "simple note", limit: 1)
        XCTAssertEqual(found.first?["importance"] as? Double, 5.0)
    }

    // MARK: - Search → Results flow

    func testSearchReturnsFormattedResults() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-search-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        // Seed data
        try db.insertChunk(id: "qc-1", content: "BrainBar quick capture feature", sessionId: "s1", project: "brainlayer", contentType: "assistant_text", importance: 7)

        let results = try QuickCaptureController.search(db: db, query: "BrainBar quick capture", limit: 5)

        XCTAssertFalse(results.formatted.isEmpty, "Should return formatted text")
        XCTAssertTrue(results.formatted.contains("## Search results"), "Should have markdown header")
        XCTAssertGreaterThan(results.count, 0, "Should find at least one result")
    }

    func testSearchOverlayResultsUsePreviewTextNotFullContent() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-search-preview-\(UUID().uuidString).db"
        defer {
            try? FileManager.default.removeItem(atPath: tempDB)
            try? FileManager.default.removeItem(atPath: tempDB + "-wal")
            try? FileManager.default.removeItem(atPath: tempDB + "-shm")
        }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        let fullContent = "BrainBar overlay preview sentinel term should not load this full content column"
        let previewText = "Short preview from preview_text"
        try db.insertChunk(
            id: "preview-overlay-1",
            content: fullContent,
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )
        try updatePreviewText(path: tempDB, chunkID: "preview-overlay-1", previewText: previewText)

        let results = try QuickCaptureController.search(db: db, query: "sentinel", limit: 5)

        XCTAssertEqual(results.results.first?["chunk_id"] as? String, "preview-overlay-1")
        XCTAssertEqual(results.results.first?["content"] as? String, previewText)
        XCTAssertFalse(results.formatted.contains(fullContent))
        XCTAssertTrue(results.formatted.contains(previewText))
    }

    func testSearchKeepsExactChunkIDFallback() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-search-exact-id-\(UUID().uuidString).db"
        defer {
            try? FileManager.default.removeItem(atPath: tempDB)
            try? FileManager.default.removeItem(atPath: tempDB + "-wal")
            try? FileManager.default.removeItem(atPath: tempDB + "-shm")
        }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(
            id: "exact-id-lookup-1",
            content: "Chunk content does not mention its identifier",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )

        let results = try QuickCaptureController.search(db: db, query: "exact-id-lookup-1", limit: 5)

        XCTAssertEqual(results.results.first?["chunk_id"] as? String, "exact-id-lookup-1")
    }

    func testSearchKeepsExactChunkIDFallbackWhenCandidatesFillLimit() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-search-exact-id-limit-\(UUID().uuidString).db"
        defer {
            try? FileManager.default.removeItem(atPath: tempDB)
            try? FileManager.default.removeItem(atPath: tempDB + "-wal")
            try? FileManager.default.removeItem(atPath: tempDB + "-shm")
        }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(
            id: "candidate-hit-1",
            content: "exact-id-priority-1 appears only in this candidate text",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )
        try db.insertChunk(
            id: "exact-id-priority-1",
            content: "Chunk content does not mention its identifier",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )

        let results = try QuickCaptureController.search(db: db, query: "exact-id-priority-1", limit: 1)

        XCTAssertEqual(results.results.first?["chunk_id"] as? String, "exact-id-priority-1")
    }

    func testSearchPreservesTagsInMappedResults() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-search-tags-\(UUID().uuidString).db"
        defer {
            try? FileManager.default.removeItem(atPath: tempDB)
            try? FileManager.default.removeItem(atPath: tempDB + "-wal")
            try? FileManager.default.removeItem(atPath: tempDB + "-shm")
        }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        try db.insertChunk(
            id: "tagged-result-1",
            content: "Tagged result should keep tag metadata",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7,
            tags: "[\"phase-2-4\", \"search-perf\"]"
        )

        let results = try QuickCaptureController.search(db: db, query: "tagged metadata", limit: 5)

        XCTAssertEqual(results.results.first?["chunk_id"] as? String, "tagged-result-1")
        XCTAssertEqual(results.results.first?["tags"] as? String, "[\"phase-2-4\", \"search-perf\"]")
        XCTAssertFalse(results.formatted.contains("phase-2-4"))
    }

    func testSearchEmptyQueryReturnsEmpty() throws {
        let tempDB = NSTemporaryDirectory() + "brainbar-search-empty-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempDB) }
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }

        let results = try QuickCaptureController.search(db: db, query: "", limit: 5)
        XCTAssertEqual(results.count, 0)
    }

    // MARK: - Panel State

    func testPanelStateTransitions() {
        let state = QuickCapturePanelState()

        XCTAssertEqual(state.mode, .capture, "Default mode should be capture")
        XCTAssertFalse(state.isVisible, "Should start hidden")

        state.show()
        XCTAssertTrue(state.isVisible)

        state.switchMode(.search)
        XCTAssertEqual(state.mode, .search)

        state.dismiss()
        XCTAssertFalse(state.isVisible)
        XCTAssertEqual(state.mode, .capture, "Dismiss should reset to capture mode")
    }

    func testPanelToggle() {
        let state = QuickCapturePanelState()

        state.toggle()
        XCTAssertTrue(state.isVisible)

        state.toggle()
        XCTAssertFalse(state.isVisible)
    }

    // MARK: - Hotkey Configuration

    func testHotkeyManagerConfiguredForF4() {
        let gesture = GestureStateMachine()
        let manager = HotkeyManager(gesture: gesture)

        // F4 keycodes: 118 (standard), 129 (media)
        manager.configure(keycodes: [118, 129], useModifierMode: false)

        // Can't test tap creation without Input Monitoring permission,
        // but we can verify configuration doesn't crash
        XCTAssertNotNil(manager)
    }

    func testHotkeyPermissionStatusRequiresInputMonitoringAndAccessibility() {
        let missingInput = HotkeyPermissionStatus(inputMonitoringGranted: false, accessibilityGranted: true)
        XCTAssertFalse(missingInput.isSatisfied)
        XCTAssertEqual(missingInput.missingPermissionsMessage, "Input Monitoring")

        let missingAccessibility = HotkeyPermissionStatus(inputMonitoringGranted: true, accessibilityGranted: false)
        XCTAssertFalse(missingAccessibility.isSatisfied)
        XCTAssertEqual(missingAccessibility.missingPermissionsMessage, "Accessibility")

        let missingBoth = HotkeyPermissionStatus(inputMonitoringGranted: false, accessibilityGranted: false)
        XCTAssertFalse(missingBoth.isSatisfied)
        XCTAssertEqual(missingBoth.missingPermissionsMessage, "Input Monitoring and Accessibility")
    }

    func testDebouncerRejectsRepeatedKeyDownsWithin300Milliseconds() {
        let debouncer = HotkeyDebouncer(windowMs: 300)

        XCTAssertTrue(debouncer.shouldProcessKeyDown(at: Date(timeIntervalSinceReferenceDate: 10)))
        XCTAssertFalse(debouncer.shouldProcessKeyDown(at: Date(timeIntervalSinceReferenceDate: 10.2)))
        XCTAssertTrue(debouncer.shouldProcessKeyDown(at: Date(timeIntervalSinceReferenceDate: 10.31)))
    }

    func testHotkeyEventDecisionConsumesMatchedNonRepeatingF4Events() {
        let consumeDown = HotkeyEventDecision.make(
            type: .keyDown,
            keycode: 118,
            autorepeat: 0,
            targetKeycodes: [118, 129],
            useModifierMode: false,
            debouncer: HotkeyDebouncer(windowMs: 300),
            now: Date(timeIntervalSinceReferenceDate: 20)
        )
        XCTAssertTrue(consumeDown.matchesHotkey)
        XCTAssertTrue(consumeDown.shouldConsumeEvent)
        XCTAssertEqual(consumeDown.action, .keyDown)

        let consumeUp = HotkeyEventDecision.make(
            type: .keyUp,
            keycode: 118,
            autorepeat: 0,
            targetKeycodes: [118, 129],
            useModifierMode: false,
            debouncer: HotkeyDebouncer(windowMs: 300),
            now: Date(timeIntervalSinceReferenceDate: 21)
        )
        XCTAssertTrue(consumeUp.matchesHotkey)
        XCTAssertTrue(consumeUp.shouldConsumeEvent)
        XCTAssertEqual(consumeUp.action, .keyUp)
    }

    func testHotkeyEventDecisionPassesThroughNonMatchingAndDebouncedEvents() {
        let debouncer = HotkeyDebouncer(windowMs: 300)
        _ = HotkeyEventDecision.make(
            type: .keyDown,
            keycode: 118,
            autorepeat: 0,
            targetKeycodes: [118],
            useModifierMode: false,
            debouncer: debouncer,
            now: Date(timeIntervalSinceReferenceDate: 30)
        )

        let debounced = HotkeyEventDecision.make(
            type: .keyDown,
            keycode: 118,
            autorepeat: 0,
            targetKeycodes: [118],
            useModifierMode: false,
            debouncer: debouncer,
            now: Date(timeIntervalSinceReferenceDate: 30.1)
        )
        XCTAssertTrue(debounced.matchesHotkey)
        XCTAssertFalse(debounced.shouldConsumeEvent)
        XCTAssertEqual(debounced.action, .none)

        let nonMatching = HotkeyEventDecision.make(
            type: .keyDown,
            keycode: 96,
            autorepeat: 0,
            targetKeycodes: [118],
            useModifierMode: false,
            debouncer: HotkeyDebouncer(windowMs: 300),
            now: Date(timeIntervalSinceReferenceDate: 31)
        )
        XCTAssertFalse(nonMatching.matchesHotkey)
        XCTAssertFalse(nonMatching.shouldConsumeEvent)
        XCTAssertEqual(nonMatching.action, .none)
    }

    func testBrainBarHotkeyFailureMessageMentionsBothPermissions() {
        let message = BrainBarAppSupport.hotkeyPermissionFailureMessage(
            permissions: HotkeyPermissionStatus(inputMonitoringGranted: false, accessibilityGranted: false)
        )

        XCTAssertTrue(message.contains("Input Monitoring"))
        XCTAssertTrue(message.contains("Accessibility"))
    }

    // MARK: - Gesture State Machine

    func testGestureKeyDownTransitionsToWaiting() {
        let gesture = GestureStateMachine()
        XCTAssertEqual(gesture.state, .idle)

        gesture.handleKeyDown()
        XCTAssertEqual(gesture.state, .waitingForHoldThreshold)
    }

    func testGestureQuickReleaseTransitionsToDoubleTapWindow() {
        let gesture = GestureStateMachine()
        gesture.handleKeyDown()
        gesture.handleKeyUp()
        XCTAssertEqual(gesture.state, .waitingForDoubleTap)
    }

    func testGestureResetClearsState() {
        let gesture = GestureStateMachine()
        gesture.handleKeyDown()
        XCTAssertNotEqual(gesture.state, .idle)
        gesture.reset()
        XCTAssertEqual(gesture.state, .idle)
    }

    // MARK: - brainbar:// URL scheme

    func testBrainBarURLParsesToggle() {
        let url = URL(string: "brainbar://toggle")!
        XCTAssertEqual(BrainBarURLAction.parse(url: url), .toggle)
    }

    func testBrainBarURLParsesSearch() {
        let url = URL(string: "brainbar://search")!
        XCTAssertEqual(BrainBarURLAction.parse(url: url), .search)
    }

    func testBrainBarURLRejectsOtherSchemes() {
        let url = URL(string: "https://example.com")!
        XCTAssertNil(BrainBarURLAction.parse(url: url))
    }
}

private func updatePreviewText(path: String, chunkID: String, previewText: String) throws {
    var sqlite: OpaquePointer?
    let rc = sqlite3_open_v2(path, &sqlite, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil)
    guard rc == SQLITE_OK, let sqlite else {
        throw NSError(domain: "QuickCaptureTests", code: Int(rc))
    }
    defer { sqlite3_close(sqlite) }

    var stmt: OpaquePointer?
    let prepareRC = sqlite3_prepare_v2(sqlite, "UPDATE chunks SET preview_text = ? WHERE id = ?", -1, &stmt, nil)
    guard prepareRC == SQLITE_OK, let stmt else {
        throw NSError(domain: "QuickCaptureTests", code: Int(prepareRC))
    }
    defer { sqlite3_finalize(stmt) }

    let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
    sqlite3_bind_text(stmt, 1, previewText, -1, transient)
    sqlite3_bind_text(stmt, 2, chunkID, -1, transient)
    let stepRC = sqlite3_step(stmt)
    guard stepRC == SQLITE_DONE else {
        throw NSError(domain: "QuickCaptureTests", code: Int(stepRC))
    }
}
