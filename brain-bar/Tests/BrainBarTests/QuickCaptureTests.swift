// QuickCaptureTests.swift — TDD tests for BrainBar Quick Capture.
//
// Tests the capture→store and search→results flows,
// hotkey manager configuration, and panel state management.

import XCTest
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
        XCTAssertTrue(results.formatted.contains("\u{250c}"), "Should have box-drawing header")
        XCTAssertGreaterThan(results.count, 0, "Should find at least one result")
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
}
