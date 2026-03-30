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
