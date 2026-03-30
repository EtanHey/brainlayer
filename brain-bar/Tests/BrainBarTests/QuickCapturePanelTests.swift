import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class QuickCapturePanelTests: XCTestCase {
    func testViewModelUsesCapturePlaceholderByDefault() throws {
        let (db, path) = try makeDatabase(name: "placeholder-capture")
        defer { cleanupDatabase(db, path: path) }

        let model = QuickCaptureViewModel(db: db, panelState: QuickCapturePanelState())

        XCTAssertEqual(model.placeholderText, "Capture an idea. Press Return to store.")
    }

    func testViewModelUsesSearchPlaceholderInSearchMode() throws {
        let (db, path) = try makeDatabase(name: "placeholder-search")
        defer { cleanupDatabase(db, path: path) }

        let panelState = QuickCapturePanelState()
        panelState.switchMode(.search)
        let model = QuickCaptureViewModel(db: db, panelState: panelState)

        XCTAssertEqual(model.placeholderText, "Search memory. Press Return to run or select.")
    }

    func testSubmitCaptureStoresChunkAndShowsConfirmation() throws {
        let (db, path) = try makeDatabase(name: "submit-capture")
        defer { cleanupDatabase(db, path: path) }

        let panelState = QuickCapturePanelState()
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.inputText = "Remember to verify the real MCP handshake"

        model.submit()

        XCTAssertEqual(model.feedback, .success("Stored in BrainLayer"))
        XCTAssertGreaterThan(model.confirmationFlashCount, 0)
        let results = try db.search(query: "real MCP handshake", limit: 5)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(model.inputText, "")
    }

    func testSubmitSearchPublishesResults() throws {
        let (db, path) = try makeDatabase(name: "submit-search")
        defer { cleanupDatabase(db, path: path) }
        try db.insertChunk(
            id: "search-1",
            content: "Quick capture panel should auto focus when shown",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )

        let panelState = QuickCapturePanelState()
        panelState.switchMode(.search)
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.inputText = "auto focus"

        model.submit()

        XCTAssertEqual(model.results.count, 1)
        XCTAssertTrue(model.feedback.isIdle)
        XCTAssertEqual(model.results.first?.title, "Quick capture panel should auto focus when shown")
        XCTAssertEqual(model.results.first?.id, "search-1", "Should use chunk_id from database, not generate random UUID")
        XCTAssertEqual(model.selectedResultID, "search-1", "Search should preselect the first result for keyboard navigation")
    }

    func testHandleInputChangeRunsSearchImmediatelyInSearchMode() throws {
        let (db, path) = try makeDatabase(name: "live-search")
        defer { cleanupDatabase(db, path: path) }
        try db.insertChunk(
            id: "live-1",
            content: "BrainBar live search should query while typing",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )

        let panelState = QuickCapturePanelState()
        panelState.switchMode(.search)
        let model = QuickCaptureViewModel(db: db, panelState: panelState)

        model.handleInputChange("live search")

        XCTAssertEqual(model.inputText, "live search")
        XCTAssertEqual(model.results.map(\.id), ["live-1"])
        XCTAssertEqual(model.selectedResultID, "live-1")
        XCTAssertTrue(model.feedback.isIdle)
    }

    func testTabTogglesBetweenCaptureAndSearchModes() throws {
        let (db, path) = try makeDatabase(name: "tab-toggle")
        defer { cleanupDatabase(db, path: path) }

        let model = QuickCaptureViewModel(db: db, panelState: QuickCapturePanelState())
        XCTAssertEqual(model.mode, .capture)

        model.toggleMode()
        XCTAssertEqual(model.mode, .search)

        model.toggleMode()
        XCTAssertEqual(model.mode, .capture)
    }

    func testArrowKeysMoveSelectedSearchResult() throws {
        let (db, path) = try makeDatabase(name: "arrow-navigation")
        defer { cleanupDatabase(db, path: path) }
        try db.insertChunk(
            id: "arrow-1",
            content: "Alpha memory",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        try db.insertChunk(
            id: "arrow-2",
            content: "Beta memory",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        let panelState = QuickCapturePanelState()
        panelState.switchMode(.search)
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.inputText = "memory"
        model.submit()

        XCTAssertEqual(model.selectedResultID, "arrow-1")

        model.moveSelectionDown()
        XCTAssertEqual(model.selectedResultID, "arrow-2")

        model.moveSelectionUp()
        XCTAssertEqual(model.selectedResultID, "arrow-1")
    }

    func testEnterSelectsHighlightedSearchResultIntoCaptureMode() throws {
        let (db, path) = try makeDatabase(name: "enter-selects-result")
        defer { cleanupDatabase(db, path: path) }
        try db.insertChunk(
            id: "enter-1",
            content: "First matching memory",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        try db.insertChunk(
            id: "enter-2",
            content: "Second matching memory",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        let panelState = QuickCapturePanelState()
        panelState.switchMode(.search)
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.inputText = "matching"
        model.submit()
        model.moveSelectionDown()

        model.submit()

        XCTAssertEqual(model.mode, .capture)
        XCTAssertEqual(model.inputText, "Second matching memory")
        XCTAssertEqual(model.results.count, 0)
        XCTAssertNil(model.selectedResultID)
    }

    func testCommandEnterForceStoresWhileRemainingInSearchMode() throws {
        let (db, path) = try makeDatabase(name: "force-store-search-mode")
        defer { cleanupDatabase(db, path: path) }

        let panelState = QuickCapturePanelState()
        panelState.switchMode(.search)
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.inputText = "Ship the keyboard-first quick capture flow"

        model.submit(forceCapture: true)

        XCTAssertEqual(model.mode, .search)
        XCTAssertEqual(model.feedback, .success("Stored in BrainLayer"))
        XCTAssertEqual(model.inputText, "")
        let results = try db.search(query: "keyboard-first quick capture", limit: 5)
        XCTAssertEqual(results.count, 1)
    }

    func testHandleInputReturnInCaptureModeStoresAndTriggersConfirmationFlash() throws {
        let (db, path) = try makeDatabase(name: "capture-return-flash")
        defer { cleanupDatabase(db, path: path) }

        let model = QuickCaptureViewModel(db: db, panelState: QuickCapturePanelState())
        model.inputText = "Return should store and flash green"

        model.handleInputReturn(modifiers: [])

        XCTAssertEqual(model.feedback, .success("Stored in BrainLayer"))
        XCTAssertEqual(model.confirmationFlashCount, 1)
        XCTAssertEqual(model.inputText, "")
        let results = try db.search(query: "flash green", limit: 5)
        XCTAssertEqual(results.count, 1)
    }

    func testPanelAppearanceRequestsFieldFocus() throws {
        let (db, path) = try makeDatabase(name: "focus-request")
        defer { cleanupDatabase(db, path: path) }

        let model = QuickCaptureViewModel(db: db, panelState: QuickCapturePanelState())
        XCTAssertEqual(model.focusRequestCount, 0)

        model.panelDidAppear()

        XCTAssertEqual(model.focusRequestCount, 1)
    }

    func testEscapeDismissesPanel() {
        let panel = QuickCapturePanel()
        var dismissCount = 0
        panel.onEscape = {
            dismissCount += 1
        }

        panel.cancelOperation(nil)

        XCTAssertEqual(dismissCount, 1)
    }

    func testDismissResetsViewModelBackToCaptureMode() throws {
        let (db, path) = try makeDatabase(name: "dismiss-reset")
        defer { cleanupDatabase(db, path: path) }

        let panelState = QuickCapturePanelState()
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.setMode(.search)

        model.dismiss()

        XCTAssertEqual(model.mode, .capture)
        XCTAssertEqual(model.placeholderText, "Capture an idea. Press Return to store.")
    }

    func testSubmitCaptureWithWhitespaceOnlyFails() throws {
        let (db, path) = try makeDatabase(name: "whitespace-capture")
        defer { cleanupDatabase(db, path: path) }

        let panelState = QuickCapturePanelState()
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.inputText = "   \n\t  "

        model.submit()

        XCTAssertEqual(model.feedback, .error("Content cannot be empty"))
        XCTAssertEqual(model.inputText, "   \n\t  ", "Should not clear input on error")
    }

    func testModeSwitchClearsResultsWhenSwitchingToCapture() throws {
        let (db, path) = try makeDatabase(name: "mode-switch-clear")
        defer { cleanupDatabase(db, path: path) }
        try db.insertChunk(
            id: "mode-1",
            content: "Test content for mode switch",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        let panelState = QuickCapturePanelState()
        panelState.switchMode(.search)
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.inputText = "mode switch"
        model.submit()

        XCTAssertGreaterThan(model.results.count, 0, "Should have search results")

        model.setMode(.capture)

        XCTAssertEqual(model.results.count, 0, "Should clear results when switching to capture mode")
        XCTAssertNil(model.selectedResultID, "Should clear the selected result when switching to capture mode")
        XCTAssertTrue(model.feedback.isIdle, "Should reset feedback")
    }

    private func makeDatabase(name: String) throws -> (BrainDatabase, String) {
        let path = NSTemporaryDirectory() + "brainbar-panel-\(name)-\(UUID().uuidString).db"
        return (BrainDatabase(path: path), path)
    }

    private func cleanupDatabase(_ db: BrainDatabase, path: String) {
        db.close()
        try? FileManager.default.removeItem(atPath: path)
    }
}
