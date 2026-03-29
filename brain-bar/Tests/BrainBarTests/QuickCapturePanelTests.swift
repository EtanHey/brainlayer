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

        XCTAssertEqual(model.placeholderText, "Search memory. Press Return to run.")
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

    private func makeDatabase(name: String) throws -> (BrainDatabase, String) {
        let path = NSTemporaryDirectory() + "brainbar-panel-\(name)-\(UUID().uuidString).db"
        return (BrainDatabase(path: path), path)
    }

    private func cleanupDatabase(_ db: BrainDatabase, path: String) {
        db.close()
        try? FileManager.default.removeItem(atPath: path)
    }
}
