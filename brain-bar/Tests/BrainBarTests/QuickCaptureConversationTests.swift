import XCTest
@testable import BrainBar

// QA #36/#37: a single click on a search hit must drill in — open the full
// conversation thread with the hit chunk highlighted — instead of doing nothing.
// These exercise the QuickCapture drill-in view-model path.
@MainActor
final class QuickCaptureConversationTests: XCTestCase {
    private func makeSeededViewModel() throws -> (QuickCaptureViewModel, BrainDatabase, String) {
        let path = NSTemporaryDirectory() + "brainbar-qc-convo-\(UUID().uuidString).db"
        let db = BrainDatabase(path: path)

        // A small thread so expandedConversation has target + surrounding context.
        try db.insertChunk(id: "c1", content: "user asks the first question", sessionId: "sess", project: "brainlayer", contentType: "user_message", importance: 5)
        try db.insertChunk(id: "c2", content: "assistant explains the xylophone behavior", sessionId: "sess", project: "brainlayer", contentType: "assistant_text", importance: 6)
        try db.insertChunk(id: "c3", content: "user replies with a follow-up", sessionId: "sess", project: "brainlayer", contentType: "user_message", importance: 5)

        let panelState = QuickCapturePanelState()
        panelState.switchMode(.search)
        let model = QuickCaptureViewModel(db: db, panelState: panelState)
        model.inputText = "xylophone"
        model.submit()
        return (model, db, path)
    }

    func testOpenConversationLoadsThreadForSearchHit() throws {
        let (model, db, path) = try makeSeededViewModel()
        defer { db.close(); try? FileManager.default.removeItem(atPath: path) }

        let hitID = try XCTUnwrap(model.results.first?.id, "search should surface the matching chunk")
        XCTAssertNil(model.conversationSelection.conversation, "no thread open before drilling in")

        model.openConversation(id: hitID)

        let conversation = try XCTUnwrap(model.conversationSelection.conversation)
        XCTAssertEqual(conversation.target.chunkID, hitID)
        XCTAssertTrue(conversation.entries.contains { $0.isTarget }, "the hit chunk is highlighted in the thread")
        XCTAssertEqual(model.selectedResultID, hitID, "drilling in also selects the row")
    }

    func testCloseConversationClearsSelection() throws {
        let (model, db, path) = try makeSeededViewModel()
        defer { db.close(); try? FileManager.default.removeItem(atPath: path) }

        let hitID = try XCTUnwrap(model.results.first?.id)
        model.openConversation(id: hitID)
        XCTAssertNotNil(model.conversationSelection.conversation)

        model.closeConversation()
        XCTAssertNil(model.conversationSelection.conversation)
    }

    func testOpenConversationForUnknownIDIsANoOp() throws {
        let (model, db, path) = try makeSeededViewModel()
        defer { db.close(); try? FileManager.default.removeItem(atPath: path) }

        model.openConversation(id: "does-not-exist")
        XCTAssertNil(model.conversationSelection.conversation)
    }
}
