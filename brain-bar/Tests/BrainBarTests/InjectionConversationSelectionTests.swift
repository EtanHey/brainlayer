import XCTest
@testable import BrainBar

final class InjectionConversationSelectionTests: XCTestCase {
    func testCloseClearsConversationAndResetsTitle() {
        var selection = InjectionConversationSelection()
        let conversation = BrainDatabase.ExpandedConversation(
            target: BrainDatabase.ConversationChunk(
                chunkID: "chunk-1",
                content: "Stored context",
                contentType: "assistant_text",
                importance: 8,
                createdAt: "2026-05-27T00:00:00Z",
                summary: "Stored context summary",
                isTarget: true
            ),
            entries: []
        )

        selection.open(conversation, title: "Stored Memory")
        selection.close()

        XCTAssertNil(selection.conversation)
        XCTAssertEqual(selection.title, InjectionConversationSelection.defaultTitle)
    }
}
