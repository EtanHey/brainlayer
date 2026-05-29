import XCTest
@testable import BrainBar

// QA #51: "copy to continue thread" must use Claude Code's resumable
// conversation UUID, not BrainLayer's internal session_id.
final class InjectionContinuationTests: XCTestCase {
    func testResumeCommandUsesConversationID() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(conversationID: "3679128a-f371-445f-82ba-b3946e2f20b6", fallbackSessionID: "session-abc"),
            "claude --resume 3679128a-f371-445f-82ba-b3946e2f20b6"
        )
    }

    func testResumeCommandTrimsWhitespace() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(conversationID: "  3679128a-f371-445f-82ba-b3946e2f20b6\n"),
            "claude --resume 3679128a-f371-445f-82ba-b3946e2f20b6"
        )
    }

    func testResumeCommandFallsBackToSessionIDWhenConversationMissing() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(conversationID: "   ", fallbackSessionID: "sess-42"),
            "claude --resume sess-42"
        )
    }

    func testResumeCommandFallsBackToContinueWhenSessionMissing() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(conversationID: "   "),
            "claude --continue"
        )
    }

    func testResumeCommandRejectsInvalidConversationID() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(conversationID: "not-a-uuid", fallbackSessionID: "sess-42"),
            "claude --resume sess-42"
        )
    }

    func testResumeCommandRejectsUnsafeFallbackSessionID() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(conversationID: "not-a-uuid", fallbackSessionID: "sess-42; rm -rf /"),
            "claude --continue"
        )
    }
}
