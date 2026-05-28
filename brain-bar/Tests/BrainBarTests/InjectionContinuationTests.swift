import XCTest
@testable import BrainBar

// QA #51: "copy to continue thread … like the repo golems command". A burst maps
// to a Claude Code session; copying a resume command lets the user pick that exact
// thread back up (mirrors the repo-golem `-c` continue pattern).
final class InjectionContinuationTests: XCTestCase {
    func testResumeCommandUsesSessionID() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(sessionID: "abc123"),
            "claude --resume abc123"
        )
    }

    func testResumeCommandTrimsWhitespace() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(sessionID: "  sess-42\n"),
            "claude --resume sess-42"
        )
    }

    func testResumeCommandFallsBackToContinueWhenSessionMissing() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(sessionID: "   "),
            "claude --continue"
        )
    }
}
