import XCTest
import AppKit
import SwiftUI
@testable import BrainBar

// QA #51: "copy to continue thread" must use Claude Code's resumable
// conversation UUID, not BrainLayer's internal session_id.
final class InjectionContinuationTests: XCTestCase {
    func testResumeCommandUsesConversationID() {
        XCTAssertEqual(
            InjectionContinuation.resumeCommand(
                conversationID: "3679128a-f371-445f-82ba-b3946e2f20b6",
                fallbackSessionID: "session-abc",
                projectPath: "/Users/etanheyman/Gits/brainlayer"
            ),
            "cd /Users/etanheyman/Gits/brainlayer && claude --resume 3679128a-f371-445f-82ba-b3946e2f20b6"
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

    @MainActor
    func testRendersCwdAwareResumeCommandQAImage() throws {
        let command = InjectionContinuation.resumeCommand(
            conversationID: "3679128a-f371-445f-82ba-b3946e2f20b6",
            fallbackSessionID: "session-abc",
            projectPath: "/Users/etanheyman/Gits/brainlayer"
        )
        let view = Text(command)
            .font(.system(size: 12, weight: .medium, design: .monospaced))
            .lineLimit(2)
            .padding(18)
            .frame(width: 620, height: 90, alignment: .leading)

        try renderContinuationPNG(view, name: "bug6-cwd-aware-resume-command.png")
    }

    @MainActor
    private func renderContinuationPNG<V: View>(_ view: V, name: String) throws {
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2
        guard let image = renderer.nsImage,
              let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let png = bitmap.representation(using: .png, properties: [:]) else {
            XCTFail("Expected renderer to produce a PNG")
            return
        }

        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("docs.local/wave3-qa/\(name)")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try png.write(to: url)
        XCTAssertGreaterThan(png.count, 1_000)
    }
}
