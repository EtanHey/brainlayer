import XCTest
@testable import BrainBar

final class AgentActivityMonitorTests: XCTestCase {
    func testParseSnapshotCountsEachAgentFamilyAndSkipsHelperNoise() {
        let snapshot = """
         2918 2.1.114 claude --dangerously-skip-permissions --resume 3679128a-f371-445f-82ba-b3946e2f20b6
        13908 node node /Users/etanheyman/.bun/bin/codex --model gpt-5.4 --dangerously-bypass-approvals-and-sandbox
        13909 codex /Users/etanheyman/.bun/install/global/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/codex/codex --model gpt-5.4 --dangerously-bypass-approvals-and-sandbox
        18001 cursor cursor agent --resume session-123
        19001 gemini gemini --model gemini-2.5-pro
         1355 Electron /Applications/Claude.app/Contents/Frameworks/Electron Framework.framework/Helpers/chrome_crashpad_handler --monitor-self-annotation=ptype=crashpad-handler
         1588 Claude\\ Helper /Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper --type=gpu-process
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .claude), 1)
        XCTAssertEqual(activity.count(for: .codex), 1)
        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.count(for: .gemini), 1)
        XCTAssertEqual(activity.totalActiveAgents, 4)
    }

    func testParseSnapshotRetainsFamiliesWithZeroCounts() {
        let activity = AgentActivityMonitor.parse("")

        XCTAssertEqual(activity.count(for: .claude), 0)
        XCTAssertEqual(activity.count(for: .codex), 0)
        XCTAssertEqual(activity.count(for: .cursor), 0)
        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
        XCTAssertEqual(activity.summaryText, "No agent CLIs live")
    }

    func testParseSnapshotSkipsSearchCommandsThatMentionAgentNames() {
        let snapshot = """
        42029 rg rg -n claude\\|codex\\|cursor\\|gemini /Users/etanheyman/Gits
        42030 awk awk BEGIN{IGNORECASE=1} /claude|codex|cursor|gemini/ {print}
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testRunSnapshotCommandDrainsLargeStdoutWithoutDeadlocking() {
        let script = "python3 -c \"print('codex ' * 20000)\""

        let snapshot = AgentActivityMonitor.runSnapshotCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", script]
        )

        XCTAssertNotNil(snapshot)
        XCTAssertGreaterThan(snapshot?.count ?? 0, 100_000)
    }
}
