import XCTest
@testable import BrainBar

final class AgentActivityMonitorTests: XCTestCase {
    func testFailedProcessSnapshotIsUnmeasuredInsteadOfQuiet() {
        let activity = AgentActivityMonitor(snapshotProvider: { nil }, executablePathResolver: { _ in nil }).sample()

        XCTAssertFalse(activity.isMeasured)
        XCTAssertEqual(activity.summaryText, "Agent activity unavailable: ps capture failed")
    }

    func testParseSnapshotCountsEachAgentFamilyAndSkipsHelperNoise() {
        let snapshot = """
         2918     1 2.1.114          claude --dangerously-skip-permissions --resume 3679128a-f371-445f-82ba-b3946e2f20b6
        13908     1 node             node /Users/dev/.bun/bin/codex --model gpt-5.4 --dangerously-bypass-approvals-and-sandbox
        13909 13908 codex            /Users/dev/.bun/install/global/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/codex/codex --model gpt-5.4 --dangerously-bypass-approvals-and-sandbox
        18001     1 cursor           cursor agent --resume session-123
        19001     1 gemini           gemini --model gemini-2.5-pro
        19002     1 agy              agy --dangerously-skip-permissions --model Gemini 3.1 Pro (High)
        19003     1 agy              /Users/dev/.local/bin/agy --dangerously-skip-permissions --model Gemini 3.1 Pro (High)
         1355     1 Electron         /Applications/Claude.app/Contents/Frameworks/Electron Framework.framework/Helpers/chrome_crashpad_handler --monitor-self-annotation=ptype=crashpad-handler
         1588     1 Claude Helper    /Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper --type=gpu-process
        """

        // Only the kernel path marks the Claude.app helpers as app-bundled (#990).
        let activity = AgentActivityMonitor.parse(snapshot, executablePath: { pid in
            [
                1355: "/Applications/Claude.app/Contents/Frameworks/Electron Framework.framework/Helpers/chrome_crashpad_handler",
                1588: "/Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper",
            ][pid]
        })

        XCTAssertEqual(activity.count(for: .claude), 1)
        XCTAssertEqual(activity.count(for: .codex), 1)
        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.count(for: .gemini), 3)
        XCTAssertEqual(activity.totalActiveAgents, 6)
    }

    func testParseSnapshotRetainsFamiliesWithZeroCounts() {
        let activity = AgentActivityMonitor.parse("")

        XCTAssertEqual(activity.count(for: .claude), 0)
        XCTAssertEqual(activity.count(for: .codex), 0)
        XCTAssertEqual(activity.count(for: .cursor), 0)
        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
        XCTAssertTrue(activity.isMeasured)
        XCTAssertEqual(activity.summaryText, "No agent sessions live")
    }

    func testPresenceLabelsSayCountsAreLiveAgentSessions() {
        let presence = AgentPresence(family: .codex, count: 12)

        XCTAssertEqual(presence.liveSessionLabel, "12 live agent sessions")
        XCTAssertEqual(presence.accessibilityLabel, "Codex: 12 live agent sessions from ps")
    }

    func testParseSnapshotSkipsSearchCommandsThatMentionAgentNames() {
        let snapshot = """
        42029     1 rg               rg -n claude\\|codex\\|cursor\\|gemini /Users/dev/Gits
        42030     1 awk              awk BEGIN{IGNORECASE=1} /claude|codex|cursor|gemini/ {print}
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotSkipsSearchCommandsThatMentionCursorAgentPhrase() {
        let snapshot = """
        42031     1 rg               rg -n cursor agent /Users/dev/Gits
        42032     1 zsh              /bin/zsh -lc ps -axo pid=,ucomm=,args= | rg cursor agent
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotCountsClaudeCliWhenPromptMentionsSearchCommands() {
        let snapshot = """
        13303     1 2.1.175          claude --dangerously-skip-permissions --append-system-prompt Use grep and ps -axo only as debugging examples
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .claude), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotDoesNotTreatNonCodexLauncherAsActualCodex() {
        let snapshot = """
        13908     1 node             node /opt/homebrew/bin/codex --model gpt-5.4
        13909 13908 codex            /Users/dev/.bun/install/global/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex --model gpt-5.4
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .codex), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotClassifiesAgyGeminiByExecutableBeforePromptMentionsClaude() {
        let snapshot = """
        19004     1 agy              /Users/dev/.local/bin/agy --dangerously-skip-permissions --model Gemini 3.1 Pro --prompt-interactive Adopt brainlayerClaude context for orchestration
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .claude), 0)
        XCTAssertEqual(activity.count(for: .gemini), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotDoesNotClassifyAgyByGeminiSubstringInsideArgument() {
        let snapshot = """
        19005     1 agy              /Users/dev/.local/bin/agy --config mygeminiconfig --prompt-interactive ordinary task
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotDoesNotClassifyAgyByPromptMentioningModelFlag() {
        let snapshot = """
        19006     1 agy              /Users/dev/.local/bin/agy --prompt-interactive Please document why --model gemini should not be parsed from prompt text
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotDoesNotClassifyAgyByPromptAliasMentioningModelFlag() {
        let snapshot = """
        19007     1 agy              /Users/dev/.local/bin/agy -i Please document why --model gemini should not be parsed from prompt text
        19008     1 agy              /Users/dev/.local/bin/agy -p Please document why --model gemini should not be parsed from prompt text
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotDoesNotClassifyWrapperByPromptMentioningGemini() {
        let snapshot = """
        19009     1 node             node /Users/dev/tmp/helper.js --prompt Please explain gemini launcher behavior
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotClassifiesCursorAgentNodeLauncherAndSkipsWorkerServer() {
        let snapshot = """
        50008     1 node             /Users/dev/.local/bin/cursor-agent --use-system-ca /Users/dev/.local/share/cursor-agent/versions/2026.06.12-01-15-52-7244546/index.js agent
        52858     1 node             /Users/dev/.local/share/cursor-agent/versions/2026.06.12-01-15-52-7244546/node /Users/dev/.local/share/cursor-agent/versions/2026.06.12-01-15-52-7244546/index.js worker-server
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotClassifiesRootCursorAgentSession() {
        let snapshot = """
        50009     1 cursor-agent     cursor-agent --yolo -p investigate live agent signals
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotClassifiesCursorCliLaunchedByFullPath() {
        let snapshot = """
        50100     1 cursor           /Users/dev/.local/bin/cursor agent --resume session-456
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotDoesNotClassifyCursorDesktopAppAsAgent() {
        let snapshot = """
        50101     1 Cursor           /Applications/Cursor.app/Contents/MacOS/Cursor --type=renderer
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 0)
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
