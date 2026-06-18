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
        19002 agy agy --dangerously-skip-permissions --model Gemini 3.1 Pro (High)
        19003 agy /Users/etanheyman/.local/bin/agy --dangerously-skip-permissions --model Gemini 3.1 Pro (High)
         1355 Electron /Applications/Claude.app/Contents/Frameworks/Electron Framework.framework/Helpers/chrome_crashpad_handler --monitor-self-annotation=ptype=crashpad-handler
         1588 Claude\\ Helper /Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper --type=gpu-process
        """

        let activity = AgentActivityMonitor.parse(snapshot)

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

    func testParseSnapshotSkipsSearchCommandsThatMentionCursorAgentPhrase() {
        let snapshot = """
        42031 rg rg -n cursor agent /Users/etanheyman/Gits
        42032 zsh /bin/zsh -lc ps -axo pid=,ucomm=,args= | rg cursor agent
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotCountsClaudeCliWhenPromptMentionsSearchCommands() {
        let snapshot = """
        13303 2.1.175 claude --dangerously-skip-permissions --append-system-prompt Use grep and ps -axo only as debugging examples
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .claude), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotDoesNotTreatNonCodexLauncherAsActualCodex() {
        let snapshot = """
        13908 node node /opt/homebrew/bin/codex --model gpt-5.4
        13909 codex /Users/etanheyman/.bun/install/global/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex --model gpt-5.4
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .codex), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotClassifiesAgyGeminiByExecutableBeforePromptMentionsClaude() {
        let snapshot = """
        19004 agy /Users/etanheyman/.local/bin/agy --dangerously-skip-permissions --model Gemini 3.1 Pro --prompt-interactive Adopt brainlayerClaude context for orchestration
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .claude), 0)
        XCTAssertEqual(activity.count(for: .gemini), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotDoesNotClassifyAgyByGeminiSubstringInsideArgument() {
        let snapshot = """
        19005 agy /Users/etanheyman/.local/bin/agy --config mygeminiconfig --prompt-interactive ordinary task
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotDoesNotClassifyAgyByPromptMentioningModelFlag() {
        let snapshot = """
        19006 agy /Users/etanheyman/.local/bin/agy --prompt-interactive Please document why --model gemini should not be parsed from prompt text
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotDoesNotClassifyAgyByPromptAliasMentioningModelFlag() {
        let snapshot = """
        19007 agy /Users/etanheyman/.local/bin/agy -i Please document why --model gemini should not be parsed from prompt text
        19008 agy /Users/etanheyman/.local/bin/agy -p Please document why --model gemini should not be parsed from prompt text
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotDoesNotClassifyWrapperByPromptMentioningGemini() {
        let snapshot = """
        19009 node node /Users/etanheyman/tmp/helper.js --prompt Please explain gemini launcher behavior
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .gemini), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func testParseSnapshotClassifiesCursorAgentNodeLauncherAndSkipsWorkerServer() {
        let snapshot = """
        50008 node /Users/etanheyman/.local/bin/cursor-agent --use-system-ca /Users/etanheyman/.local/share/cursor-agent/versions/2026.06.12-01-15-52-7244546/index.js agent
        52858 node /Users/etanheyman/.local/share/cursor-agent/versions/2026.06.12-01-15-52-7244546/node /Users/etanheyman/.local/share/cursor-agent/versions/2026.06.12-01-15-52-7244546/index.js worker-server
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotClassifiesRootCursorAgentSession() {
        let snapshot = """
        50009 cursor-agent cursor-agent --yolo -p investigate live agent signals
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotClassifiesCursorCliLaunchedByFullPath() {
        let snapshot = """
        50100 cursor /Users/etanheyman/.local/bin/cursor agent --resume session-456
        """

        let activity = AgentActivityMonitor.parse(snapshot)

        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func testParseSnapshotDoesNotClassifyCursorDesktopAppAsAgent() {
        let snapshot = """
        50101 Cursor /Applications/Cursor.app/Contents/MacOS/Cursor --type=renderer
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
