import XCTest
@testable import BrainBar

/// #971: "64 agent processes live" counted ps ROWS. The Runtime row now counts agent
/// SESSIONS — one per top-level CLI session, with no counted agent among its ancestors —
/// and excludes app-bundled helpers, the Codex app-server daemons, bridges/proxies and
/// every child of a session.
///
/// Fixtures are synthetic rows in the exact `ps -axo pid=,ppid=,ucomm=,args=` layout
/// (right-aligned PIDs, `ucomm` in a 16-column field), shaped after a live capture.
final class AgentSessionCountTests: XCTestCase {
    static func psRow(_ pid: Int, _ ppid: Int, _ ucomm: String, _ args: String) -> String {
        let comm = String(ucomm.prefix(16)).padding(toLength: 16, withPad: " ", startingAt: 0)
        return String(format: "%5d %5d ", pid, ppid) + comm + " " + args
    }

    private static let row = psRow

    /// A machine with 6 real sessions (2 Claude, 2 Codex, 1 Gemini, 1 Cursor) and a
    /// crowd of processes the old row-counter also counted.
    static let busyMachine = [
        row(100, 1, "zsh", "-zsh"),
        row(101, 1, "zsh", "-zsh"),
        row(102, 1, "zsh", "-zsh"),
        row(103, 1, "zsh", "-zsh"),
        row(104, 1, "zsh", "-zsh"),
        row(105, 1, "zsh", "-zsh"),
        // Claude session A and everything it spawned.
        row(200, 100, "2.1.281", "claude --dangerously-skip-permissions --model claude-opus-5-5[1m]"),
        row(201, 200, "zsh", "/bin/zsh -c source /Users/dev/.claude/shell-snapshots/snapshot-zsh.sh && claude -p summarize"),
        row(202, 200, "brainlayer-mcp-stdio-bridge", "/opt/homebrew/bin/brainlayer-mcp-stdio-bridge"),
        row(203, 200, "socat", "socat STDIO UNIX-CONNECT:/tmp/brainbar.sock"),
        row(204, 201, "2.1.281", "claude -p summarize the diff"),
        row(205, 200, "node", "node /Users/dev/mcp/cmuxlayer/dist/index.js"),
        // Claude session B.
        row(300, 101, "2.1.283", "claude --continue --model claude-opus-5-5[1m]"),
        // Not a session: the Chrome native-messaging host.
        row(310, 1, "2.1.275", "/Users/dev/.local/bin/claude --chrome-native-host"),
        // Codex session (native CLI) and its children.
        row(400, 102, "codex", "codex --dangerously-bypass-approvals-and-sandbox resume 0000-synthetic"),
        row(401, 400, "codex-code-mode-host", "/Users/dev/.codex/packages/standalone/releases/0.157.1/bin/codex-code-mode-host"),
        row(402, 400, "node", "/Users/dev/.cache/codex-runtimes/node/bin/node /Users/dev/.codex/plugins/cache/plugin.js"),
        row(403, 400, "SkyComputerUseClient", "/Users/dev/.codex/computer-use/Codex Computer Use.app/Contents/MacOS/SkyComputerUseClient"),
        // Codex session launched through the bun wrapper: wrapper + native child = 1.
        row(410, 103, "node", "node /Users/dev/.bun/bin/codex --model gpt-5.4"),
        row(411, 410, "codex", "/Users/dev/.bun/install/global/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/codex/codex --model gpt-5.4"),
        // Not sessions: the Codex app-server daemons.
        row(420, 1, "codex", "/Users/dev/.codex/packages/app-server-daemon/releases/0.157.1/bin/codex app-server --listen unix://"),
        row(421, 1, "codex", "/Users/dev/.codex/packages/app-server-daemon/releases/0.157.1/bin/codex app-server daemon pid-update-loop"),
        // Not sessions: ChatGPT.app's bundled Codex framework.
        row(500, 1, "ChatGPT", "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT"),
        row(501, 500, "Codex (Renderer)", "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer) --type=renderer"),
        row(502, 500, "Codex (Renderer)", "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer) --type=renderer"),
        row(503, 500, "Codex (Service)", "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Helpers/Codex (Service).app/Contents/MacOS/Codex (Service) --type=utility"),
        row(504, 500, "codex", "/Applications/ChatGPT.app/Contents/Resources/codex -c features.code_mode_host=true app-server"),
        row(505, 504, "codex-code-mode-host", "/Applications/ChatGPT.app/Contents/Resources/codex-code-mode-host"),
        row(506, 1, "codex", "/Applications/ChatGPT.app/Contents/Resources/codex --orphaned-helper"),
        // Gemini session.
        row(600, 104, "gemini", "gemini --model gemini-2.5-pro"),
        // Cursor session and its worker-server.
        row(700, 105, "node", "/Users/dev/.local/bin/cursor-agent --use-system-ca /Users/dev/.local/share/cursor-agent/versions/1/index.js agent"),
        row(701, 700, "node", "/Users/dev/.local/share/cursor-agent/versions/1/node /Users/dev/.local/share/cursor-agent/versions/1/index.js worker-server"),
        // Not sessions: proxies/bridges started on their own.
        row(800, 1, "mcplayer", "/Users/dev/.local/bin/mcplayer proxy --spawn /Users/dev/.local/bin/claude mcp serve"),
        row(801, 1, "brainlayer-mcp-stdio-bridge", "/opt/homebrew/bin/brainlayer-mcp-stdio-bridge --spawn codex mcp"),
        row(802, 1, "socat", "socat STDIO EXEC:/Users/dev/.local/bin/claude mcp serve"),
    ].joined(separator: "\n")

    func test_counts_one_per_top_level_session_on_a_busy_machine() {
        let activity = AgentActivityMonitor.parse(Self.busyMachine)

        XCTAssertEqual(activity.count(for: .claude), 2)
        XCTAssertEqual(activity.count(for: .codex), 2)
        XCTAssertEqual(activity.count(for: .gemini), 1)
        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 6)
    }

    func test_chatgpt_app_codex_helpers_are_never_sessions() {
        let rows = Self.busyMachine.split(separator: "\n").filter { $0.contains("/Applications/ChatGPT.app") }
        XCTAssertEqual(rows.count, 7)

        let activity = AgentActivityMonitor.parse(rows.joined(separator: "\n"))

        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func test_a_cli_nested_under_another_session_is_part_of_that_session() {
        let activity = AgentActivityMonitor.parse([
            Self.row(10, 1, "zsh", "-zsh"),
            Self.row(11, 10, "2.1.281", "claude --model claude-opus-5-5[1m]"),
            Self.row(12, 11, "zsh", "/bin/zsh -c codex exec review"),
            Self.row(13, 12, "codex", "codex exec review the diff"),
            Self.row(14, 13, "gemini", "gemini --model gemini-2.5-pro -p check"),
        ].joined(separator: "\n"))

        XCTAssertEqual(activity.count(for: .claude), 1)
        XCTAssertEqual(activity.totalActiveAgents, 1, "descendants of a counted session are not new sessions")
    }

    func test_a_parent_cycle_in_the_process_table_cannot_hang_the_count() {
        let activity = AgentActivityMonitor.parse([
            Self.row(20, 21, "2.1.281", "claude --model x"),
            Self.row(21, 20, "zsh", "-zsh"),
        ].joined(separator: "\n"))

        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    func test_ucomm_with_spaces_does_not_shift_the_args_column() {
        let activity = AgentActivityMonitor.parse(
            Self.row(30, 1, "Claude Helper", "/Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper --type=gpu-process")
        )

        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    func test_runtime_row_shows_sessions_with_the_per_cli_breakdown_and_a_definition() {
        let activity = AgentActivityMonitor.parse(Self.busyMachine)

        XCTAssertEqual(activity.summaryText, "6 agent sessions live")
        XCTAssertEqual(activity.breakdownText, "2 Claude · 2 Codex · 1 Gemini · 1 Cursor")
        XCTAssertEqual(activity.runtimeRowText, "6 sessions · 2 Claude · 2 Codex · 1 Gemini · 1 Cursor")
        XCTAssertEqual(
            AgentActivitySnapshot.countingDefinition,
            "Agents = top-level Claude, Codex, Gemini and Cursor CLI sessions; app helpers, MCP bridges and each session's child processes are not counted."
        )
    }

    func test_monitor_captures_the_parent_pid_column() {
        XCTAssertEqual(AgentActivityMonitor.psArguments, ["-axo", "pid=,ppid=,ucomm=,args="])
    }
}
