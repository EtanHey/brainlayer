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
    static let busyMachine: String = ([String]([
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
    ])).joined(separator: "\n")

    /// The kernel's executable path for the busy machine's app-bundled rows, as
    /// `proc_pidpath` answers live. Only the path decides bundling (#990); every other row's
    /// path is outside any bundle, so it resolves to nil here.
    static let busyMachinePaths: [Int32: String] = [
        403: "/Users/dev/.codex/computer-use/Codex Computer Use.app/Contents/MacOS/SkyComputerUseClient",
        500: "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT",
        501: "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer)",
        502: "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer)",
        503: "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Helpers/Codex (Service).app/Contents/MacOS/Codex (Service)",
        504: "/Applications/ChatGPT.app/Contents/Resources/codex",
        505: "/Applications/ChatGPT.app/Contents/Resources/codex-code-mode-host",
        506: "/Applications/ChatGPT.app/Contents/Resources/codex",
    ]

    static func parseBusyMachine(_ snapshot: String = busyMachine) -> AgentActivitySnapshot {
        AgentActivityMonitor.parse(snapshot, executablePath: { busyMachinePaths[$0] })
    }

    func test_counts_one_per_top_level_session_on_a_busy_machine() {
        let activity = Self.parseBusyMachine()

        XCTAssertEqual(activity.count(for: .claude), 2)
        XCTAssertEqual(activity.count(for: .codex), 2)
        XCTAssertEqual(activity.count(for: .gemini), 1)
        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 6)
    }

    func test_chatgpt_app_codex_helpers_are_never_sessions() {
        let rows = Self.busyMachine.split(separator: "\n").filter { $0.contains("/Applications/ChatGPT.app") }
        XCTAssertEqual(rows.count, 7)

        let activity = Self.parseBusyMachine(rows.joined(separator: "\n"))

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
            Self.row(30, 1, "Claude Helper", "/Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper --type=gpu-process"),
            executablePath: { _ in "/Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper" }
        )

        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    /// Live capture 2026-09-28: a detached perl that rewrites `$0` (cmuxlayer's inbox
    /// tail) makes ps print its leftover environment as args, including a
    /// `.../claude CMUX_CLAUDE_WRAPPER...` path. Shells and text tools are never the
    /// session process; the real CLI is always its own row.
    func test_shells_and_argv_rewriting_helpers_are_never_sessions() {
        let activity = AgentActivityMonitor.parse([
            Self.row(40, 1, "perl", "cmuxlayer-inbox-tail:0123abcd      CMUX_BUNDLE_ID=com.example.app CMUX_CLAUDE_WRAPPER_SHIM=/var/folders/xx/T/shim/claude CMUX_CLAUDE_WRAPPER=1"),
            Self.row(41, 40, "tail", "tail -n0 -F /Users/dev/.cmux/agents/brainlayerClaude-0000/inbox.jsonl"),
            Self.row(42, 1, "bash", "/bin/bash /Users/dev/.claude/skills/collab-monitor/scripts/collab-monitor.sh run"),
            Self.row(50, 1, "zsh", "/bin/zsh -c brainlayerClaude -s --resume 0000"),
            Self.row(51, 50, "2.1.281", "claude --dangerously-skip-permissions --resume 0000"),
        ].joined(separator: "\n"))

        XCTAssertEqual(activity.count(for: .claude), 1, "only the claude process itself is the session")
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    /// #977 R1 B1: an interactive CLI launched with no arguments is a session. Its
    /// whole args field is the binary, so detection must match argv[0] itself.
    func test_bare_interactive_clis_with_no_arguments_are_sessions() {
        let activity = AgentActivityMonitor.parse([
            Self.row(60, 1, "zsh", "-zsh"),
            Self.row(61, 60, "2.1.281", "claude"),
            Self.row(62, 60, "2.1.281", "/Users/dev/.local/bin/claude"),
            Self.row(63, 60, "codex", "codex"),
            Self.row(64, 60, "codex", "/opt/homebrew/bin/codex"),
            Self.row(65, 60, "gemini", "gemini"),
            Self.row(66, 60, "node", "/Users/dev/.npm-global/bin/gemini"),
            Self.row(67, 60, "cursor-agent", "cursor-agent"),
        ].joined(separator: "\n"))

        XCTAssertEqual(activity.count(for: .claude), 2)
        XCTAssertEqual(activity.count(for: .codex), 2)
        XCTAssertEqual(activity.count(for: .gemini), 2)
        XCTAssertEqual(activity.count(for: .cursor), 1)
        XCTAssertEqual(activity.totalActiveAgents, 7)
    }

    /// #977 R1 B2: a mode is a process role at a fixed argv position, not a word
    /// anywhere in the args. Prompt text that mentions a mode must not hide a session.
    func test_mode_words_inside_a_prompt_do_not_hide_a_headless_session() {
        let activity = AgentActivityMonitor.parse([
            Self.row(70, 1, "zsh", "-zsh"),
            Self.row(71, 70, "2.1.281", "claude -p summarize the app-server logs"),
            Self.row(72, 70, "codex", "codex exec review why mcp serve fails"),
            Self.row(73, 70, "gemini", "gemini -p fix the mcp-server config and --chrome-native-host flag"),
        ].joined(separator: "\n"))

        XCTAssertEqual(activity.count(for: .claude), 1)
        XCTAssertEqual(activity.count(for: .codex), 1)
        XCTAssertEqual(activity.count(for: .gemini), 1)
        XCTAssertEqual(activity.totalActiveAgents, 3)
    }

    func test_real_non_session_roles_are_still_excluded_by_argv_position() {
        let activity = AgentActivityMonitor.parse([
            Self.row(80, 1, "codex", "/Users/dev/.codex/packages/app-server-daemon/releases/0.157.1/bin/codex app-server --listen unix://"),
            Self.row(81, 1, "codex", "/Users/dev/.codex/bin/codex -c features.code_mode_host=true app-server --analytics-default-enabled"),
            Self.row(82, 1, "codex", "/Users/dev/.codex/bin/codex --config model=x mcp-server"),
            Self.row(83, 1, "2.1.281", "claude mcp serve"),
            Self.row(84, 1, "2.1.275", "/Users/dev/.local/bin/claude --chrome-native-host"),
        ].joined(separator: "\n"))

        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    /// #977 R2 B3: a server role belongs to one binary. `app-server`/`mcp-server` are
    /// Codex roles and `mcp serve`/`--chrome-native-host` are Claude roles; the same
    /// word as another CLI's first positional is that CLI's prompt or subcommand.
    func test_server_roles_are_scoped_to_the_binary_that_owns_them() {
        let activity = AgentActivityMonitor.parse([
            Self.row(90, 1, "zsh", "-zsh"),
            Self.row(91, 90, "gemini", "gemini app-server"),
            Self.row(92, 90, "gemini", "gemini mcp-server tidy the config"),
            Self.row(93, 90, "codex", "codex mcp serve"),
            Self.row(94, 90, "gemini", "gemini --chrome-native-host"),
            Self.row(95, 90, "2.1.281", "claude app-server"),
        ].joined(separator: "\n"))

        XCTAssertEqual(activity.count(for: .gemini), 3)
        XCTAssertEqual(activity.count(for: .codex), 1)
        XCTAssertEqual(activity.count(for: .claude), 1)
        XCTAssertEqual(activity.totalActiveAgents, 5)
    }

    /// Parses rows with the kernel's executable path for each PID, as `proc_pidpath`
    /// answers live. Paths are synthetic; a PID missing from `paths` resolves to nil.
    private static func parse(_ rows: [String], paths: [Int32: String]) -> AgentActivitySnapshot {
        AgentActivityMonitor.parse(rows.joined(separator: "\n"), executablePath: { paths[$0] })
    }

    /// #984: an executable inside a `*.app/Contents/` bundle is an app helper wherever the
    /// bundle lives, not only under `/Applications`. Live on both Macs, Codex's computer-use
    /// helper under `~/.codex` counted as a Codex session: its args split on whitespace
    /// leave argv[0] as `…/.codex/computer-use/codex`, which reads as a bare `codex` CLI.
    func test_app_bundled_helpers_outside_applications_are_never_sessions() {
        let sky = "/Users/u/.codex/computer-use/Codex Computer Use.app/Contents"
        let activity = Self.parse([
            Self.row(900, 1, "SkyComputerUseService", "\(sky)/MacOS/SkyComputerUseService"),
            Self.row(901, 1, "SkyComputerUseClient", "\(sky)/SharedSupport/SkyComputerUseClient.app/Contents/MacOS/SkyComputerUseClient computer-history mcp"),
            Self.row(902, 1, "codex", "/Users/u/Applications/ChatGPT.app/Contents/Resources/codex --orphaned-helper"),
            Self.row(903, 1, "codex", "/opt/vendor/Foo.app/Contents/Resources/codex"),
            Self.row(904, 1, "claude", "/Users/u/Library/Application Support/Foo.app/Contents/MacOS/claude --model x"),
        ], paths: [
            900: "\(sky)/MacOS/SkyComputerUseService",
            901: "\(sky)/SharedSupport/SkyComputerUseClient.app/Contents/MacOS/SkyComputerUseClient",
            902: "/Users/u/Applications/ChatGPT.app/Contents/Resources/codex",
            903: "/opt/vendor/Foo.app/Contents/Resources/codex",
            904: "/Users/u/Library/Application Support/Foo.app/Contents/MacOS/claude",
        ])

        XCTAssertEqual(activity.count(for: .codex), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    /// #987 R1 B1/B2 and R2 B1: args text cannot say where argv[0] ends. A space may be
    /// inside the path ("My App.app") or between arguments, `ucomm` may be a version string
    /// ("2.1.281"), and an earlier component may share the file name. The kernel's
    /// executable path is unambiguous, so it alone decides.
    func test_the_kernel_executable_path_decides_app_bundling() {
        let activity = Self.parse([
            Self.row(920, 1, "2.1.281", "/Users/u/Foo.app/Contents/Resources/claude --model x"),
            Self.row(921, 1, "codex", "/Users/u/codex/Foo.app/Contents/Resources/codex"),
            Self.row(922, 1, "SkyComputerUseService", "/Users/u/SkyComputerUseService/Codex Computer Use.app/Contents/MacOS/SkyComputerUseService"),
            Self.row(923, 1, "2.1.281", "/Users/u/My App.app/Contents/Resources/claude --x"),
            Self.row(924, 1, "codex", "/Users/u/codex/Codex App.app/Contents/Resources/codex --x"),
        ], paths: [
            920: "/Users/u/Foo.app/Contents/Resources/claude",
            921: "/Users/u/codex/Foo.app/Contents/Resources/codex",
            922: "/Users/u/SkyComputerUseService/Codex Computer Use.app/Contents/MacOS/SkyComputerUseService",
            923: "/Users/u/My App.app/Contents/Resources/claude",
            924: "/Users/u/codex/Codex App.app/Contents/Resources/codex",
        ])

        XCTAssertEqual(activity.count(for: .claude), 0)
        XCTAssertEqual(activity.count(for: .codex), 0)
        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    /// #987 R3: when the kernel cannot say (the process exited, or permission is denied),
    /// the row is not app-bundled. Args text is never consulted in its place.
    func test_an_unresolvable_executable_path_is_not_app_bundled() {
        let activity = Self.parse([
            Self.row(950, 1, "SkyComputerUseService", "/Users/u/.codex/computer-use/Codex Computer Use.app/Contents/MacOS/SkyComputerUseService"),
        ], paths: [:])

        XCTAssertEqual(activity.count(for: .codex), 1)
    }

    /// #987 R1 N1 policy: Python.app-hosted processes are app-bundled like any
    /// `*.app/Contents/` executable, whatever script they host. That covers the framework
    /// Python, and also Homebrew's, whose kernel path resolves into its own Python.app. No
    /// agent CLI is Python-hosted today. A Python outside any bundle (a uv standalone
    /// build) still has its script detected.
    func test_python_app_hosted_processes_are_app_bundled_by_policy() {
        let activity = Self.parse([
            Self.row(930, 1, "zsh", "-zsh"),
            Self.row(931, 930, "Python", "/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python /Users/u/.local/bin/gemini --model x"),
            Self.row(932, 930, "python3.13", "/Users/u/.local/share/uv/python/cpython-3.13-macos-aarch64-none/bin/python3.13 /Users/u/.local/bin/gemini --model x"),
        ], paths: [
            930: "/bin/zsh",
            931: "/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python",
            932: "/Users/u/.local/share/uv/python/cpython-3.13-macos-aarch64-none/bin/python3.13",
        ])

        XCTAssertEqual(activity.count(for: .gemini), 1, "only the non-bundled python's gemini counts")
        XCTAssertEqual(activity.totalActiveAgents, 1)
    }

    /// #984, #987 R2 B2: only the executable's own path decides. A real CLI whose ARGUMENTS
    /// name an app bundle, even one ending in its own `ucomm`, is still a session, and so is
    /// one launched through an interpreter outside a bundle.
    func test_a_cli_that_merely_mentions_an_app_bundle_is_still_a_session() {
        let activity = Self.parse([
            Self.row(910, 1, "zsh", "-zsh"),
            Self.row(911, 910, "codex", "/opt/homebrew/bin/codex --add-dir /Users/u/Foo.app/Contents/Resources"),
            Self.row(912, 910, "codex", "/opt/homebrew/bin/codex exec review Foo.app/Contents/Info.plist"),
            Self.row(913, 910, "node", "node /Users/u/.bun/bin/codex --model gpt-5.4"),
            Self.row(914, 910, "2.1.281", "/Users/u/.local/bin/claude --add-dir /Users/u/Applications/X.app/Contents"),
            Self.row(915, 910, "codex", "codex exec inspect /Users/u/Foo.app/Contents/MacOS/codex"),
            Self.row(916, 910, "2.1.281", "/opt/homebrew/bin/claude --add-dir /Users/u/Foo.app/Contents/2.1.281"),
        ], paths: [
            910: "/bin/zsh",
            911: "/opt/homebrew/bin/codex",
            912: "/opt/homebrew/bin/codex",
            913: "/opt/homebrew/Cellar/node/24.0.0/bin/node",
            914: "/Users/u/.local/share/claude/versions/2.1.281",
            915: "/opt/homebrew/bin/codex",
            916: "/opt/homebrew/bin/claude",
        ])

        XCTAssertEqual(activity.count(for: .codex), 4)
        XCTAssertEqual(activity.count(for: .claude), 2)
        XCTAssertEqual(activity.totalActiveAgents, 6)
    }

    // MARK: #990: the bundle decision is the kernel path's alone

    /// #990: `/Applications/…` plus any later `.app/` argument used to mark the row app-bundled
    /// from args text, before the kernel path was consulted. A CLI installed under
    /// /Applications (outside any bundle) whose arguments name a bundle is a session.
    func test_a_cli_under_applications_with_a_later_app_argument_is_a_session() {
        let activity = Self.parse([
            Self.row(960, 1, "codex", "/Applications/Tools/codex --add-dir /Users/u/Foo.app/Contents"),
        ], paths: [960: "/Applications/Tools/codex"])

        XCTAssertEqual(activity.count(for: .codex), 1)
    }

    /// #990: a noise word in a real CLI's ARGUMENTS never drops it. The row is recognised by
    /// its argv[0] binary name, so "/Applications/Claude.app", "crashpad" or "grep " in its
    /// args are just arguments.
    func test_noise_words_in_a_real_clis_arguments_do_not_hide_it() {
        let activity = Self.parse([
            Self.row(970, 1, "2.1.281", "/opt/homebrew/bin/claude --add-dir /Applications/Claude.app/Contents/Resources"),
            Self.row(971, 1, "2.1.281", "/Users/u/.local/bin/claude -p why does crashpad restart"),
            Self.row(972, 1, "codex", "/opt/homebrew/bin/codex exec grep the logs for claude helper"),
        ], paths: [
            970: "/opt/homebrew/bin/claude",
            971: "/Users/u/.local/share/claude/versions/2.1.281",
            972: "/opt/homebrew/bin/codex",
        ])

        XCTAssertEqual(activity.count(for: .claude), 2)
        XCTAssertEqual(activity.count(for: .codex), 1)
    }

    /// #990: the genuine non-session filters still hold, now decided by the kernel path or
    /// the row's own executable, never by a later argument.
    func test_genuine_non_session_filters_still_hold() {
        let activity = Self.parse([
            Self.row(980, 1, "rg", "rg -n /Applications/Claude.app /claude /Users/u"),
            Self.row(981, 1, "grep", "grep -r codex exec /Users/u"),
            Self.row(982, 1, "Claude Helper", "/Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper --type=gpu-process"),
            Self.row(983, 1, "codex", "/Applications/ChatGPT.app/Contents/Resources/codex --orphaned-helper"),
            Self.row(984, 1, "2.1.281", "/System/Library/Private/claude --model x"),
            Self.row(985, 1, "ps", "ps -axo pid=,ppid=,ucomm=,args= claude codex"),
        ], paths: [
            980: "/opt/homebrew/bin/rg",
            981: "/usr/bin/grep",
            982: "/Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper",
            983: "/Applications/ChatGPT.app/Contents/Resources/codex",
            984: "/System/Library/Private/claude",
            985: "/bin/ps",
        ])

        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    // MARK: #982: two more non-session roles

    /// #982: Codex global flags, value-taking or boolean, may precede the role. The role is the
    /// first positional after every flag, so `codex --profile work app-server` is the daemon.
    func test_codex_global_flags_before_the_role_are_skipped() {
        let activity = Self.parse([
            Self.row(1000, 1, "codex", "codex --profile work app-server"),
            Self.row(1001, 1, "codex", "codex -m gpt-5 --search mcp-server"),
            Self.row(1002, 1, "codex", "codex --enable code_mode --disable web_search app-server --listen unix://"),
            Self.row(1003, 1, "codex", "/opt/homebrew/bin/codex -s workspace-write -a never app-server"),
            Self.row(1004, 1, "codex", "codex -C /Users/u/repo --add-dir /Users/u/other --oss --local-provider ollama mcp-server"),
            Self.row(1005, 1, "codex", "codex --model=gpt-5 --remote ws://127.0.0.1:1 --remote-auth-token-env TOKEN -i shot.png app-server"),
        ], paths: [:])

        XCTAssertEqual(activity.totalActiveAgents, 0)
    }

    /// #982: `remote-control` manages the app-server daemon (Codex) or bridges sessions to
    /// claude.ai (Claude); it is not itself a session. Claude's `--remote-control` FLAG starts
    /// an interactive session and still counts, as does the word for another CLI.
    func test_remote_control_roles_are_not_sessions() {
        let activity = Self.parse([
            Self.row(1010, 1, "codex", "codex remote-control start"),
            Self.row(1011, 1, "codex", "codex --profile work remote-control"),
            Self.row(1012, 1, "2.1.281", "claude remote-control"),
            Self.row(1013, 1, "2.1.281", "/Users/u/.local/bin/claude remote-control --name laptop"),
            Self.row(1020, 1, "2.1.281", "claude --remote-control laptop"),
            Self.row(1021, 1, "codex", "codex --profile work exec review the diff"),
            Self.row(1022, 1, "codex", "codex --profile work"),
            Self.row(1023, 1, "gemini", "gemini remote-control"),
        ], paths: [:])

        XCTAssertEqual(activity.count(for: .claude), 1)
        XCTAssertEqual(activity.count(for: .codex), 2)
        XCTAssertEqual(activity.count(for: .gemini), 1)
        XCTAssertEqual(activity.totalActiveAgents, 4)
    }

    func test_runtime_row_shows_sessions_with_the_per_cli_breakdown_and_a_definition() {
        let activity = Self.parseBusyMachine()

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
