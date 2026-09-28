import AppKit
import Foundation

enum AgentFamily: String, CaseIterable, Sendable {
    case claude
    case codex
    case cursor
    case gemini

    var label: String {
        switch self {
        case .claude:
            return "Claude"
        case .codex:
            return "Codex"
        case .cursor:
            return "Cursor"
        case .gemini:
            return "Gemini"
        }
    }

    var accentColor: NSColor {
        switch self {
        case .claude:
            return .systemIndigo
        case .codex:
            return .systemBlue
        case .cursor:
            return .systemGreen
        case .gemini:
            return .systemTeal
        }
    }
}

struct AgentPresence: Sendable, Equatable {
    let family: AgentFamily
    let count: Int

    var isActive: Bool { count > 0 }

    var liveSessionLabel: String {
        count == 1 ? "1 live agent session" : "\(DashboardMetricFormatter.integerString(count)) live agent sessions"
    }

    var accessibilityLabel: String {
        "\(family.label): \(liveSessionLabel) from ps"
    }
}

struct AgentActivitySnapshot: Sendable, Equatable {
    let presences: [AgentPresence]
    let measurement: MetricEvidenceReadability

    init(
        presences: [AgentPresence],
        measurement: MetricEvidenceReadability = .readable
    ) {
        self.presences = presences
        self.measurement = measurement
    }

    static let empty = AgentActivitySnapshot(
        presences: AgentFamily.allCases.map { AgentPresence(family: $0, count: 0) }
    )

    static func unavailable(_ reason: String) -> AgentActivitySnapshot {
        AgentActivitySnapshot(
            presences: AgentFamily.allCases.map { AgentPresence(family: $0, count: 0) },
            measurement: .unreadable(reason)
        )
    }

    var isMeasured: Bool { measurement.isReadable }

    var totalActiveAgents: Int {
        presences.reduce(0) { $0 + $1.count }
    }

    func count(for family: AgentFamily) -> Int {
        presences.first(where: { $0.family == family })?.count ?? 0
    }

    var summaryText: String {
        if case let .unreadable(reason) = measurement {
            return "Agent activity unavailable: \(reason)"
        }
        switch totalActiveAgents {
        case 0:
            return "No agent sessions live"
        case 1:
            return "1 agent session live"
        default:
            return "\(DashboardMetricFormatter.integerString(totalActiveAgents)) agent sessions live"
        }
    }

    /// Per-CLI counts in a fixed order, zeros included, so the row never reflows.
    var breakdownText: String {
        Self.breakdownOrder
            .map { "\(DashboardMetricFormatter.integerString(count(for: $0))) \($0.label)" }
            .joined(separator: " · ")
    }

    /// The Runtime card's "Agents" value (#971).
    var runtimeRowText: String {
        guard isMeasured else { return summaryText }
        let total = DashboardMetricFormatter.integerString(totalActiveAgents)
        let sessions = totalActiveAgents == 1 ? "1 session" : "\(total) sessions"
        return "\(sessions) · \(breakdownText)"
    }

    static let breakdownOrder: [AgentFamily] = [.claude, .codex, .gemini, .cursor]

    static let countingDefinition =
        "Agents = top-level Claude, Codex, Gemini and Cursor CLI sessions; app helpers, MCP bridges and each session's child processes are not counted."
}

final class AgentActivityMonitor {
    private let snapshotProvider: @Sendable () -> String?

    init(snapshotProvider: @escaping @Sendable () -> String? = AgentActivityMonitor.captureProcessSnapshot) {
        self.snapshotProvider = snapshotProvider
    }

    func sample() -> AgentActivitySnapshot {
        guard let snapshot = snapshotProvider() else { return .unavailable("ps capture failed") }
        return Self.parse(snapshot)
    }

    struct ProcessRow: Equatable {
        let pid: Int32
        let parentPID: Int32
        /// Lowercased `ucomm` (16-character short name, may contain spaces).
        let executable: String
        /// Lowercased `args`.
        let command: String
    }

    /// `ucomm` is printed left-justified in a fixed MAXCOMLEN (16) column, so it can be
    /// split off by width even when it contains spaces ("Codex (Service)").
    static let ucommColumnWidth = 16

    /// Counts agent SESSIONS (#971): a row is a candidate when it looks like an agent
    /// CLI and is not an app-bundled helper, bridge/proxy, or non-session mode; a
    /// candidate counts only when none of its ancestors is itself a candidate, so a
    /// session's wrappers, MCP children and nested CLIs fold into that one session.
    static func parse(_ snapshot: String) -> AgentActivitySnapshot {
        let rows = snapshot.split(whereSeparator: \.isNewline).compactMap(parseRow)
        var parentByPID: [Int32: Int32] = [:]
        var candidates: [Int32: AgentFamily] = [:]
        for row in rows {
            parentByPID[row.pid] = row.parentPID
            if let family = candidateFamily(row) {
                candidates[row.pid] = family
            }
        }

        var counts = Dictionary(uniqueKeysWithValues: AgentFamily.allCases.map { ($0, 0) })
        for (pid, family) in candidates where !hasCandidateAncestor(pid, parentByPID: parentByPID, candidates: candidates) {
            counts[family, default: 0] += 1
        }

        return AgentActivitySnapshot(
            presences: AgentFamily.allCases.map { AgentPresence(family: $0, count: counts[$0, default: 0]) }
        )
    }

    static func parseRow(_ rawLine: Substring) -> ProcessRow? {
        var rest = rawLine.drop(while: \.isWhitespace)
        guard let pidEnd = rest.firstIndex(where: \.isWhitespace),
              let pid = Int32(rest[..<pidEnd]) else { return nil }
        rest = rest[pidEnd...].drop(while: \.isWhitespace)
        guard let parentEnd = rest.firstIndex(where: \.isWhitespace),
              let parentPID = Int32(rest[..<parentEnd]) else { return nil }
        rest = rest[rest.index(after: parentEnd)...]

        let executable: Substring
        let command: Substring
        if rest.count > ucommColumnWidth,
           rest[rest.index(rest.startIndex, offsetBy: ucommColumnWidth)] == " " {
            executable = rest.prefix(ucommColumnWidth)
            command = rest.dropFirst(ucommColumnWidth + 1)
        } else {
            let parts = rest.split(maxSplits: 1, omittingEmptySubsequences: true, whereSeparator: \.isWhitespace)
            executable = parts.first ?? ""
            command = parts.count > 1 ? parts[1] : ""
        }
        return ProcessRow(
            pid: pid,
            parentPID: parentPID,
            executable: executable.trimmingCharacters(in: .whitespaces).lowercased(),
            command: command.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        )
    }

    private static func candidateFamily(_ row: ProcessRow) -> AgentFamily? {
        let executable = row.executable
        let command = row.command
        guard !command.isEmpty,
              !isAppBundled(executable: executable, command: command),
              !nonSessionExecutables.contains(executable),
              !isBridgeOrProxy(executable: executable, command: command),
              !isNonSessionMode(command),
              !isIgnoredProcess(executable: executable, command: command)
        else { return nil }
        return detectActualFamily(executable: executable, command: command)
            ?? bareCLIFamily(command)
            ?? detectWrapperFamily(executable: executable, command: command)
    }

    private static let bareCLIFamilies: [String: AgentFamily] = [
        "claude": .claude,
        "codex": .codex,
        "gemini": .gemini,
        "cursor-agent": .cursor,
    ]

    /// A CLI started with no arguments has args == its binary (`claude`,
    /// `/Users/x/.local/bin/claude`), so the detectors that look for "claude " miss it.
    private static func bareCLIFamily(_ command: String) -> AgentFamily? {
        guard let argv0 = command.split(whereSeparator: \.isWhitespace).first else { return nil }
        let name = argv0.split(separator: "/").last.map(String.init) ?? String(argv0)
        return bareCLIFamilies[name]
    }

    private static func hasCandidateAncestor(
        _ pid: Int32,
        parentByPID: [Int32: Int32],
        candidates: [Int32: AgentFamily]
    ) -> Bool {
        var visited: Set<Int32> = [pid]
        var current = parentByPID[pid]
        while let ancestor = current, ancestor > 1, visited.insert(ancestor).inserted {
            if candidates[ancestor] != nil { return true }
            current = parentByPID[ancestor]
        }
        return false
    }

    /// Helpers shipped inside a desktop app bundle (ChatGPT.app's Codex framework and
    /// bundled `codex`, Claude.app, Cursor.app) are never CLI sessions — wherever the
    /// bundle lives (#984: Codex's computer-use helper under `~/.codex/computer-use/`).
    private static func isAppBundled(executable: String, command: String) -> Bool {
        if (command.hasPrefix("/applications/") && command.contains(".app/"))
            || command.hasPrefix("/system/") {
            return true
        }
        // argv[0] may contain spaces ("Codex Computer Use.app"), so it cannot be split
        // off on whitespace. `ucomm` is its basename (truncated to 16 characters), so
        // the executable's directory is everything before the first "/<ucomm>"; only
        // that directory decides, never a bundle path that appears in the arguments.
        guard command.hasPrefix("/"), !executable.isEmpty,
              let basename = command.range(of: "/" + executable) else { return false }
        return command[..<basename.lowerBound].contains(".app/contents/")
    }

    /// Shells, launch wrappers and text tools are never the session process: a real
    /// CLI session is always its own row (`ucomm` = the Claude version string, codex,
    /// node, gemini, agy …), so dropping these loses no session. It also drops a
    /// detached helper that rewrote `$0` (cmuxlayer's inbox tail), whose ps args then
    /// show its leftover environment, `CMUX_CLAUDE_WRAPPER_SHIM=…/claude` included.
    private static let nonSessionExecutables: Set<String> = [
        "sh", "bash", "zsh", "fish", "dash", "login", "env", "sudo", "nohup", "timeout",
        "caffeinate", "script", "tmux", "screen", "perl", "tail", "cat", "sed", "less",
    ]

    private static let bridgeExecutables: Set<String> = ["brainlayer-mcp-stdio-bridge", "socat", "mcplayer"]

    private static func isBridgeOrProxy(executable: String, command: String) -> Bool {
        let launcher = command.split(whereSeparator: \.isWhitespace).first.map(String.init) ?? ""
        let launcherName = launcher.split(separator: "/").last.map(String.init) ?? launcher
        // `ucomm` is truncated to 16 characters ("brainlayer-mcp-s").
        return bridgeExecutables.contains(launcherName)
            || bridgeExecutables.contains { String($0.prefix(ucommColumnWidth)) == executable }
    }

    /// Agent binaries run in a role that is not an interactive or headless session.
    /// Each role belongs to the binary that defines it (#977 R2 B3) and is read from
    /// its argv POSITION, never from words anywhere in the args, where they may just be
    /// prompt text (#977 R1 B2):
    /// - `claude --chrome-native-host` (first argument) and `claude mcp serve`;
    /// - `codex app-server` / `codex mcp-server`, after any `-c`/`--config` pairs.
    private static func isNonSessionMode(_ command: String) -> Bool {
        let tokens = command.split(whereSeparator: \.isWhitespace)
        guard let argv0 = tokens.first else { return false }
        let binary = argv0.split(separator: "/").last.map(String.init) ?? String(argv0)
        let arguments = Array(tokens.dropFirst())

        switch binary {
        case "claude":
            if arguments.first == "--chrome-native-host" { return true }
            return arguments.count >= 2 && arguments[0] == "mcp" && arguments[1] == "serve"
        case "codex":
            var index = 0
            while index < arguments.count {
                let argument = arguments[index]
                if argument == "-c" || argument == "--config" {
                    index += 2
                } else if argument.hasPrefix("-c=") || argument.hasPrefix("--config=") {
                    index += 1
                } else {
                    break
                }
            }
            guard index < arguments.count else { return false }
            return arguments[index] == "app-server" || arguments[index] == "mcp-server"
        default:
            return false
        }
    }

    static func runSnapshotCommand(executableURL: URL, arguments: [String]) -> String? {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments

        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()

        do {
            try process.run()
        } catch {
            return nil
        }

        // Drain stdout before waiting so verbose process lists cannot fill the pipe and deadlock tests.
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static let psArguments = ["-axo", "pid=,ppid=,ucomm=,args="]

    private static func captureProcessSnapshot() -> String? {
        runSnapshotCommand(
            executableURL: URL(fileURLWithPath: "/bin/ps"),
            arguments: psArguments
        )
    }

    private static func isIgnoredProcess(executable: String, command: String) -> Bool {
        if isAgentEntrypoint(executable: executable, command: command) {
            return false
        }

        let noiseTokens = [
            "/applications/claude.app",
            "claude helper",
            "crashpad",
            "cursoruiviewservice",
            "rg -n",
            "awk ",
            "grep ",
            "ps -axo",
        ]
        if noiseTokens.contains(where: { command.contains($0) }) {
            return true
        }
        if executable == "rg" || executable == "awk" || executable == "grep" {
            return true
        }
        return false
    }

    private static func isAgentEntrypoint(executable: String, command: String) -> Bool {
        if executable == "agy" || executable == "codex" || executable == "cursor" {
            return true
        }
        if command.hasPrefix("claude ")
            || command.hasPrefix("codex ")
            || command.hasPrefix("gemini ") {
            return true
        }
        if commandHasCursorCLIEntryPoint(command) {
            return true
        }
        if command.contains("/.bun/bin/codex")
            || commandHasCursorAgentSession(command) {
            return true
        }
        return false
    }

    private static func detectActualFamily(executable: String, command: String) -> AgentFamily? {
        if executable == "agy" && commandHasModelToken(command, familyToken: "gemini") {
            return .gemini
        }
        if command.hasPrefix("claude ") || command.contains("/claude ") || command.contains(" brainlayerclaude") {
            return .claude
        }
        if command.hasPrefix("codex ")
            || command.contains("/codex/codex ")
            || (executable == "codex" && command.contains("/bin/codex "))
            || command.contains(" brainlayercodex") {
            return .codex
        }
        if commandHasCursorCLIEntryPoint(command)
            || commandHasCursorAgentSession(command)
            || command.contains(" brainlayercursor") {
            return .cursor
        }
        if command.hasPrefix("gemini ")
            || command.contains("/gemini ")
            || command.contains(" brainlayergemini") {
            return .gemini
        }
        return nil
    }

    private static func commandHasModelToken(_ command: String, familyToken: String) -> Bool {
        let tokens = command.split { character in
            character.isWhitespace || character == "="
        }
        let promptFlags = ["--prompt", "--prompt-interactive", "--message", "-p", "-i"]
        for index in tokens.indices {
            if promptFlags.contains(String(tokens[index])) {
                return false
            }
            guard tokens[index] == "--model" && index + 1 < tokens.endIndex else {
                continue
            }
            let model = tokens[index + 1]
            if model == familyToken || model.hasPrefix("\(familyToken)-") {
                return true
            }
        }
        return false
    }

    private static func detectWrapperFamily(executable: String, command: String) -> AgentFamily? {
        guard executable == "node" || executable == "bun" || executable == "python" || executable == "python3" else {
            return nil
        }
        if command.contains("/.bun/bin/codex") {
            return .codex
        }
        if commandHasCursorAgentSession(command) {
            return .cursor
        }
        if commandHasExecutableToken(command, executableName: "gemini") {
            return .gemini
        }
        return nil
    }

    private static func commandHasExecutableToken(_ command: String, executableName: String) -> Bool {
        let tokens = command.split(whereSeparator: \.isWhitespace).map(String.init)
        let promptFlags = ["--prompt", "--prompt-interactive", "--message", "-p", "-i"]
        for token in tokens {
            if promptFlags.contains(token) {
                return false
            }
            if token == executableName || token.hasSuffix("/\(executableName)") {
                return true
            }
        }
        return false
    }

    private static func commandHasCursorAgentSession(_ command: String) -> Bool {
        guard command.contains("cursor-agent") else { return false }

        let tokens = command.split(whereSeparator: \.isWhitespace).map(String.init)
        if tokens.contains("worker-server") {
            return false
        }
        if let launcher = tokens.first,
           launcher == "cursor-agent" || launcher.hasSuffix("/cursor-agent") {
            return true
        }
        for index in tokens.indices where tokens[index] == "agent" {
            guard index > tokens.startIndex else { continue }
            let launcher = tokens[tokens.index(before: index)]
            if launcher == "index.js"
                || launcher.hasSuffix("/index.js")
                || launcher == "cursor-agent"
                || launcher.hasSuffix("/cursor-agent") {
                return true
            }
        }
        return false
    }

    private static func commandHasCursorCLIEntryPoint(_ command: String) -> Bool {
        let tokens = command.split(whereSeparator: \.isWhitespace).map(String.init)
        guard tokens.count >= 2 else { return false }

        let launcher = tokens[0]
        return (launcher == "cursor" || launcher.hasSuffix("/cursor")) && tokens[1] == "agent"
    }
}
