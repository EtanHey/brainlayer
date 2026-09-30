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

final class AgentActivityMonitor: Sendable {
    private let snapshotProvider: @Sendable () -> String?
    private let executablePathResolver: @Sendable (Int32) -> String?

    /// Both seams are required (#990 N1): the live pair runs `ps` and `proc_pidpath`, so it is
    /// named explicitly (`.live`) where production builds a monitor, and never a default a
    /// test could inherit.
    init(
        snapshotProvider: @escaping @Sendable () -> String?,
        executablePathResolver: @escaping @Sendable (Int32) -> String?
    ) {
        self.snapshotProvider = snapshotProvider
        self.executablePathResolver = executablePathResolver
    }

    /// The live process table: `ps` rows, and the kernel's executable path per candidate PID.
    static let live = AgentActivityMonitor(
        snapshotProvider: AgentActivityMonitor.captureProcessSnapshot,
        executablePathResolver: AgentActivityMonitor.kernelExecutablePath
    )

    func sample() -> AgentActivitySnapshot {
        guard let snapshot = snapshotProvider() else { return .unavailable("ps capture failed") }
        return Self.parse(snapshot, executablePath: executablePathResolver)
    }

    struct ProcessRow: Equatable {
        let pid: Int32
        let parentPID: Int32
        /// Lowercased `ucomm` (16-character short name, may contain spaces).
        let executable: String
        /// Lowercased `args`.
        let command: String
        /// The kernel's path for this PID's executable, resolved for a row that names an agent
        /// CLI, before it is classified. Nil when unresolved.
        var executablePath: String? = nil
    }

    /// `ucomm` is printed left-justified in a fixed MAXCOMLEN (16) column, so it can be
    /// split off by width even when it contains spaces ("Codex (Service)").
    static let ucommColumnWidth = 16

    /// Counts agent SESSIONS (#971): a row is a candidate when it looks like an agent
    /// CLI and is not an app-bundled helper, bridge/proxy, or non-session mode; a
    /// candidate counts only when none of its ancestors is itself a candidate, so a
    /// session's wrappers, MCP children and nested CLIs fold into that one session.
    /// `executablePath` answers where a PID's executable lives (#987); it alone decides app
    /// bundling (#990), because args text cannot say where a spaced argv[0] ends and a later
    /// argument may name any bundle.
    static func parse(
        _ snapshot: String,
        executablePath: (Int32) -> String? = { _ in nil }
    ) -> AgentActivitySnapshot {
        let rows = snapshot.split(whereSeparator: \.isNewline).compactMap(parseRow)
        var parentByPID: [Int32: Int32] = [:]
        var candidates: [Int32: AgentFamily] = [:]
        for var row in rows {
            parentByPID[row.pid] = row.parentPID
            guard mentionsAnAgent(row) else { continue }
            // The kernel path is resolved BEFORE classification (#1019 R1): it marks where a
            // spaced argv[0] ends, so the binary, its role and its flags are read correctly.
            row.executablePath = executablePath(row.pid)
            guard let family = candidateFamily(row, Invocation(command: row.command, kernelPath: row.executablePath)),
                  !isAppOrSystemExecutable(row.executablePath) else { continue }
            candidates[row.pid] = family
        }

        var counts = Dictionary(uniqueKeysWithValues: AgentFamily.allCases.map { ($0, 0) })
        for (pid, family) in candidates where !hasCandidateAncestor(pid, parentByPID: parentByPID, candidates: candidates) {
            counts[family, default: 0] += 1
        }

        return AgentActivitySnapshot(
            presences: AgentFamily.allCases.map { AgentPresence(family: $0, count: counts[$0, default: 0]) }
        )
    }

    /// argv[0] and the arguments after it. ps prints argv joined by spaces, so a path with a space
    /// ("AI Tools") would split into two tokens; when the args begin with the kernel's executable
    /// path, that whole path is argv[0] (#1019 R1). Otherwise argv[0] is the first token, which is
    /// the case for bare names and for symlinked launchers the kernel resolved elsewhere.
    struct Invocation: Equatable {
        let argv0: String
        let arguments: [String]

        init(command: String, kernelPath: String?) {
            let path = kernelPath?.lowercased() ?? ""
            let rest: Substring
            if !path.isEmpty, command == path || command.hasPrefix(path + " ") {
                argv0 = path
                rest = command.dropFirst(path.count)
            } else {
                let first = command.split(maxSplits: 1, omittingEmptySubsequences: true, whereSeparator: \.isWhitespace)
                argv0 = first.first.map(String.init) ?? ""
                rest = first.count > 1 ? first[1] : ""
            }
            arguments = rest.split(whereSeparator: \.isWhitespace).map(String.init)
        }

        /// The executable's file name, e.g. `claude` for `/Users/u/AI Tools/claude`.
        var binary: String { argv0.split(separator: "/").last.map(String.init) ?? argv0 }
    }

    /// Only rows that name an agent CLI somewhere are classified, and only those pay for a
    /// kernel path lookup.
    private static func mentionsAnAgent(_ row: ProcessRow) -> Bool {
        ["claude", "codex", "gemini", "cursor", "agy"].contains { row.command.contains($0) || row.executable.contains($0) }
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

    private static func candidateFamily(_ row: ProcessRow, _ invocation: Invocation) -> AgentFamily? {
        let executable = row.executable
        let command = row.command
        guard !command.isEmpty,
              !nonSessionExecutables.contains(executable),
              !isBridgeOrProxy(executable: executable, invocation: invocation),
              !isNonSessionMode(invocation),
              !isIgnoredProcess(executable: executable, invocation: invocation)
        else { return nil }
        return detectActualFamily(executable: executable, command: command)
            ?? bareCLIFamilies[invocation.binary]
            ?? detectWrapperFamily(executable: executable, command: command)
    }

    /// A CLI named by its argv[0] file name (`claude`, `/Users/u/AI Tools/claude`), including one
    /// started with no arguments, which the "claude " detectors miss.
    private static let bareCLIFamilies: [String: AgentFamily] = [
        "claude": .claude,
        "codex": .codex,
        "gemini": .gemini,
        "cursor-agent": .cursor,
    ]

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

    /// Helpers shipped inside a desktop app bundle (ChatGPT.app's Codex framework and bundled
    /// `codex`, Claude.app, Cursor.app) and system executables are never CLI sessions. Only
    /// the kernel path decides (#990): a CLI under /Applications that is not in a bundle, or
    /// whose arguments name one, is a session.
    private static func isAppOrSystemExecutable(_ executablePath: String?) -> Bool {
        isInsideAppBundle(executablePath) || executablePath?.lowercased().hasPrefix("/system/") == true
    }

    /// An executable inside a `*.app/Contents/` bundle is an app helper wherever the bundle
    /// lives (#984: Codex's computer-use helper under `~/.codex/computer-use/`). That
    /// includes any Python.app-hosted process, framework or Homebrew (#987 N1); no agent CLI
    /// is Python-hosted today. An unresolved path is not app-bundled.
    private static func isInsideAppBundle(_ executablePath: String?) -> Bool {
        guard let executablePath else { return false }
        let directories = executablePath.lowercased().split(separator: "/").dropLast()
        return zip(directories, directories.dropFirst()).contains { $0.hasSuffix(".app") && $1 == "contents" }
    }

    /// The kernel's path for the executable `pid` is running (`proc_pidpath`, symlinks
    /// resolved). Unlike args text, a space in it is always part of the path. Nil when the
    /// process has exited or the kernel refuses; the caller then does not treat the row as
    /// app-bundled.
    static func kernelExecutablePath(_ pid: Int32) -> String? {
        var buffer = [CChar](repeating: 0, count: 4 * Int(MAXPATHLEN)) // PROC_PIDPATHINFO_MAXSIZE
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else { return nil }
        return String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
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

    private static func isBridgeOrProxy(executable: String, invocation: Invocation) -> Bool {
        // `ucomm` is truncated to 16 characters ("brainlayer-mcp-s").
        return bridgeExecutables.contains(invocation.binary)
            || bridgeExecutables.contains { String($0.prefix(ucommColumnWidth)) == executable }
    }

    /// Agent binaries run in a role that is not an interactive or headless session.
    /// Each role belongs to the binary that defines it (#977 R2 B3) and is read from
    /// its argv POSITION, never from words anywhere in the args, where they may just be
    /// prompt text (#977 R1 B2):
    /// - `claude --chrome-native-host` (first argument), `claude mcp serve` and
    ///   `claude remote-control` (#982; the `--remote-control` FLAG is a session);
    /// - `codex app-server` / `mcp-server` / `remote-control` (#982), after every global flag.
    private static func isNonSessionMode(_ invocation: Invocation) -> Bool {
        let arguments = invocation.arguments
        switch invocation.binary {
        case "claude":
            if arguments.first == "--chrome-native-host" || arguments.first == "remote-control" { return true }
            return arguments.count >= 2 && arguments[0] == "mcp" && arguments[1] == "serve"
        case "codex":
            var index = 0
            while index < arguments.count, arguments[index].hasPrefix("-") {
                // A value-taking flag written as two tokens consumes its value too; `--x=value`
                // and boolean flags are one token.
                index += codexValueFlags.contains(arguments[index]) ? 2 : 1
            }
            guard index < arguments.count else { return false }
            return codexNonSessionRoles.contains(arguments[index])
        default:
            return false
        }
    }

    private static let codexNonSessionRoles: Set<String> = ["app-server", "mcp-server", "remote-control"]

    /// `codex --help` global options that take a value (#982), lowercased as ps args are read
    /// (`-C` is `-c`, which also takes a value). Every other flag is boolean.
    private static let codexValueFlags: Set<String> = [
        "-c", "--config", "--enable", "--disable", "--remote", "--remote-auth-token-env",
        "-i", "--image", "-m", "--model", "--local-provider", "-p", "--profile",
        "-s", "--sandbox", "--cd", "--add-dir", "-a", "--ask-for-approval",
    ]

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

    /// Helpers and text tools, recognised by the executable (`ucomm`) or argv[0] only (#990,
    /// #1019 R1 B2). A word like "grep" or "crashpad" later in the ARGUMENTS is just an argument:
    /// it never drops a real CLI. Claude.app's processes are also dropped by their kernel path.
    private static func isIgnoredProcess(executable: String, invocation: Invocation) -> Bool {
        let noiseNames = ["claude helper", "crashpad", "cursoruiviewservice"]
        if noiseNames.contains(where: { executable.contains($0) || invocation.argv0.contains($0) }) {
            return true
        }
        return ["rg", "awk", "grep", "ps"].contains(executable)
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
