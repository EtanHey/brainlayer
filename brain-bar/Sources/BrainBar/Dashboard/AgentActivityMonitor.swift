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
}

struct AgentActivitySnapshot: Sendable, Equatable {
    let presences: [AgentPresence]

    static let empty = AgentActivitySnapshot(
        presences: AgentFamily.allCases.map { AgentPresence(family: $0, count: 0) }
    )

    var totalActiveAgents: Int {
        presences.reduce(0) { $0 + $1.count }
    }

    func count(for family: AgentFamily) -> Int {
        presences.first(where: { $0.family == family })?.count ?? 0
    }

    var summaryText: String {
        switch totalActiveAgents {
        case 0:
            return "No agent CLIs live"
        case 1:
            return "1 agent CLI live"
        default:
            return "\(totalActiveAgents) agent CLIs live"
        }
    }
}

final class AgentActivityMonitor {
    private let snapshotProvider: @Sendable () -> String?

    init(snapshotProvider: @escaping @Sendable () -> String? = AgentActivityMonitor.captureProcessSnapshot) {
        self.snapshotProvider = snapshotProvider
    }

    func sample() -> AgentActivitySnapshot {
        guard let snapshot = snapshotProvider() else { return .empty }
        return Self.parse(snapshot)
    }

    static func parse(_ snapshot: String) -> AgentActivitySnapshot {
        var actualCounts = Dictionary(uniqueKeysWithValues: AgentFamily.allCases.map { ($0, 0) })
        var wrapperCounts = Dictionary(uniqueKeysWithValues: AgentFamily.allCases.map { ($0, 0) })

        for rawLine in snapshot.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }
            let parts = line.split(maxSplits: 2, omittingEmptySubsequences: true) { $0 == " " || $0 == "\t" }
            guard parts.count == 3 else { continue }

            let executable = String(parts[1]).lowercased()
            let command = String(parts[2]).trimmingCharacters(in: .whitespacesAndNewlines)
            let lowerCommand = command.lowercased()

            guard !isIgnoredProcess(executable: executable, command: lowerCommand) else { continue }

            if let family = detectActualFamily(executable: executable, command: lowerCommand) {
                actualCounts[family, default: 0] += 1
                continue
            }

            if let family = detectWrapperFamily(executable: executable, command: lowerCommand) {
                wrapperCounts[family, default: 0] += 1
            }
        }

        let presences = AgentFamily.allCases.map { family in
            let actual = actualCounts[family, default: 0]
            let wrappers = wrapperCounts[family, default: 0]
            return AgentPresence(family: family, count: actual > 0 ? actual : wrappers)
        }

        return AgentActivitySnapshot(presences: presences)
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

    private static func captureProcessSnapshot() -> String? {
        runSnapshotCommand(
            executableURL: URL(fileURLWithPath: "/bin/ps"),
            arguments: ["-axo", "pid=,ucomm=,args="]
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
        if command.contains(" gemini") {
            return .gemini
        }
        return nil
    }

    private static func commandHasCursorAgentSession(_ command: String) -> Bool {
        guard command.contains("cursor-agent") else { return false }

        let tokens = command.split(whereSeparator: \.isWhitespace).map(String.init)
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
