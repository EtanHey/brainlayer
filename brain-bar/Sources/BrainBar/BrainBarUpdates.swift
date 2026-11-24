import Foundation
import Combine
import SwiftUI
import Darwin

enum BrainBarReleaseFeed {
    static func components(_ value: String) -> [Int]? {
        let parts = value.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3, parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) }) else { return nil }
        let numbers = parts.compactMap { Int($0) }
        return numbers.count == 3 ? numbers : nil
    }
    static func newer(_ candidate: String, than installed: String) -> Bool {
        guard let a = components(candidate), let b = components(installed) else { return false }
        return b.lexicographicallyPrecedes(a)
    }
    static func tapVersion(_ text: String, cask: Bool) throws -> String {
        let code = text.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
        guard code.first == (cask ? "cask \"brainbar\" do" : "class Brainlayer < Formula"), code.last == "end" else {
            throw URLError(.cannotParseResponse)
        }
        if cask {
            // BrainBar has conditional postflight activation; its version must precede all hooks.
            guard code.count > 2, code[1].hasPrefix("version ") else { throw URLError(.cannotParseResponse) }
            guard code.filter({ $0.hasPrefix("version") }).count == 1 else { throw URLError(.cannotParseResponse) }
        }
        let pattern = cask ? #"(?m)^\s*version "([0-9]+\.[0-9]+\.[0-9]+)"\s*$"# : #"(?m)^\s*url "https://files\.pythonhosted\.org/[^"\n]+/brainlayer-([0-9]+\.[0-9]+\.[0-9]+)\.tar\.gz"\s*$"#
        let regex = try NSRegularExpression(pattern: pattern)
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        guard matches.count == 1, let range = Range(matches[0].range(at: 1), in: text) else { throw URLError(.cannotParseResponse) }
        return String(text[range])
    }
    static func latest() async throws -> String {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 5
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        func fetch(_ url: String) async throws -> Data {
            let (data, response) = try await session.data(from: URL(string: url)!)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            return data
        }
        let release = try await fetch("https://api.github.com/repos/EtanHey/brainlayer/releases/latest")
        let json = try JSONSerialization.jsonObject(with: release) as? [String: Any]
        guard let tag = json?["tag_name"] as? String, tag.hasPrefix("v"),
              components(String(tag.dropFirst())) != nil else { throw URLError(.cannotParseResponse) }
        let version = String(tag.dropFirst())
        for path in ["Casks/brainbar.rb", "Formula/brainlayer.rb"] {
            let data = try await fetch("https://raw.githubusercontent.com/EtanHey/homebrew-layers/main/\(path)")
            let text = String(decoding: data, as: UTF8.self)
            guard try tapVersion(text, cask: path.hasPrefix("Casks")) == version else { throw URLError(.cannotParseResponse) }
        }
        return version
    }
}

@MainActor final class BrainBarUpdates: ObservableObject {
    enum Phase: String { case idle, checking, waiting, installing, failed }
    @Published var phase: Phase = .idle
    @Published var offer: String?
    @Published var detail = ""
    @Published var snoozed = false
    @Published private(set) var canCancel = false
    let installed: String
    private let stateRoot: String
    private let defaults: UserDefaults
    private let feed: () async throws -> String
    private let now: () -> Date
    private let launch: (String, String, String) throws -> Void
    private var operation: String?
    private var lastCheck: Date?
    private var lastAttempt: Date?
    private var checking = false
    private var launchedAt: Date?
    nonisolated static let root = NSHomeDirectory() + "/Library/Application Support/BrainBar"
    var busy: () -> String? = { "Job status unavailable" }

    init(installed: String = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
         stateRoot: String = BrainBarUpdates.root,
         defaults: UserDefaults = .standard,
         now: @escaping () -> Date = Date.init,
         feed: @escaping () async throws -> String = BrainBarReleaseFeed.latest,
         launch: @escaping (String, String, String) throws -> Void = BrainBarUpdates.spawn) {
        self.stateRoot = stateRoot
        self.installed = installed; self.defaults = defaults; self.now = now; self.feed = feed; self.launch = launch
        if let saved = defaults.string(forKey: "update.offer"), BrainBarReleaseFeed.newer(saved, than: installed) { offer = saved }
    }

    func check(force: Bool = false) async {
        guard !checking, phase != .installing, phase != .failed || force else { return }
        if !force, let lastCheck, now().timeIntervalSince(lastCheck) < 6 * 3600 { return }
        checking = true; phase = .checking
        lastAttempt = now()
        defer { checking = false }
        do {
            let latest = try await feed()
            guard BrainBarReleaseFeed.components(latest) != nil else { throw URLError(.cannotParseResponse) }
            lastCheck = now()
            if BrainBarReleaseFeed.newer(latest, than: installed),
               offer == nil || BrainBarReleaseFeed.newer(latest, than: offer!) {
                offer = latest; defaults.set(latest, forKey: "update.offer"); snoozed = false
            }
            phase = .idle; detail = offer == nil ? "Up to date" : ""
            if operation != nil { install() }
        } catch { phase = .idle; detail = "Could not check" }
    }

    func install() {
        guard phase != .installing, let offer else { return }
        if let job = busy() { phase = .waiting; detail = "Will update when \(job) finishes"; operation = operation ?? UUID().uuidString; return }
        let id = UUID().uuidString
        do {
            try publishJobs()
            try launch(offer, id, stateRoot + "/update-jobs.json")
            operation = id; launchedAt = now(); phase = .installing; canCancel = true; detail = "Updating…"
        } catch { phase = .failed; detail = "Could not start update" }
    }

    func cancel() {
        guard let operation else { return }
        if phase == .waiting { self.operation = nil; phase = .idle; detail = ""; return }
        // The helper checks this only before formula mutation. It is never killed mid-swap.
        _ = FileManager.default.createFile(atPath: stateRoot + "/cancel-" + operation, contents: Data())
    }

    func tick() async {
        if phase == .installing {
            try? publishJobs()
            if let text = try? String(contentsOfFile: stateRoot + "/update-state", encoding: .utf8) {
                let lines = text.components(separatedBy: "\n")
                if lines.count >= 3, lines[0] == operation {
                    canCancel = lines[1] == "preflight"
                    switch lines[1] {
                    case "success": phase = .idle; detail = "Updated — relaunching"
                    case "failed", "recovery-required": phase = .failed; detail = lines[2]
                    case "waiting": phase = .waiting; detail = lines[2]
                    case "cancelled": phase = .idle; operation = nil; detail = "Cancelled"
                    default: detail = "Updating…"
                    }
                } else if let launchedAt, now().timeIntervalSince(launchedAt) > 15 {
                    phase = .failed; detail = "Update did not acquire custody. See update.log."
                }
            } else if let launchedAt, now().timeIntervalSince(launchedAt) > 15 {
                phase = .failed; detail = "Update helper did not start. See update.log."
            }
        } else if phase == .waiting, busy() == nil { install() }
        if lastAttempt == nil || now().timeIntervalSince(lastAttempt!) >= 6 * 3600 { await check() }
    }

    func resumeAttempt() {
        guard let text = try? String(contentsOfFile: stateRoot + "/update-state", encoding: .utf8) else { return }
        let lines = text.components(separatedBy: "\n")
        guard lines.count >= 3 else { phase = .failed; detail = "Unreadable update state"; return }
        switch lines[1] {
        case "success", "cancelled", "waiting": break
        case "failed", "recovery-required": phase = .failed; detail = lines[2]
        default:
            operation = lines[0]
            if let text = try? String(contentsOfFile: stateRoot + "/update.lock/pid", encoding: .utf8),
               let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)), kill(pid, 0) == 0 {
                phase = .installing; detail = "Update in progress — see update.log"
            } else {
                phase = .failed; detail = "Interrupted update requires recovery. See update-state and update.log."
            }
        }
    }

    private func publishJobs() throws {
        try FileManager.default.createDirectory(atPath: stateRoot, withIntermediateDirectories: true,
                                              attributes: [.posixPermissions: 0o700])
        let data = try JSONSerialization.data(withJSONObject: ["at": now().timeIntervalSince1970, "busy": busy() ?? ""])
        try data.write(to: URL(fileURLWithPath: stateRoot + "/update-jobs.json"), options: .atomic)
    }

    nonisolated static func spawn(_ version: String, _ id: String, _ jobs: String) throws {
        let python = "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python"
        // Resolve the shipped resource using the installed interpreter, then copy outside the keg before upgrade.
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = ["-c", #"set -e; mkdir -p "$3"; source_path=$("$0" -c 'import importlib.resources; print(importlib.resources.files("brainlayer") / "launchd/brainbar-update.sh")'); cp "$source_path" "$3/update-$2.sh"; BRAINBAR_UPDATE_JOBS="$4" nohup /bin/bash "$3/update-$2.sh" "$1" "$2" >>"$3/update.log" 2>&1 </dev/null &"#, python, version, id, Self.root, jobs]
        try p.run(); p.waitUntilExit()
        guard p.terminationStatus == 0 else { throw CocoaError(.executableLoad) }
    }
}

struct BrainBarUpdateRow: View {
    @ObservedObject var updates: BrainBarUpdates
    var header = false
    var body: some View {
        if !header || (updates.offer != nil && !updates.snoozed) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("BrainBar \(updates.installed)" + (updates.offer.map { " → \($0)" } ?? ""))
                        .font(NDType.body)
                    if !updates.detail.isEmpty { Text(updates.detail).font(NDType.caption).foregroundStyle(ND.t2).textSelection(.enabled) }
                }
                Spacer(minLength: 0)
                if updates.phase == .waiting || (updates.phase == .installing && updates.canCancel) {
                    NDButton(label: "Cancel", small: true) { updates.cancel() }
                } else if updates.phase == .installing {
                    Text("Installing…").font(NDType.caption).foregroundStyle(ND.t2)
                } else if updates.phase == .checking {
                    Text("Checking…").font(NDType.caption).foregroundStyle(ND.t2)
                } else if updates.offer != nil {
                    NDButton(label: "Update", small: true) { updates.install() }
                    if header { NDButton(label: "Later", small: true) { updates.snoozed = true } }
                } else {
                    NDButton(label: "Check for updates", small: true) { Task { await updates.check(force: true) } }
                }
            }.padding(.horizontal, 12).padding(.vertical, 10)
        }
    }
}
