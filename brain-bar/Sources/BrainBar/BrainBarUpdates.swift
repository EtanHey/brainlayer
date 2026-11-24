import Foundation
import Combine
import SwiftUI
import Darwin

enum BrainBarReleaseFeed {
    enum FeedError: Error { case tapPending }
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
            guard try tapVersion(text, cask: path.hasPrefix("Casks")) == version else { throw FeedError.tapPending }
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
    @Published private(set) var installerAvailable = false
    private var installerPath: String?
    let installed: String
    private let stateRoot: String
    private let defaults: UserDefaults
    private let feed: () async throws -> String
    private let now: () -> Date
    private let launch: (String, String, String, String) throws -> Void
    private var operation: String?
    private var lastCheck: Date?
    private var lastAttempt: Date?
    private var lifecycleGeneration = 0
    private var checking = false
    private var launchedAt: Date?
    private var transientResultUntil: Date?
    nonisolated static let root = NSHomeDirectory() + "/Library/Application Support/BrainBar"
    var busy: () -> String? = { "Job status unavailable" }

    init(installerPath: String? = nil, installed: String = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
         stateRoot: String = BrainBarUpdates.root,
         defaults: UserDefaults = .standard,
         now: @escaping () -> Date = Date.init,
         feed: @escaping () async throws -> String = BrainBarReleaseFeed.latest,
         launch: @escaping (String, String, String, String) throws -> Void = BrainBarUpdates.spawn) {
        self.installerPath = installerPath; installerAvailable = installerPath != nil
        self.stateRoot = stateRoot
        self.installed = installed; self.defaults = defaults; self.now = now; self.feed = feed; self.launch = launch
        if let saved = defaults.string(forKey: "update.offer"), BrainBarReleaseFeed.newer(saved, than: installed) { offer = saved }
    }

    func prepareInstaller() async {
        guard installerPath == nil else { return }
        let path = await Task.detached {
            let p = Process(); let pipe = Pipe()
            p.executableURL = URL(fileURLWithPath: "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python")
            p.arguments = ["-c", "import importlib.resources; print(importlib.resources.files(\"brainlayer\") / \"launchd/brainbar-update.sh\")"]
            p.standardOutput = pipe; p.standardError = FileHandle.nullDevice
            do { try p.run() } catch { return Optional<String>.none }
            let data = pipe.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit()
            let path = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            return p.terminationStatus == 0 && FileManager.default.isExecutableFile(atPath: path) ? path : nil
        }.value
        installerPath = path; installerAvailable = path != nil
        if path == nil { detail = "Update unavailable — reinstall brainlayer" }
    }

    func check(force: Bool = false) async {
        guard !checking, phase != .installing else { return }
        if !force, let lastCheck, now().timeIntervalSince(lastCheck) < 6 * 3600 { return }
        let generation = lifecycleGeneration
        let previousPhase = phase, previousDetail = detail
        checking = true
        if force { phase = .checking }
        lastAttempt = now()
        defer { checking = false }
        do {
            let latest = try await feed()
            guard generation == lifecycleGeneration else { return }
            guard BrainBarReleaseFeed.components(latest) != nil else { throw URLError(.cannotParseResponse) }
            lastCheck = now()
            let superseded = offer == nil || BrainBarReleaseFeed.newer(latest, than: offer!)
            if BrainBarReleaseFeed.newer(latest, than: installed), superseded {
                offer = latest; defaults.set(latest, forKey: "update.offer"); snoozed = false
            }
            phase = previousPhase == .failed && !superseded ? .failed : .idle
            detail = phase == .failed ? previousDetail : (force && offer == nil ? "Up to date" : "")
            transientResultUntil = detail == "Up to date" ? now().addingTimeInterval(5) : nil
            if !installerAvailable { detail = "Update unavailable — reinstall brainlayer" }
            if previousPhase == .waiting, operation != nil { install() }
        } catch BrainBarReleaseFeed.FeedError.tapPending {
            guard generation == lifecycleGeneration else { return }
            phase = previousPhase; detail = "Release is reaching Homebrew"
        } catch {
            guard generation == lifecycleGeneration else { return }
            phase = previousPhase; detail = previousPhase == .failed ? previousDetail : "Could not check"
        }
    }

    func install() {
        guard phase != .installing, let offer, let installerPath else { return }
        lifecycleGeneration += 1
        if let job = busy() { phase = .waiting; detail = job == "Job status unavailable" ? "Will update when job status is available" : "Will update when \(job) finishes"; operation = operation ?? UUID().uuidString; return }
        let id = operation ?? UUID().uuidString
        do {
            try publishJobs()
            try launch(offer, id, stateRoot + "/update-jobs.json", installerPath)
            operation = id; launchedAt = now(); phase = .installing; canCancel = true; detail = "Updating…"
        } catch { phase = .failed; detail = "Could not start update" }
    }

    func cancel() {
        guard let operation else { return }
        lifecycleGeneration += 1
        if phase == .waiting { self.operation = nil; phase = .idle; detail = ""; return }
        // The helper checks this only before formula mutation. It is never killed mid-swap.
        canCancel = false; detail = "Cancelling…"
        _ = FileManager.default.createFile(atPath: stateRoot + "/cancel-" + operation, contents: Data())
    }

    func tick() async {
        if let until = transientResultUntil, now() >= until {
            if detail == "Up to date" { detail = "" }
            transientResultUntil = nil
        }
        if phase == .installing {
            try? publishJobs()
            if let text = try? String(contentsOfFile: stateRoot + "/update-state", encoding: .utf8) {
                let lines = text.components(separatedBy: "\n")
                if lines.count >= 3, lines[0] == operation {
                    canCancel = lines[1] == "preflight" && detail != "Cancelling…"
                    switch lines[1] {
                    case "success": operation = nil; phase = .idle; detail = "Updated — relaunching"
                    case "failed", "recovery-required": operation = nil; phase = .failed; detail = lines[2]
                    case "waiting": phase = .waiting; lastAttempt = now(); lastCheck = nil; detail = lines[2]
                    case "cancelled": phase = .idle; operation = nil; detail = "Cancelled"
                    default: if detail != "Cancelling…" { detail = "Updating…" }
                    }
                } else if let launchedAt, now().timeIntervalSince(launchedAt) > 15 {
                    phase = .failed; detail = "Update did not acquire custody. See update.log."
                }
            } else if let launchedAt, now().timeIntervalSince(launchedAt) > 15 {
                phase = .failed; detail = "Update helper did not start. See update.log."
            }
        }
        if lastAttempt == nil || now().timeIntervalSince(lastAttempt!) >= 6 * 3600 { await check() }
    }

    func resumeAttempt() {
        guard let text = try? String(contentsOfFile: stateRoot + "/update-state", encoding: .utf8) else { return }
        let lines = text.components(separatedBy: "\n")
        guard lines.count >= 3 else { phase = .failed; detail = "Unreadable update state"; return }
        switch lines[1] {
        case "success", "cancelled", "waiting": break
        case "failed", "recovery-required": operation = nil; phase = .failed; detail = lines[2]
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

    nonisolated static func spawn(_ version: String, _ id: String, _ jobs: String, _ script: String) throws {
        try spawn(version, id, jobs, script, root: Self.root) { arguments in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
            process.arguments = arguments
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { throw CocoaError(.executableRuntimeMismatch) }
        }
    }

    nonisolated static func spawn(_ version: String, _ id: String, _ jobs: String, _ script: String,
                                 root: String, submit: ([String]) throws -> Void) throws {
        let files = FileManager.default
        guard BrainBarReleaseFeed.components(version) != nil,
              !id.isEmpty, id.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }),
              files.isExecutableFile(atPath: script) else { throw CocoaError(.fileReadNoPermission) }
        try files.createDirectory(atPath: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let copy = root + "/update-" + id + ".sh"
        if !files.fileExists(atPath: copy) { try files.copyItem(atPath: script, toPath: copy) }
        // A one-shot job has independent launchd custody and cannot respawn after exit, including failure.
        let plist = root + "/update-job-" + id + ".plist"
        let job: [String: Any] = [
            "Label": "com.brainlayer.brainbar-update." + id + "." + UUID().uuidString,
            "ProgramArguments": ["/usr/bin/env", "BRAINBAR_UPDATE_JOBS=" + jobs, "/bin/bash", copy, version, id],
            "RunAtLoad": true, "LaunchOnlyOnce": true,
            "StandardOutPath": root + "/update.log", "StandardErrorPath": root + "/update.log"
        ]
        try PropertyListSerialization.data(fromPropertyList: job, format: .xml, options: 0)
            .write(to: URL(fileURLWithPath: plist), options: .atomic)
        try submit(["bootstrap", "gui/" + String(getuid()), plist])
    }
}

struct BrainBarUpdateRow: View {
    @ObservedObject var updates: BrainBarUpdates
    var header = false
    var showDetails: (() -> Void)? = nil
    var body: some View {
        if !header || (updates.offer != nil && !updates.snoozed) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(updates.offer == nil && !header ? "BrainBar" : "BrainBar \(updates.installed)" + (updates.offer.map { " → \($0)" } ?? ""))
                        .font(NDType.body)
                    if header && updates.phase == .failed {
                        Text("Update failed").font(NDType.caption).foregroundStyle(ND.t2)
                    } else if !updates.detail.isEmpty {
                        Text(updates.detail).font(NDType.caption).foregroundStyle(ND.t2).textSelection(.enabled)
                    }
                }
                Spacer(minLength: 0)
                if updates.offer == nil { Text(updates.installed).font(NDType.body).foregroundStyle(ND.t2) }
                if header && updates.phase == .failed, let showDetails {
                    NDButton(label: "Details", small: true, action: showDetails)
                } else if updates.phase == .waiting || (updates.phase == .installing && updates.canCancel) {
                    NDButton(label: "Cancel", small: true) { updates.cancel() }
                } else if updates.phase == .installing {
                    Text("Installing…").font(NDType.caption).foregroundStyle(ND.t2)
                } else if updates.phase == .checking {
                    Text("Checking…").font(NDType.caption).foregroundStyle(ND.t2)
                } else if updates.offer != nil && updates.installerAvailable {
                    NDButton(label: "Update", filled: header, small: true) { updates.install() }
                    if header { Button("Later") { updates.snoozed = true }.buttonStyle(.plain).font(NDType.caption).foregroundStyle(ND.t2) }
                } else {
                    NDButton(label: "Check", small: true) { Task { await updates.check(force: true) } }
                }
            }.padding(.horizontal, 12).padding(.vertical, 10)
        }
    }
}
