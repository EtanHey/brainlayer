import Foundation

// "Reconnect Google Drive" (Etan: "click once and never again"). BrainBar reads and renews the
// Drive access that BrainLayer's backups use ONLY through the backend's CLI contract:
//   brainlayer backup auth --status --json → {"state":"valid"|"expiring"|"missing"|"invalid",
//                                            "reason":…, "expires_at":…, "days_left":…}
//   brainlayer backup auth --json          → {"status":"ok"|"cancelled"|"timeout"|"error", "reason":…}
// It never reads the token file and never shows a token.

struct BrainLayerCLIResult: Equatable, Sendable {
    let terminationStatus: Int32
    let stdout: String
}

/// Runs `brainlayer <arguments>`. Nil when the CLI cannot be found or launched. Injected in tests.
protocol BrainLayerCLIRunning: Sendable {
    func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult?
}

/// Drive access as `--status --json` reports it; `unknown` when that report is unavailable.
struct DriveAuthStatus: Equatable, Sendable {
    enum State: String, Sendable { case valid, expiring, missing, invalid, unknown }
    let state: State
    let reason: String?
    let expiresAt: Date?

    static func parse(_ result: BrainLayerCLIResult?) -> DriveAuthStatus {
        guard let result else { return unknown("brainlayer CLI not found") }
        guard let object = DriveAuthJSON.lastObject(in: result.stdout) else {
            return unknown(result.terminationStatus == 0
                ? "brainlayer backup auth --status returned no status"
                : "brainlayer backup auth --status failed (exit \(result.terminationStatus))")
        }
        let raw = object["state"] as? String ?? ""
        guard let state = State(rawValue: raw), state != .unknown else {
            return unknown("unrecognised Drive state \"\(DriveAuthJSON.sanitized(raw) ?? "")\"")
        }
        return DriveAuthStatus(
            state: state,
            reason: DriveAuthJSON.sanitized(object["reason"] as? String),
            expiresAt: DriveAuthJSON.date(object["expires_at"])
        )
    }

    static func unknown(_ reason: String) -> DriveAuthStatus {
        DriveAuthStatus(state: .unknown, reason: reason, expiresAt: nil)
    }
}

/// How a reconnect (`backup auth --json`) ended.
enum DriveAuthOutcome: Equatable, Sendable {
    case ok
    case cancelled(String)
    case timeout(String)
    case error(String)

    static func parse(_ result: BrainLayerCLIResult?) -> DriveAuthOutcome {
        guard let result else { return .error("brainlayer CLI not found") }
        guard let object = DriveAuthJSON.lastObject(in: result.stdout), let status = object["status"] as? String else {
            return .error("brainlayer backup auth failed (exit \(result.terminationStatus))")
        }
        let reason = DriveAuthJSON.sanitized(object["reason"] as? String) ?? "no reason given"
        switch status {
        case "ok": return .ok
        case "cancelled": return .cancelled(reason)
        case "timeout": return .timeout(reason)
        case "error": return .error(reason)
        default: return .error("unrecognised reconnect status \"\(DriveAuthJSON.sanitized(status) ?? "")\"")
        }
    }
}

/// What the Backups page and the Dashboard banner show. Pure, so every state is testable.
struct DriveAuthPresentation: Equatable, Sendable {
    enum Tone: Equatable, Sendable { case connected, expiring, attention, unknown }
    let line: String
    let detail: String?
    let tone: Tone
    let showsReconnect: Bool
    let reconnectEnabled: Bool
    /// True when the Dashboard should show the Drive banner.
    let needsAttention: Bool

    static let buttonTitle = "Reconnect Google Drive"

    static func derive(
        status: DriveAuthStatus,
        isReconnecting: Bool,
        lastOutcome: DriveAuthOutcome?,
        now: Date,
        formatDate: (Date) -> String
    ) -> DriveAuthPresentation {
        var line: String
        var detail = status.reason
        let tone: Tone
        var showsReconnect = true
        switch status.state {
        case .valid:
            line = status.expiresAt.map { "Connected: renews by \(formatDate($0))" } ?? "Connected"
            detail = nil
            tone = .connected
            showsReconnect = false
        case .expiring:
            // The app stays in Google's Testing mode, so consent lasts 7 days; day 6 asks again.
            let hours = status.expiresAt.map { max(1, Int(($0.timeIntervalSince(now) / 3_600).rounded(.up))) }
            line = hours.map { "Drive access expires in \($0) h. Reconnect" } ?? "Drive access expires soon. Reconnect"
            tone = .expiring
        case .missing:
            line = "Google Drive is not connected. Backups can't upload."
            tone = .attention
        case .invalid:
            line = "Drive access expired or was revoked. Backups can't upload."
            tone = .attention
        case .unknown:
            line = "Drive status unknown — \(status.reason ?? "no reason given")"
            detail = nil
            tone = .unknown
            // No button that cannot work: the CLI or its status report is unavailable.
            showsReconnect = false
        }
        if showsReconnect {
            if isReconnecting {
                detail = "Waiting for Google consent in your browser…"
            } else if let lastOutcome {
                switch lastOutcome {
                case .ok: break
                case let .cancelled(reason): detail = "Reconnect was cancelled: \(reason)"
                case let .timeout(reason): detail = "Reconnect timed out: \(reason)"
                case let .error(reason): detail = "Reconnect failed: \(reason)"
                }
            }
        }
        return DriveAuthPresentation(
            line: line,
            detail: detail,
            tone: tone,
            showsReconnect: showsReconnect,
            reconnectEnabled: showsReconnect && !isReconnecting,
            needsAttention: showsReconnect
        )
    }
}

enum DriveAuthJSON {
    /// The last line of stdout that is a JSON object, so a stray log line cannot hide the report.
    static func lastObject(in output: String) -> [String: Any]? {
        for line in output.split(whereSeparator: \.isNewline).reversed() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("{"),
                  let object = (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8))) as? [String: Any]
            else { continue }
            return object
        }
        return nil
    }

    static func date(_ value: Any?) -> Date? {
        guard let text = value as? String, !text.isEmpty else { return nil }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fractional.date(from: text) ?? ISO8601DateFormatter().date(from: text)
    }

    /// Google OAuth access tokens, refresh tokens and client secrets, which SecretScrubber does
    /// not cover. A reason is shown to the user, so it must never carry one.
    private static let googleCredentialPatterns = [
        #"ya29\.[A-Za-z0-9_\-\.]+"#,
        #"1//[A-Za-z0-9_\-]+"#,
        #"GOCSPX-[A-Za-z0-9_\-]+"#,
    ]

    /// A reason safe to show: Google credentials and anything SecretScrubber recognises are
    /// redacted, whitespace is trimmed and the length is capped. Empty means no reason.
    static func sanitized(_ text: String?) -> String? {
        guard var text = text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return nil }
        for pattern in googleCredentialPatterns {
            text = text.replacingOccurrences(of: pattern, with: "[REDACTED:google-credential]", options: .regularExpression)
        }
        text = SecretScrubber.scrub(text).text
        return text.count > 200 ? String(text.prefix(199)) + "…" : text
    }
}

/// The live runner: `brainlayer` resolved explicitly (a GUI app has no shell PATH), stdout
/// captured, stderr discarded, and killed after `timeout`.
struct ProcessBrainLayerCLIRunner: BrainLayerCLIRunning {
    var environment: [String: String] = ProcessInfo.processInfo.environment

    /// `BRAINLAYER_CLI` when set (and it must exist), else the Homebrew formula's stable
    /// `opt/brainlayer/bin/brainlayer` under `HOMEBREW_PREFIX`, /opt/homebrew or /usr/local.
    static func resolveExecutable(environment: [String: String], isExecutable: (String) -> Bool) -> String? {
        if let explicit = environment["BRAINLAYER_CLI"], !explicit.isEmpty {
            return isExecutable(explicit) ? explicit : nil
        }
        let prefixes = [environment["HOMEBREW_PREFIX"], "/opt/homebrew", "/usr/local"].compactMap { $0 }.filter { !$0.isEmpty }
        return prefixes.map { "\($0)/opt/brainlayer/bin/brainlayer" }.first(where: isExecutable)
    }

    func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
        guard let executable = Self.resolveExecutable(
            environment: environment,
            isExecutable: { FileManager.default.isExecutableFile(atPath: $0) }
        ) else { return nil }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        let deadline = DispatchWorkItem { if process.isRunning { process.terminate() } }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + timeout, execute: deadline)
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        deadline.cancel()
        return BrainLayerCLIResult(terminationStatus: process.terminationStatus, stdout: String(decoding: data, as: UTF8.self))
    }
}

/// The one Drive-access state BrainBar shows, shared by the Backups page and the Dashboard banner.
/// It runs the CLI off the main thread. Without a runner (tests, previews, the daemon) it never
/// runs anything.
@MainActor
final class BrainBarDriveAuthModel: ObservableObject {
    @Published private(set) var status = DriveAuthStatus.unknown("not checked yet")
    @Published private(set) var isReconnecting = false
    @Published private(set) var lastOutcome: DriveAuthOutcome?
    private(set) var lastCheckedAt: Date?

    private var runner: (any BrainLayerCLIRunning)?
    private let now: @Sendable () -> Date

    nonisolated static let statusArguments = ["backup", "auth", "--status", "--json"]
    nonisolated static let reconnectArguments = ["backup", "auth", "--json"]
    nonisolated static let statusTimeout: TimeInterval = 60
    /// Longer than the CLI's own consent timeout, so the CLI reports "timeout" itself.
    nonisolated static let reconnectTimeout: TimeInterval = 330

    init(runner: (any BrainLayerCLIRunning)?, now: @escaping @Sendable () -> Date = Date.init) {
        self.runner = runner
        self.now = now
    }

    /// Production installs the live runner once at launch.
    func install(runner: any BrainLayerCLIRunning) {
        self.runner = runner
    }

    func refreshStatus() async {
        guard let runner else {
            status = .unknown("Drive status is not checked in this process")
            return
        }
        let result = await Task.detached(priority: .utility) {
            runner.run(Self.statusArguments, timeout: Self.statusTimeout)
        }.value
        status = DriveAuthStatus.parse(result)
        lastCheckedAt = now()
    }

    /// Re-checks when the last check is older than `maxAge` (the status command starts Python).
    func refreshIfStale(maxAge: TimeInterval = 600) async {
        if let lastCheckedAt, now().timeIntervalSince(lastCheckedAt) < maxAge { return }
        await refreshStatus()
    }

    /// One click: opens Google consent in the browser through the CLI. On `ok` it re-reads the
    /// status; any other outcome keeps the state and the button, with the reason.
    func reconnect() async {
        guard !isReconnecting, let runner else { return }
        isReconnecting = true
        lastOutcome = nil
        let result = await Task.detached(priority: .userInitiated) {
            runner.run(Self.reconnectArguments, timeout: Self.reconnectTimeout)
        }.value
        let outcome = DriveAuthOutcome.parse(result)
        lastOutcome = outcome
        isReconnecting = false
        if outcome == .ok { await refreshStatus() }
    }

    func presentation(formatDate: (Date) -> String) -> DriveAuthPresentation {
        DriveAuthPresentation.derive(
            status: status, isReconnecting: isReconnecting, lastOutcome: lastOutcome, now: now(), formatDate: formatDate
        )
    }

#if DEBUG
    /// Render seam: a fixed state for the DEBUG render harness, with no CLI run.
    func setForPreview(status: DriveAuthStatus, isReconnecting: Bool = false, lastOutcome: DriveAuthOutcome? = nil) {
        self.status = status
        self.isReconnecting = isReconnecting
        self.lastOutcome = lastOutcome
        lastCheckedAt = now()
    }
#endif
}
