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
    /// The runner stopped the CLI at its timeout; it did not finish on its own.
    var timedOut = false
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
            if result.timedOut { return unknown("brainlayer backup auth --status did not finish in time") }
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
            if result.timedOut { return .timeout("brainlayer backup auth did not finish in time") }
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
/// captured, stderr discarded. It returns at its timeout whatever the child does (#1028 review
/// B1): the CLI runs in its own process group, stdout is read without blocking on it, and at the
/// deadline the whole group gets SIGTERM, a grace, then SIGKILL.
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
        return Self.spawn(executable: executable, arguments: arguments, environment: environment, timeout: timeout)
    }

    /// SIGTERM to SIGKILL.
    static let terminationGrace: TimeInterval = 2
    /// After SIGKILL, how long to wait for the exit to be reaped.
    private static let killWait: TimeInterval = 1
    /// After the CLI exits, how long a descendant may keep its stdout open before it is stopped.
    private static let drainWait: TimeInterval = 0.5

    /// Runs `executable` and returns within `timeout + grace + ~2 s`, or nil when it cannot start.
    static func spawn(
        executable: String,
        arguments: [String],
        environment: [String: String],
        timeout: TimeInterval,
        grace: TimeInterval = terminationGrace
    ) -> BrainLayerCLIResult? {
        var pipeFDs: [Int32] = [-1, -1]
        guard pipe(&pipeFDs) == 0 else { return nil }
        let readFD = pipeFDs[0], writeFD = pipeFDs[1]
        _ = fcntl(readFD, F_SETFD, FD_CLOEXEC)

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_adddup2(&actions, writeFD, STDOUT_FILENO)
        posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0)

        var attributes: posix_spawnattr_t?
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        // Its own process group, so a timeout stops the CLI and whatever it started; and no
        // BrainBar descriptor except the three above crosses into it.
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT))
        posix_spawnattr_setpgroup(&attributes, 0)

        let argv = ([executable] + arguments).map { strdup($0) } + [nil]
        let envp = environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { (argv + envp).forEach { free($0) } }

        var pid: pid_t = 0
        let spawned = posix_spawn(&pid, executable, &actions, &attributes, argv, envp)
        close(writeFD)
        guard spawned == 0 else {
            close(readFD)
            return nil
        }

        let output = CLIOutputReader(fd: readFD)
        let drained = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .utility).async {
            output.drain()
            drained.signal()
        }
        let child = CLIChild(pid: pid)
        let exited = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .utility).async {
            child.reap()
            exited.signal()
        }

        var timedOut = false
        if exited.wait(timeout: .now() + timeout) == .timedOut {
            timedOut = true
            child.signalGroup(SIGTERM)
            if exited.wait(timeout: .now() + grace) == .timedOut {
                child.signalGroup(SIGKILL)
                _ = exited.wait(timeout: .now() + killWait)
            }
        }
        // Whatever the CLI wrote is kept; a descendant still holding its stdout is stopped, not waited for.
        if drained.wait(timeout: .now() + drainWait) == .timedOut {
            child.signalGroup(SIGKILL, evenAfterExit: true)
            output.stop()
            _ = drained.wait(timeout: .now() + drainWait)
        }
        return BrainLayerCLIResult(
            terminationStatus: child.terminationStatus ?? 128 + SIGKILL,
            stdout: String(decoding: output.data, as: UTF8.self),
            timedOut: timedOut
        )
    }
}

/// Reads a pipe to EOF in short polls, so it can be told to stop, and closes it when done.
private final class CLIOutputReader: @unchecked Sendable {
    private let fd: Int32
    private let lock = NSLock()
    private var buffer = Data()
    private var stopRequested = false

    init(fd: Int32) { self.fd = fd }

    var data: Data { lock.withLock { buffer } }
    func stop() { lock.withLock { stopRequested = true } }

    func drain() {
        defer { close(fd) }
        var chunk = [UInt8](repeating: 0, count: 65_536)
        while !lock.withLock({ stopRequested }) {
            var descriptor = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let ready = poll(&descriptor, 1, 50)
            if ready == 0 { continue }
            if ready < 0 {
                if errno == EINTR { continue }
                return
            }
            let count = read(fd, &chunk, chunk.count)
            if count > 0 {
                lock.withLock { buffer.append(contentsOf: chunk[0..<count]) }
            } else if count == 0 || (errno != EINTR && errno != EAGAIN) {
                return
            }
        }
    }
}

/// The spawned CLI: reaped exactly once, and never signalled by pid once reaped, when its pid may
/// already belong to someone else.
private final class CLIChild: @unchecked Sendable {
    private let pid: pid_t
    private let lock = NSLock()
    private var status: Int32?

    init(pid: pid_t) { self.pid = pid }

    func reap() {
        var raw: Int32 = 0
        while waitpid(pid, &raw, 0) == -1 {
            guard errno == EINTR else { return }
        }
        lock.withLock { status = raw }
    }

    /// The shell convention: the exit code, or 128 + the signal that ended it.
    var terminationStatus: Int32? {
        lock.withLock {
            status.map { raw in raw & 0x7f == 0 ? (raw >> 8) & 0xff : 128 + (raw & 0x7f) }
        }
    }

    /// Signals the CLI's process group. After the CLI has been reaped the group only exists while
    /// a descendant is still in it, and a group id is not reused while its group exists; so the
    /// stray-descendant cleanup (`evenAfterExit`) may still address it.
    func signalGroup(_ signal: Int32, evenAfterExit: Bool = false) {
        lock.withLock {
            if status == nil || evenAfterExit { _ = kill(-pid, signal) }
        }
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
    /// Bumped by every status read; a read that a newer one has overtaken is dropped.
    private var statusGeneration = 0

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
        statusGeneration += 1
        let generation = statusGeneration
        let result = await Task.detached(priority: .utility) {
            runner.run(Self.statusArguments, timeout: Self.statusTimeout)
        }.value
        // An older read (say, one begun before a reconnect) never overwrites a newer one.
        guard generation == statusGeneration else { return }
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
