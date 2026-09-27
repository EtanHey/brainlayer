import AppKit
import Darwin
import Foundation

public final class BrainBarLifecycleWatchdog: @unchecked Sendable {
    public struct Configuration: Sendable {
        let watchedName: String
        let heartbeatPath: String
        let staleTimeout: TimeInterval
        let checkInterval: TimeInterval
        let terminateGraceInterval: TimeInterval
        let relaunchCommand: RelaunchCommand
        /// Every restart decision is appended here as well as to NSLog, so a
        /// restart is explainable from the daemon's own log.
        let eventLogPath: String?

        public init(
            watchedName: String,
            heartbeatPath: String,
            staleTimeout: TimeInterval = 45,
            checkInterval: TimeInterval = 10,
            terminateGraceInterval: TimeInterval = 2,
            relaunchCommand: RelaunchCommand,
            eventLogPath: String? = nil
        ) {
            self.watchedName = watchedName
            self.heartbeatPath = heartbeatPath
            self.staleTimeout = staleTimeout
            self.checkInterval = checkInterval
            self.terminateGraceInterval = terminateGraceInterval
            self.relaunchCommand = relaunchCommand
            self.eventLogPath = eventLogPath
        }
    }

    /// Clocks the heartbeat is written and judged with. Staleness is measured on
    /// CLOCK_UPTIME_RAW, which pauses while the Mac sleeps, so a heartbeat that
    /// could not advance during sleep does not look stale on wake. The wall
    /// clock is kept only for legacy heartbeats that carry no uptime.
    public struct HeartbeatClock: Sendable {
        public let wallNow: @Sendable () -> Date
        public let uptimeNanos: @Sendable () -> UInt64
        /// `kern.bootsessionuuid`. Uptime values from different boots are not
        /// comparable. `kern.boottime` is not used because XNU shifts it when
        /// the wall clock is stepped, e.g. by NTP right after wake.
        public let bootSession: @Sendable () -> String

        public init(
            wallNow: @escaping @Sendable () -> Date,
            uptimeNanos: @escaping @Sendable () -> UInt64,
            bootSession: @escaping @Sendable () -> String
        ) {
            self.wallNow = wallNow
            self.uptimeNanos = uptimeNanos
            self.bootSession = bootSession
        }

        public static let system = HeartbeatClock(
            wallNow: { Date() },
            uptimeNanos: { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) },
            bootSession: { BrainBarLifecycleWatchdog.systemBootSession }
        )
    }

    public struct RelaunchCommand: Sendable {
        let executablePath: String
        let arguments: [String]

        public static func launchctlKickstart(label: String) -> RelaunchCommand {
            RelaunchCommand(
                executablePath: "/bin/launchctl",
                arguments: ["kickstart", "-k", "gui/\(getuid())/\(label)"]
            )
        }

        public static func openBundle(_ bundlePath: String) -> RelaunchCommand {
            RelaunchCommand(executablePath: "/usr/bin/open", arguments: ["-n", bundlePath])
        }
    }

    public static let uiHeartbeatPath = "/tmp/brainbar-ui.heartbeat"
    public static let daemonHeartbeatPath = "/tmp/brainbar-daemon.heartbeat"
    public static let uiLaunchAgentLabel = "com.brainlayer.brainbar"
    public static let daemonLaunchAgentLabel = "com.brainlayer.brainbar-daemon"
    public static let daemonSocketPath = "/tmp/brainbar.sock"
    public static let daemonDebugLogPath = "/tmp/brainbar-debug.log"
    /// How long the daemon gets to answer a `ping` before a stale heartbeat
    /// is treated as a hang.
    public static let daemonProbeTimeout: TimeInterval = 5

    private let configuration: Configuration
    private let processProvider: @Sendable () -> [pid_t]
    private let terminateProcess: @Sendable (pid_t, Int32) -> Void
    private let relaunch: @Sendable (RelaunchCommand) -> Void
    private let clock: HeartbeatClock
    /// A positive liveness check run before any restart. When it answers, a
    /// stale heartbeat alone never kills the process. nil means the watched
    /// process has nothing to probe (the UI app), and awake-time staleness is
    /// the failure signal on its own.
    private let livenessProbe: (@Sendable () -> Bool)?
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.lifecycle-watchdog", qos: .utility)
    private var timer: DispatchSourceTimer?
    private var isRestarting = false
    private var loggedStaleButAnswering = false

    init(
        configuration: Configuration,
        processProvider: @escaping @Sendable () -> [pid_t],
        terminateProcess: @escaping @Sendable (pid_t, Int32) -> Void = { pid, signal in
            _ = Darwin.kill(pid, signal)
        },
        relaunch: @escaping @Sendable (RelaunchCommand) -> Void = { command in
            _ = BrainBarLifecycleWatchdog.run(command: command)
        },
        clock: HeartbeatClock = .system,
        livenessProbe: (@Sendable () -> Bool)? = nil
    ) {
        self.configuration = configuration
        self.processProvider = processProvider
        self.terminateProcess = terminateProcess
        self.relaunch = relaunch
        self.clock = clock
        self.livenessProbe = livenessProbe
    }

    var hasLivenessProbe: Bool { livenessProbe != nil }
    var eventLogPath: String? { configuration.eventLogPath }

    public func start() {
        queue.async { [weak self] in
            guard let self, self.timer == nil else { return }
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(
                deadline: .now() + self.configuration.checkInterval,
                repeating: self.configuration.checkInterval
            )
            timer.setEventHandler { [weak self] in
                self?.check()
            }
            self.timer = timer
            timer.resume()
        }
    }

    public func stop() {
        queue.sync {
            timer?.cancel()
            timer = nil
            isRestarting = false
        }
    }

    /// Runs one check synchronously on the watchdog queue.
    func checkNow() {
        queue.sync { check() }
    }

    private func check() {
        guard !isRestarting else { return }
        let age = Self.heartbeatAwakeAge(atPath: configuration.heartbeatPath, clock: clock)
        guard age > configuration.staleTimeout else {
            loggedStaleButAnswering = false
            return
        }

        let name = configuration.watchedName
        let staleness = age.isFinite
            ? String(format: "heartbeat stale for %.0fs awake (limit %.0fs)", age, configuration.staleTimeout)
            : "heartbeat missing, unreadable, or from a previous boot"
        let evidence: String
        if let livenessProbe {
            guard !livenessProbe() else {
                if !loggedStaleButAnswering {
                    loggedStaleButAnswering = true
                    log("\(name) \(staleness) but its liveness probe answered; not restarting")
                }
                return
            }
            evidence = "liveness probe failed"
        } else {
            evidence = "no liveness probe for this process"
        }
        loggedStaleButAnswering = false

        let pids = processProvider()
        guard !pids.isEmpty else {
            isRestarting = true
            log("\(name) \(staleness), \(evidence), and no process is running; requesting launch")
            relaunch(configuration.relaunchCommand)
            queue.asyncAfter(deadline: .now() + configuration.terminateGraceInterval) { [weak self] in
                self?.isRestarting = false
            }
            return
        }

        isRestarting = true
        log("\(name) \(staleness), \(evidence); terminating PIDs \(pids.map(String.init).joined(separator: ","))")
        pids.forEach { terminateProcess($0, SIGTERM) }
        let stalePIDs = pids
        queue.asyncAfter(deadline: .now() + configuration.terminateGraceInterval) { [weak self] in
            guard let self else { return }
            stalePIDs.forEach { self.terminateProcess($0, SIGKILL) }
            self.relaunch(self.configuration.relaunchCommand)
            self.queue.asyncAfter(deadline: .now() + self.configuration.terminateGraceInterval) { [weak self] in
                self?.isRestarting = false
            }
        }
    }

    private func log(_ message: String) {
        NSLog("[BrainBarWatchdog] %@", message)
        guard let path = configuration.eventLogPath else { return }
        Self.appendEventLogLine("[BrainBarWatchdog] \(message)", to: path, now: clock.wallNow())
    }

    static func appendEventLogLine(_ message: String, to path: String, now: Date) {
        let line = "[\(ISO8601DateFormatter().string(from: now))] \(message)\n"
        let fd = open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0o644)
        guard fd >= 0 else { return }
        defer { close(fd) }
        _ = line.withCString { write(fd, $0, strlen($0)) }
    }

    /// Legacy wall-clock check, kept for heartbeats that carry no uptime.
    public static func isHeartbeatStale(atPath path: String, now: Date, timeout: TimeInterval) -> Bool {
        let clock = HeartbeatClock(
            wallNow: { now },
            uptimeNanos: HeartbeatClock.system.uptimeNanos,
            bootSession: HeartbeatClock.system.bootSession
        )
        return heartbeatAwakeAge(atPath: path, clock: clock) > timeout
    }

    /// Seconds the Mac has been awake since the heartbeat was written.
    /// `.infinity` when the heartbeat is missing, unreadable, or from another
    /// boot. A legacy heartbeat (wall-clock only) falls back to its mtime age,
    /// where sleep does count; the liveness probe is the backstop for that.
    public static func heartbeatAwakeAge(atPath path: String, clock: HeartbeatClock) -> TimeInterval {
        guard let payload = try? String(contentsOfFile: path, encoding: .utf8) else {
            return .infinity
        }
        var uptime: UInt64?
        var boot: String?
        for line in payload.split(separator: "\n") {
            if line.hasPrefix("uptime_ns=") {
                uptime = UInt64(line.dropFirst("uptime_ns=".count))
            } else if line.hasPrefix("boot=") {
                boot = String(line.dropFirst("boot=".count))
            }
        }
        guard let uptime, let boot else {
            guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
                  let modifiedAt = attributes[.modificationDate] as? Date
            else {
                return .infinity
            }
            return max(0, clock.wallNow().timeIntervalSince(modifiedAt))
        }
        let now = clock.uptimeNanos()
        guard boot == clock.bootSession(), now >= uptime else {
            return .infinity
        }
        return TimeInterval(now - uptime) / 1_000_000_000
    }

    /// Line 1 stays the wall-clock epoch so any legacy reader keeps working.
    public static func writeHeartbeat(to path: String, clock: HeartbeatClock = .system) {
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let payload = """
        \(clock.wallNow().timeIntervalSince1970)
        uptime_ns=\(clock.uptimeNanos())
        boot=\(clock.bootSession())

        """
        try? payload.write(to: url, atomically: true, encoding: .utf8)
    }

    static let systemBootSession: String = {
        var size = 0
        guard sysctlbyname("kern.bootsessionuuid", nil, &size, nil, 0) == 0, size > 0 else {
            return "unknown"
        }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname("kern.bootsessionuuid", &buffer, &size, nil, 0) == 0 else {
            return "unknown"
        }
        return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
    }()

    /// True only when the socket answers a framed MCP `ping` with a result for
    /// our id within `timeout`. A successful connect() is not an answer: the
    /// kernel completes it into the listen backlog even when the daemon is wedged.
    public static func socketAnswersPing(path: String, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var noSigPipe: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(path.utf8)
        guard pathBytes.count < MemoryLayout.size(ofValue: addr.sun_path) else { return false }
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            raw.copyBytes(from: pathBytes)
            raw[pathBytes.count] = 0
        }
        let connected = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if connected != 0 {
            guard errno == EINPROGRESS, waitFor(fd, events: Int16(POLLOUT), until: deadline) else { return false }
            var socketError: Int32 = 0
            var length = socklen_t(MemoryLayout<Int32>.size)
            guard getsockopt(fd, SOL_SOCKET, SO_ERROR, &socketError, &length) == 0, socketError == 0 else {
                return false
            }
        }

        let id = "brainbar-watchdog-probe-\(UUID().uuidString)"
        let body = "{\"jsonrpc\":\"2.0\",\"id\":\"\(id)\",\"method\":\"ping\"}"
        let request = Array("Content-Length: \(body.utf8.count)\r\n\r\n\(body)".utf8)
        var sent = 0
        while sent < request.count {
            let n = request.withUnsafeBytes { write(fd, $0.baseAddress! + sent, $0.count - sent) }
            if n > 0 {
                sent += n
            } else if n < 0, errno == EAGAIN || errno == EINTR {
                guard waitFor(fd, events: Int16(POLLOUT), until: deadline) else { return false }
            } else {
                return false
            }
        }

        var response = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while waitFor(fd, events: Int16(POLLIN), until: deadline) {
            let n = read(fd, &buffer, buffer.count)
            if n > 0 {
                response.append(contentsOf: buffer[0..<n])
                if pingResponse(response, answers: id) { return true }
            } else if n < 0, errno == EAGAIN || errno == EINTR {
                continue
            } else {
                return false
            }
        }
        return false
    }

    static func pingResponse(_ data: Data, answers id: String) -> Bool {
        let body = data.range(of: Data("\r\n\r\n".utf8)).map { data[$0.upperBound...] } ?? data[...]
        guard let object = try? JSONSerialization.jsonObject(with: Data(body)) as? [String: Any] else {
            return false
        }
        return object["id"] as? String == id && object["result"] != nil
    }

    private static func waitFor(_ fd: Int32, events: Int16, until deadline: Date) -> Bool {
        while true {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { return false }
            var descriptor = pollfd(fd: fd, events: events, revents: 0)
            let ready = poll(&descriptor, 1, Int32(min(remaining * 1000, Double(Int32.max)).rounded(.up)))
            if ready > 0 { return descriptor.revents & (events | Int16(POLLHUP) | Int16(POLLERR)) != 0 }
            if ready == 0 { return false }
            if errno != EINTR { return false }
        }
    }

    public static func makeHeartbeatTimer(path: String, interval: TimeInterval, queue: DispatchQueue) -> DispatchSourceTimer {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: interval)
        timer.setEventHandler {
            writeHeartbeat(to: path)
        }
        timer.resume()
        return timer
    }

    public static func runningPIDs(named executableName: String, bundleIdentifiers: [String] = []) -> [pid_t] {
        let currentPID = ProcessInfo.processInfo.processIdentifier
        let appPIDs = NSWorkspace.shared.runningApplications.compactMap { app -> pid_t? in
            guard app.processIdentifier != currentPID else { return nil }
            if bundleIdentifiers.contains(app.bundleIdentifier ?? "") {
                return app.processIdentifier
            }
            if app.localizedName == executableName || app.executableURL?.lastPathComponent == executableName {
                return app.processIdentifier
            }
            return nil
        }
        if !appPIDs.isEmpty {
            return Array(Set(appPIDs))
        }

        return pgrep(executableName).filter { $0 != currentPID }
    }

    private static func pgrep(_ executableName: String) -> [pid_t] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        process.arguments = ["-x", executableName]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return []
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0,
              let output = String(data: data, encoding: .utf8)
        else {
            return []
        }
        return output
            .components(separatedBy: .newlines)
            .compactMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            .map { pid_t($0) }
    }

    private static func run(command: RelaunchCommand) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command.executablePath)
        process.arguments = command.arguments
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            NSLog("[BrainBarWatchdog] Failed to run %@ %@: %@", command.executablePath, command.arguments.joined(separator: " "), String(describing: error))
            return false
        }
    }

    public static func makeDaemonWatchdog(socketPath: String = daemonSocketPath) -> BrainBarLifecycleWatchdog {
        BrainBarLifecycleWatchdog(
            configuration: Configuration(
                watchedName: "BrainBarDaemon",
                heartbeatPath: daemonHeartbeatPath,
                relaunchCommand: .launchctlKickstart(label: daemonLaunchAgentLabel),
                eventLogPath: daemonDebugLogPath
            ),
            processProvider: {
                runningPIDs(named: "BrainBarDaemon", bundleIdentifiers: ["com.brainlayer.brainbar-daemon", "com.brainlayer.BrainBarDaemon"])
            },
            livenessProbe: {
                socketAnswersPing(path: socketPath, timeout: daemonProbeTimeout)
            }
        )
    }

    public static func makeUIWatchdog(bundlePath: String = Bundle.main.bundlePath) -> BrainBarLifecycleWatchdog {
        BrainBarLifecycleWatchdog(
            configuration: Configuration(
                watchedName: "BrainBar",
                heartbeatPath: uiHeartbeatPath,
                relaunchCommand: .launchctlKickstart(label: uiLaunchAgentLabel),
                eventLogPath: daemonDebugLogPath
            ),
            processProvider: {
                runningPIDs(named: "BrainBar", bundleIdentifiers: ["com.brainlayer.BrainBar"])
            },
            relaunch: { command in
                let launchctlSucceeded = run(command: command)
                if !launchctlSucceeded, FileManager.default.fileExists(atPath: bundlePath) {
                    _ = run(command: .openBundle(bundlePath))
                }
            }
        )
    }
}
