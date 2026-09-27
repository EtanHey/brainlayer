import Darwin
import Foundation

/// Per-process facts the daemon monitor reads. Injected so the identity rules are
/// testable against a scripted process table, with no real process spawned.
protocol DaemonProcessInspecting: Sendable {
    /// Every PID on the machine, or `nil` when the process table cannot be read.
    func processIDs() -> [pid_t]?
    /// The 16-character `p_comm`; a cheap prefilter, never proof of identity.
    func shortName(of pid: pid_t) -> String?
    func executablePath(of pid: pid_t) -> String?
    /// Whether `pid` holds a LISTENING Unix socket bound to `path`; `nil` when its
    /// file descriptors cannot be read, which is "not measured", never "no".
    func isListening(_ pid: pid_t, onUnixSocket path: String) -> Bool?
    /// `nil` when the process is gone or its task info is unreadable.
    func processInfo(of pid: pid_t) -> DaemonProcessInfo?
}

struct DaemonProcessInfo: Sendable, Equatable {
    let rssBytes: UInt64
    let startedAt: Date
    let openSockets: Int
}

/// Why there is no daemon snapshot. `down` is a confirmed outage; `unknown` means the
/// monitor could not measure (#976 R2 B3) and must never be shown as an outage.
enum DaemonUnavailability: Sendable, Equatable {
    case down(String)
    case unknown(String)
}

/// A monitor reading: a snapshot when the daemon is running, otherwise why not, so
/// the Runtime rows can say why instead of a bare "Unavailable".
struct DaemonHealthReading: Sendable, Equatable {
    let snapshot: DaemonHealthSnapshot?
    let unavailability: DaemonUnavailability?
}

/// Which process IS the production daemon (#972, #976 R1): the one listening on the
/// BrainBar socket whose executable is the installed app's `BrainBarDaemon`. A
/// basename match, a pidfile, a DEV/scratch build or a reused PID is never proof.
struct DaemonIdentity: Sendable {
    static let executableName = "BrainBarDaemon"
    static let installedExecutablePath = "/Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon"

    let socketPath: String

    enum Resolution: Equatable {
        case found(pid_t)
        case unavailable(DaemonUnavailability)
    }

    static func isInstalledExecutable(_ path: String) -> Bool {
        path == installedExecutablePath
    }

    func isProductionDaemon(_ pid: pid_t, inspector: any DaemonProcessInspecting) -> Bool {
        guard pid > 0,
              let path = inspector.executablePath(of: pid),
              Self.isInstalledExecutable(path) else { return false }
        return inspector.isListening(pid, onUnixSocket: socketPath) == true
    }

    func resolve(inspector: any DaemonProcessInspecting) -> Resolution {
        guard let pids = inspector.processIDs() else {
            return .unavailable(.unknown("process table unreadable"))
        }
        let named = pids.filter { $0 > 0 && inspector.shortName(of: $0) == Self.executableName }
        let listening = Dictionary(uniqueKeysWithValues: named.map {
            ($0, inspector.isListening($0, onUnixSocket: socketPath))
        })
        let installed = named.filter { inspector.executablePath(of: $0).map(Self.isInstalledExecutable) == true }
        if let pid = installed.first(where: { listening[$0] == .some(true) }) {
            return .found(pid)
        }
        if let impostor = named.first(where: { listening[$0] == .some(true) }) {
            let path = inspector.executablePath(of: impostor) ?? "path unreadable"
            return .unavailable(.down("\(socketPath) is served by a non-installed BrainBarDaemon (\(path))"))
        }
        // A daemon whose sockets could not be read may be the one serving: unmeasured.
        if let unread = installed.first(where: { listening[$0] == .some(nil) }) {
            return .unavailable(.unknown("sockets of BrainBarDaemon PID \(unread) unreadable"))
        }
        if !installed.isEmpty {
            return .unavailable(.down("BrainBarDaemon is running but not serving \(socketPath)"))
        }
        return .unavailable(.down("BrainBarDaemon not running"))
    }
}

final class DaemonHealthMonitor: @unchecked Sendable {
    private enum Source {
        /// Test seam: trust one PID as the daemon without identity checks.
        case pinned(pid_t)
        case production(DaemonIdentity)
    }

    private let source: Source
    private let inspector: any DaemonProcessInspecting
    private let now: @Sendable () -> Date
    private let lock = NSLock()
    private var verifiedPID: pid_t?

    init(
        inspector: any DaemonProcessInspecting = LiveDaemonProcessInspector(),
        socketPath: String = BrainBarServer.defaultSocketPath(),
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.source = .production(DaemonIdentity(socketPath: socketPath))
        self.inspector = inspector
        self.now = now
    }

    /// Watches one fixed PID with no identity check. Test seam only.
    init(targetPID: pid_t) {
        self.source = .pinned(targetPID)
        self.inspector = LiveDaemonProcessInspector()
        self.now = Date.init
    }

    func sample() -> DaemonHealthSnapshot? {
        read().snapshot
    }

    /// Blocking (a process-table scan when the verified PID is gone); callers on the
    /// main actor must run it off-main.
    func read() -> DaemonHealthReading {
        lock.withLock {
            switch source {
            case .pinned(let pid):
                guard pid > 0, inspector.executablePath(of: pid) != nil else {
                    return DaemonHealthReading(snapshot: nil, unavailability: .down("PID \(pid) not running"))
                }
                return DaemonHealthReading(snapshot: snapshot(for: pid), unavailability: nil)
            case .production(let identity):
                return readProduction(identity)
            }
        }
    }

    private func readProduction(_ identity: DaemonIdentity) -> DaemonHealthReading {
        // Re-verified every sample, so a reused PID or a daemon that stopped serving
        // the socket is dropped instead of trusted.
        if let pid = verifiedPID, identity.isProductionDaemon(pid, inspector: inspector) {
            return DaemonHealthReading(snapshot: snapshot(for: pid), unavailability: nil)
        }
        let lastPID = verifiedPID
        verifiedPID = nil
        switch identity.resolve(inspector: inspector) {
        case .found(let pid):
            verifiedPID = pid
            return DaemonHealthReading(snapshot: snapshot(for: pid), unavailability: nil)
        case .unavailable(.down(let reason)):
            let described = lastPID.map { "\(reason) (last PID \($0) exited)" } ?? reason
            return DaemonHealthReading(snapshot: nil, unavailability: .down(described))
        case .unavailable(.unknown(let reason)):
            return DaemonHealthReading(snapshot: nil, unavailability: .unknown(reason))
        }
    }

    private func snapshot(for pid: pid_t) -> DaemonHealthSnapshot {
        guard let info = inspector.processInfo(of: pid) else {
            return DaemonHealthSnapshot(
                pid: pid,
                isResponsive: false,
                rssBytes: 0,
                uptime: 0,
                openConnections: 0,
                lastSeenAt: nil,
                startedAt: nil
            )
        }
        return DaemonHealthSnapshot(
            pid: pid,
            isResponsive: true,
            rssBytes: info.rssBytes,
            uptime: max(0, now().timeIntervalSince(info.startedAt)),
            openConnections: info.openSockets,
            lastSeenAt: nil,
            startedAt: info.startedAt
        )
    }
}

/// Reads the real process table in-process via libproc: no subprocess, so nothing
/// here can stall on launchctl (#974). Socket ownership comes from the process's own
/// file descriptors (`PROC_PIDFDSOCKETINFO`); nothing connects to the socket.
struct LiveDaemonProcessInspector: DaemonProcessInspecting {
    func processIDs() -> [pid_t]? {
        let capacity = Int(proc_listallpids(nil, 0))
        guard capacity > 0 else { return nil }
        var pids = [pid_t](repeating: 0, count: capacity + 64)
        let count = pids.withUnsafeMutableBytes { buffer in
            proc_listallpids(buffer.baseAddress, Int32(buffer.count))
        }
        guard count > 0 else { return nil }
        return Array(pids.prefix(Int(count)))
    }

    func shortName(of pid: pid_t) -> String? {
        var info = proc_bsdshortinfo()
        let size = Int32(MemoryLayout.size(ofValue: info))
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            proc_pidinfo(pid, PROC_PIDT_SHORTBSDINFO, 0, pointer, size)
        }
        guard result == size else { return nil }
        return withUnsafeBytes(of: info.pbsi_comm) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

    func executablePath(of pid: pid_t) -> String? {
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN) * 4)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else { return nil }
        return String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    func isListening(_ pid: pid_t, onUnixSocket path: String) -> Bool? {
        guard let descriptors = socketDescriptors(pid) else { return nil }
        for fd in descriptors {
            var info = socket_fdinfo()
            let size = Int32(MemoryLayout<socket_fdinfo>.size)
            guard proc_pidfdinfo(pid, fd, PROC_PIDFDSOCKETINFO, &info, size) == size,
                  info.psi.soi_family == AF_UNIX,
                  (Int32(info.psi.soi_options) & SO_ACCEPTCONN) != 0 else { continue }
            let bound = withUnsafeBytes(of: info.psi.soi_proto.pri_un.unsi_addr.ua_sun.sun_path) { raw in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            if Self.socketPathsMatch(bound, path) { return true }
        }
        return false
    }

    func processInfo(of pid: pid_t) -> DaemonProcessInfo? {
        var task = proc_taskinfo()
        let taskSize = Int32(MemoryLayout.size(ofValue: task))
        let taskResult = withUnsafeMutablePointer(to: &task) { pointer in
            proc_pidinfo(pid, PROC_PIDTASKINFO, 0, pointer, taskSize)
        }
        var bsd = proc_bsdinfo()
        let bsdSize = Int32(MemoryLayout.size(ofValue: bsd))
        let bsdResult = withUnsafeMutablePointer(to: &bsd) { pointer in
            proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, pointer, bsdSize)
        }
        guard taskResult == taskSize, bsdResult == bsdSize else { return nil }
        let startedAt = Date(
            timeIntervalSince1970: TimeInterval(bsd.pbi_start_tvsec) +
                (TimeInterval(bsd.pbi_start_tvusec) / 1_000_000)
        )
        return DaemonProcessInfo(
            rssBytes: task.pti_resident_size,
            startedAt: startedAt,
            openSockets: socketDescriptors(pid)?.count ?? 0
        )
    }

    /// `/tmp` is a symlink to `/private/tmp`; the kernel reports the path as bound.
    static func socketPathsMatch(_ bound: String, _ expected: String) -> Bool {
        guard !bound.isEmpty else { return false }
        return bound == expected || canonical(bound) == canonical(expected)
    }

    private static func canonical(_ path: String) -> String {
        path.hasPrefix("/private/") ? String(path.dropFirst("/private".count)) : path
    }

    /// `nil` when the descriptor table cannot be read (gone, or not permitted).
    private func socketDescriptors(_ pid: pid_t) -> [Int32]? {
        let needed = Int(proc_pidinfo(pid, PROC_PIDLISTFDS, 0, nil, 0))
        guard needed > 0 else { return nil }
        let stride = MemoryLayout<proc_fdinfo>.stride
        var fds = [proc_fdinfo](repeating: proc_fdinfo(), count: needed / stride + 16)
        let bytes = fds.withUnsafeMutableBytes { buffer in
            proc_pidinfo(pid, PROC_PIDLISTFDS, 0, buffer.baseAddress, Int32(buffer.count))
        }
        guard bytes > 0 else { return nil }
        return fds.prefix(Int(bytes) / stride)
            .filter { Int32($0.proc_fdtype) == PROX_FDTYPE_SOCKET }
            .map(\.proc_fd)
    }
}

/// The Runtime card's "Daemon" and "Last seen" strings, kept out of the view so the
/// restart / outage wording is unit-testable.
enum DaemonRuntimeRows {
    /// The Daemon row exactly as the Runtime card renders it.
    @MainActor
    static func daemonText(collector: StatsCollector) -> String {
        daemonText(daemon: collector.daemon, unavailability: collector.daemonUnavailability)
    }

    static func daemonText(daemon: DaemonHealthSnapshot?, unavailability: DaemonUnavailability?) -> String {
        guard let daemon else {
            switch unavailability {
            case .down(let reason)?: return "Down — \(reason)"
            case .unknown(let reason)?: return "Unknown — \(reason)"
            case nil: return "Checking…"
            }
        }
        guard daemon.isResponsive else {
            return "PID \(daemon.pid) · Unavailable (process info unreadable)"
        }
        let sockets = daemon.openConnections == 1 ? "1 socket" : "\(daemon.openConnections) sockets"
        return "PID \(daemon.pid) · up \(uptimeText(daemon.uptime)) · \(sockets)"
    }

    /// `lastAnswerAt` is the last brain-bus frame from the daemon, so it stays
    /// meaningful while the daemon is down ("12m ago"), unlike a sample time.
    static func lastSeenText(lastAnswerAt: Date?, now: Date) -> String {
        guard let lastAnswerAt else { return "No socket answer yet" }
        return DashboardMetricFormatter.relativeEventString(lastEventAt: lastAnswerAt, now: now)
    }

    /// Rows the definition list paints in the attention colour.
    static func isAttention(_ text: String) -> Bool {
        text.localizedCaseInsensitiveContains("unavailable") || text.hasPrefix("Down")
    }

    static func uptimeText(_ uptime: TimeInterval) -> String {
        let totalMinutes = Int(max(0, uptime) / 60)
        let days = totalMinutes / (24 * 60)
        let hours = (totalMinutes / 60) % 24
        let minutes = totalMinutes % 60
        if days > 0 { return "\(days)d \(hours)h" }
        if hours > 0 { return "\(hours)h \(minutes)m" }
        return "\(minutes)m"
    }
}
