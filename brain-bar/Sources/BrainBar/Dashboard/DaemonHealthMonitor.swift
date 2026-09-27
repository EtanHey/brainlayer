import Darwin
import Foundation

/// How the monitor finds BrainBarDaemon. `isDaemon` is the cheap per-sample identity
/// check for a cached PID; `resolveDaemonPID` is the fresh lookup, run only when the
/// cached PID has exited or is no longer the daemon (#972: the PID was once captured
/// at app launch and never re-resolved, so every daemon restart blanked the rows).
protocol DaemonPIDResolving: Sendable {
    func isDaemon(_ pid: pid_t) -> Bool
    func resolveDaemonPID() -> pid_t?
}

/// A monitor reading: a snapshot when the daemon is running, otherwise the reason it
/// is not, so the Runtime rows can say why instead of a bare "Unavailable".
struct DaemonHealthReading: Sendable, Equatable {
    let snapshot: DaemonHealthSnapshot?
    let downReason: String?
}

final class DaemonHealthMonitor: @unchecked Sendable {
    private let resolver: any DaemonPIDResolving
    private let lock = NSLock()
    private var cachedPID: pid_t?

    init(pidResolver: any DaemonPIDResolving) {
        self.resolver = pidResolver
    }

    /// Watches one fixed PID. Test seam only — production resolves live.
    convenience init(targetPID: pid_t) {
        self.init(pidResolver: FixedDaemonPIDResolver(pid: targetPID))
    }

    func sample() -> DaemonHealthSnapshot? {
        read().snapshot
    }

    /// Resolves the daemon PID for this sample. Blocking (a process-table scan on a
    /// cache miss); callers on the main actor must run it off-main.
    func read() -> DaemonHealthReading {
        lock.withLock {
            if let pid = cachedPID, isAlive(pid), resolver.isDaemon(pid) {
                return DaemonHealthReading(snapshot: snapshot(for: pid), downReason: nil)
            }

            let exitedPID = cachedPID
            cachedPID = nil
            guard let pid = resolver.resolveDaemonPID(), pid > 0, isAlive(pid) else {
                let reason = exitedPID.map {
                    "BrainBarDaemon not running (PID \($0) exited; no replacement found)"
                } ?? "BrainBarDaemon not running (no daemon process found)"
                return DaemonHealthReading(snapshot: nil, downReason: reason)
            }
            cachedPID = pid
            return DaemonHealthReading(snapshot: snapshot(for: pid), downReason: nil)
        }
    }

    private func isAlive(_ pid: pid_t) -> Bool {
        pid > 0 && kill(pid, 0) == 0
    }

    private func snapshot(for pid: pid_t) -> DaemonHealthSnapshot {
        guard let taskInfo = taskInfo(pid),
              let bsdInfo = bsdInfo(pid) else {
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

        let startTime = Date(
            timeIntervalSince1970: TimeInterval(bsdInfo.pbi_start_tvsec) +
                (TimeInterval(bsdInfo.pbi_start_tvusec) / 1_000_000)
        )

        return DaemonHealthSnapshot(
            pid: pid,
            isResponsive: true,
            rssBytes: taskInfo.pti_resident_size,
            uptime: max(0, Date().timeIntervalSince(startTime)),
            openConnections: countOpenSocketDescriptors(pid),
            lastSeenAt: nil,
            startedAt: startTime
        )
    }

    private func taskInfo(_ pid: pid_t) -> proc_taskinfo? {
        var info = proc_taskinfo()
        let size = Int32(MemoryLayout.size(ofValue: info))
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            proc_pidinfo(pid, PROC_PIDTASKINFO, 0, pointer, size)
        }
        guard result == size else { return nil }
        return info
    }

    private func bsdInfo(_ pid: pid_t) -> proc_bsdinfo? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout.size(ofValue: info))
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, pointer, size)
        }
        guard result == size else { return nil }
        return info
    }

    private func countOpenSocketDescriptors(_ pid: pid_t) -> Int {
        var fdInfos = Array(repeating: proc_fdinfo(), count: 256)
        let bytesRead = fdInfos.withUnsafeMutableBytes { rawBuffer in
            proc_pidinfo(
                pid,
                PROC_PIDLISTFDS,
                0,
                rawBuffer.baseAddress,
                Int32(rawBuffer.count)
            )
        }

        guard bytesRead > 0 else { return 0 }
        let infoCount = Int(bytesRead) / MemoryLayout<proc_fdinfo>.stride
        return fdInfos.prefix(infoCount).reduce(into: 0) { count, info in
            if Int32(info.proc_fdtype) == PROX_FDTYPE_SOCKET {
                count += 1
            }
        }
    }
}

struct FixedDaemonPIDResolver: DaemonPIDResolving {
    let pid: pid_t

    func isDaemon(_ candidate: pid_t) -> Bool {
        candidate == pid
    }

    func resolveDaemonPID() -> pid_t? {
        pid > 0 ? pid : nil
    }
}

/// Finds the live BrainBarDaemon in-process: the pidfile when it names the daemon,
/// else a `proc_listallpids` scan for the `BrainBarDaemon` executable. No launchctl
/// subprocess — #974 showed a launchctl sweep can stall, and this runs every time
/// the daemon restarts.
struct LiveDaemonPIDResolver: DaemonPIDResolving {
    static let executableName = "BrainBarDaemon"
    var pidFilePath = "/tmp/brainbar-daemon.pid"

    struct ProcessEntry: Equatable {
        let pid: pid_t
        let parentPID: pid_t
        let executablePath: String
    }

    func isDaemon(_ pid: pid_t) -> Bool {
        Self.executablePath(of: pid).map(Self.isDaemonExecutable) ?? false
    }

    func resolveDaemonPID() -> pid_t? {
        if let pid = Self.pidFromFile(pidFilePath), isDaemon(pid) {
            return pid
        }
        return Self.selectDaemonPID(from: Self.processTable())
    }

    /// Prefers the launchd-owned daemon (parent PID 1); a stray copy started by hand
    /// only counts when launchd has none.
    static func selectDaemonPID(from entries: [ProcessEntry]) -> pid_t? {
        let daemons = entries.filter { $0.pid > 0 && isDaemonExecutable($0.executablePath) }
        return (daemons.first { $0.parentPID == 1 } ?? daemons.first)?.pid
    }

    static func isDaemonExecutable(_ path: String) -> Bool {
        URL(fileURLWithPath: path).lastPathComponent == executableName
    }

    static func pidFromFile(_ path: String) -> pid_t? {
        guard let contents = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        let token = contents.trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: CharacterSet.whitespacesAndNewlines)
            .first ?? ""
        guard let rawPID = Int32(token), rawPID > 0 else { return nil }
        return pid_t(rawPID)
    }

    /// One `SHORTBSDINFO` call per PID yields the short name and parent; the full
    /// path is fetched only for PIDs whose short name is already `BrainBarDaemon`.
    private static func processTable() -> [ProcessEntry] {
        let capacity = max(Int(proc_listallpids(nil, 0)), 0) + 64
        var pids = [pid_t](repeating: 0, count: capacity)
        let count = pids.withUnsafeMutableBytes { buffer in
            proc_listallpids(buffer.baseAddress, Int32(buffer.count))
        }
        guard count > 0 else { return [] }
        return pids.prefix(Int(count)).compactMap { pid in
            guard pid > 0, let info = shortInfo(pid), shortName(info) == executableName,
                  let path = executablePath(of: pid) else { return nil }
            return ProcessEntry(pid: pid, parentPID: pid_t(info.pbsi_ppid), executablePath: path)
        }
    }

    private static func shortInfo(_ pid: pid_t) -> proc_bsdshortinfo? {
        var info = proc_bsdshortinfo()
        let size = Int32(MemoryLayout.size(ofValue: info))
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            proc_pidinfo(pid, PROC_PIDT_SHORTBSDINFO, 0, pointer, size)
        }
        return result == size ? info : nil
    }

    private static func shortName(_ info: proc_bsdshortinfo) -> String {
        withUnsafeBytes(of: info.pbsi_comm) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

    private static func executablePath(of pid: pid_t) -> String? {
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN) * 4)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else { return nil }
        return String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

/// The Runtime card's "Daemon" and "Last seen" strings, kept out of the view so the
/// restart / outage wording is unit-testable.
enum DaemonRuntimeRows {
    static func daemonText(daemon: DaemonHealthSnapshot?, downReason: String?) -> String {
        guard let daemon else {
            guard let downReason else { return "Checking…" }
            return "Down — \(downReason)"
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
