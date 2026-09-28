import Darwin
import Foundation
import os

/// Where BrainBar's diagnostics go.
///
/// Unified logging is the always-on sink:
/// `log show --predicate 'subsystem == "com.brainlayer.brainbar"'`.
/// Two files sit beside it, and neither ever carries a request, response,
/// query, stored content or token:
/// - the lifecycle log (always on, small): server starts and watchdog restart
///   decisions, so a restart stays explainable across process lifetimes;
/// - the debug log (`/tmp/brainbar-debug.log`): a connection/framing trace,
///   written only when the daemon's environment has `BRAINBAR_DEBUG_LOG=1`.
public enum BrainBarLog {
    public static let subsystem = "com.brainlayer.brainbar"

    public static func logger(_ category: String) -> Logger {
        Logger(subsystem: subsystem, category: category)
    }

    public static let debugLogFlag = "BRAINBAR_DEBUG_LOG"
    public static let debugLogPath = "/tmp/brainbar-debug.log"
    /// 5 MB current + one 5 MB rotated generation.
    public static let debugLogMaxBytes = 5 * 1024 * 1024
    public static let lifecycleLogMaxBytes = 256 * 1024

    public static var lifecycleLogPath: String {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Logs/BrainBar/lifecycle.log")
            .path
    }

    public static func isDebugLogEnabled(environment: [String: String]) -> Bool {
        environment[debugLogFlag] == "1"
    }

    /// The flag-gated debug log, or nil when the flag is not exactly "1".
    public static func debugLogFile(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        path: String = debugLogPath
    ) -> BrainBarLogFile? {
        guard isDebugLogEnabled(environment: environment) else { return nil }
        return BrainBarLogFile(path: path, maxBytes: debugLogMaxBytes)
    }

    public static func lifecycleLogFile(path: String = lifecycleLogPath) -> BrainBarLogFile {
        BrainBarLogFile(path: path, maxBytes: lifecycleLogMaxBytes)
    }
}

/// An append-only, owner-only (0600) log file with one rotated generation.
///
/// Opened with `O_NOFOLLOW` and refused unless it is a regular file owned by
/// this user, so a planted symlink in `/tmp` can neither receive lines nor be
/// chmod'ed. Mode 0600 is enforced on every open, which also tightens a file
/// an older build created 0644. Callers pass only counts, states, PIDs and
/// fixed strings; this type does not scrub.
public final class BrainBarLogFile: @unchecked Sendable {
    public let path: String
    public let maxBytes: Int
    private let lock = NSLock()

    public var rotatedPath: String { path + ".1" }

    public init(path: String, maxBytes: Int) {
        self.path = path
        self.maxBytes = max(1, maxBytes)
    }

    public func append(_ message: String, now: Date = Date()) {
        let line = Array("[\(ISO8601DateFormatter().string(from: now))] \(message)\n".utf8)
        lock.lock()
        defer { lock.unlock() }
        guard var fd = openOwned(create: true) else { return }
        var info = stat()
        if fstat(fd, &info) == 0, info.st_size > 0, Int(info.st_size) + line.count > maxBytes {
            close(fd)
            // rename(2) replaces the previous generation atomically.
            _ = rename(path, rotatedPath)
            guard let reopened = openOwned(create: true) else { return }
            fd = reopened
        }
        defer { close(fd) }
        _ = line.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
    }

    /// Chmods an existing file to 0600 without creating, writing or truncating it.
    public func tightenExisting() {
        lock.lock()
        defer { lock.unlock() }
        if let fd = openOwned(create: false) { close(fd) }
    }

    private func openOwned(create: Bool) -> Int32? {
        if create {
            let parent = (path as NSString).deletingLastPathComponent
            if !parent.isEmpty, !FileManager.default.fileExists(atPath: parent) {
                try? FileManager.default.createDirectory(
                    atPath: parent,
                    withIntermediateDirectories: true,
                    attributes: [.posixPermissions: 0o700]
                )
            }
        }
        var flags = O_WRONLY | O_APPEND | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK
        if create { flags |= O_CREAT }
        let fd = open(path, flags, 0o600)
        guard fd >= 0 else { return nil }
        var info = stat()
        guard fstat(fd, &info) == 0,
              info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(),
              fchmod(fd, 0o600) == 0
        else {
            close(fd)
            return nil
        }
        return fd
    }
}
