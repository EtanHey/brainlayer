import Darwin
import Foundation

enum BrainBarJobOwnership: Equatable, Sendable {
    case legacy, supervisor, unknown(String)

    static let directory = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".local/share/brainlayer/supervisor")

    static func read(journalURL: URL = directory.appendingPathComponent("migration.json")) -> Self {
        do {
            // Rollback payloads, published plists and the manifest make the live journal >100 KB.
            // 4 MiB leaves room for that inventory while keeping the private read finite.
            guard let data = try readOwnedFile(journalURL, privateMode: true, limit: 4 * 1024 * 1024) else { return .legacy }
            guard let row = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let phase = row["phase"] as? String else { throw CocoaError(.fileReadCorruptFile) }
            switch phase {
            case "migrating", "migrated", "reverting": return .supervisor
            case "reverted", "rolled_back": return .legacy
            default: throw CocoaError(.fileReadCorruptFile)
            }
        } catch {
            return .unknown("Can't determine job owner: migration journal is unreadable or untrusted")
        }
    }

    // Same custody checks as install.sh's journal guard. Open every path segment with NOFOLLOW:
    // a parent link must not turn an apparently private journal into a different owner's file.
    static func readOwnedFile(_ url: URL, privateMode: Bool, limit: Int) throws -> Data? {
        var fd = Darwin.open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard fd >= 0 else { throw CocoaError(.fileReadNoPermission) }
        defer { Darwin.close(fd) }
        let parts = url.pathComponents.dropFirst()
        guard !parts.contains(".."), !parts.contains(".") else { throw CocoaError(.fileReadNoPermission) }
        for (index, part) in parts.enumerated() {
            let last = index == parts.count - 1
            let next = openat(fd, part, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | (last ? 0 : O_DIRECTORY))
            if next < 0 {
                if errno == ENOENT { return nil }
                throw CocoaError(.fileReadNoPermission)
            }
            Darwin.close(fd)
            fd = next
        }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(), info.st_nlink == 1,
              info.st_mode & (privateMode ? 0o077 : 0o022) == 0,
              info.st_size <= limit else { throw CocoaError(.fileReadNoPermission) }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 8192)
        while true {
            let count = Darwin.read(fd, &buffer, buffer.count)
            guard count >= 0 else { throw CocoaError(.fileReadUnknown) }
            if count == 0 { return data }
            guard data.count + count <= limit else { throw CocoaError(.fileReadTooLarge) }
            data.append(contentsOf: buffer.prefix(count))
        }
    }
}

struct BrainBarSupervisorRuntime: Decodable, Sendable {
    let configured: Bool
    let paused: Bool
    let pid: Int32?
    let starts: Int
    let failures: Int
    let last_exit_code: Int32?
    let last_run_at: Double?
    let next_run_at: Double?
    let error: String?

    static let cli = "/opt/homebrew/opt/brainlayer/bin/brainlayer"

    static func decode(_ output: String) throws -> Self {
        struct Response: Decodable { let source: String; let runtime: BrainBarSupervisorRuntime }
        let response = try JSONDecoder().decode(Response.self, from: Data(output.utf8))
        let value = response.runtime
        guard response.source == "supervisor", value.starts >= 0, value.failures >= 0,
              value.pid == nil || value.pid! > 1,
              [value.last_run_at, value.next_run_at].compactMap({ $0 }).allSatisfy({ $0.isFinite && $0 >= 0 })
        else { throw CocoaError(.fileReadCorruptFile) }
        return value
    }
}

struct BrainBarCalendarHistory: Sendable {
    struct Run: Sendable {
        let at: Date
        let code: Int32
        let status: String
        var startedAt: Date? = nil
        var reason: String? = nil
        var replaces: String? = nil
        var scheduledDate: String? = nil
    }
    var runs: [String: Run] = [:]
    var error: String? = nil
    var latestScheduledDate: String? = nil
    static let empty = Self()

    static func read(directory: URL = BrainBarJobOwnership.directory.appendingPathComponent("calendar")) -> Self {
        do {
            guard let data = try BrainBarJobOwnership.readOwnedFile(
                directory.appendingPathComponent("runs.jsonl"), privateMode: false, limit: 4 * 1024 * 1024
            ) else { return .empty }
            var result = Self()
            var starts: [String: Date] = [:]
            for line in String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline) {
                guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
                else { throw CocoaError(.fileReadCorruptFile) }
                if row["event"] as? String == "spawn", let step = row["step"] as? String,
                   let ts = row["ts"] as? Double, ts.isFinite {
                    starts[step] = Date(timeIntervalSince1970: ts)
                }
                if row["event"] as? String == "runner" {
                    starts.removeAll()
                    result.latestScheduledDate = row["scheduled_date"] as? String
                }
                guard row["event"] as? String == "finished" else { continue }
                guard let step = row["step"] as? String, let ts = row["ts"] as? Double, ts.isFinite,
                      let status = row["status"] as? String else { throw CocoaError(.fileReadCorruptFile) }
                // Shutdown records the unstarted remainder as skipped/null. It did not run,
                // so preserve the step's preceding result rather than invalidating all history.
                let startedAt = starts.removeValue(forKey: step)
                if status == "skipped", row["returncode"] is NSNull { continue }
                guard let code = row["returncode"] as? Int32,
                      ["ok", "failed", "deferred", "stopped"].contains(status)
                else { throw CocoaError(.fileReadCorruptFile) }
                result.runs[step] = Run(at: Date(timeIntervalSince1970: ts), code: code, status: status,
                    startedAt: startedAt, reason: row["reason"] as? String, replaces: row["replaces"] as? String,
                    scheduledDate: result.latestScheduledDate)
            }
            return result
        } catch { return Self(error: "Maintenance calendar history is unreadable") }
    }
}
