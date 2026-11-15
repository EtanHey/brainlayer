import Foundation

/// One attempt on a backup lane: a finished, verified run (ok) or a failed one.
struct NDAttempt: Equatable, Sendable {
    var at: Date
    var ok: Bool
}

/// A week of backup attempts (L2). A lane whose log could not be read, or held a damaged line inside the window, is
/// unknown: the rail draws the dashed line to "now", never empty and never fine.
enum NDBackupHistory: Equatable, Sendable {
    case unread
    case read(NDBackupLanes)
}

struct NDBackupLanes: Equatable, Sendable {
    /// nil: that log could not be read, or held damaged lines inside the window.
    var local: [NDAttempt]?
    var drive: [NDAttempt]?
    var conversations: [NDAttempt]?
    var integrity: [NDAttempt]?
    /// Damaged lines skipped, per log (counted, never shown as a value).
    var skippedDatabaseLines = 0
    var skippedConversationLines = 0
}

/// L2's reader: read-only, off the main thread, the last 8 days or the last 200 lines (whichever is smaller), real rows
/// only, grouped by `backup_invocation_id`, damaged lines skipped and counted, `BRAINLAYER_BACKUP_LOG_PATH` honoured
/// (through `BrainBarBackupSources.Paths.live`), never written.
enum NDBackupHistoryReader {
    static let windowS: TimeInterval = 8 * 86_400
    static let maxLines = 200
    /// 200 lines of a few hundred bytes each; the tail is read at most this far back.
    static let maxBytes = 512 * 1024

    struct Log: Sendable {
        var lines: [Substring]
        var unreadable: Bool
    }

    /// The tail of a log as lines, the newest last. A tail that starts mid-line drops that partial first line.
    static func tail(_ url: URL) -> Log {
        switch BrainBarStableFileReader.readTail(url, maxBytes: maxBytes) {
        case .unreadable, .changedWhileReading:
            return Log(lines: [], unreadable: true)
        case let .read(data, _):
            guard var text = String(data: data, encoding: .utf8) else { return Log(lines: [], unreadable: true) }
            if data.count >= maxBytes, let newline = text.firstIndex(of: "\n") { text = String(text[text.index(after: newline)...]) }
            let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
            return Log(lines: Array(lines.suffix(maxLines)), unreadable: false)
        }
    }

    static func read(databaseLog: URL, conversationLog: URL, now: Date) -> NDBackupHistory {
        let database = parseDatabase(tail(databaseLog), now: now)
        let conversations = parseConversations(tail(conversationLog), now: now)
        return .read(NDBackupLanes(local: database?.local, drive: database?.drive, conversations: conversations?.attempts,
                                   integrity: database?.integrity, skippedDatabaseLines: database?.skipped ?? 0,
                                   skippedConversationLines: conversations?.skipped ?? 0))
    }

    struct DatabaseLanes: Equatable, Sendable {
        var local: [NDAttempt]?, drive: [NDAttempt]?, integrity: [NDAttempt]?
        var skipped: Int
    }

    /// `backup-daily.log`: one row per attempt, `backup_log_provenance == "real"` only, grouped by invocation (the newest
    /// row of an invocation speaks for it). Nil lanes when the log is unreadable or damaged inside the window.
    static func parseDatabase(_ log: Log, now: Date) -> DatabaseLanes? {
        guard !log.unreadable else { return nil }
        var skipped = 0
        var byInvocation: [String: (at: Date, row: [String: Any])] = [:]
        var order: [String] = []
        for (index, line) in log.lines.enumerated() {
            guard let row = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { skipped += 1; continue }
            guard row["backup_log_provenance"] as? String == "real" else { continue }
            guard let at = BackupLogReader.isoDate(row["attempted_at"]) else { skipped += 1; continue }
            guard now.timeIntervalSince(at) <= windowS else { continue }
            // N01: a real row that carries none of the lanes' results is damaged, not an empty night.
            guard row["local_verified"] is Bool || row["uploaded"] is Bool || row["local_integrity"] is String else {
                skipped += 1; continue
            }
            let key = (row["backup_invocation_id"] as? String) ?? "line-\(index)"
            if byInvocation[key] == nil { order.append(key) }
            byInvocation[key] = (at, row)
        }
        guard skipped == 0 else { return DatabaseLanes(local: nil, drive: nil, integrity: nil, skipped: skipped) }
        var local: [NDAttempt] = [], drive: [NDAttempt] = [], integrity: [NDAttempt] = []
        for key in order {
            guard let (at, row) = byInvocation[key] else { continue }
            if let verified = row["local_verified"] as? Bool { local.append(.init(at: at, ok: verified)) }
            // A Drive dot only for an attempt that tried Drive: uploaded and verified with no md5 mismatch is a copy.
            if let uploaded = row["uploaded"] as? Bool {
                let ok = uploaded && row["verified"] as? Bool == true && row["drive_md5_match"] as? Bool != false
                drive.append(.init(at: at, ok: ok))
            }
            if let value = row["local_integrity"] as? String { integrity.append(.init(at: at, ok: value == "ok")) }
        }
        return DatabaseLanes(local: local, drive: drive, integrity: integrity, skipped: 0)
    }

    /// `jsonl-backup.log`: `attempted_at`, `uploaded`, `verified`, `status`. A pending row (waiting to upload) is not an
    /// attempt that finished either way, so it draws no dot.
    static func parseConversations(_ log: Log, now: Date) -> (attempts: [NDAttempt]?, skipped: Int)? {
        guard !log.unreadable else { return nil }
        var skipped = 0
        var attempts: [NDAttempt] = []
        for line in log.lines {
            guard let row = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { skipped += 1; continue }
            guard let at = BackupLogReader.isoDate(row["attempted_at"]) else { skipped += 1; continue }
            guard now.timeIntervalSince(at) <= windowS else { continue }
            if row["status"] as? String == "pending" { continue }
            guard row["uploaded"] is Bool || row["verified"] is Bool else { skipped += 1; continue }  // N01
            attempts.append(.init(at: at, ok: row["uploaded"] as? Bool == true && row["verified"] as? Bool == true))
        }
        return (skipped == 0 ? attempts : nil, skipped)
    }
}
