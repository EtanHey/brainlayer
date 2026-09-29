import Foundation

/// When a backup LaunchAgent runs, read from its INSTALLED plist (#968).
enum BackupCadence: Equatable, Sendable {
    /// `weekday` uses launchd's numbering: 0 and 7 are Sunday.
    case calendar(hour: Int, minute: Int, weekday: Int?)
    case interval(seconds: Int)

    var text: String {
        switch self {
        case let .calendar(hour, minute, weekday):
            let time = String(format: "%02d:%02d", hour, minute)
            guard let weekday else { return "daily at \(time)" }
            return "weekly on \(Self.weekdays[weekday % 7]) at \(time)"
        case let .interval(seconds):
            if seconds % 3_600 == 0 { return Self.every(seconds / 3_600, "hour") }
            if seconds % 60 == 0 { return Self.every(seconds / 60, "minute") }
            return Self.every(seconds, "second")
        }
    }

    /// launchd weekday order (0 = Sunday); fixed so the text never depends on the user's locale.
    private static let weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    private static func every(_ count: Int, _ unit: String) -> String {
        count == 1 ? "every \(unit)" : "every \(count) \(unit)s"
    }

    /// The next calendar match after `date`. An interval job's next run depends on launchd's own
    /// clock, so it is not guessed.
    func nextRun(after date: Date, calendar: Calendar) -> Date? {
        guard case let .calendar(hour, minute, weekday) = self else { return nil }
        var components = DateComponents(hour: hour, minute: minute, second: 0)
        if let weekday { components.weekday = (weekday % 7) + 1 }
        return calendar.nextDate(after: date, matching: components, matchingPolicy: .nextTime)
    }
}

enum BackupScheduleRead: Equatable, Sendable {
    case scheduled([BackupCadence])
    case unknown(reason: String)

    static func parse(plist data: Data?, path: String) -> BackupScheduleRead {
        guard let data else { return .unknown(reason: "no LaunchAgent installed at \(path)") }
        guard let plist = (try? PropertyListSerialization.propertyList(from: data, format: nil)) as? [String: Any] else {
            return .unknown(reason: "\(path) is not a readable plist")
        }
        let entries = (plist["StartCalendarInterval"] as? [[String: Int]])
            ?? (plist["StartCalendarInterval"] as? [String: Int]).map { [$0] }
        if let entries, !entries.isEmpty {
            let cadences = entries.compactMap { entry -> BackupCadence? in
                guard let hour = entry["Hour"], let minute = entry["Minute"] else { return nil }
                return .calendar(hour: hour, minute: minute, weekday: entry["Weekday"])
            }
            guard cadences.count == entries.count else {
                return .unknown(reason: "\(path) has a StartCalendarInterval without Hour and Minute")
            }
            return .scheduled(cadences)
        }
        if let seconds = plist["StartInterval"] as? Int, seconds > 0 {
            return .scheduled([.interval(seconds: seconds)])
        }
        return .unknown(reason: "\(path) has no StartCalendarInterval or StartInterval")
    }

    var text: String {
        switch self {
        case let .scheduled(cadences): cadences.map(\.text).joined(separator: " and ")
        case let .unknown(reason): "Schedule unknown — \(reason)"
        }
    }

    func nextRun(after date: Date, calendar: Calendar) -> Date? {
        guard case let .scheduled(cadences) = self else { return nil }
        return cadences.compactMap { $0.nextRun(after: date, calendar: calendar) }.min()
    }
}

/// The last run a backup job actually recorded in its own log. Never inferred from a schedule.
struct BackupRunReceipt: Equatable, Sendable {
    let at: Date
    let verified: Bool?
}

enum BackupLogReader {
    enum Kind: Sendable { case databaseBackup, transcriptArchive, weeklyMaintenance }

    static func lastRun(_ kind: Kind, log data: Data?) -> BackupRunReceipt? {
        guard let data, let text = String(data: data, encoding: .utf8) else { return nil }
        for line in text.split(separator: "\n").reversed() {
            guard let row = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { continue }
            switch kind {
            case .databaseBackup:
                // Test runs append to their own log path but carry provenance; only real rows count.
                guard row["backup_log_provenance"] as? String == "real",
                      let at = isoDate(row["attempted_at"]) else { continue }
                return BackupRunReceipt(at: at, verified: row["verified"] as? Bool ?? false)
            case .transcriptArchive:
                guard let at = isoDate(row["attempted_at"]) else { continue }
                return BackupRunReceipt(at: at, verified: row["verified"] as? Bool ?? false)
            case .weeklyMaintenance:
                // Only a completed full pass; progress rows (backup waits) and dry runs don't count.
                guard row["mode"] as? String == "full", row["dry_run"] as? Bool == false,
                      let at = isoDate(row["ts"]) else { continue }
                return BackupRunReceipt(at: at, verified: nil)
            }
        }
        return nil
    }

    /// Python's `datetime.isoformat()`: microseconds are optional, the UTC offset is present.
    static func isoDate(_ value: Any?) -> Date? {
        guard let raw = value as? String, !raw.isEmpty else { return nil }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fractional.date(from: raw) ?? ISO8601DateFormatter().date(from: raw)
    }
}

enum BackupLocalFiles {
    /// The newest local DB snapshot (`YYYY-MM-DD.db` or `.db.gz`) in `names`.
    static func latestSnapshot(in names: [String]) -> String? {
        latest(names, matching: #"^\d{4}-\d{2}-\d{2}\.db(\.gz)?$"#)
    }

    /// The newest local transcript archive (`claude-jsonl-YYYY-MM-DD.tar.gz`) in `names`.
    static func latestArchive(in names: [String]) -> String? {
        latest(names, matching: #"^claude-jsonl-\d{4}-\d{2}-\d{2}\.tar\.gz$"#)
    }

    /// The names carry ISO dates, so the lexically greatest is the newest.
    private static func latest(_ names: [String], matching pattern: String) -> String? {
        names.filter { $0.range(of: pattern, options: .regularExpression) != nil }.max()
    }
}
