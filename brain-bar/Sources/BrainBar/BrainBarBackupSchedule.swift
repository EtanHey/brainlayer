import Foundation

/// When a backup LaunchAgent runs, read from its INSTALLED plist (#968).
enum BackupCadence: Equatable, Sendable {
    /// One `StartCalendarInterval` entry. Omitted fields are launchd wildcards; `weekday` uses
    /// launchd's numbering (0 and 7 are Sunday). Only representable shapes are constructed.
    case calendar(hour: Int?, minute: Int, weekday: Int? = nil, day: Int? = nil, month: Int? = nil)
    case interval(seconds: Int)

    var text: String {
        switch self {
        case let .calendar(hour, minute, weekday, day, month):
            guard let hour else { return String(format: "hourly at :%02d", minute) }
            let time = String(format: "%02d:%02d", hour, minute)
            if let month, let day { return "yearly on \(Self.months[month - 1]) \(day) at \(time)" }
            if let day { return "monthly on day \(day) at \(time)" }
            if let weekday { return "weekly on \(Self.weekdays[weekday % 7]) at \(time)" }
            return "daily at \(time)"
        case let .interval(seconds):
            if seconds % 3_600 == 0 { return Self.every(seconds / 3_600, "hour") }
            if seconds % 60 == 0 { return Self.every(seconds / 60, "minute") }
            return Self.every(seconds, "second")
        }
    }

    /// Fixed English names so the text never depends on the user's locale.
    private static let weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    private static let months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    private static func every(_ count: Int, _ unit: String) -> String {
        count == 1 ? "every \(unit)" : "every \(count) \(unit)s"
    }

    /// The next calendar match after `date`. An interval job's next run depends on launchd's own
    /// clock, so it is not guessed.
    func nextRun(after date: Date, calendar: Calendar) -> Date? {
        guard case let .calendar(hour, minute, weekday, day, month) = self else { return nil }
        var components = DateComponents(month: month, day: day, hour: hour, minute: minute, second: 0)
        if let weekday { components.weekday = (weekday % 7) + 1 }
        return calendar.nextDate(after: date, matching: components, matchingPolicy: .nextTime)
    }

    /// Builds a cadence from one `StartCalendarInterval` dict, or explains why it can't be shown.
    fileprivate static func fromCalendarEntry(_ entry: [String: Any]) -> Result<BackupCadence, ScheduleProblem> {
        let ranges: [(key: String, range: ClosedRange<Int>)] = [
            ("Minute", 0...59), ("Hour", 0...23), ("Day", 1...31), ("Weekday", 0...7), ("Month", 1...12),
        ]
        var values: [String: Int] = [:]
        for (key, range) in ranges {
            guard let raw = entry[key] else { continue }
            guard let value = raw as? Int else { return .failure(.invalid("non-integer \(key)")) }
            guard range.contains(value) else { return .failure(.invalid("invalid \(key) \(value)")) }
            values[key] = value
        }
        let unknownKeys = Set(entry.keys).subtracting(ranges.map(\.key))
        if let key = unknownKeys.sorted().first { return .failure(.notRepresentable("unknown key \(key)")) }
        guard let minute = values["Minute"] else { return .failure(.notRepresentable("no Minute: every minute")) }
        if values["Day"] != nil, values["Weekday"] != nil {
            return .failure(.notRepresentable("Day with Weekday"))
        }
        if values["Month"] != nil, values["Day"] == nil { return .failure(.notRepresentable("Month without Day")) }
        if values["Hour"] == nil, values.keys.contains(where: { $0 != "Minute" }) {
            return .failure(.notRepresentable("no Hour with a date constraint"))
        }
        return .success(.calendar(
            hour: values["Hour"], minute: minute, weekday: values["Weekday"], day: values["Day"], month: values["Month"]
        ))
    }
}

fileprivate enum ScheduleProblem: Error {
    case invalid(String)
    case notRepresentable(String)

    var text: String {
        switch self {
        case let .invalid(reason): reason
        case let .notRepresentable(reason): "a schedule that is not representable (\(reason))"
        }
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
        var cadences: [BackupCadence] = []
        // launchd evaluates StartCalendarInterval and StartInterval independently: keep both.
        if let calendarValue = plist["StartCalendarInterval"] {
            let entries: [[String: Any]]
            if let single = calendarValue as? [String: Any] {
                entries = [single]
            } else if let array = calendarValue as? [Any], let dicts = array as? [[String: Any]], !dicts.isEmpty {
                entries = dicts
            } else {
                return .unknown(reason: "\(path) has a StartCalendarInterval that is neither a dict nor an array of dicts")
            }
            for (index, entry) in entries.enumerated() {
                switch BackupCadence.fromCalendarEntry(entry) {
                case let .success(cadence):
                    cadences.append(cadence)
                case let .failure(problem):
                    let suffix = entries.count > 1 ? " (entry \(index + 1))" : ""
                    return .unknown(reason: "\(path) has \(problem.text)\(suffix)")
                }
            }
        }
        if let intervalValue = plist["StartInterval"] {
            guard let seconds = intervalValue as? Int, seconds > 0 else {
                return .unknown(reason: "\(path) has an invalid StartInterval")
            }
            cadences.append(.interval(seconds: seconds))
        }
        guard !cadences.isEmpty else {
            return .unknown(reason: "\(path) has no StartCalendarInterval or StartInterval")
        }
        return .scheduled(cadences)
    }

    var text: String {
        switch self {
        case let .scheduled(cadences): cadences.map(\.text).joined(separator: " and ")
        case let .unknown(reason): "Schedule unknown — \(reason)"
        }
    }

    /// The soonest calendar match. When an interval also drives the job, launchd's own clock may
    /// fire it sooner, so no next run is claimed.
    func nextRun(after date: Date, calendar: Calendar) -> Date? {
        guard case let .scheduled(cadences) = self else { return nil }
        if cadences.contains(where: { if case .interval = $0 { true } else { false } }) { return nil }
        return cadences.compactMap { $0.nextRun(after: date, calendar: calendar) }.min()
    }
}

/// The last run a backup job actually recorded in its own log. Never inferred from a schedule.
struct BackupRunReceipt: Equatable, Sendable {
    let at: Date
    let verified: Bool?
}

/// A weekly pass that wrote its final row but did not complete (#1015 review B1).
struct BackupIncompleteAttempt: Equatable, Sendable {
    let at: Date
    let reason: String
}

enum BackupLogReader {
    /// A completed full pass: its final (non-dry-run) row records the VACUUM it ran. Newer rows carry
    /// `backup_status: "verified"`; rows from before #1002 carry no backup_status at all. A final row
    /// whose backup was unavailable or whose post-backup gate failed aborted before VACUUM.
    private static func isCompletedFullPass(_ row: [String: Any]) -> Bool {
        guard row["mode"] as? String == "full", row["dry_run"] as? Bool == false,
              row["vacuum_after_bytes"] is NSNumber else { return false }
        let status = row["backup_status"] as? String
        return status == nil || status == "verified"
    }

    /// The newest full-pass attempt that wrote a final row without completing, when it is newer than
    /// the last completed pass. Progress rows and dry runs are not attempts.
    static func lastIncompleteAttempt(_ kind: Kind, log data: Data?) -> BackupIncompleteAttempt? {
        guard kind == .weeklyMaintenance, let data, let text = String(data: data, encoding: .utf8) else { return nil }
        for line in text.split(separator: "\n").reversed() {
            guard let row = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any],
                  row["mode"] as? String == "full", row["dry_run"] as? Bool == false else { continue }
            if isCompletedFullPass(row) { return nil }
            guard let at = isoDate(row["ts"]) else { continue }
            let reason = switch row["backup_status"] as? String {
            case "unavailable": "backup unavailable, VACUUM skipped"
            case "gates_failed": "post-backup safety gate failed, VACUUM skipped"
            case let status?: "backup \(status), VACUUM skipped"
            case nil: "VACUUM did not run"
            }
            return BackupIncompleteAttempt(at: at, reason: reason)
        }
        return nil
    }

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
                guard isCompletedFullPass(row), let at = isoDate(row["ts"]) else { continue }
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
