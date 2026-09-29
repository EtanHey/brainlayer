import Foundation

/// What the Python watcher last wrote to `watcher-health.json` beside the resolved DB.
/// `poll_count` and `updated_at` advance on every poll (~60-95 s apart); see AGENTS.md
/// "Real-time JSONL Watcher".
struct WatcherHealthFile: Sendable, Equatable {
    var updatedAt: Date
    var pollCount: Int
    var alertReasons: [String] = []
    var dbProbeFailed = false
    var maxOffsetLagBytes = 0
    var fileIngestionFailureCount = 0
    /// How many failures the producer listed in detail (it caps the list at 100). nil = not reported.
    var fileIngestionFailuresListed: Int?
    var earliestFileIngestionFailureAt: Date?
    var quarantinedRecordCount = 0
    /// How many quarantined records the producer listed in detail (capped at 100). nil = not reported.
    var quarantinedRecordsListed: Int?
    var earliestQuarantinedRecordAt: Date?
}

enum WatcherHealthFileRead: Sendable, Equatable {
    case readable(WatcherHealthFile)
    case missing(path: String)
    case unreadable(path: String, reason: String)
}

/// Whether launchd says `com.brainlayer.watch` is running.
enum WatcherLaunchdEvidence: Sendable, Equatable {
    case running
    case notRunning(String)
    case unavailable(String)

    /// The Dashboard's view: the launchctl process probe for `com.brainlayer.watch`.
    init(process: WatcherProcessProbeResult?) {
        switch process {
        case .running:
            self = .running
        case .absent:
            self = .notRunning("launchd reports com.brainlayer.watch is not running")
        case let .failure(detail):
            self = .unavailable(detail)
        case nil:
            self = .unavailable("watcher process evidence unavailable")
        }
    }
}

/// One concrete reason the watcher needs attention: what is wrong, since when, what to do.
struct WatcherHealthIssue: Sendable, Equatable {
    let what: String
    let since: Date?
    let action: String
    /// Set when `since` is only a lower bound on the true start (the producer's detail list was
    /// capped), e.g. "100 of 150 listed". The text then reads "since at least …".
    var sinceBoundNote: String? = nil
}

/// The one watcher-health truth (#966). Every surface that shows watcher health renders this
/// instead of deriving its own verdict from partial evidence.
enum WatcherHealthStatus: Sendable, Equatable {
    case running(heartbeatAt: Date)
    case degraded(issues: [WatcherHealthIssue], heartbeatAt: Date)
    case stopped(reason: String)
    case unknown(reason: String)

    /// The watcher writes once per poll, ~60-95 s apart (measured 91 s). Five minutes without
    /// a heartbeat is several missed polls, not normal spacing.
    static let staleHeartbeatSeconds: TimeInterval = 300

    static let logHint = "~/Library/Logs/brainlayer/watch.err.log"

    static func derive(launchd: WatcherLaunchdEvidence, file: WatcherHealthFileRead?, now: Date) -> Self {
        switch launchd {
        case let .notRunning(detail):
            // launchd is authoritative for "is the process there"; a fresh file cannot revive it.
            // launchd records no stop time. A heartbeat proves the watcher was alive AT that time,
            // so it only bounds the stop from above: report it as a fact, never as a duration.
            let since: String = if case let .readable(health) = file {
                "stop time unknown; last heartbeat \(age(health.updatedAt, now: now))"
            } else {
                "stop time unknown — no readable heartbeat"
            }
            return .stopped(
                reason: "Watcher is not running (\(detail)) · \(since) · Restart it from Settings → Jobs → Ingest, or check \(logHint)"
            )
        case .running:
            switch file {
            case nil:
                return .unknown(reason: "Watcher is running, but its health file has not been read yet.")
            case let .missing(path):
                return .unknown(reason: "Watcher is running, but its health file is missing at \(path).")
            case let .unreadable(path, reason):
                return .unknown(reason: "Watcher is running, but its health file at \(path) is unreadable: \(reason).")
            case let .readable(health):
                return fromHeartbeat(health, now: now)
            }
        case let .unavailable(detail):
            // The heartbeat is the canonical liveness surface: a fresh one stands on its own.
            if case let .readable(health) = file, !isStale(health, now: now) {
                return fromHeartbeat(health, now: now)
            }
            let heartbeat: String
            switch file {
            case let .readable(health):
                heartbeat = "the last heartbeat was \(DashboardMetricFormatter.relativeEventString(lastEventAt: health.updatedAt, now: now).lowercased())"
            case let .missing(path):
                heartbeat = "the health file is missing at \(path)"
            case let .unreadable(path, reason):
                heartbeat = "the health file at \(path) is unreadable: \(reason)"
            case nil:
                heartbeat = "the health file has not been read yet"
            }
            return .unknown(reason: "launchd status is unavailable (\(detail)) and \(heartbeat).")
        }
    }

    private static func isStale(_ health: WatcherHealthFile, now: Date) -> Bool {
        now.timeIntervalSince(health.updatedAt) > staleHeartbeatSeconds
    }

    private static func fromHeartbeat(_ health: WatcherHealthFile, now: Date) -> Self {
        if isStale(health, now: now) {
            // Alerts in a stale file describe the past; the only current fact is the silence.
            return .degraded(issues: [WatcherHealthIssue(
                what: "Watcher heartbeat stopped updating",
                since: health.updatedAt,
                action: "Check \(logHint), then restart the Watcher job"
            )], heartbeatAt: health.updatedAt)
        }
        var issues: [WatcherHealthIssue] = []
        for reason in health.alertReasons {
            switch reason {
            case "coverage_drop":
                issues.append(WatcherHealthIssue(
                    what: "Watcher is reading transcripts but few chunks are landing in the database",
                    since: nil,
                    action: "Check \(logHint) for write errors"
                ))
            case "offset_lag":
                issues.append(WatcherHealthIssue(
                    what: "Watcher is \(megabytes(health.maxOffsetLagBytes)) behind on transcripts",
                    since: nil,
                    action: "It usually catches up on its own; if it keeps growing, check \(logHint)"
                ))
            case "file_ingestion_failure":
                let count = max(health.fileIngestionFailureCount, 1)
                issues.append(WatcherHealthIssue(
                    what: "\(count) transcript \(count == 1 ? "file" : "files") could not be ingested",
                    since: health.earliestFileIngestionFailureAt,
                    action: "See file_ingestion_failures in watcher-health.json",
                    sinceBoundNote: cappedNote(listed: health.fileIngestionFailuresListed, total: count)
                ))
            case "quarantined_record":
                let count = max(health.quarantinedRecordCount, 1)
                issues.append(WatcherHealthIssue(
                    what: "\(count) transcript \(count == 1 ? "record" : "records") quarantined",
                    since: health.earliestQuarantinedRecordAt,
                    action: "Review quarantined_records in watcher-health.json",
                    sinceBoundNote: cappedNote(listed: health.quarantinedRecordsListed, total: count)
                ))
            default:
                issues.append(WatcherHealthIssue(
                    what: "Watcher reported alert \(reason)",
                    since: nil,
                    action: "Check \(logHint)"
                ))
            }
        }
        if health.dbProbeFailed {
            issues.append(WatcherHealthIssue(
                what: "Watcher cannot read the database to confirm its writes",
                since: nil,
                action: "Check the database path and \(logHint)"
            ))
        }
        return issues.isEmpty
            ? .running(heartbeatAt: health.updatedAt)
            : .degraded(issues: issues, heartbeatAt: health.updatedAt)
    }

    /// The producer lists at most 100 failure/quarantine details (failures: first 100 by path;
    /// quarantines: the newest 100). A shorter list than the count means the listed earliest is
    /// only a lower bound on when the problem started.
    private static func cappedNote(listed: Int?, total: Int) -> String? {
        guard let listed, listed < total else { return nil }
        return "\(listed) of \(total) listed"
    }

    private static func megabytes(_ bytes: Int) -> String {
        let mb = Double(bytes) / 1_048_576
        return mb >= 1 ? "\(Int(mb.rounded())) MB" : "\(bytes) bytes"
    }

    var title: String {
        switch self {
        case .running: "Watcher running"
        case .degraded: "Watcher needs attention"
        case .stopped: "Watcher stopped"
        case .unknown: "Watcher status unknown"
        }
    }

    /// Degraded and stopped are real problems. Unknown is an honest "can't tell", never a claim
    /// that the watcher is down.
    var needsAttention: Bool {
        switch self {
        case .degraded, .stopped: true
        case .running, .unknown: false
        }
    }

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }

    /// One line, the same on every surface: what · since when · what to do.
    func reasonText(now: Date) -> String? {
        switch self {
        case .running:
            return nil
        case let .stopped(reason), let .unknown(reason):
            return reason
        case let .degraded(issues, heartbeatAt):
            guard let first = issues.first else { return nil }
            let when = first.since.map { since in
                first.sinceBoundNote.map { "since at least \(Self.age(since, now: now)) (\($0))" }
                    ?? "since \(Self.age(since, now: now))"
            } ?? "as of \(Self.age(heartbeatAt, now: now))"
            let more = issues.count > 1 ? " (+\(issues.count - 1) more)" : ""
            return "\(first.what) · \(when) · \(first.action)\(more)"
        }
    }

    private static func age(_ date: Date, now: Date) -> String {
        let relative = DashboardMetricFormatter.relativeEventString(lastEventAt: date, now: now)
        return relative == "Just now" ? "just now" : relative
    }
}

enum WatcherHealthReader {
    static let pathOverrideKey = "BRAINLAYER_WATCHER_HEALTH_PATH"

    /// `watcher-health.json` beside the resolved DB, unless BRAINLAYER_WATCHER_HEALTH_PATH overrides it.
    static func url(dbPath: String, environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let override = environment[pathOverrideKey], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath: dbPath).deletingLastPathComponent().appendingPathComponent("watcher-health.json")
    }

    /// The watcher's OWN health path, resolved the way the watcher resolves it (#1014): the watcher's
    /// launchd environment overlaid by the env file `brainlayer-env-run.sh` exports over it, then
    /// `BRAINLAYER_WATCHER_HEALTH_PATH`, else beside `BRAINLAYER_DB`, else beside the canonical DB.
    /// BrainBar's own `BRAINBAR_DB_PATH` names BrainBar's database, not the watcher's, so it never
    /// steers this.
    static func watcherURL(environment: [String: String], envFile: String?, home: URL) -> URL {
        var effective = environment
        for (key, value) in envFileValues(envFile) { effective[key] = value }
        func expand(_ path: String) -> String {
            path == "~" ? home.path : path.hasPrefix("~/") ? home.path + path.dropFirst() : path
        }
        let dbPath = effective["BRAINLAYER_DB"].flatMap { $0.isEmpty ? nil : expand($0) }
            ?? home.appendingPathComponent(".local/share/brainlayer/brainlayer.db").path
        let override = effective[pathOverrideKey].flatMap { $0.isEmpty ? nil : expand($0) }
        return url(dbPath: dbPath, environment: override.map { [pathOverrideKey: $0] } ?? [:])
    }

    /// The live path: this process's environment plus the env file the watcher's LaunchAgent loads.
    static func resolvedURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        let envFilePath = environment["BRAINLAYER_ENV_FILE"].flatMap { $0.isEmpty ? nil : $0 }
            ?? home.appendingPathComponent(".config/brainlayer/brainlayer.env").path
        return watcherURL(
            environment: environment,
            envFile: try? String(contentsOfFile: envFilePath, encoding: .utf8),
            home: home
        )
    }

    /// The same simple `KEY=value` / `export KEY="value"` lines `brainlayer-env-run.sh` exports.
    /// Command substitutions are skipped there too.
    private static func envFileValues(_ text: String?) -> [String: String] {
        guard let text else { return [:] }
        var values: [String: String] = [:]
        for raw in text.split(whereSeparator: \.isNewline) {
            var line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }
            if line.hasPrefix("export ") { line = String(line.dropFirst("export ".count)) }
            guard let equals = line.firstIndex(of: "=") else { continue }
            let key = line[..<equals].trimmingCharacters(in: .whitespaces)
            var value = line[line.index(after: equals)...].trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty, !value.contains("$("), !value.contains("`") else { continue }
            if value.count >= 2, let first = value.first, first == value.last, first == "\"" || first == "'" {
                value = String(value.dropFirst().dropLast())
            }
            values[key] = value
        }
        return values
    }

    static func read(url: URL) -> WatcherHealthFileRead {
        guard FileManager.default.fileExists(atPath: url.path) else { return .missing(path: url.path) }
        do {
            return parse(try Data(contentsOf: url), path: url.path)
        } catch {
            return .unreadable(path: url.path, reason: "could not read the file (\((error as NSError).localizedDescription))")
        }
    }

    static func parse(_ data: Data, path: String) -> WatcherHealthFileRead {
        guard let payload = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return .unreadable(path: path, reason: "not a JSON object")
        }
        // The watcher writes these together in one atomic snapshot; a snapshot missing any of them
        // cannot prove a live, healthy watcher (#1013 review B3).
        let missing = ["updated_at", "poll_count", "alert_reasons"].filter { payload[$0] == nil || payload[$0] is NSNull }
        guard missing.isEmpty else {
            return .unreadable(path: path, reason: "missing required field(s): \(missing.joined(separator: ", "))")
        }
        // Every present field must carry the producer's type. Coercing or dropping a bad value
        // (e.g. `"alert_reasons": [17]` read as zero alerts) would let a malformed file read as
        // Running (#1013 round-2 B3).
        guard let updatedAt = date(payload["updated_at"]) else {
            return .unreadable(path: path, reason: "updated_at must be an ISO-8601 timestamp")
        }
        guard let rawReasons = payload["alert_reasons"] as? [Any],
              let alertReasons = rawReasons as? [String], alertReasons.count == rawReasons.count else {
            return .unreadable(path: path, reason: "alert_reasons must be a list of strings")
        }
        let integerKeys = [
            "poll_count", "max_offset_lag_bytes", "file_ingestion_failure_count", "quarantined_record_count_total",
        ]
        var integers: [String: Int] = [:]
        for key in integerKeys {
            guard let value = payload[key] else { continue }
            guard let integer = strictInteger(value) else {
                return .unreadable(path: path, reason: "\(key) must be an integer")
            }
            integers[key] = integer
        }
        var dbProbeFailed = false
        if let value = payload["db_probe_failed"] {
            guard let flag = strictBool(value) else {
                return .unreadable(path: path, reason: "db_probe_failed must be a boolean")
            }
            dbProbeFailed = flag
        }
        for key in ["file_ingestion_failures", "quarantined_records"] {
            if let value = payload[key], !(value is [Any]) {
                return .unreadable(path: path, reason: "\(key) must be a list")
            }
        }
        return .readable(WatcherHealthFile(
            updatedAt: updatedAt,
            pollCount: integers["poll_count"] ?? 0,
            alertReasons: alertReasons,
            dbProbeFailed: dbProbeFailed,
            maxOffsetLagBytes: integers["max_offset_lag_bytes"] ?? 0,
            fileIngestionFailureCount: integers["file_ingestion_failure_count"] ?? 0,
            fileIngestionFailuresListed: (payload["file_ingestion_failures"] as? [Any])?.count,
            earliestFileIngestionFailureAt: earliestObservedAt(payload["file_ingestion_failures"]),
            quarantinedRecordCount: integers["quarantined_record_count_total"] ?? 0,
            quarantinedRecordsListed: (payload["quarantined_records"] as? [Any])?.count,
            earliestQuarantinedRecordAt: earliestObservedAt(payload["quarantined_records"])
        ))
    }

    /// JSON numbers arrive as NSNumber, which bridges `true` to 1 and `1` to true. Only a
    /// non-boolean whole number is an integer here.
    private static func strictInteger(_ value: Any) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        return value as? Int
    }

    private static func strictBool(_ value: Any) -> Bool? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }

    private static func earliestObservedAt(_ value: Any?) -> Date? {
        (value as? [Any])?
            .compactMap { ($0 as? [String: Any])?["observed_at"] }
            .compactMap(date)
            .min()
    }

    /// Python's `datetime.isoformat()`: microseconds are optional, the offset is always present.
    static func date(_ value: Any?) -> Date? {
        guard let raw = (value as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: raw) { return date }
        if let date = ISO8601DateFormatter().date(from: raw) { return date }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXXXX"
        return formatter.date(from: raw)
    }
}
