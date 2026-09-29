import Foundation

/// What the Python watcher last wrote to `watcher-health.json` beside the resolved DB.
/// `poll_count` and `updated_at` advance on every poll (~60-95 s apart); see AGENTS.md
/// "Real-time JSONL Watcher".
struct WatcherHealthFile: Sendable, Equatable {
    var updatedAt: Date
    var pollCount: Int?
    var alertReasons: [String] = []
    var dbProbeFailed = false
    var maxOffsetLagBytes = 0
    var fileIngestionFailureCount = 0
    var earliestFileIngestionFailureAt: Date?
    var quarantinedRecordCount = 0
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
            return .stopped(reason: "Watcher is not running (\(detail)). Restart it from Settings → Jobs → Ingest, or check \(logHint).")
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
                    action: "See file_ingestion_failures in watcher-health.json"
                ))
            case "quarantined_record":
                let count = max(health.quarantinedRecordCount, 1)
                issues.append(WatcherHealthIssue(
                    what: "\(count) transcript \(count == 1 ? "record" : "records") quarantined",
                    since: health.earliestQuarantinedRecordAt,
                    action: "Review quarantined_records in watcher-health.json"
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
            let when = first.since.map { "since \(Self.age($0, now: now))" } ?? "as of \(Self.age(heartbeatAt, now: now))"
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
        guard let updatedAt = date(payload["updated_at"]) else {
            return .unreadable(path: path, reason: "no parseable updated_at")
        }
        return .readable(WatcherHealthFile(
            updatedAt: updatedAt,
            pollCount: payload["poll_count"] as? Int,
            alertReasons: (payload["alert_reasons"] as? [Any])?.compactMap { $0 as? String } ?? [],
            dbProbeFailed: payload["db_probe_failed"] as? Bool ?? false,
            maxOffsetLagBytes: payload["max_offset_lag_bytes"] as? Int ?? 0,
            fileIngestionFailureCount: payload["file_ingestion_failure_count"] as? Int ?? 0,
            earliestFileIngestionFailureAt: earliestObservedAt(payload["file_ingestion_failures"]),
            quarantinedRecordCount: payload["quarantined_record_count_total"] as? Int ?? 0,
            earliestQuarantinedRecordAt: earliestObservedAt(payload["quarantined_records"])
        ))
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
