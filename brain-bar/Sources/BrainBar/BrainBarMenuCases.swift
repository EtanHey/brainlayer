#if DEBUG
import CryptoKit
import Foundation

/// The lead's `presentation-cases.json` (phase 8), read once for the table test and the render harness, so a case's
/// capture is built one way only: decoded from its own `inputs`, literally, named as the truth table names them.
enum BrainBarMenuCases {
    /// The plan's `phase-8-native-build/presentation-cases.json`, byte for byte.
    static let sha256 = "65e4e76375265706bd23e73e6869d5c387d2d0f221ccfb8218bab7d1286882a6"

    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    /// The copy the tests bundle. The harness reads it beside the sources, or from `BRAINBAR_MENU_CASES`.
    static func sourceURL() -> URL {
        if let path = ProcessInfo.processInfo.environment["BRAINBAR_MENU_CASES"], !path.isEmpty { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../Tests/BrainBarTests/Fixtures/presentation-cases.json").standardizedFileURL
    }

    static func digest(of url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    }

    /// The cases, refused unless the file is the lead's unchanged copy.
    static func load(_ url: URL) throws -> [[String: Any]] {
        let found = try digest(of: url)
        guard found == sha256 else { throw Failure("\(url.path): sha256 \(found), expected the lead's \(sha256)") }
        let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        guard let cases = doc?["cases"] as? [[String: Any]] else { throw Failure("\(url.path): no cases array") }
        return cases
    }

    /// A time as the file writes it; some carry a remark after the timestamp, as in "…+02:00 (synthetic; …)".
    static func date(_ value: Any?) -> Date? {
        (value as? String).flatMap { ISO8601DateFormatter().date(from: String($0.prefix { $0 != " " })) }
    }

    private static func required<T>(_ value: T?, _ what: String) throws -> T {
        guard let value else { throw Failure("case inputs: \(what) is missing or malformed") }
        return value
    }

    static func capture(_ inputs: [String: Any]) throws -> BrainBarMenuCapture {
        // S8's `now` is null (first launch): any time does.
        let now = try required(date(inputs["now"]) ?? date("2026-11-12T12:48:00+02:00"), "now")
        var c = BrainBarMenuCapture(now: now, timeZone: try required(TimeZone(secondsFromGMT: 7200), "time zone"))
        let verdict = inputs["verdict"] as? [String: Any]
        let checks = inputs["checks"] as? [String: [String: Any]] ?? [:]
        if verdict?["decodable"] as? Bool == true {
            // The file's convention: any of the 11 checks not given is green, with blind false and unknown_ticks 0.
            let built = try BrainBarMenuPresenter.known.map { name -> BrainBarHealthVerdict.Check in
                let k = checks[name] ?? [:]
                return .init(name: name, state: try required(.init(rawValue: k["state"] as? String ?? "green"), "\(name).state"),
                             ageS: k["ageS"] as? Double, thresholdS: k["thresholdS"] as? Double, since: k["since"] as? String,
                             note: "", unknownTicks: 0, blind: false)
            }
            guard verdict?["fresh"] as? Bool == true else { throw Failure("case inputs: a decodable verdict must be fresh") }
            c.verdict = BrainBarHealthVerdict(dot: .green, checks: built, notes: [],
                                              evaluatedAt: try required(date(verdict?["evaluatedAt"]), "verdict.evaluatedAt"))
        }
        c.isChecking = inputs["app.isChecking"] as? Bool ?? false
        c.wakeCatchUpPending = inputs["app.wakeCatchUpPending"] as? Bool ?? false
        c.backupsMeasured = (inputs["backupsObservation.state"] as? String ?? "measured") == "measured"
        c.lastKnownBackupAt = date(inputs["backupsObservation.lastMeasured.newestVerifiedCopyAt"])
        if let w = inputs["watcherFile"] as? [String: Any] {
            c.watcher = .init(readable: try required(w["readable"] as? Bool, "watcherFile.readable"),
                              fresh: try required(w["fresh"] as? Bool, "watcherFile.fresh"),
                              alertReasons: try required(w["alertReasons"] as? [String], "watcherFile.alertReasons"),
                              dbProbeFailed: try required(w["dbProbeFailed"] as? Bool, "watcherFile.dbProbeFailed"))
        }
        if inputs["localSnapshot.uploadState"] as? String == "in_progress" {
            c.upload = .init(startedAt: try required(date(inputs["uploadProcessStartedAt"]), "uploadProcessStartedAt"),
                             live: try required(inputs["uploadProcessLive"] as? Bool, "uploadProcessLive"))
        }
        if let r = checks["backup_offsite"]?["recovering"] as? [String: Any] {
            c.offsiteRecovering = .init(progress: r["progress"] as? Double, eta: date(r["eta"]),
                                        startedAt: try required(date(r["started_at"]), "recovering.started_at"),
                                        updatedAt: try required(date(r["updated_at"]), "recovering.updated_at"),
                                        cadenceS: try required(r["cadence_s"] as? Double, "recovering.cadence_s"))
        }
        c.lagEpisodeStart = date(inputs["uiClock.offsetLagEpisodeStart"])
        c.maintenanceNightFailures = inputs["maintenance.night.consecutiveFailures"] as? Int ?? 0
        // "invalid (wording only: not missing)" and the like: the state is the first word.
        if let state = inputs["DriveAuthModel.state"] as? String {
            c.driveAuthModel = try required(DriveAuthStatus.State(rawValue: String(state.prefix { $0 != " " })), "DriveAuthModel.state")
        }
        c.backupDailyFailures = inputs["job.backupDaily.consecutiveFailures"] as? Int ?? 0
        c.backupDailyNextRunAt = date(inputs["job.backupDaily.nextRunAt"])
        c.jsonlBackupFailures = inputs["job.jsonlBackup.consecutiveFailures"] as? Int ?? 0
        c.keywordCoverage = inputs["collector.signalCoverage.keyword"] as? Double
        return c
    }
}
#endif
