import Foundation

/// The job-alert state the Python jobs keep in `job-alerts.json` (`brainlayer.job_alerts`): one
/// reason per failing job key, removed by the job's next clean run.
///
/// `observability.json` copies one of these into `backups.error_type` as `job_alert:<reason>`, but
/// only when its producer next runs. Reconciling with this file makes a clean run clear the alert
/// in every BrainBar surface right away, and names the job, so Show log opens that job's log.
struct BrainBarJobAlerts: Equatable, Sendable {
    /// The LaunchAgent that writes `observability.json`; its environment names the alert file.
    static let producerLabel = "com.brainlayer.observability"
    static let pathEnvironmentKey = "BRAINLAYER_JOB_ALERT_PATH"
    static let observabilityPrefix = "job_alert:"

    /// Job key → reason, e.g. `maintenance-light` → "BrainLayer light maintenance failed; …".
    let active: [String: String]

    /// The same path `job_alerts.alert_path` resolves: the override, else beside the database.
    static func url(dbPath: String, environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let override = environment[pathEnvironmentKey], !override.isEmpty {
            return URL(fileURLWithPath: BadgeStateReader.expandedTildePath(override))
        }
        return URL(fileURLWithPath: BadgeStateReader.expandedTildePath(dbPath))
            .deletingLastPathComponent()
            .appendingPathComponent("job-alerts.json")
    }

    /// Nil when the file is missing, unreadable or not a map of job key → reason string: unknown,
    /// never "no alerts", so a malformed file can never clear a live alert.
    static func read(url: URL) -> Self? {
        guard let data = FileManager.default.contents(atPath: url.path),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        var active: [String: String] = [:]
        for (key, value) in object {
            guard let reason = value as? String else { return nil }
            active[key] = reason
        }
        return Self(active: active)
    }

    /// The job alert's reason exactly as the job wrote it, before display sanitizing; the key
    /// lookup must use this, since the shown text may be redacted or shortened.
    static func rawReason(_ document: ObservabilityDocument) -> String? {
        guard let errorType = document.backups.errorType, errorType.hasPrefix(observabilityPrefix) else { return nil }
        return String(errorType.dropFirst(observabilityPrefix.count))
    }

    func key(for reason: String) -> String? {
        active.filter { $0.value == reason }.keys.min()
    }

    /// The document as of this alert state. A job alert no job still reports is cleared; if another
    /// job still reports one, that alert is shown instead. Any other backup error is left alone.
    func reconcile(_ document: ObservabilityDocument) -> ObservabilityDocument {
        guard let errorType = document.backups.errorType, errorType.hasPrefix(Self.observabilityPrefix) else {
            return document
        }
        let shown = String(errorType.dropFirst(Self.observabilityPrefix.count))
        if active.values.contains(shown) { return document }
        let remaining = active.keys.min().flatMap { active[$0] }
        return document.replacingBackupsErrorType(remaining.map { Self.observabilityPrefix + $0 })
    }

    /// The alert file the producer of `document` reads (see `readReconciled`).
    static func producerURL(
        for document: ObservabilityDocument,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        readFile: (URL) -> Data? = { FileManager.default.contents(atPath: $0.path) }
    ) -> URL {
        let producer = BrainLayerJobEnvironment.effective(
            plist: readFile(home.appendingPathComponent("Library/LaunchAgents/\(producerLabel).plist")),
            brainBarEnvironment: environment, home: home, readFile: readFile
        )
        return url(dbPath: document.dbPath, environment: producer)
    }

    /// Reconciles with a state that may be unknown; an unknown state leaves the document as read.
    static func reconcile(_ document: ObservabilityDocument, with alerts: Self?) -> ObservabilityDocument {
        alerts?.reconcile(document) ?? document
    }
}

extension ObservabilityDocument {
    func replacingBackupsErrorType(_ errorType: String?) -> Self {
        let b = backups
        return Self(
            schemaVersion: schemaVersion, generatedAt: generatedAt, dbPath: dbPath, windowHours: windowHours,
            stores: stores, emitters: emitters, authorUnknown: authorUnknown,
            backups: .init(
                state: b.state, reason: b.reason, inputs: b.inputs, freshness: b.freshness,
                thresholdHours: b.thresholdHours, retentionInvariant: b.retentionInvariant,
                survivingArchives30D: b.survivingArchives30D, errorType: errorType,
                lastVerifiedUpload: b.lastVerifiedUpload, dbSnapshot: b.dbSnapshot, launchd: b.launchd
            )
        )
    }
}

extension ObservabilityReader {
    /// What BrainBar shows: the document, reconciled with the live job-alert state the producer
    /// read: `BRAINLAYER_JOB_ALERT_PATH` from the producer's own environment (its LaunchAgent and
    /// env file, then BrainBar's), else beside the producer's database.
    static func readReconciled(
        url: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        readFile: (URL) -> Data? = { FileManager.default.contents(atPath: $0.path) }
    ) -> ObservabilityReadResult {
        let result = read(url: url)
        guard case let .readable(document) = result else { return result }
        let alertsURL = BrainBarJobAlerts.producerURL(for: document, environment: environment, home: home, readFile: readFile)
        let alerts = BrainBarJobAlerts.read(url: alertsURL)
        return .readable(BrainBarJobAlerts.reconcile(document, with: alerts))
    }
}
