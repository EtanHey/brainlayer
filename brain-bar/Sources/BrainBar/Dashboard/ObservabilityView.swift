import Combine
import Foundation
import SwiftUI

struct BadgeStateDocument: Codable, Sendable {
    let schemaVersion: Int
    let generatedAt: Date
    let alerts: Alerts

    struct Alerts: Codable, Sendable {
        let state: String
        let reason: String
        let inputs: [Input]
        let expectedFirstRunBy: Date?
        let badgeOn: Bool?
        let active: [Issue]?
        let suppressed: [Issue]?
    }

    struct Input: Codable, Sendable {}
    struct Issue: Codable, Sendable {
        let code: String
        let severity: String
        let message: String
        let episode: Episode?
    }

    struct Episode: Codable, Sendable {
        let kind, owner, episodeId: String
        let startedAt, observedAt, deadline: Date
        let expectedCodes: [String]

        func allows(_ issue: Issue, now: Date) -> Bool {
            // Keep these bounds aligned with badge_state.py. No sleep grace,
            // host-load heuristic, or snapshot exemption for unrelated faults.
            issue.severity == "critical" && issue.code == "brain_search_canary_failed"
                && ["snapshot", "maintenance"].contains(kind)
                && !owner.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && !episodeId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && expectedCodes.contains(issue.code)
                && startedAt <= observedAt && observedAt <= now && now < deadline
                && deadline.timeIntervalSince(startedAt) <= 3600
                && now.timeIntervalSince(observedAt) <= 300
        }
    }
}

struct BadgeStatePresentation: Equatable, Sendable {
    let badgeOn: Bool
    let reason: String
    let activeCodes: [String]
    /// Each active issue's message, in `activeCodes` order; `reason` joins them.
    var activeMessages: [String] = []
    /// Empty only for legacy in-memory callers, whose active issues are critical.
    var activeSeverities: [String] = []
    var vectorScan: BrainBarVectorScan? = nil

    static func failVisible(_ reason: String) -> Self {
        .init(badgeOn: true, reason: reason, activeCodes: [])
    }
}

final class BadgeReadHistory: @unchecked Sendable {
    private var previousReadAt: Date?
    private var gracedDocumentAt: Date?
    private var graceUntil: Date?

    func missedPreviousRead(now: Date, cadence: ObservabilityCadence) -> Bool {
        defer { previousReadAt = now }
        guard let previousReadAt else { return false }
        return now < previousReadAt || now.timeIntervalSince(previousReadAt) >= cadence.interval * 2
    }

    func sleepGrace(now: Date, cadence: ObservabilityCadence, documentGeneratedAt: Date, missedPreviousRead: Bool) -> Date? {
        if missedPreviousRead, gracedDocumentAt != documentGeneratedAt {
            gracedDocumentAt = documentGeneratedAt
            graceUntil = now.addingTimeInterval(cadence.interval)
        } else if let graceUntil, now > graceUntil {
            self.graceUntil = nil
        }
        return gracedDocumentAt == documentGeneratedAt ? graceUntil : nil
    }
}

enum BadgeStateReader {
    static func expandedTildePath(_ path: String) -> String {
        guard path.hasPrefix("~") else { return path }
        let userAndSuffix = path.dropFirst()
        let separator = userAndSuffix.firstIndex(of: "/")
        let user = separator.map { String(userAndSuffix[..<$0]) } ?? String(userAndSuffix)
        let suffix = separator.map { String(userAndSuffix[$0...]) } ?? ""
        let home = user.isEmpty
            ? FileManager.default.homeDirectoryForCurrentUser
            : FileManager.default.homeDirectory(forUser: user)
        guard let home else { return path }
        return home.path + suffix
    }

    static func url(
        dbPath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        if let override = environment["BRAINLAYER_BADGE_STATE_PATH"], !override.isEmpty {
            return URL(fileURLWithPath: expandedTildePath(override))
        }
        return URL(fileURLWithPath: expandedTildePath(dbPath))
            .resolvingSymlinksInPath().deletingLastPathComponent()
            .appendingPathComponent("badge-state.json")
    }

    static func read(
        url: URL,
        now: Date,
        cadence: ObservabilityCadence,
        history: BadgeReadHistory? = nil
    ) -> BadgeStatePresentation {
        let missedPreviousRead = history?.missedPreviousRead(now: now, cadence: cadence) ?? false
        // An unknown cadence falls back to its conservative interval; staleness
        // past twice that interval still fails visible below.
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .failVisible("Badge state missing without a pending_first_run marker.")
        }
        do {
            let data = try Data(contentsOf: url)
            let envelope = try JSONDecoder().decode(BadgeSchemaEnvelope.self, from: data)
            guard envelope.schemaVersion == 1 else {
                return .failVisible("Unsupported badge schema_version \(envelope.schemaVersion); expected 1.")
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            decoder.dateDecodingStrategy = .custom { value in
                let raw = try value.singleValueContainer().decode(String.self)
                let fractional = ISO8601DateFormatter()
                fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                if let date = fractional.date(from: raw) ?? ISO8601DateFormatter().date(from: raw) { return date }
                throw DecodingError.dataCorruptedError(
                    in: try value.singleValueContainer(),
                    debugDescription: "Invalid badge date-time"
                )
            }
            let document = try decoder.decode(BadgeStateDocument.self, from: data)
            let age = now.timeIntervalSince(document.generatedAt)
            guard age >= 0 else {
                return .failVisible("Badge state timestamp is in the future.")
            }
            let graceUntil = history?.sleepGrace(
                now: now, cadence: cadence, documentGeneratedAt: document.generatedAt,
                missedPreviousRead: missedPreviousRead)
            let sleepGraceApplies = graceUntil.map { now <= $0 } == true
            if document.alerts.state == "pending_first_run" {
                guard let expected = document.alerts.expectedFirstRunBy, !document.alerts.reason.isEmpty else {
                    return .failVisible("Badge pending_first_run state is invalid.")
                }
                // The Python producer truncates generated_at to whole seconds but
                // preserves fractional seconds in expected_first_run_by.
                guard expected <= document.generatedAt.addingTimeInterval(601) else {
                    return .failVisible("Badge pending_first_run deadline exceeds the allowed 600 seconds plus timestamp precision slack.")
                }
                if now <= expected || sleepGraceApplies {
                    return .init(badgeOn: false, reason: document.alerts.reason, activeCodes: [])
                }
                return .failVisible("Badge producer missed expected_first_run_by.")
            }
            guard age <= cadence.interval * 2 || sleepGraceApplies else {
                return .failVisible("Badge state is stale.")
            }
            guard document.alerts.state == "measured", document.alerts.reason.isEmpty else {
                let reason = document.alerts.reason.isEmpty ? "Badge alert state is unknown." : document.alerts.reason
                return .failVisible(reason)
            }
            guard let badgeOn = document.alerts.badgeOn, let active = document.alerts.active,
                  badgeOn == active.contains(where: { $0.severity != "warning" }) else {
                return .failVisible("Badge alert state is internally inconsistent.")
            }
            // Re-evaluate on every read: an expired or unsupported suppression
            // must become visible even when the producer has not run again.
            let restored = (document.alerts.suppressed ?? []).filter { issue in
                issue.episode?.allows(issue, now: now) != true
            }
            let visible = active + restored
            return .init(
                badgeOn: visible.contains { $0.severity != "warning" },
                reason: visible.map(\.message).joined(separator: "; "),
                activeCodes: visible.map(\.code),
                activeMessages: visible.map(\.message),
                activeSeverities: visible.map(\.severity),
                vectorScan: BrainBarVectorScan.read(beside: url)
            )
        } catch {
            return .failVisible("Badge state unreadable: \(error.localizedDescription)")
        }
    }
}

private struct BadgeSchemaEnvelope: Decodable {
    let schemaVersion: Int
    enum CodingKeys: String, CodingKey { case schemaVersion = "schema_version" }
}

struct ObservabilityDocument: Codable, Sendable {
    let schemaVersion: Int
    let generatedAt: Date
    let dbPath: String
    let windowHours: Int
    let stores: Stores
    let emitters: Emitters
    let authorUnknown: AuthorUnknown
    let backups: Backups

    struct Stores: Codable, Sendable {
        let state: String, reason: String
        let inputs: [Input]
        let totalChunks: Int?
        let inWindow: Window?
    }
    struct Window: Codable, Sendable {
        let count: Int
        let byHour: [HourBucket]?
    }
    struct HourBucket: Codable, Sendable {
        let hour: Date
        let count: Int
    }
    struct Emitters: Codable, Sendable {
        let state: String, reason: String
        let inputs: [Input]
        let byEmitter: [Emitter]?
        let bySourceClass: [SourceClass]?
        let hiddenFromDefaultSearch: Int?
    }
    struct Emitter: Codable, Sendable { let emitter: String, countInWindow: Int }
    struct SourceClass: Codable, Sendable { let sourceClass: String?; let count, inWindow: Int }
    struct AuthorUnknown: Codable, Sendable {
        let state: String, reason: String
        let inputs: [Input]
        let neverClassified: Share?
        let classifiedUnknown: Share?
    }
    struct Share: Codable, Sendable { let count: Int, share: Double }
    struct Backups: Codable, Sendable {
        let state: String, reason: String
        let inputs: [Input]
        let freshness: String?
        let thresholdHours: Double?
        let retentionInvariant: String?
        let survivingArchives30D: Int?
        let errorType: String?
        let lastVerifiedUpload: LastVerifiedUpload?
        let dbSnapshot: DBSnapshot?
        let launchd: Launchd?
        var transcriptAttempt: TranscriptAttempt? = nil
        var localSnapshot: LocalSnapshot? = nil

        func isRecent(_ date: Date, now: Date) -> Bool {
            guard let thresholdHours, thresholdHours > 0 else { return false }
            return (0...thresholdHours * 3600).contains(now.timeIntervalSince(date))
        }

        func hasRecentVerifiedNoOp(now: Date) -> Bool {
            guard let attempt = transcriptAttempt, attempt.status == "no-op",
                  attempt.verified, !attempt.uploaded, isRecent(attempt.at, now: now) else { return false }
            return true
        }

        func verifiedIdle(now: Date) -> Bool {
            state == "measured" && freshness == "fresh" && retentionInvariant == "PASS" &&
                (errorType?.hasPrefix("jsonl_backup_attempt_") != true) && hasRecentVerifiedNoOp(now: now) &&
                lastVerifiedUpload?.verified != false
        }

        func localUploadPending(now: Date) -> Bool {
            guard let value = localSnapshot, value.localVerified,
                  !value.uploaded, let lastAt = value.lastAt,
                  ["pending", "in_progress"].contains(value.uploadState),
                  value.uploadError == nil || value.uploadError == "",
                  isRecent(lastAt, now: now),
                  let offsite = dbSnapshot, offsite.verified, isRecent(offsite.lastAt, now: now) else { return false }
            return true
        }
    }
    struct TranscriptAttempt: Codable, Sendable {
        let at: Date
        let status: String
        let verified, uploaded: Bool
    }
    struct LastVerifiedUpload: Codable, Sendable {
        let at: Date, ageHours: Double
        let archiveId: String
        let verified: Bool
    }
    struct DBSnapshot: Codable, Sendable {
        let lastAt: Date
        let destination: String
        let verified: Bool
    }
    struct LocalSnapshot: Codable, Sendable {
        let lastAt: Date?
        let destination: String?
        let localVerified, uploaded: Bool
        let uploadState: String
        let uploadError: String?
        let localIntegrity, localIntegrityPragma: String?
    }
    struct Launchd: Codable, Sendable {
        let label: String
        let bootstrapped, disabledDirPresent: Bool
    }
    struct Input: Codable, Sendable { let path: String, status: String }
}

enum ObservabilityReadResult: Sendable {
    case readable(ObservabilityDocument)
    case unreadable(String)
}

struct ObservabilityCadence: Sendable {
    let interval: TimeInterval
    let assumption: String?

    static func known(_ interval: TimeInterval) -> Self {
        .init(interval: interval, assumption: nil)
    }
}

enum ObservabilityReader {
    static let installedHealthCheckCadence = healthCheckCadence()

    static func url(
        dbPath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        if let override = environment["BRAINLAYER_OBSERVABILITY_PATH"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath: dbPath).deletingLastPathComponent()
            .appendingPathComponent("observability.json")
    }

    static func read(url: URL) -> ObservabilityReadResult {
        do {
            let data = try Data(contentsOf: url)
            let envelope = try JSONDecoder().decode(SchemaEnvelope.self, from: data)
            guard [1, 2].contains(envelope.schemaVersion) else {
                return .unreadable("Unsupported schema_version \(envelope.schemaVersion); expected 1 or 2.")
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            decoder.dateDecodingStrategy = .custom { value in
                let raw = try value.singleValueContainer().decode(String.self)
                let fractional = ISO8601DateFormatter()
                fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                if let date = fractional.date(from: raw) ?? ISO8601DateFormatter().date(from: raw) { return date }
                throw DecodingError.dataCorruptedError(in: try value.singleValueContainer(), debugDescription: "Invalid date-time")
            }
            let document = try decoder.decode(ObservabilityDocument.self, from: data)
            guard document.windowHours == 24, !document.dbPath.isEmpty else {
                return .unreadable("Observability root fields are invalid.")
            }
            return .readable(document)
        } catch {
            return .unreadable("Observability data unreadable: \(error.localizedDescription)")
        }
    }

    static func healthCheckCadence(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ObservabilityCadence {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let plistPath = environment["BRAINLAYER_HEALTH_CHECK_PLIST_PATH"] ??
            home.appendingPathComponent("Library/LaunchAgents/com.brainlayer.health-check.plist").path
        if let data = FileManager.default.contents(atPath: plistPath),
           let plist = try? PropertyListSerialization.propertyList(from: data, format: nil),
           let interval = validInterval((plist as? [String: Any])?["StartInterval"]) {
            return .known(interval)
        }
        // The supervisor now owns the health check: jobs.json keys each job by label.
        let jobsPath = environment["BRAINLAYER_SUPERVISOR_JOBS_PATH"] ??
            home.appendingPathComponent(".local/share/brainlayer/supervisor/jobs.json").path
        if let data = FileManager.default.contents(atPath: jobsPath),
           let jobs = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
           let job = jobs["com.brainlayer.health-check"] as? [String: Any],
           (job["enabled"] as? Bool) != false,
           let interval = validInterval((job["plist"] as? [String: Any])?["StartInterval"]) {
            return .known(interval)
        }
        return .init(interval: 300, assumption: "cadence unknown, assuming 300s")
    }

    private static func validInterval(_ value: Any?) -> TimeInterval? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        let interval = number.doubleValue
        // Untrustworthy schedules use the disclosed 300s fallback. Bounding the
        // TimeInterval also keeps the 2x freshness window and refresh timer finite.
        return interval.isFinite && (30.0...3600.0).contains(interval) ? interval : nil
    }

    private struct SchemaEnvelope: Decodable {
        let schemaVersion: Int
        enum CodingKeys: String, CodingKey { case schemaVersion = "schema_version" }
    }
}

enum ObservabilityCardTone: String, Equatable, Sendable { case standard, neutral, amber }
enum ObservabilityStatusTone: String, Equatable, Hashable, Sendable { case green, red, neutral }

struct ObservabilityStatusLine: Equatable, Sendable {
    let text: String
    let tone: ObservabilityStatusTone
}

struct ObservabilityStatusRows: View {
    let lines: [ObservabilityStatusLine]
    var textColor = Color.primary

    var body: some View {
        ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
            HStack(alignment: .firstTextBaseline, spacing: 7) {
                BrainBarStatusDot(color: color(line.tone))
                Text(line.text).foregroundStyle(textColor)
            }
        }
    }

    private func color(_ tone: ObservabilityStatusTone) -> Color {
        switch tone {
        case .green: Color(nsColor: BrainBarDesignTokens.Colors.statusOK)
        case .red: Color(nsColor: BrainBarDesignTokens.Colors.statusError)
        case .neutral: .secondary
        }
    }
}

struct ObservabilityBackupStatus: Equatable, Sendable {
    let upload, snapshot, job, freshness: ObservabilityStatusLine
    let retention, archives: ObservabilityStatusLine
    let error: ObservabilityStatusLine?
    /// `error` is a job alert (#1031): the failed job's own sentence for Etan.
    var errorIsJobAlert = false
    var snapshotUpload: ObservabilityStatusLine? = nil
    var uploadInProgress = false
    var uploadQueued = false
    var uploadUnconfirmed = false
    var freshnessCheckTitle = "Up to date"
    var freshnessCheckValue = "Unknown"

    var lines: [ObservabilityStatusLine] {
        [upload, snapshot] + [snapshotUpload].compactMap { $0 } + [job, freshness, retention, archives] + [error].compactMap { $0 }
    }

    /// The one red line that speaks for the backups: a job alert, the failed job's own sentence,
    /// leads whatever else is red (#1029 review B2); otherwise the first red line.
    var attentionLine: ObservabilityStatusLine? {
        if errorIsJobAlert, let error, error.tone == .red { return error }
        return lines.first { $0.tone == .red }
    }
}

struct ObservabilitySnapshot: Sendable {
    struct Card: Sendable {
        let title: String, subtitle: String?, detail: String
        let note: String?
        let tone: ObservabilityCardTone
        let statusLines: [ObservabilityStatusLine]
    }
    let generatedAt: Date, ageText: String, isStale: Bool, cards: [Card]
}

enum ObservabilityPresentation {
    static func snapshot(
        document: ObservabilityDocument,
        now: Date,
        cadence: ObservabilityCadence,
        locale: Locale = .current
    ) -> ObservabilitySnapshot {
        let age = max(0, now.timeIntervalSince(document.generatedAt))
        let stale = age > cadence.interval * 2
        let chunks = card(
            "Chunks", "What BrainLayer indexed",
            document.stores.state, document.stores.reason, stale, note: cadence.assumption
        ) {
            guard let total = document.stores.totalChunks, let recent = document.stores.inWindow?.count else { return nil }
            let base = "\(number(total, locale: locale)) chunks indexed · \(number(recent, locale: locale)) in the last 24 h — everything BrainLayer has read, all sources"
            guard document.authorUnknown.state == "measured",
                  let never = document.authorUnknown.neverClassified else {
                let reason = document.authorUnknown.reason.isEmpty ? "attribution unavailable" : document.authorUnknown.reason
                return "\(base)\nAttribution unmeasurable — \(reason)"
            }
            return "\(base)\n\(number(never.count, locale: locale)) chunks not yet attributed to a person or source class"
        }
        let stores = card(
            "Stores", "What agents wrote via brain_store",
            document.emitters.state, document.emitters.reason, stale
        ) {
            guard let emitters = document.emitters.byEmitter else { return nil }
            guard let mcp = emitters.first(where: { $0.emitter == "mcp" }) else {
                return "No MCP brain_store count in this document — what agents wrote via brain_store"
            }
            return "\(number(mcp.countInWindow, locale: locale)) MCP brain_store writes in the last 24 h — what agents wrote via brain_store"
        }
        let emitters = card(
            "Emitters", "Where indexed memory came from",
            document.emitters.state, document.emitters.reason, stale
        ) {
            guard let classes = document.emitters.bySourceClass,
                  let emitterRows = document.emitters.byEmitter else { return nil }
            func classCount(_ name: String?) -> Int {
                classes.first(where: { $0.sourceClass == name })?.count ?? 0
            }
            let subagents = classCount("subagent") + classCount("brain-worker")
            let mcp = emitterRows.first(where: { $0.emitter == "mcp" })?.countInWindow
            let mcpText = mcp.map { "\(number($0, locale: locale)) MCP brain_store writes in the last 24 h" }
                ?? "MCP brain_store count unavailable in this document"
            return [
                "\(number(classCount("cli-agent"), locale: locale)) chunks from CLI agents",
                mcpText,
                "\(number(subagents, locale: locale)) chunks from subagents",
                "\(number(classCount("desktop"), locale: locale)) chunks from desktop apps hidden from search",
                "\(number(classCount("fleet-coordination"), locale: locale)) chunks from fleet coordination",
                "\(number(classCount(nil), locale: locale)) unclassified chunks",
            ].joined(separator: "\n")
        }
        let backupTone: ObservabilityCardTone = document.backups.freshness == "unknown" ||
            document.backups.retentionInvariant == "unknown" ? .neutral :
            (document.backups.freshness == "stale" || document.backups.retentionInvariant == "FAIL" ? .amber : .standard)
        let backupStatus = backupStatus(for: document.backups, locale: locale, now: now)
        let backups = card(
            "Backups", "Where copies are, when they last verified, and whether recovery is healthy",
            document.backups.state, document.backups.reason, stale, tone: backupTone,
            statusLines: backupStatus.lines
        ) {
            guard let freshness = document.backups.freshness,
                  let retention = document.backups.retentionInvariant else { return nil }
            _ = freshness
            _ = retention
            return backupStatus.lines.map(\.text).joined(separator: "\n")
        }
        return ObservabilitySnapshot(
            generatedAt: document.generatedAt,
            ageText: age < 60 ? "just now" : "\(Int(age / 60))m old",
            isStale: stale,
            cards: [chunks, stores, emitters, backups]
        )
    }

    private static func card(
        _ title: String, _ subtitle: String, _ state: String, _ reason: String, _ stale: Bool,
        tone: ObservabilityCardTone = .standard,
        note: String? = nil,
        statusLines: [ObservabilityStatusLine] = [],
        measured: () -> String?
    ) -> ObservabilitySnapshot.Card {
        guard state == "measured", let detail = measured() else {
            let detail = reason.isEmpty ? "unmeasurable" : "unmeasurable — \(reason)"
            return .init(title: title, subtitle: subtitle, detail: detail, note: note, tone: .neutral, statusLines: [])
        }
        return .init(
            title: title, subtitle: subtitle, detail: detail, note: note,
            tone: stale ? .amber : tone, statusLines: statusLines
        )
    }

    static func number(_ value: Int, locale: Locale = .current) -> String {
        DashboardMetricFormatter.integerString(value, locale: locale)
    }

    static func backupStatus(
        for backups: ObservabilityDocument.Backups,
        locale: Locale = .current,
        jobStatus: ObservabilityStatusLine? = nil,
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> ObservabilityBackupStatus {
        let idle = backups.verifiedIdle(now: now)
        var upload = backups.lastVerifiedUpload.map { value in
            ObservabilityStatusLine(
                text: "\(value.verified ? "Last verified transcript upload" : "Last transcript upload (NOT verified)"): \(localDate(value.at)) (\(hours(max(0, now.timeIntervalSince(value.at) / 3600))) ago) · transcript archive \(value.archiveId)",
                tone: value.verified ? .green : .red
            )
        } ?? .init(text: "No verified transcript upload on record", tone: .red)

        if idle, let attempt = backups.transcriptAttempt {
            let since = backups.lastVerifiedUpload.map { " since \(idleDate($0.at, calendar: calendar, locale: locale))" } ?? ""
            upload = .init(text: "No new transcripts\(since) · checked \(relative(attempt.at, now: now, locale: locale))", tone: .neutral)
        }
        let staleTranscript = !idle && backups.thresholdHours != nil &&
            backups.lastVerifiedUpload.map { $0.verified && !backups.isRecent($0.at, now: now) } == true
        if staleTranscript { upload = .init(text: upload.text, tone: .red) }
        if backups.lastVerifiedUpload.map({ $0.at > now }) == true {
            upload = .init(text: "Last transcript upload has an invalid timestamp", tone: .red)
        }

        var snapshot = backups.dbSnapshot.map { value in
            ObservabilityStatusLine(
                text: "Latest DB snapshot (\(value.verified ? "verified" : "NOT verified")): \(localDate(value.lastAt)) → \(value.destination)",
                tone: value.verified ? .green : .red)
        } ?? .init(text: "No verified DB snapshot on record", tone: .red)
        // Judge the receipt at presentation time too: a cached producer's "fresh" flag cannot
        // keep an offsite copy green after its declared freshness window has elapsed.
        let staleOffsite = (backups.thresholdHours != nil || backups.localSnapshot != nil || backups.transcriptAttempt != nil || backups.freshness == "stale") &&
            backups.dbSnapshot.map { $0.verified && !backups.isRecent($0.lastAt, now: now) } == true
        let offsiteAge = backups.dbSnapshot.map { relative($0.lastAt, now: now, locale: locale) } ?? "unknown"
        let pendingState = backups.localSnapshot.map { ["pending", "in_progress"].contains($0.uploadState) } == true
        let staleText: String
        if let date = backups.dbSnapshot?.lastAt, now >= date {
            let days = Int(now.timeIntervalSince(date) / 86400)
            let age = days > 0 ? "\(number(days, locale: locale)) \(days == 1 ? "day" : "days") old" : "outside the freshness window (\(offsiteAge))"
            staleText = "Last offsite DB copy is \(age)" + (pendingState ? "; upload still pending" : "")
        } else {
            staleText = "Last offsite DB copy has an invalid timestamp"
        }
        if staleOffsite { snapshot = .init(text: staleText, tone: .red) }
        let snapshotUpload: ObservabilityStatusLine?
        if let value = backups.localSnapshot {
            if let error = value.uploadError, !error.isEmpty {
                snapshotUpload = .init(text: "DB upload \(value.uploadState): \(error)", tone: .red)
            } else if ["failed", "abandoned"].contains(value.uploadState) {
                snapshotUpload = .init(text: "DB upload \(value.uploadState): No failure reason provided", tone: .red)
            } else if value.uploadState == "superseded", backups.dbSnapshot?.verified == true {
                snapshotUpload = .init(text: "Local snapshot superseded by a verified offsite copy", tone: .neutral)
            } else if value.uploadState == "nothing_pending" {
                snapshotUpload = .init(text: missingLocalSnapshot, tone: .red)
            } else if pendingState && staleOffsite {
                snapshotUpload = .init(text: staleText, tone: .red)
            } else if backups.localUploadPending(now: now) {
                snapshotUpload = .init(text: value.uploadState == "pending" ? "Upload queued" : "Upload checkpoint saved; current activity unknown", tone: .neutral)
            } else if !value.localVerified || value.lastAt.map({ !backups.isRecent($0, now: now) }) != false ||
                        !value.uploaded || value.uploadState != "verified" {
                snapshotUpload = .init(text: "Local DB snapshot missing verification or stale", tone: .red)
            } else {
                snapshotUpload = nil
            }
        } else {
            snapshotUpload = nil
        }

        let job: ObservabilityStatusLine
        let jobLabel = backups.launchd?.label ?? "com.brainlayer.jsonl-backup"
        if let jobStatus {
            job = jobStatus
        } else if backups.launchd?.bootstrapped == true {
            job = .init(text: "Transcript backup (\(jobLabel)): loaded", tone: .green)
        } else if backups.launchd?.disabledDirPresent == true {
            job = .init(text: "Transcript backup (\(jobLabel)): NOT loaded — parked in .disabled-retention-P0", tone: .red)
        } else {
            job = .init(text: "Transcript backup (\(jobLabel)): NOT loaded", tone: .red)
        }

        let threshold = backups.thresholdHours.map { number(Int($0), locale: locale) } ?? "unknown"
        let plainThreshold = backups.thresholdHours.map { "\(Int($0)) hours" }
        let freshness: ObservabilityStatusLine
        let freshnessCheckValue: String
        var freshnessCheckTitle = "Up to date"
        switch backups.freshness {
        case _ where backups.localSnapshot?.uploadState == "nothing_pending":
            freshness = .init(text: missingLocalSnapshot, tone: .red)
            freshnessCheckValue = freshness.text
        case _ where staleOffsite:
            freshness = .init(text: staleText, tone: .red)
            freshnessCheckValue = staleText
        case _ where staleTranscript:
            freshness = .init(text: "Backup freshness (DB + transcript): stale (> \(threshold) h)", tone: .red)
            freshnessCheckValue = plainThreshold.map { "A copy is older than \($0)" } ?? "A copy is out of date"
        case "fresh" where idle:
            freshness = .init(text: "Checked \(relative(backups.transcriptAttempt!.at, now: now, locale: locale)); nothing new", tone: .neutral)
            freshnessCheckTitle = "Transcript coverage"
            freshnessCheckValue = freshness.text
        case "fresh" where backups.localSnapshot != nil:
            freshness = .init(text: "Transcript backup is current", tone: .green)
            freshnessCheckValue = freshness.text
        case "fresh":
            freshness = .init(text: "Backup freshness (DB + transcript): fresh (within \(threshold) h)", tone: .green)
            freshnessCheckValue = plainThreshold.map { "Both copies are under \($0) old" } ?? "Both copies are recent"
        case "stale":
            freshness = .init(text: "Backup freshness (DB + transcript): stale (> \(threshold) h)", tone: .red)
            freshnessCheckValue = plainThreshold.map { "A copy is older than \($0)" } ?? "A copy is out of date"
        default:
            freshness = .init(text: "Backup freshness (DB + transcript): unknown", tone: .red)
            freshnessCheckValue = "Unknown"
        }

        let retentionValue = backups.retentionInvariant ?? "unknown"
        let retention = ObservabilityStatusLine(
            text: "Transcript retention invariant: \(retentionValue)",
            tone: retentionValue == "PASS" ? .green : .red
        )
        let archives: ObservabilityStatusLine
        if let archiveCount = backups.survivingArchives30D {
            archives = .init(
                text: "\(number(archiveCount, locale: locale)) verified transcript \(archiveCount == 1 ? "archive" : "archives") in the last 30 days",
                tone: idle ? .neutral : archiveCount > 0 ? .green : .red
            )
        } else {
            archives = .init(text: "Verified transcript archives in the last 30 days: unknown", tone: .red)
        }
        let error = backups.errorType.flatMap { value -> ObservabilityStatusLine? in
            guard !value.isEmpty else { return nil }
            if value == "drive_credentials_restored_backup_pending" {
                return .init(text: "Google Drive credentials restored — next backup pending", tone: .neutral)
            }
            // A job alert (#1031) carries a sentence written for Etan: show it as written.
            if value.hasPrefix(jobAlertPrefix) {
                let reason = DriveAuthJSON.sanitized(String(value.dropFirst(jobAlertPrefix.count)))
                return .init(text: reason ?? "Backup alert with no reason given", tone: .red)
            }
            // The DB copy's own verdict (#1420) is a sentence, not "an error the backup reported".
            if let sentence = dbSnapshotSentence(value, thresholdHours: backups.thresholdHours) {
                return .init(text: sentence, tone: .red)
            }
            let kind = value.hasPrefix("jsonl_backup_attempt_") ? "Transcript" : "DB"
            return .init(text: "\(kind) backup error: \(errorText(value))", tone: .red)
        }
        return .init(
            upload: upload, snapshot: snapshot, job: job, freshness: freshness,
            retention: retention, archives: archives, error: error,
            errorIsJobAlert: backups.errorType?.hasPrefix(jobAlertPrefix) == true, snapshotUpload: snapshotUpload, uploadInProgress: backups.localUploadPending(now: now) && backups.localSnapshot?.uploadState == "pending",
            uploadQueued: backups.localSnapshot?.uploadState == "pending",
            uploadUnconfirmed: backups.localUploadPending(now: now) && backups.localSnapshot?.uploadState == "in_progress",
            freshnessCheckTitle: freshnessCheckTitle, freshnessCheckValue: freshnessCheckValue
        )
    }

    /// Verified receipt dates shared by Dashboard, Backups, and menu consumers. `lastAt` is
    /// snapshot creation; upload `at` is the successful attempt's timestamp, never log mtime.
    static func backupReceiptLines(
        for backups: ObservabilityDocument.Backups,
        locale: Locale = .current,
        now: Date,
        calendar: Calendar = .current
    ) -> [ObservabilityStatusLine] {
        let status = backupStatus(for: backups, locale: locale, now: now, calendar: calendar)
        func moment(_ date: Date, prefix: String = "last good ") -> String {
            let formatter = DateFormatter()
            formatter.calendar = calendar
            formatter.locale = locale
            formatter.timeZone = calendar.timeZone
            let today = calendar.isDate(date, inSameDayAs: now)
            let sameYear = calendar.isDate(date, equalTo: now, toGranularity: .year)
            formatter.dateFormat = today ? "HH:mm" : sameYear ? "MMM d 'at' HH:mm" : "MMM d, yyyy 'at' HH:mm"
            let label = date > now ? "invalid timestamp " : prefix
            return label + (today ? "today " : "") + formatter.string(from: date)
        }
        let database = ObservabilityStatusLine(
            text: "Database · Drive · \(backups.dbSnapshot.flatMap { $0.verified ? moment($0.lastAt) : nil } ?? "no verified copy")",
            tone: backups.dbSnapshot.map { $0.lastAt > now } == true ? .red : status.snapshot.tone)
        let transcript = ObservabilityStatusLine(
            text: backups.verifiedIdle(now: now)
                ? "Transcripts · Drive · no changes\(backups.lastVerifiedUpload.map { " since " + moment($0.at, prefix: "") } ?? "")"
                : "Transcripts · Drive · \(backups.lastVerifiedUpload.flatMap { $0.verified ? moment($0.at) : nil } ?? "no verified copy")",
            tone: backups.lastVerifiedUpload.map { $0.at > now } == true ? .red : status.upload.tone)
        return [database] + [status.snapshotUpload].compactMap { $0 } + [transcript]
    }

    static func relative(_ date: Date, now: Date, locale: Locale = .current) -> String {
        guard now.timeIntervalSince(date) >= 60 else { return "just now" }
        let formatter = RelativeDateTimeFormatter()
        formatter.locale = locale
        formatter.unitsStyle = .full
        return formatter.localizedString(for: date, relativeTo: now)
    }

    private static let missingLocalSnapshot = "No local snapshot found to verify. Run backup-daily or check the snapshot job."

    private static func idleDate(_ date: Date, calendar: Calendar, locale: Locale) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = locale
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "MMM d"
        return formatter.string(from: date)
    }

    private static func localDate(_ date: Date) -> String {
        date.formatted(date: .abbreviated, time: .shortened)
    }

    private static func hours(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.maximumFractionDigits = value.rounded() == value ? 0 : 1
        formatter.locale = .current
        return "\(formatter.string(from: NSNumber(value: value)) ?? String(value)) h"
    }

    private static let jobAlertPrefix = "job_alert:"

    /// `observability_backup._db_snapshot_freshness` codes, in plain words; nil for any other value.
    static func dbSnapshotSentence(_ errorType: String, thresholdHours: Double?) -> String? {
        let code = errorType.split(separator: ":", maxSplits: 1).first.map(String.init) ?? errorType
        switch code {
        case "db_snapshot_stale":
            let window = thresholdHours.map { "\(Int($0)) hours" } ?? "its freshness window"
            return "The database copy in Google Drive is older than \(window)"
        case "db_snapshot_missing": return "No verified database copy in Google Drive"
        case "db_snapshot_unverified": return "The newest database copy in Google Drive failed verification"
        case "db_snapshot_clock_skew": return "The newest database copy is dated in the future; check this Mac's clock"
        default: return nil
        }
    }

    private static func errorText(_ value: String) -> String {
        switch value {
        case "drive_credentials_missing": "Google Drive credentials missing — re-auth needed"
        case "FileNotFoundError": "Backup input file missing"
        default: value.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}

struct ObservabilityDashboardView: View {
    let result: ObservabilityReadResult
    var now = Date()
    var cadence = ObservabilityCadence.known(300)

    var body: some View {
        switch result {
        case let .readable(document):
            let snapshot = ObservabilityPresentation.snapshot(document: document, now: now, cadence: cadence)
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Memory & backups").font(.title2.bold())
                        Text("The human-readable totals behind this dashboard")
                            .font(.caption)
                            .foregroundStyle(Color.secondary)
                    }
                    Spacer()
                    Text("Generated \(snapshot.generatedAt.formatted()) · \(snapshot.ageText)")
                        .foregroundStyle(snapshot.isStale ? Color.orange : Color.secondary)
                }
                LazyVGrid(columns: [.init(.flexible()), .init(.flexible())], spacing: 14) {
                    ForEach(Array(snapshot.cards.enumerated()), id: \.offset) { _, card in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(card.title).font(.headline)
                            if let subtitle = card.subtitle {
                                Text(subtitle)
                                    .font(.caption)
                                    .foregroundStyle(Color.secondary)
                            }
                            if card.statusLines.isEmpty {
                                Text(card.detail)
                                    .foregroundStyle(card.tone == .neutral ? Color.secondary : Color.primary)
                            } else {
                                ObservabilityStatusRows(lines: card.statusLines)
                            }
                            if let note = card.note {
                                Text(note).font(.caption).foregroundStyle(Color.secondary)
                            }
                        }
                        .frame(maxWidth: .infinity, minHeight: 132, alignment: .topLeading)
                        .padding(16).background(fill(card.tone), in: RoundedRectangle(cornerRadius: 14))
                        .accessibilityIdentifier("card.\(identifier(card.title)).tone=\(card.tone.rawValue)")
                    }
                }
            }
            .accessibilityIdentifier("brainbar.dashboard.observability")
        case let .unreadable(reason):
            ContentUnavailableView("Observability unreadable", systemImage: "questionmark.circle", description: Text(reason))
        }
    }

    private func fill(_ tone: ObservabilityCardTone) -> Color {
        switch tone {
        case .standard: Color.blue.opacity(0.16)
        case .neutral: Color.gray.opacity(0.16)
        case .amber: Color.orange.opacity(0.20)
        }
    }

    private func identifier(_ title: String) -> String {
        title.lowercased().replacingOccurrences(of: "-", with: "_")
    }
}

struct ObservabilityTechnicalDetailsView: View {
    let result: ObservabilityReadResult

    var body: some View {
        switch result {
        case let .readable(document):
            let window = "\(document.windowHours) h"
            VStack(alignment: .leading, spacing: 12) {
                Text("Observability")
                    .font(.headline)
                detail("Document", "schema v\(document.schemaVersion) · \(document.windowHours) h window")
                detail("Generated", document.generatedAt.formatted())
                detail("Database", document.dbPath)

                Divider()
                Text("Memory provenance")
                    .font(.subheadline.bold())
                if document.emitters.state == "measured" {
                    if let mcp = document.emitters.byEmitter?.first(where: { $0.emitter == "mcp" }) {
                        detail("Agent writes · \(window)", "\(mcp.countInWindow) via brain_store")
                    }
                    ForEach(Array((document.emitters.byEmitter ?? []).enumerated()), id: \.offset) { _, row in
                        if row.emitter != "mcp" {
                            detail("\(row.emitter) · \(window)", "\(row.countInWindow) indexed chunks")
                        }
                    }
                    ForEach(Array((document.emitters.bySourceClass ?? []).enumerated()), id: \.offset) { _, row in
                        detail("\(row.sourceClass ?? "unclassified") · \(window)", "\(row.inWindow) indexed chunks")
                    }
                } else {
                    let reason = document.emitters.reason.isEmpty ? "Emitter provenance unavailable" : document.emitters.reason
                    detail("Emitters", reason)
                }
                if document.authorUnknown.state == "measured" {
                    if let unknown = document.authorUnknown.neverClassified {
                        detail("Unattributed · total", "\(unknown.count) indexed chunks · \(percent(unknown.share)) of indexed chunks")
                    }
                } else {
                    let reason = document.authorUnknown.reason.isEmpty ? "Attribution unavailable" : document.authorUnknown.reason
                    detail("Attribution", reason)
                }

                Divider()
                Text("Backup internals")
                    .font(.subheadline.bold())
                if let launchd = document.backups.launchd {
                    detail("Launch agent", "\(launchd.label) · \(launchd.bootstrapped ? "loaded" : "not loaded")")
                }
                if let archiveID = document.backups.lastVerifiedUpload?.archiveId {
                    detail("Drive file ID", archiveID)
                }
                if let invariant = document.backups.retentionInvariant {
                    detail("Retention invariant", invariant)
                }
                if let archives = document.backups.survivingArchives30D {
                    detail("Retention", "\(archives) surviving transcript archives in 30 d")
                }
                if let error = document.backups.errorType {
                    detail("Backup error", error.replacingOccurrences(of: "_", with: " "))
                }
                ForEach(Array(document.backups.inputs.enumerated()), id: \.offset) { _, input in
                    detail("Backup input", "\(input.status) · \(input.path)")
                }
            }
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(Color.brainBarTextSecondary)
            .accessibilityIdentifier("brainbar.dashboard.observability-details")
        case let .unreadable(reason):
            Text("Observability unreadable: \(reason)")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.orange)
        }
    }

    private func detail(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(label)
                .foregroundStyle(Color.brainBarTextPrimary)
                .frame(width: 112, alignment: .leading)
            Text(value)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
    }

    private func percent(_ share: Double) -> String {
        share.formatted(.percent.precision(.fractionLength(1)))
    }
}

struct ObservabilityLiveView: View {
    enum Reader {
        typealias Operation = @Sendable (URL) async -> ObservabilityReadResult

        static func read(
            url: URL,
            using operation: @escaping Operation = { ObservabilityReader.readReconciled(url: $0) }
        ) async -> ObservabilityReadResult {
            return await operation(url)
        }

        static func watch(
            url: URL,
            every interval: Duration = .seconds(30),
            using operation: @escaping Reader.Operation = { ObservabilityReader.readReconciled(url: $0) }
        ) -> AsyncStream<ObservabilityReadResult> {
            AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
                let producer = Task {
                    defer { continuation.finish() }
                    while !Task.isCancelled {
                        let next = await Reader.read(url: url, using: operation)
                        guard !Task.isCancelled else { return }
                        continuation.yield(next)
                        do {
                            try await Task.sleep(for: interval)
                        } catch {
                            return
                        }
                    }
                }

                continuation.onTermination = { @Sendable _ in
                    producer.cancel()
                }
            }
        }
    }

    let dbPath: String
    private let cadence = ObservabilityReader.installedHealthCheckCadence
    @State private var result: ObservabilityReadResult = .unreadable("Loading observability data.")

    var body: some View {
        ObservabilityDashboardView(result: result, cadence: cadence)
            .task(id: dbPath) {
                let url = ObservabilityReader.url(dbPath: dbPath)
                for await next in Reader.watch(url: url) {
                    result = next
                }
            }
    }
}

struct BrainBarVectorScan: Decodable, Equatable, Sendable {
    let completeness: String
    let scanned: Int
    let missing: Int
    let reason: String

    var routinePartial: Bool {
        // Keep budget reasons aligned with routine_partial in src/brainlayer/health_check.py.
        completeness == "partial" && scanned > 0 && missing >= 0 && missing <= scanned
            && ["instruction_limit", "time_limit", "page_limit", "window_end"].contains(reason)
    }

    var detail: String {
        if completeness == "complete" {
            return "Background vector check complete. \(missing.formatted()) \(missing == 1 ? "chunk is" : "chunks are") waiting for vectors."
        }
        guard scanned > 0 else {
            return "The vector check could not finish. The current backlog is not verified by this check."
        }
        return "Background vector check in progress. Checked \(scanned.formatted()) \(scanned == 1 ? "chunk" : "chunks"); \(missing == 0 ? "none" : missing.formatted()) in this sample \(missing == 1 ? "is" : "are") waiting for vectors."
    }

    static func read(beside badgeURL: URL) -> Self? {
        struct State: Decodable { let missing_embedding_scan: BrainBarVectorScan }
        let url = badgeURL.deletingLastPathComponent().appendingPathComponent("health-check-state.json")
        guard let data = try? Data(contentsOf: url),
              let scan = try? JSONDecoder().decode(State.self, from: data).missing_embedding_scan,
              scan.scanned >= 0, scan.missing >= 0, scan.missing <= scan.scanned else { return nil }
        return scan
    }
}

struct BrainBarVectorCheck: Equatable, Sendable {
    static let unverified = "Vector backlog could not be verified. The background check is still in progress."
    let count: Int?
    let checkedAt: Date?
    let error: String?

    var isAwaitingFirstResult: Bool {
        count == nil && checkedAt == nil && error == nil
    }

    func isFresh(now: Date) -> Bool {
        guard count != nil, let checkedAt, error == nil else { return false }
        return (0...600).contains(now.timeIntervalSince(checkedAt))
    }

    func summary(now: Date) -> String {
        if isAwaitingFirstResult { return "Checking vector backlog…" }
        guard let count, let checkedAt else { return Self.unverified }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "HH:mm 'UTC'"
        let time = formatter.string(from: checkedAt)
        return isFresh(now: now)
            ? "\(count.formatted()) \(count == 1 ? "chunk" : "chunks") waiting for vectors · checked \(time)"
            : "Last measured backlog: \(count.formatted()) · checked \(time)"
    }
}
