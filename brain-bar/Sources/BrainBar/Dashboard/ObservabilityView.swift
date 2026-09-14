import Combine
import Foundation
import SwiftUI

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
    struct Window: Codable, Sendable { let count: Int }
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
            guard envelope.schemaVersion == 1 else {
                return .unreadable("Unsupported schema_version \(envelope.schemaVersion); expected 1.")
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
                return .unreadable("Observability schema v1 root fields are invalid.")
            }
            return .readable(document)
        } catch {
            return .unreadable("Observability data unreadable: \(error.localizedDescription)")
        }
    }

    static func healthCheckCadence(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ObservabilityCadence {
        let path = environment["BRAINLAYER_HEALTH_CHECK_PLIST_PATH"] ??
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/LaunchAgents/com.brainlayer.health-check.plist").path
        guard let data = FileManager.default.contents(atPath: path),
              let plist = try? PropertyListSerialization.propertyList(from: data, format: nil),
              let interval = (plist as? [String: Any])?["StartInterval"] as? NSNumber else {
            return .init(interval: 300, assumption: "cadence unknown, assuming 300s")
        }
        return .known(interval.doubleValue)
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
                Circle().fill(color(line.tone)).frame(width: 7, height: 7)
                Text(line.text).foregroundStyle(textColor)
            }
        }
    }

    private func color(_ tone: ObservabilityStatusTone) -> Color {
        switch tone {
        case .green: .green
        case .red: .red
        case .neutral: .secondary
        }
    }
}

struct ObservabilityBackupStatus: Equatable, Sendable {
    let upload, snapshot, job, freshness: ObservabilityStatusLine
    let retention, archives: ObservabilityStatusLine
    let error: ObservabilityStatusLine?

    var lines: [ObservabilityStatusLine] {
        [upload, snapshot, job, freshness, retention, archives] + [error].compactMap { $0 }
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
        let backupStatus = backupStatus(for: document.backups, locale: locale)
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
        locale: Locale = .current
    ) -> ObservabilityBackupStatus {
        let upload = backups.lastVerifiedUpload.map { value in
            ObservabilityStatusLine(
                text: "\(value.verified ? "Last verified transcript upload" : "Last transcript upload (NOT verified)"): \(localDate(value.at)) (\(hours(value.ageHours)) ago) · transcript archive \(value.archiveId)",
                tone: value.verified ? .green : .red
            )
        } ?? .init(text: "No verified transcript upload on record", tone: .red)

        let snapshot = backups.dbSnapshot.map { value in
            ObservabilityStatusLine(
                text: "Latest DB snapshot (\(value.verified ? "verified" : "NOT verified")): \(localDate(value.lastAt)) → \(value.destination)",
                tone: value.verified ? .green : .red
            )
        } ?? .init(text: "No verified DB snapshot on record", tone: .red)

        let job: ObservabilityStatusLine
        let jobLabel = backups.launchd?.label ?? "com.brainlayer.jsonl-backup"
        if backups.launchd?.bootstrapped == true {
            job = .init(text: "Transcript backup (\(jobLabel)): loaded", tone: .green)
        } else if backups.launchd?.disabledDirPresent == true {
            job = .init(text: "Transcript backup (\(jobLabel)): NOT loaded — parked in .disabled-retention-P0", tone: .red)
        } else {
            job = .init(text: "Transcript backup (\(jobLabel)): NOT loaded", tone: .red)
        }

        let threshold = backups.thresholdHours.map { number(Int($0), locale: locale) } ?? "unknown"
        let freshness: ObservabilityStatusLine
        switch backups.freshness {
        case "fresh": freshness = .init(text: "Backup freshness (DB + transcript): fresh (within \(threshold) h)", tone: .green)
        case "stale": freshness = .init(text: "Backup freshness (DB + transcript): stale (> \(threshold) h)", tone: .red)
        default: freshness = .init(text: "Backup freshness (DB + transcript): unknown", tone: .red)
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
                tone: archiveCount > 0 ? .green : .red
            )
        } else {
            archives = .init(text: "Verified transcript archives in the last 30 days: unknown", tone: .red)
        }
        let error = backups.errorType.flatMap { value -> ObservabilityStatusLine? in
            guard !value.isEmpty else { return nil }
            if value == "drive_credentials_restored_backup_pending" {
                return .init(text: "Google Drive credentials restored — next backup pending", tone: .neutral)
            }
            let kind = value.hasPrefix("jsonl_backup_attempt_") ? "Transcript" : "DB"
            return .init(text: "\(kind) backup error: \(errorText(value))", tone: .red)
        }
        return .init(
            upload: upload, snapshot: snapshot, job: job, freshness: freshness,
            retention: retention, archives: archives, error: error
        )
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

struct ObservabilityLiveView: View {
    enum Reader {
        typealias Operation = @Sendable (URL) async -> ObservabilityReadResult

        static func read(
            url: URL,
            using operation: @escaping Operation = { ObservabilityReader.read(url: $0) }
        ) async -> ObservabilityReadResult {
            return await operation(url)
        }
    }

    @MainActor
    enum Loader {
        static func load(
            replacing previous: Task<Void, Never>?,
            url: URL,
            using operation: @escaping Reader.Operation = { ObservabilityReader.read(url: $0) },
            apply: @escaping @MainActor (ObservabilityReadResult) -> Void
        ) -> Task<Void, Never> {
            previous?.cancel()
            return Task { @MainActor in
                let next = await Reader.read(url: url, using: operation)
                guard !Task.isCancelled else { return }
                apply(next)
            }
        }
    }

    let dbPath: String
    private let cadence = ObservabilityReader.installedHealthCheckCadence
    @State private var result: ObservabilityReadResult = .unreadable("Loading observability data.")
    @State private var readTask: Task<Void, Never>?
    private let refresh = Timer.publish(every: 30, on: .main, in: .common).autoconnect()

    var body: some View {
        ObservabilityDashboardView(result: result, cadence: cadence)
            .onAppear(perform: reload)
            .onReceive(refresh) { _ in reload() }
            .onDisappear { readTask?.cancel() }
    }

    private func reload() {
        let url = ObservabilityReader.url(dbPath: dbPath)
        readTask = Loader.load(replacing: readTask, url: url) {
            result = $0
        }
    }
}
