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
    struct SourceClass: Codable, Sendable {
        let sourceClass: String?
        let count: Int
        let inWindow: Int
    }
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
        let retentionInvariant: String?
        let survivingArchives30D: Int?
        let errorType: String?
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

struct ObservabilitySnapshot: Sendable {
    struct Card: Sendable {
        let title: String, subtitle: String, detail: String
        let note: String?
        let tone: ObservabilityCardTone
    }
    let generatedAt: Date, ageText: String, isStale: Bool, cards: [Card]
}

enum ObservabilityPresentation {
    static func snapshot(
        document: ObservabilityDocument,
        now: Date,
        cadence: ObservabilityCadence
    ) -> ObservabilitySnapshot {
        let age = max(0, now.timeIntervalSince(document.generatedAt))
        let stale = age > cadence.interval * 2
        let stores = card(
            "Stores", "Is memory still growing?",
            document.stores.state, document.stores.reason, stale, note: cadence.assumption
        ) {
            guard let total = document.stores.totalChunks, let recent = document.stores.inWindow?.count else { return nil }
            return "\(number(total)) chunks total (all sources, incl. archived)\n" +
                "\(number(recent)) added in the last 24 h"
        }
        let emitters = card(
            "Emitters", "Where is memory coming from?",
            document.emitters.state, document.emitters.reason, stale
        ) {
            guard let classes = document.emitters.bySourceClass,
                  let emitters = document.emitters.byEmitter,
                  let hidden = document.emitters.hiddenFromDefaultSearch else { return nil }
            func classCount(_ name: String) -> Int {
                classes.first { $0.sourceClass == name }?.count ?? 0
            }
            let mcpStores = emitters.first { $0.emitter == "mcp" }?.countInWindow ?? 0
            return [
                "\(number(classCount("cli-agent"))) from CLI agents",
                "\(number(mcpStores)) from MCP brain_store in the last 24 h",
                "\(number(classCount("subagent"))) from subagents",
                "\(number(classCount("desktop"))) desktop (hidden from search by default)",
                "\(number(hidden)) hidden from search by default",
            ].joined(separator: "\n")
        }
        let authors = card(
            "Author-unknown", "Can person filters trust the provenance?",
            document.authorUnknown.state, document.authorUnknown.reason, stale
        ) {
            guard let never = document.authorUnknown.neverClassified,
                  let classified = document.authorUnknown.classifiedUnknown else { return nil }
            return "\(number(never.count)) chunks never classified (no source/provenance yet)\n" +
                "\(number(classified.count)) chunks classified as unknown author\n" +
                "Why it matters: these never match a person filter"
        }
        let backupTone: ObservabilityCardTone = document.backups.freshness == "unknown" ||
            document.backups.retentionInvariant == "unknown" ? .neutral :
            (document.backups.freshness == "stale" || document.backups.retentionInvariant == "FAIL" ? .amber : .standard)
        let backups = card(
            "Backups", "Are recoverable copies current and running?",
            document.backups.state, document.backups.reason, stale, tone: backupTone
        ) {
            guard let freshness = document.backups.freshness,
                  let retention = document.backups.retentionInvariant else { return nil }
            var parts = [freshness, "retention \(retention)"]
            if let archives = document.backups.survivingArchives30D {
                parts.append("\(number(archives)) verified \(archives == 1 ? "archive" : "archives") in the last 30 days")
            }
            if let errorType = document.backups.errorType, !errorType.isEmpty {
                parts.append(errorType)
            }
            return parts.joined(separator: " · ")
        }
        return ObservabilitySnapshot(
            generatedAt: document.generatedAt,
            ageText: age < 60 ? "just now" : "\(Int(age / 60))m old",
            isStale: stale,
            cards: [stores, emitters, authors, backups]
        )
    }

    private static func card(
        _ title: String, _ subtitle: String, _ state: String, _ reason: String, _ stale: Bool,
        tone: ObservabilityCardTone = .standard,
        note: String? = nil,
        measured: () -> String?
    ) -> ObservabilitySnapshot.Card {
        guard state == "measured", let detail = measured() else {
            let detail = reason.isEmpty ? "unmeasurable" : "unmeasurable — \(reason)"
            return .init(title: title, subtitle: subtitle, detail: detail, note: note, tone: .neutral)
        }
        return .init(title: title, subtitle: subtitle, detail: detail, note: note, tone: stale ? .amber : tone)
    }

    static func number(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = .current
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
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
                    Text("Observability").font(.title2.bold())
                    Spacer()
                    Text("Generated \(snapshot.generatedAt.formatted()) · \(snapshot.ageText)")
                        .foregroundStyle(snapshot.isStale ? Color.orange : Color.secondary)
                }
                LazyVGrid(columns: [.init(.flexible()), .init(.flexible())], spacing: 14) {
                    ForEach(Array(snapshot.cards.enumerated()), id: \.offset) { _, card in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(card.title).font(.headline)
                            Text(card.subtitle)
                                .font(.caption)
                                .foregroundStyle(Color.secondary)
                            Text(card.detail)
                                .foregroundStyle(card.tone == .neutral ? Color.secondary : Color.primary)
                            if let note = card.note {
                                Text(note).font(.caption).foregroundStyle(Color.secondary)
                            }
                        }
                        .frame(maxWidth: .infinity, minHeight: 110, alignment: .topLeading)
                        .padding(16).background(fill(card.tone), in: RoundedRectangle(cornerRadius: 14))
                        .accessibilityIdentifier("card.\(identifier(card.title)).tone=\(card.tone.rawValue)")
                    }
                }
                Spacer()
            }
            .padding(20)
            .background(Color(nsColor: .windowBackgroundColor))
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
