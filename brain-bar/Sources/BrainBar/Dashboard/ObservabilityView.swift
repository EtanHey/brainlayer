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
        let hiddenFromDefaultSearch: Int?
    }
    struct Emitter: Codable, Sendable { let emitter: String, countInWindow: Int }
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
        let survivingArchives30d: Int?
        let errorType: String?
    }
    struct Input: Codable, Sendable { let path: String, status: String }
}

enum ObservabilityReadResult: Sendable {
    case readable(ObservabilityDocument)
    case unreadable(String)
}

enum ObservabilityReader {
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
    ) -> TimeInterval {
        let path = environment["BRAINLAYER_HEALTH_CHECK_PLIST_PATH"] ??
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/LaunchAgents/com.brainlayer.health-check.plist").path
        guard let data = FileManager.default.contents(atPath: path),
              let plist = try? PropertyListSerialization.propertyList(from: data, format: nil),
              let interval = (plist as? [String: Any])?["StartInterval"] as? NSNumber else {
            return 300
        }
        return interval.doubleValue
    }

    private struct SchemaEnvelope: Decodable {
        let schemaVersion: Int
        enum CodingKeys: String, CodingKey { case schemaVersion = "schema_version" }
    }
}

enum ObservabilityCardTone: Equatable, Sendable { case standard, neutral, amber }

struct ObservabilitySnapshot: Sendable {
    struct Card: Sendable {
        let title: String, detail: String
        let tone: ObservabilityCardTone
    }
    let generatedAt: Date, ageText: String, isStale: Bool, cards: [Card]
}

enum ObservabilityPresentation {
    static func snapshot(
        document: ObservabilityDocument,
        now: Date,
        cadence: TimeInterval
    ) -> ObservabilitySnapshot {
        let age = max(0, now.timeIntervalSince(document.generatedAt))
        let stale = age > cadence * 2
        let stores = card("Stores", document.stores.state, document.stores.reason, stale) {
            guard let total = document.stores.totalChunks, let recent = document.stores.inWindow?.count else { return nil }
            return "\(total) total · \(recent) in 24h"
        }
        let emitters = card("Emitters", document.emitters.state, document.emitters.reason, stale) {
            guard let sources = document.emitters.byEmitter, let hidden = document.emitters.hiddenFromDefaultSearch else { return nil }
            let top = sources.first.map { "\($0.emitter) \($0.countInWindow)" } ?? "none in 24h"
            return "\(top) · \(hidden) hidden"
        }
        let authors = card("Author-unknown", document.authorUnknown.state, document.authorUnknown.reason, stale) {
            guard let never = document.authorUnknown.neverClassified,
                  let classified = document.authorUnknown.classifiedUnknown else { return nil }
            return "\(never.count) never classified · \(classified.count) unknown"
        }
        let backupTone: ObservabilityCardTone = document.backups.freshness == "unknown" ||
            document.backups.retentionInvariant == "unknown" ? .neutral :
            (document.backups.freshness == "stale" || document.backups.retentionInvariant == "FAIL" ? .amber : .standard)
        let backups = card("Backups", document.backups.state, document.backups.reason, stale, tone: backupTone) {
            guard let freshness = document.backups.freshness,
                  let retention = document.backups.retentionInvariant else { return nil }
            return "\(freshness) · retention \(retention)"
        }
        return ObservabilitySnapshot(
            generatedAt: document.generatedAt,
            ageText: age < 60 ? "just now" : "\(Int(age / 60))m old",
            isStale: stale,
            cards: [stores, emitters, authors, backups]
        )
    }

    private static func card(
        _ title: String, _ state: String, _ reason: String, _ stale: Bool,
        tone: ObservabilityCardTone = .standard,
        measured: () -> String?
    ) -> ObservabilitySnapshot.Card {
        guard state == "measured", let detail = measured() else {
            let detail = reason.isEmpty ? "unmeasurable" : "unmeasurable — \(reason)"
            return .init(title: title, detail: detail, tone: .neutral)
        }
        return .init(title: title, detail: detail, tone: stale ? .amber : tone)
    }
}

struct ObservabilityDashboardView: View {
    let result: ObservabilityReadResult
    var now = Date()
    var cadence = ObservabilityReader.healthCheckCadence()

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
                            Text(card.detail)
                                .foregroundStyle(card.tone == .neutral ? Color.secondary : Color.primary)
                        }
                        .frame(maxWidth: .infinity, minHeight: 110, alignment: .topLeading)
                        .padding(16).background(fill(card.tone), in: RoundedRectangle(cornerRadius: 14))
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
}

struct ObservabilityLiveView: View {
    let dbPath: String
    @State private var result: ObservabilityReadResult = .unreadable("Loading observability data.")
    private let refresh = Timer.publish(every: 30, on: .main, in: .common).autoconnect()

    var body: some View {
        ObservabilityDashboardView(result: result)
            .onAppear(perform: reload)
            .onReceive(refresh) { _ in reload() }
    }

    private func reload() {
        result = ObservabilityReader.read(url: ObservabilityReader.url(dbPath: dbPath))
    }
}
