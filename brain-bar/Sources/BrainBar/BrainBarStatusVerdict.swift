import Foundation

/// The shared status verdict as the surfaces compiled into both BrainBar targets see it (S5).
/// `BrainBarStatusSnapshot` (app target) derives it; the menu, its icon and the Dashboard render it.
struct BrainBarStatusVerdict: Equatable, Sendable {
    /// Dot-worthy faults, most severe first. Non-empty means the menu-bar dot.
    let attention: [String]
    /// Visible neutral notices; never a dot.
    let notices: [String]
    /// Nothing has been measured yet.
    let isChecking: Bool

    var dotOverride: BrainBarRenderedDot? = nil
    var dot: BrainBarRenderedDot { dotOverride ?? (needsAttention ? .red : .green) }

    var needsAttention: Bool { !attention.isEmpty }

    /// The Dashboard headline: a count only for dot issues; notices are neutral; "All good" only
    /// when everything was measured and nothing is pending.
    var dashboardStatus: BrainBarOnePageStatus {
        if let first = attention.first {
            return .init(headline: attention.count == 1 ? "1 thing needs you" : "\(attention.count) things need you",
                         reason: ([first] + notices).joined(separator: "; "), tone: .amber)
        }
        if !notices.isEmpty { return .init(headline: "Status notice", reason: notices.joined(separator: "; "), tone: .neutral) }
        if isChecking { return .init(headline: "Checking…", reason: nil, tone: .neutral) }
        return .init(headline: "All good", reason: nil, tone: .green)
    }

    var tooltip: String {
        if needsAttention { return "BrainBar — needs attention: " + attention.joined(separator: "; ") }
        return dot == .amber ? "BrainBar — notice: " + notices.joined(separator: "; ") : "BrainBar"
    }
}

/// The color actually drawn; green means no overlay on the sparkline.
enum BrainBarRenderedDot: String, Codable, Sendable { case green, amber, red }

struct BrainBarHealthVerdict: Codable, Equatable, Sendable {
    enum State: String, Codable, Sendable { case green, amber, red, unknown }
    struct Check: Codable, Equatable, Sendable {
        let name: String
        let state: State
        var ageS: Double? = nil
        var thresholdS: Double? = nil
        var since: String? = nil
        let note: String
        var text: String { "\(Self.labels[name] ?? name): \(note.isEmpty ? state.rawValue : note)" }
        static let labels = [
            "backup_local": "Local database backup", "backup_offsite": "Offsite (Drive) database backup",
            "transcripts": "Transcript backup", "drive_auth": "Drive authorization", "integrity": "Database integrity",
            "ingest": "Transcript ingest", "embedding": "Embedding", "stores": "Memory stores",
            "search": "Memory search", "disk": "Disk space for backups", "monitor": "Health monitor",
        ]
    }
    struct Note: Codable, Equatable, Sendable { let code, message: String }
    let dot: State
    let checks: [Check]
    let notes: [Note]
    let evaluatedAt: Date
    var renderedDot: BrainBarRenderedDot {
        switch dot { case .green: .green; case .amber, .unknown: .amber; case .red: .red }
    }
    func isFresh(now: Date) -> Bool { (0...900).contains(now.timeIntervalSince(evaluatedAt)) }
    func fallbackNotice(now: Date) -> String {
        let age = now.timeIntervalSince(evaluatedAt)
        return age < 0 ? "Health check result is dated in the future" : "Health check result is \(Int(age / 60)) min old"
    }
}

struct BrainBarUIStatusReceipt: Codable, Equatable {
    let renderedAt: Date
    let dot: BrainBarRenderedDot
    let source: String
    let verdictEvaluatedAt: Date?
    let reasons: [String]
    let appVersion: String
    let gitCommit: String

    static func url(dbPath: String, environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        BadgeStateReader.url(dbPath: dbPath, environment: environment).deletingLastPathComponent()
            .appendingPathComponent("brainbar-ui-status.json")
    }

    func write(to url: URL) throws {
        let iso = ISO8601DateFormatter()
        let document: [String: Any] = [
            "rendered_at": iso.string(from: renderedAt), "dot": dot.rawValue, "source": source,
            "verdict_evaluated_at": verdictEvaluatedAt.map { iso.string(from: $0) } as Any? ?? NSNull(),
            "reasons": reasons, "app_version": appVersion, "git_commit": gitCommit,
        ]
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys]).write(to: url, options: .atomic)
    }
}
