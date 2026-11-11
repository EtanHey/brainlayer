import Foundation

/// The shared status verdict as the surfaces compiled into both BrainBar targets see it (S5).
/// `BrainBarStatusSnapshot` (app target) derives it; the menu, its icon and the Dashboard render it.
struct BrainBarStatusVerdict: Equatable, Sendable {
    /// Dot-worthy faults, most severe first. Non-empty means the menu-bar dot.
    let attention: [String]
    /// Visible neutral notices; never a dot.
    let notices: [String]
    /// Historical rows remain on the Dashboard/menu but do not count as current work.
    var oldNotices: [String] = []
    /// Nothing has been measured yet.
    let isChecking: Bool

    var dotOverride: BrainBarRenderedDot? = nil
    var healthVerdictIsStale = false
    /// Active run failures are independent of the age of the last verified copy.
    var backupRunFailures: [BrainBarBackupRunFailure] = []
    var dot: BrainBarRenderedDot { dotOverride ?? (needsAttention ? .red : .green) }

    var needsAttention: Bool { !attention.isEmpty }

    /// The Dashboard headline: a count only for dot issues; notices are neutral; "All good" only
    /// when everything was measured and nothing is pending.
    var dashboardStatus: BrainBarOnePageStatus {
        if let dotOverride {
            let headline = overrideHeadline(dotOverride)
            let details = attention + notices + oldNotices
            return .init(headline: headline, reason: details.isEmpty ? nil
                : details.joined(separator: "; "), tone: dotOverride == .green ? .green : .amber)
        }
        if let first = attention.first {
            return .init(headline: attention.count == 1 ? "1 thing needs you" : "\(attention.count) things need you",
                         reason: ([first] + notices).joined(separator: "; "), tone: .amber)
        }
        if !notices.isEmpty || !oldNotices.isEmpty {
            return .init(headline: "Status notice", reason: (notices + oldNotices).joined(separator: "; "), tone: .neutral)
        }
        if isChecking { return .init(headline: "Checking…", reason: nil, tone: .neutral) }
        return .init(headline: "All good", reason: nil, tone: .green)
    }

    /// Every distinct current row; historical rows never count (compact headline, 2026-11-08).
    var currentRowCount: Int { Set(attention + notices).count }

    /// The menu and Dashboard headline under a verdict dot: a true count of the rows below it.
    /// Without any row it falls back to the dot's own words.
    func overrideHeadline(_ dot: BrainBarRenderedDot) -> String {
        let count = currentRowCount
        switch dot {
        case .green: return safeHeadline
        case .red, .amber:
            guard count > 0 else { return dot == .red ? "Needs attention" : "Check soon" }
            return count == 1 ? "1 thing needs you" : "\(count) things need you"
        }
    }

    var safeHeadline: String {
        let count = Set(attention + notices).count
        return count == 0 ? "All good" : "Memory safe · \(count) \(count == 1 ? "thing" : "things") to look at"
    }

    var tooltip: String {
        if needsAttention { return "BrainBar — needs attention: " + attention.joined(separator: "; ") }
        return dot == .amber ? "BrainBar — notice: " + notices.joined(separator: "; ") : "BrainBar"
    }
}

struct BrainBarBackupRunFailure: Equatable, Sendable {
    enum Target: String, Sendable { case database, transcripts }
    let target: Target
    let text: String
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
            "backup_local": "Database snapshot (verified)", "backup_offsite": "Offsite (Drive) database backup",
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
    func isFresh(now: Date) -> Bool { (0...900).contains(BrainBarSleepHistory.shared.elapsed(since: evaluatedAt, now: now)) }
    var staleRenderedDot: BrainBarRenderedDot { dot == .red ? .red : .amber }
    func fallbackNotice(now: Date, calendar: Calendar = .current) -> String {
        let age = now.timeIntervalSince(evaluatedAt)
        if age < 0 { return "Health check result is dated in the future" }
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm"
        return "Health check stale since \(formatter.string(from: evaluatedAt))"
    }
}

struct BrainBarUIStatusReceipt: Codable, Equatable, Sendable {
    let renderedAt: Date
    let dot: BrainBarRenderedDot
    let source: String
    let verdictEvaluatedAt: Date?
    let reasons: [String]
    let appVersion: String
    let gitCommit: String

    func hasSameContent(as other: Self) -> Bool {
        dot == other.dot && source == other.source && verdictEvaluatedAt == other.verdictEvaluatedAt
            && reasons == other.reasons && appVersion == other.appVersion && gitCommit == other.gitCommit
    }

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

/// All publication state and file I/O belong to this serial utility queue.
final class BrainBarUIStatusPublisher: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.ui-status", qos: .utility)
    private var lastReceipt: BrainBarUIStatusReceipt?
    private var lastURL: URL?
    private let writer: @Sendable (BrainBarUIStatusReceipt, URL) throws -> Void

    init(writer: @escaping @Sendable (BrainBarUIStatusReceipt, URL) throws -> Void = { try $0.write(to: $1) }) {
        self.writer = writer
    }

    func publish(_ receipt: BrainBarUIStatusReceipt, to url: URL) {
        queue.async { [self] in
            if let previous = lastReceipt, lastURL == url,
               previous.hasSameContent(as: receipt),
               (0..<300).contains(receipt.renderedAt.timeIntervalSince(previous.renderedAt)) { return }
            do {
                try writer(receipt, url)
                lastReceipt = receipt
                lastURL = url
            } catch {
                NSLog("BrainBar UI status receipt write failed: %@", error.localizedDescription)
            }
        }
    }

    /// Drain on shutdown so the final receipt cannot outlive its runtime directory.
    func flush() { queue.sync {} }
}

/// Deadlines use mach uptime through DispatchTime, which stops during system sleep.
struct BrainBarWakeCatchUp {
    private var deadline: Double? = nil
    mutating func sleeping() { deadline = .infinity }
    mutating func woke(uptime: Double, grace: Double = 180) { deadline = uptime + grace }
    func isPending(uptime: Double) -> Bool { deadline.map { uptime < $0 } ?? false }
}

/// Wall timestamps remain traceable; only the age used for liveness excludes known suspension.
final class BrainBarSleepHistory: @unchecked Sendable {
    static let shared = BrainBarSleepHistory()
    private let lock = NSLock()
    private var intervals: [(Date, Date)] = []
    private var sleepingAt: Date?

    func sleeping(at date: Date) {
        lock.lock(); defer { lock.unlock() }
        if sleepingAt == nil { sleepingAt = date }
    }

    func woke(at date: Date) {
        lock.lock(); defer { lock.unlock() }
        if let start = sleepingAt, date >= start { intervals.append((start, date)) }
        sleepingAt = nil
    }

    func elapsed(since date: Date, now: Date) -> TimeInterval {
        lock.lock(); defer { lock.unlock() }
        let ranges = intervals + (sleepingAt.map { [($0, now)] } ?? [])
        let suspended = ranges.reduce(0.0) { total, range in
            total + max(0, min(now, range.1).timeIntervalSince(max(date, range.0)))
        }
        return now.timeIntervalSince(date) - suspended
    }
}
