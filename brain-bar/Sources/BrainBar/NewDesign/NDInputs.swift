import Foundation

/// Everything the new design draws for one moment, sampled from what the app already holds (L5). The icon, the
/// dropdown and the window all read one `NDInputs` through one `NDModel` (R1). Pure values only: fixtures and tests
/// build it directly; production builds it in `NDStore` from the runtime.
struct NDInputs: Sendable {
    var capture: BrainBarMenuCapture
    /// The three runtime warnings the presentation does not carry (a job alert between health checks, an unverified
    /// vector check, an unloaded watcher job), read from the shared status snapshot as the old menu read them.
    var extras: [NDExtra] = []
    /// Total memories; nil before the first read.
    var memories: Int? = nil
    /// Searchable by meaning: vector-indexed share of what can be searched (app's own read). Nil when not measured.
    var searchable: Double? = nil
    /// Waiting saves: queue depth and the oldest's time (app's own read). Nil depth when not measured.
    var savesWaiting: Int? = nil
    var oldestSaveQueuedAt: Date? = nil
    /// The watcher's heartbeat age, read by the app itself (R20: kept when the verdict is stale).
    var watcherHeartbeatAgeS: Double? = nil
    /// The dropdown waveform: the last hour in 24 buckets.
    var menuWave: NDWave? = nil
    /// The hero waveform, 60 buckets per range.
    var heroWaves: [NDRange: NDWave] = [:]
    /// Live sessions per agent and memories attributed to it in the last 24 h (field `saves`); nil members are unknown.
    var agents: NDAgents = .unknown
    /// The background jobs as launchd and their logs report them; nil when not sampled yet.
    var jobs: [NDJobInput]? = nil
    /// A week of backup attempts from the backup logs (L2).
    var backupHistory: NDBackupHistory = .unread
    /// The last nightly and weekly upkeep runs (drawing scale only, L3).
    var nightlyLastRun: NDRun = .unread
    var weeklyLastRun: NDRun = .unread
    /// Google Drive, for wording and the reconnect state only (F14: never a status source).
    var driveReconnecting = false
    /// The newest local copy, for the Backups page's "Newest copy" and Show in Finder.
    var newestLocalCopyAt: Date? = nil
    /// R1.5 change 8 (L45): the REAL number of compressed database copies found on this Mac (not the keep setting).
    var localCopiesKept: Int? = nil
    var canRevealLocalCopy = false
    /// A copy of the report failed: the copy row says so until it goes (R12).
    var copyReportFailed = false
}

/// R1.5 (L42, L45): what the backup does with the copies on this Mac, from `src/brainlayer/backup_daily.py`. It keeps the
/// newest `DEFAULT_LOCAL_COMPRESSED_KEEP` compressed copies and removes an older one only when Google Drive holds a checked
/// copy (`prune_local_compressed_snapshots`). A test reads the Python file and fails if the two disagree.
enum NDBackupFacts {
    static let keepNewest = 3
}

enum NDPage: String, CaseIterable, Sendable, Identifiable {
    case overview, backups, jobs
    var id: String { rawValue }
    var title: String {
        switch self {
        case .overview: "Overview"
        case .backups: "Backups"
        case .jobs: "Jobs"
        }
    }
}

enum NDRange: String, CaseIterable, Sendable, Identifiable {
    case hour, threeHours, day
    var id: String { rawValue }
    var minutes: Int {
        switch self {
        case .hour: 60
        case .threeHours: 180
        case .day: 1440
        }
    }
    var label: String {
        switch self {
        case .hour: "1 h"
        case .threeHours: "3 h"
        case .day: "24 h"
        }
    }
    /// The axis under the hero: oldest, middle, now.
    var axis: (String, String, String) {
        switch self {
        case .hour: ("60 min ago", "30 min ago", "now")
        case .threeHours: ("3 h ago", "90 min ago", "now")
        case .day: ("24 h ago", "12 h ago", "now")
        }
    }
}

/// Writes per bucket, oldest first. Saves by agents above the line, conversations read below (V1).
struct NDWave: Equatable, Sendable {
    var saves: [Int]
    var reads: [Int]
    /// False when the database could not be read: the waveform is unknown (dashed), never empty.
    var readable = true
    /// When the newest bucket ends; a later end by one bucket width is a shift (motion 5).
    var endsAt: Date? = nil
    /// The newest bucket's index on the wave's fixed bucket grid (`endsAt` / bucket width), set by `linked(after:)`.
    var sequence: Int? = nil
    /// Review B1: the bucket that left on the left edge when this sample is exactly one bucket later than the one it
    /// replaced. The moving layer keeps it, sliding out under the plot's clip, so no hole opens.
    var outgoing: Bucket? = nil

    struct Bucket: Equatable, Sendable {
        var saves: Int
        var reads: Int
    }

    var savesTotal: Int { saves.reduce(0, +) }
    var readsTotal: Int { reads.reduce(0, +) }
    var count: Int { max(saves.count, reads.count) }

    /// The bucket grid index of a wave ending at `end` with `count` buckets over `minutes`.
    static func sequence(endsAt end: Date?, minutes: Int, count: Int) -> Int? {
        guard let end, count > 0 else { return nil }
        let width = Double(minutes * 60) / Double(count)
        return Int((end.timeIntervalSinceReferenceDate / width + 1e-6).rounded(.down))
    }

    /// This sample as it follows `previous` (the one on screen): its grid index, and the bucket that slid out when it
    /// is exactly one bucket later with the same bucket count. Any other change (first sample, a gap, a range switch,
    /// an unreadable read) has no outgoing bucket and is drawn in place.
    func linked(after previous: NDWave?, minutes: Int) -> NDWave {
        var next = self
        next.outgoing = nil
        next.sequence = Self.sequence(endsAt: endsAt, minutes: minutes, count: count)
        guard readable, let seq = next.sequence, let previous, previous.readable, previous.count == count,
              (previous.sequence ?? Self.sequence(endsAt: previous.endsAt, minutes: minutes, count: count)) == seq - 1
        else { return next }
        next.outgoing = Bucket(saves: previous.saves.first ?? 0, reads: previous.reads.first ?? 0)
        return next
    }
}

struct NDAgents: Equatable, Sendable {
    struct Agent: Equatable, Sendable, Identifiable {
        var id: String { family.rawValue }
        var family: AgentFamily
        /// Live sessions now; nil when the monitor could not measure.
        var sessions: Int?
        /// Saves in the last 24 h; nil when the emitter counts are not readable.
        var saves: Int?
    }
    var agents: [Agent]

    static let unknown = NDAgents(agents: AgentFamily.allCases.map { Agent(family: $0, sessions: nil, saves: nil) })
}

struct NDJobInput: Equatable, Sendable, Identifiable {
    var id: String { job.rawValue }
    var job: BrainLayerLaunchdJob
    /// The config's wish; nil when the config could not be read.
    var enabled: Bool?
    var continuous: Bool
    var lastRun: Date?
    var nextRun: Date?
    var loaded: Bool?
    /// What the last run recorded (F02): a success dot needs `.ok`.
    var lastOutcome: NDRunOutcome = .unknown
}

/// A job's last run as its records show it.
enum NDRunOutcome: Equatable, Sendable { case ok, failed, running, unknown }

/// One finished run of a job, or none on record, or its log could not be read.
enum NDRun: Equatable, Sendable {
    case ran(Date)
    case never
    case unread
}

/// A runtime warning outside the presentation (COMMON-BRIEF § "What every direction must cover").
struct NDExtra: Equatable, Sendable, Identifiable {
    enum Level: Int, Comparable, Sendable {
        case notice, fault
        static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }
    }
    var id: String
    var level: Level
    /// The status snapshot's own words for it (what the old menu showed).
    var text: String
    var thing: BrainBarMenuPresenter.Thing?
    var jobs: Set<BrainLayerLaunchdJob> = []

    /// The snapshot's issues in the three families, plus a watcher job that is wanted but not loaded. Old issues never count.
    static func derive(snapshot: BrainBarStatusSnapshot, watchSetting: BrainLayerLaunchdJobSetting?) -> [NDExtra] {
        var out: [NDExtra] = []
        for issue in snapshot.issues where !issue.isOld {
            let level: Level = issue.severity == .attention ? .fault : .notice
            if issue.alertKey != nil, issue.component != .watcher {
                out.append(.init(id: issue.id, level: level, text: issue.text, thing: thing(for: issue), jobs: issue.jobs))
            } else if issue.id == "health.vector-coverage" {
                out.append(.init(id: issue.id, level: .notice, text: issue.text, thing: .finding))
            } else if issue.component == .watcher, !["watcher.alerts", "watcher.ingest"].contains(issue.id) {
                out.append(.init(id: issue.id, level: level, text: issue.text, thing: .remembering, jobs: [.watch]))
            }
        }
        if let setting = watchSetting, setting.enabled, setting.loadState == .unloaded,
           !out.contains(where: { $0.thing == .remembering && $0.jobs.contains(.watch) }) {
            out.append(.init(id: "nd.watch-unloaded", level: .fault, text: BrainBarMenuPresenter.Say.generic.text,
                             thing: .remembering, jobs: [.watch]))
        }
        var seen = Set<String>()
        return out.filter { seen.insert($0.text).inserted }.sorted { $0.level > $1.level }
    }

    static func thing(for issue: BrainBarStatusIssue) -> BrainBarMenuPresenter.Thing? {
        let key = issue.alertKey ?? ""
        switch issue.component {
        case .backups, .drive, .database, .transcripts: return .safe
        case .maintenance: return .upkeep
        case .watcher, .ingest: return .remembering
        case .health: return issue.id == "health.vector-coverage" ? .finding : nil
        case .jobs, .configuration:
            if key.contains("backup") || key.contains("jsonl") || key.contains("snapshot") { return .safe }
            if key.contains("maintenance") || key.contains("wal") || key.contains("decay") || key.contains("fts") { return .upkeep }
            if key.contains("index") || key.contains("watch") { return .remembering }
            return nil
        }
    }
}
