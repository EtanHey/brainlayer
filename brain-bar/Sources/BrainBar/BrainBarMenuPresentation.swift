import Foundation

/// One capture of what the app holds at one moment (truth table U1). The presenter reads only this:
/// no globals, clocks or files. Nothing in production builds one yet (phase 8, PR 1).
struct BrainBarMenuCapture: Sendable {
    struct Watcher: Sendable {
        var readable = true, fresh = true
        var alertReasons: [String] = []
        var dbProbeFailed = false
    }

    /// `localSnapshot.uploadState == "in_progress"`, with `uploadProcessStartedAt` / `uploadProcessLive`.
    struct Upload: Sendable { var startedAt: Date, live: Bool }
    /// `recovering {kind: upload}` on `backup_offsite`. Not carried by `Check` today: stability PR-2.
    struct Recovering: Sendable {
        var progress: Double?, eta: Date?
        var startedAt: Date, updatedAt: Date
        var cadenceS: Double
    }

    var now: Date
    var calendar: Calendar
    var verdict: BrainBarHealthVerdict? = nil
    /// Awake seconds since `verdict.evaluatedAt`, from the caller's sleep history; fresh is 0...900.
    var verdictAwakeAgeS: TimeInterval = 0
    var isChecking = false
    var wakeCatchUpPending = false
    var driveAuth: DriveAuthStatus.State = .valid
    /// The app's backups observation: measured (true) or unavailable (false).
    var backupsMeasured = true
    /// The newest verified copy time from the last measured observation. No kept value today.
    var lastKnownBackupAt: Date? = nil
    var watcher: Watcher? = Watcher()
    var upload: Upload? = nil
    var backupDailyFailures = 0, jsonlBackupFailures = 0, maintenanceNightFailures = 0
    /// The collector's keyword signal coverage, 0...1.
    var keywordCoverage: Double? = nil
    var offsiteRecovering: Recovering? = nil
    /// Start of the watcher's `offset_lag` episode: a UI clock today, `lag_bad_since` after stability PR-3.
    var lagEpisodeStart: Date? = nil
}

/// Everything the M menu and icon show for one capture (phase-8 PRESENTATION-SPEC.md).
struct BrainBarMenuPresentation: Equatable, Sendable {
    enum Form: String, Sendable { case calm, needs }
    enum Icon: String, Sendable { case resting, marked }
    enum Sign: String, Sendable { case ok, busy, wait, marked }
    enum ActionKind: String, Sendable { case reconnectDrive = "reconnect_drive", connectDrive = "connect_drive", copyReport = "copy_report" }
    struct Action: Equatable, Sendable { var kind: ActionKind, label: String }
    enum RowKind: String, Sendable {
        case headline, separator
        case headlineDetail = "headline_detail", needsYou = "needs_you", statusLine = "status_line"
        case factLine = "fact_line", everythingElse = "everything_else", standardItem = "standard_item"
    }

    struct Row: Equatable, Sendable {
        var kind: RowKind
        var strings: [String] = []
        var action: Action? = nil
        var sign: Sign? = nil
        var dividerBefore = false
        var spinner = false
        var progressPercent: Int? = nil
        var keyEquivalent: String? = nil
        var optionAlternate: String? = nil
    }

    var form: Form
    var headline: String
    var count: Int
    var icon: Icon
    var rows: [Row]
}

