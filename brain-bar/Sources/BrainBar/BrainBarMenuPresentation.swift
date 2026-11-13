import Foundation

/// One capture of what the app holds at one moment (truth table U1). The presenter reads only this:
/// no globals, clocks or files. Nothing in production builds one yet (phase 8, P1 of five).
/// Each PR adds only the inputs its own rows read (P2: Drive model, job streaks, keyword coverage).
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
    /// The words' time zone. Month names and day boundaries are always Gregorian.
    var timeZone: TimeZone
    var verdict: BrainBarHealthVerdict? = nil
    /// Awake seconds since `verdict.evaluatedAt`, from the caller's sleep history; fresh is 0...900.
    var verdictAwakeAgeS: TimeInterval = 0
    var isChecking = false
    var wakeCatchUpPending = false
    /// The app's backups observation: measured (true) or unavailable (false).
    var backupsMeasured = true
    /// The newest verified copy time from the last measured observation (S4). No kept value today.
    var lastKnownBackupAt: Date? = nil
    var watcher: Watcher? = Watcher()
    var upload: Upload? = nil
    var offsiteRecovering: Recovering? = nil
    /// Start of the watcher's `offset_lag` episode: a UI clock today, `lag_bad_since` after stability PR-3.
    var lagEpisodeStart: Date? = nil
    var maintenanceNightFailures = 0
    /// `DriveAuthModel.state`: wording only (`missing` reads "isn’t connected"). Never a status source (F14).
    var driveAuthModel: DriveAuthStatus.State? = nil
    /// `job.backupDaily` (the database backup) and `job.jsonlBackup` (transcripts): failure streaks and the next run.
    var backupDailyFailures = 0
    var backupDailyNextRunAt: Date? = nil
    var jsonlBackupFailures = 0
    /// `collector.signalCoverage.keyword`, 0...1 (F07's exact-words sentence). Not in today's status inputs.
    var keywordCoverage: Double? = nil
}

/// Everything the M menu and icon show for one capture (phase-8 PRESENTATION-SPEC.md).
struct BrainBarMenuPresentation: Equatable, Sendable {
    enum Form: String, Sendable { case calm, needs }
    enum Icon: String, Sendable { case resting, marked }
    /// The cases file's `mark`: A2's signs, A1's disc, and A1's row marks.
    enum Mark: String, Sendable { case ok, busy, wait, marked, alertCircle = "alert-circle", checkmark }
    enum RowState: String, Sendable { case behind, cantTell = "cant-tell" }
    enum ActionKind: String, Sendable {
        case reconnectDrive = "reconnect_drive", connectDrive = "connect_drive", copyReport = "copy_report"
        case openStorageSettings = "open_storage_settings"
    }
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
        var mark: Mark? = nil
        var state: RowState? = nil
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
