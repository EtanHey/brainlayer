import AppKit
import Foundation

/// The Finder and pasteboard side effects, injected so tests never touch either.
protocol BrainBarWorkspaceActing: Sendable {
    func reveal(_ url: URL)
    func copy(_ text: String)
}

struct BrainBarWorkspace: BrainBarWorkspaceActing {
    func reveal(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

struct BrainBarBackupScheduleRow: Equatable, Identifiable, Sendable {
    var id: String { title }
    let title: String
    let cadence: String
    let lastRun: String
    let nextRun: String
    /// The latest local copy, or nil when there is none; Reveal and Copy are hidden then.
    let localCopy: URL?

    func reveal(using workspace: any BrainBarWorkspaceActing) {
        if let localCopy { workspace.reveal(localCopy) }
    }

    func copyPath(using workspace: any BrainBarWorkspaceActing) {
        if let localCopy { workspace.copy(localCopy.path) }
    }
}

/// Where each backup's schedule, log and local copies live. Readers are injected so unit tests
/// never touch ~/Library or the real logs.
struct BrainBarBackupSources: Sendable {
    struct Paths: Sendable {
        let launchAgents: URL
        let databaseLog: URL
        let archiveLog: URL
        let maintenanceLog: URL
        let snapshotDirectory: URL
        let archiveDirectory: URL

        /// The same paths the Python jobs use: the DB log and snapshots beside the resolved DB,
        /// each honouring its BRAINLAYER_* override.
        static func live(
            databasePath: String,
            environment: [String: String] = ProcessInfo.processInfo.environment,
            home: URL = FileManager.default.homeDirectoryForCurrentUser
        ) -> Paths {
            let data = home.appendingPathComponent(".local/share/brainlayer")
            func override(_ key: String, _ fallback: URL) -> URL {
                environment[key].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0) } ?? fallback
            }
            return Paths(
                launchAgents: home.appendingPathComponent("Library/LaunchAgents"),
                databaseLog: override(
                    "BRAINLAYER_BACKUP_LOG_PATH",
                    URL(fileURLWithPath: databasePath).deletingLastPathComponent().appendingPathComponent("logs/backup-daily.log")
                ),
                archiveLog: data.appendingPathComponent("logs/jsonl-backup.log"),
                maintenanceLog: override("BRAINLAYER_MAINTENANCE_LOG_PATH", data.appendingPathComponent("logs/maintenance.log")),
                snapshotDirectory: override("BRAINLAYER_BACKUP_STAGING_DIR", data.appendingPathComponent("backups")),
                archiveDirectory: data.appendingPathComponent("jsonl-backups")
            )
        }
    }

    let paths: Paths
    let readFile: @Sendable (URL) -> Data?
    let listDirectory: @Sendable (URL) -> [String]

    static func live(databasePath: String) -> Self {
        Self(
            paths: .live(databasePath: databasePath),
            readFile: { FileManager.default.contents(atPath: $0.path) },
            listDirectory: { (try? FileManager.default.contentsOfDirectory(atPath: $0.path)) ?? [] }
        )
    }

    func rows(now: Date, calendar: Calendar, formatDate: (Date) -> String) -> [BrainBarBackupScheduleRow] {
        [
            row("Database", job: .backupDaily, log: paths.databaseLog, kind: .databaseBackup,
                localCopy: BackupLocalFiles.latestSnapshot(in: listDirectory(paths.snapshotDirectory))
                    .map { paths.snapshotDirectory.appendingPathComponent($0) },
                now: now, calendar: calendar, formatDate: formatDate),
            row("Transcripts", job: .jsonlBackup, log: paths.archiveLog, kind: .transcriptArchive,
                localCopy: BackupLocalFiles.latestArchive(in: listDirectory(paths.archiveDirectory))
                    .map { paths.archiveDirectory.appendingPathComponent($0) },
                now: now, calendar: calendar, formatDate: formatDate),
            row("Weekly maintenance", job: .maintenanceWeekly, log: paths.maintenanceLog, kind: .weeklyMaintenance,
                localCopy: nil, now: now, calendar: calendar, formatDate: formatDate),
        ]
    }

    private func row(
        _ title: String,
        job: BrainLayerLaunchdJob,
        log: URL,
        kind: BackupLogReader.Kind,
        localCopy: URL?,
        now: Date,
        calendar: Calendar,
        formatDate: (Date) -> String
    ) -> BrainBarBackupScheduleRow {
        let plist = paths.launchAgents.appendingPathComponent("\(job.launchdLabel).plist")
        let schedule = BackupScheduleRead.parse(plist: readFile(plist), path: plist.path)
        let lastRun = BackupLogReader.lastRun(kind, log: readFile(log)).map { receipt in
            let verification = receipt.verified.map { $0 ? " · verified" : " · NOT verified" } ?? ""
            return "Last run \(formatDate(receipt.at))\(verification)"
        } ?? "No run recorded in \(log.lastPathComponent)"
        let nextRun = schedule.nextRun(after: now, calendar: calendar).map { "Next run \(formatDate($0))" } ?? "Next run unknown"
        return BrainBarBackupScheduleRow(
            title: title, cadence: schedule.text, lastRun: lastRun, nextRun: nextRun, localCopy: localCopy
        )
    }
}
