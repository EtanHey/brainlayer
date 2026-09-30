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

        /// The same paths the Python jobs use (#1016 R1 B1). Each value comes from the environment
        /// its job runs with: the job's installed LaunchAgent `EnvironmentVariables`, overlaid by the
        /// env file that agent names (`brainlayer-env-run.sh` exports it last), then BrainBar's own
        /// environment, then the job's default. A leading `~` is expanded.
        static func live(
            databasePath: String,
            environment: [String: String] = ProcessInfo.processInfo.environment,
            home: URL = FileManager.default.homeDirectoryForCurrentUser,
            readFile: (URL) -> Data? = { FileManager.default.contents(atPath: $0.path) }
        ) -> Paths {
            let launchAgents = home.appendingPathComponent("Library/LaunchAgents")
            let data = home.appendingPathComponent(".local/share/brainlayer")
            func path(_ job: BrainLayerLaunchdJob, _ key: String, _ fallback: URL) -> URL {
                let jobEnvironment = BrainLayerJobEnvironment.effective(
                    plist: readFile(launchAgents.appendingPathComponent("\(job.launchdLabel).plist")),
                    brainBarEnvironment: environment,
                    home: home,
                    readFile: readFile
                )
                guard let value = jobEnvironment[key], !value.isEmpty else { return fallback }
                return URL(fileURLWithPath: BrainLayerJobEnvironment.expandTilde(value, home: home))
            }
            return Paths(
                launchAgents: launchAgents,
                databaseLog: path(
                    .backupDaily, "BRAINLAYER_BACKUP_LOG_PATH",
                    URL(fileURLWithPath: databasePath).deletingLastPathComponent().appendingPathComponent("logs/backup-daily.log")
                ),
                archiveLog: path(.jsonlBackup, "BRAINLAYER_JSONL_BACKUP_LOG_PATH", data.appendingPathComponent("logs/jsonl-backup.log")),
                maintenanceLog: path(.maintenanceWeekly, "BRAINLAYER_MAINTENANCE_LOG_PATH", data.appendingPathComponent("logs/maintenance.log")),
                snapshotDirectory: path(.backupDaily, "BRAINLAYER_BACKUP_STAGING_DIR", data.appendingPathComponent("backups")),
                archiveDirectory: path(.jsonlBackup, "BRAINLAYER_JSONL_BACKUP_STAGING_DIR", data.appendingPathComponent("jsonl-backups"))
            )
        }
    }

    let paths: Paths
    let readFile: @Sendable (URL) -> Data?
    let listDirectory: @Sendable (URL) -> [String]
    let isRegularFile: @Sendable (URL) -> Bool
    /// A file's modification time; stamps a maintenance job's newest stdout record.
    var modificationDate: @Sendable (URL) -> Date? = {
        (try? FileManager.default.attributesOfItem(atPath: $0.path))?[.modificationDate] as? Date
    }

    /// A regular file, read without following a symlink (`attributesOfItem` is lstat), so a link
    /// that could point outside the backups directory never gets Reveal or Copy (#1016 R1 B2).
    static func isRegularFile(_ url: URL) -> Bool {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType) == .typeRegular
    }

    /// The newest candidate in `directory` that is a regular file; directories, symlinks and names
    /// that vanished are skipped.
    private func latestLocalCopy(in directory: URL, latest: ([String]) -> String?) -> URL? {
        let regular = listDirectory(directory).filter { isRegularFile(directory.appendingPathComponent($0)) }
        return latest(regular).map { directory.appendingPathComponent($0) }
    }

    static func live(databasePath: String) -> Self {
        Self(
            paths: .live(databasePath: databasePath),
            readFile: { FileManager.default.contents(atPath: $0.path) },
            listDirectory: { (try? FileManager.default.contentsOfDirectory(atPath: $0.path)) ?? [] },
            isRegularFile: { isRegularFile($0) }
        )
    }

    func rows(now: Date, calendar: Calendar, formatDate: (Date) -> String) -> [BrainBarBackupScheduleRow] {
        [
            row("Database", job: .backupDaily, log: paths.databaseLog, kind: .databaseBackup,
                localCopy: latestLocalCopy(in: paths.snapshotDirectory, latest: BackupLocalFiles.latestSnapshot),
                now: now, calendar: calendar, formatDate: formatDate),
            row("Transcripts", job: .jsonlBackup, log: paths.archiveLog, kind: .transcriptArchive,
                localCopy: latestLocalCopy(in: paths.archiveDirectory, latest: BackupLocalFiles.latestArchive),
                now: now, calendar: calendar, formatDate: formatDate),
            row("Weekly maintenance", job: .maintenanceWeekly, log: paths.maintenanceLog, kind: .weeklyMaintenance,
                localCopy: nil, now: now, calendar: calendar, formatDate: formatDate),
        ]
    }

    /// The Maintenance card's evidence: the weekly's last completed pass (the same #1015 reader the
    /// Weekly maintenance row uses) and, per maintenance job, ONLY the newest record its last run
    /// printed to the LaunchAgent's StandardOutPath, stamped with the log's modification time so the
    /// card can check it belongs to the observed run. An incomplete newest record is never replaced
    /// by an older one.
    func maintenanceEvidence() -> BrainLayerMaintenanceEvidence {
        var records: [BrainLayerLaunchdJob: BrainLayerMaintenanceEvidence.RunRecord] = [:]
        for job in BrainLayerLaunchdJobGroup.maintenance.jobs {
            records[job] = newestRunRecord(job)
        }
        return BrainLayerMaintenanceEvidence(weeklyCompletion: weeklyCompletion(), runRecords: records)
    }

    private func weeklyCompletion() -> BrainLayerMaintenanceEvidence.WeeklyCompletion {
        // Unreadable, not UTF-8, or no parseable row: that is not "no completed pass" (N1).
        guard let data = readFile(paths.maintenanceLog), let text = String(data: data, encoding: .utf8),
              text.split(whereSeparator: \.isNewline).contains(where: {
                  (try? JSONSerialization.jsonObject(with: Data($0.utf8))) is [String: Any]
              })
        else { return .unread }
        return BackupLogReader.lastRun(.weeklyMaintenance, log: data).map { .completed($0.at) } ?? .noneRecorded
    }

    private func newestRunRecord(_ job: BrainLayerLaunchdJob) -> BrainLayerMaintenanceEvidence.RunRecord {
        guard let plist = readFile(paths.launchAgents.appendingPathComponent("\(job.launchdLabel).plist"))
            .flatMap({ try? PropertyListSerialization.propertyList(from: $0, format: nil) as? [String: Any] })
        else { return .unavailable("its LaunchAgent could not be read") }
        guard let stdoutPath = plist["StandardOutPath"] as? String, !stdoutPath.isEmpty else {
            return .unavailable("its LaunchAgent names no StandardOutPath")
        }
        let log = URL(fileURLWithPath: stdoutPath)
        guard let data = readFile(log) else { return .unavailable("\(log.lastPathComponent) could not be read") }
        guard let writtenAt = modificationDate(log) else {
            return .unavailable("\(log.lastPathComponent) has no modification time")
        }
        // The log only grows; its tail holds the last run. Only the newest non-blank line counts.
        let text = String(decoding: data.suffix(64 * 1024), as: UTF8.self)
        guard let newest = text.split(whereSeparator: \.isNewline).last(where: {
            !$0.trimmingCharacters(in: .whitespaces).isEmpty
        }),
            let row = (try? JSONSerialization.jsonObject(with: Data(newest.utf8))) as? [String: Any],
            let status = row["status"] as? String
        else { return .unavailable("the newest record in \(log.lastPathComponent) is incomplete") }
        guard status == "aborted" else { return .notAnAbort(writtenAt: writtenAt) }
        guard let reason = row["reason"] as? String, !reason.isEmpty else {
            return .unavailable("the newest record in \(log.lastPathComponent) is incomplete")
        }
        return .aborted(reason: reason, writtenAt: writtenAt)
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
        let logData = readFile(log)
        let completed = BackupLogReader.lastRun(kind, log: logData).map { receipt in
            let verification = receipt.verified.map { $0 ? " · verified" : " · NOT verified" } ?? ""
            return "Last run \(formatDate(receipt.at))\(verification)"
        } ?? "No run recorded in \(log.lastPathComponent)"
        // An aborted pass never becomes the last run; a newer one is shown beside it.
        let lastRun = BackupLogReader.lastIncompleteAttempt(kind, log: logData).map { attempt in
            "\(completed) · last attempt \(formatDate(attempt.at)) not completed: \(attempt.reason)"
        } ?? completed
        let nextRun = schedule.nextRun(after: now, calendar: calendar).map { "Next run \(formatDate($0))" } ?? "Next run unknown"
        return BrainBarBackupScheduleRow(
            title: title, cadence: schedule.text, lastRun: lastRun, nextRun: nextRun, localCopy: localCopy
        )
    }
}

/// A BrainLayer launchd job's effective environment, built the way `brainlayer-env-run.sh` builds
/// it: the installed plist's `EnvironmentVariables`, then the env file it names (else BrainBar's
/// `BRAINLAYER_ENV_FILE`, else the default) exported over them. BrainBar's own environment only
/// fills keys the job does not set.
enum BrainLayerJobEnvironment {
    static func effective(
        plist: Data?,
        brainBarEnvironment: [String: String],
        home: URL,
        readFile: (URL) -> Data?
    ) -> [String: String] {
        let agent = plist
            .flatMap { try? PropertyListSerialization.propertyList(from: $0, format: nil) as? [String: Any] }
        var job = (agent?["EnvironmentVariables"] as? [String: Any])?.compactMapValues { $0 as? String } ?? [:]
        let envFile = [job["BRAINLAYER_ENV_FILE"], brainBarEnvironment["BRAINLAYER_ENV_FILE"]]
            .lazy.compactMap { $0 }.first { !$0.isEmpty }
            .map { expandTilde($0, home: home) }
            ?? home.appendingPathComponent(".config/brainlayer/brainlayer.env").path
        if let text = readFile(URL(fileURLWithPath: envFile)).flatMap({ String(data: $0, encoding: .utf8) }) {
            for (key, value) in envFileValues(text) { job[key] = value }
        }
        return brainBarEnvironment.merging(job) { _, jobValue in jobValue }
    }

    static func expandTilde(_ path: String, home: URL) -> String {
        path == "~" ? home.path : path.hasPrefix("~/") ? home.path + path.dropFirst() : path
    }

    /// The simple `KEY=value` / `export KEY="value"` lines `brainlayer-env-run.sh` exports; command
    /// substitutions are skipped there too.
    static func envFileValues(_ text: String) -> [String: String] {
        var values: [String: String] = [:]
        for raw in text.split(whereSeparator: \.isNewline) {
            var line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }
            if line.hasPrefix("export ") { line = String(line.dropFirst("export ".count)) }
            guard let equals = line.firstIndex(of: "=") else { continue }
            let key = line[..<equals].trimmingCharacters(in: .whitespaces)
            var value = line[line.index(after: equals)...].trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty, !value.contains("$("), !value.contains("`") else { continue }
            if value.count >= 2, let first = value.first, first == value.last, first == "\"" || first == "'" {
                value = String(value.dropFirst().dropLast())
            }
            values[key] = value
        }
        return values
    }
}
