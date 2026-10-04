import Foundation

/// Show log for a job alert. UI-only: it needs the job log paths `BrainBarBackupSources` resolves.
extension BrainBarJobAlerts {
    /// The log a job writes, by its alert key; nil for a job without a log BrainBar knows.
    static func logURL(forKey key: String?, paths: BrainBarBackupSources.Paths) -> URL? {
        switch key {
        case let key? where key.hasPrefix("maintenance-"): paths.maintenanceLog
        case "backup-daily": paths.databaseLog
        case "jsonl-backup": paths.archiveLog
        default: nil
        }
    }

    /// Show log: open the job's log in its default app (Console). When there is no log file yet,
    /// or the job is unknown, reveal the logs folder instead, so the button never does nothing.
    static func showLog(
        forKey key: String?,
        paths: BrainBarBackupSources.Paths,
        workspace: any BrainBarWorkspaceActing,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) {
        if let log = logURL(forKey: key, paths: paths), fileExists(log) {
            workspace.open(log)
        } else {
            workspace.reveal((logURL(forKey: key, paths: paths) ?? paths.maintenanceLog).deletingLastPathComponent())
        }
    }
}
