import Foundation

/// The acknowledged result shared by the menu, Dashboard, and Backups page.
enum BrainBarJobAlertLogResult: Equatable, Sendable {
    case opened(URL)
    /// No log file yet; the message says whether its folder opened.
    case missing(message: String)
    case failed(message: String)

    var message: String? {
        switch self {
        case .opened: nil
        case let .missing(message), let .failed(message): message
        }
    }
}

/// Show log for a job alert. UI-only: it needs the job log paths `BrainBarBackupSources` resolves.
extension BrainBarJobAlerts {
    /// The log a job writes, by its alert key; nil for a job without a log BrainBar knows.
    static func logURL(forKey key: String?, paths: BrainBarBackupSources.Paths) -> URL? {
        switch key {
        case let key? where ["index", "backup-snapshot", "p0-counter", "wal-checkpoint", "decay", "repair-fts"].contains(key):
            paths.calendarLogDirectory.appendingPathComponent("\(key).err.log")
        // The nightly LaunchAgent may name its own log; the weekly's is the shared default.
        case "maintenance-light": paths.nightlyMaintenanceLog ?? paths.maintenanceLog
        case let key? where key.hasPrefix("maintenance-"): paths.maintenanceLog
        case "backup-daily": paths.databaseLog
        case "jsonl-backup": paths.archiveLog
        default: nil
        }
    }

    /// The sentence a surface shows when the job has no log file yet.
    static func missingLogMessage(forKey key: String?) -> String {
        switch key {
        case let key? where ["index", "backup-snapshot", "p0-counter", "wal-checkpoint", "decay", "repair-fts"].contains(key):
            "No \(key) log yet; it's written on the next run."
        case let key? where key.hasPrefix("maintenance-"): "No maintenance log yet; it's written on the next run."
        case "backup-daily": "No database backup log yet; it's written on the next run."
        case "jsonl-backup": "No transcript backup log yet; it's written on the next run."
        default: "BrainBar doesn't know this job's log."
        }
    }

    /// Open the log, or its existing parent directory. Never claim success on workspace refusal.
    @discardableResult
    static func showLog(
        forKey key: String?,
        paths: BrainBarBackupSources.Paths,
        workspace: any BrainBarWorkspaceActing,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) },
        directoryExists: (URL) -> Bool = {
            var isDirectory: ObjCBool = false
            return FileManager.default.fileExists(atPath: $0.path, isDirectory: &isDirectory) && isDirectory.boolValue
        }
    ) -> BrainBarJobAlertLogResult {
        let log = logURL(forKey: key, paths: paths)
        if let log, fileExists(log) {
            return workspace.openLog(log) ? .opened(log) : .failed(message: "Could not open log: \(log.path)")
        }
        let folder = (log ?? paths.maintenanceLog).deletingLastPathComponent()
        let missing = missingLogMessage(forKey: key)
        guard directoryExists(folder) else {
            return .missing(message: "\(missing) The logs folder does not exist: \(folder.path)")
        }
        guard workspace.openLog(folder) else {
            return .failed(message: "\(missing) Could not open the logs folder: \(folder.path)")
        }
        return .missing(message: "\(missing) Opened the logs folder.")
    }

    /// Show log for the job alert in `document`: the job key is looked up on the raw reason in the
    /// alert file the document's producer reads.
    @discardableResult
    static func showLog(
        for document: ObservabilityDocument,
        paths: BrainBarBackupSources.Paths,
        workspace: any BrainBarWorkspaceActing
    ) -> BrainBarJobAlertLogResult {
        let key = rawReason(document).flatMap { read(url: producerURL(for: document))?.key(for: $0) }
        return showLog(forKey: key, paths: paths, workspace: workspace)
    }

    /// The Dashboard's Show log: the result's sentence lands on the panel, where the item shows it.
    @MainActor
    static func showLog(
        for document: ObservabilityDocument,
        paths: BrainBarBackupSources.Paths,
        workspace: any BrainBarWorkspaceActing,
        panelState: BrainBarDashboardPanelState
    ) {
        let message = showLog(for: document, paths: paths, workspace: workspace).message
        panelState.jobAlertLogNote = BrainBarJobAlertLogNote(reason: rawReason(document), message: message)
    }
}
