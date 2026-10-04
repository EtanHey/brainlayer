import Foundation

/// What Show log did, so every surface can say it (Codex #1062 r1 B1): a missing log is named,
/// never a silent folder reveal.
enum BrainBarJobAlertLogResult: Equatable, Sendable {
    case opened(URL)
    /// No log file yet (or a job BrainBar has no log for); the logs folder is revealed if it exists.
    case missing(message: String)

    var message: String? {
        if case let .missing(message) = self { message } else { nil }
    }
}

/// Show log for a job alert. UI-only: it needs the job log paths `BrainBarBackupSources` resolves.
extension BrainBarJobAlerts {
    /// The log a job writes, by its alert key; nil for a job without a log BrainBar knows.
    static func logURL(forKey key: String?, paths: BrainBarBackupSources.Paths) -> URL? {
        switch key {
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
        case let key? where key.hasPrefix("maintenance-"): "No maintenance log yet; it's written on the next run."
        case "backup-daily": "No database backup log yet; it's written on the next run."
        case "jsonl-backup": "No transcript backup log yet; it's written on the next run."
        default: "BrainBar doesn't know this job's log; showing the logs folder."
        }
    }

    /// Show log: open the job's log in its default app (Console). With no log file yet, or an
    /// unknown job, reveal the logs folder when it exists and return the sentence to show.
    @discardableResult
    static func showLog(
        forKey key: String?,
        paths: BrainBarBackupSources.Paths,
        workspace: any BrainBarWorkspaceActing,
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) -> BrainBarJobAlertLogResult {
        if let log = logURL(forKey: key, paths: paths), fileExists(log) {
            workspace.open(log)
            return .opened(log)
        }
        let folder = (logURL(forKey: key, paths: paths) ?? paths.maintenanceLog).deletingLastPathComponent()
        if fileExists(folder) { workspace.reveal(folder) }
        return .missing(message: missingLogMessage(forKey: key))
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
        panelState.jobAlertLogMessage = showLog(for: document, paths: paths, workspace: workspace).message
    }
}
