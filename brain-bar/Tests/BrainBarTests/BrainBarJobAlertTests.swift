import XCTest
@testable import BrainBar

/// Lead ruling 2026-10-04: Etan saw "BrainLayer light maintenance failed; check the maintenance
/// log" four times. Each alert shows once per screen, carries its own words, offers Show log, and
/// a clean later run clears it everywhere.
@MainActor
final class BrainBarJobAlertTests: XCTestCase {
    private let alert = "BrainLayer light maintenance failed; check the maintenance log"
    private let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent("brainbar-job-alert-\(UUID().uuidString)", isDirectory: true)

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("logs"), withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func writeAlerts(_ json: String) throws -> URL {
        let url = root.appendingPathComponent("job-alerts.json")
        try Data(json.utf8).write(to: url)
        return url
    }

    private func document(errorType: String?) -> ObservabilityDocument {
        guard case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else {
            fatalError("fixture must be readable")
        }
        let b = base.backups
        return ObservabilityDocument(
            schemaVersion: 1, generatedAt: base.generatedAt, dbPath: root.appendingPathComponent("brainlayer.db").path,
            windowHours: 24, stores: base.stores, emitters: base.emitters, authorUnknown: base.authorUnknown,
            backups: .init(
                state: b.state, reason: b.reason, inputs: b.inputs, freshness: b.freshness,
                thresholdHours: b.thresholdHours, retentionInvariant: b.retentionInvariant,
                survivingArchives30D: b.survivingArchives30D, errorType: errorType,
                lastVerifiedUpload: b.lastVerifiedUpload, dbSnapshot: b.dbSnapshot, launchd: b.launchd
            )
        )
    }

    // MARK: reading the job-alert state

    func test_job_alerts_read_the_python_file_and_fail_unknown_not_empty() throws {
        let url = try writeAlerts(#"{"maintenance-light": "\#(alert)"}"#)
        XCTAssertEqual(BrainBarJobAlerts.read(url: url)?.active, ["maintenance-light": alert])
        XCTAssertEqual(BrainBarJobAlerts.read(url: url)?.key(for: alert), "maintenance-light")
        XCTAssertNil(BrainBarJobAlerts.read(url: root.appendingPathComponent("missing.json")))
        _ = try writeAlerts("not json")
        XCTAssertNil(BrainBarJobAlerts.read(url: url), "an unreadable file is unknown, never 'no alerts'")
    }

    func test_the_alert_file_sits_beside_the_producers_database_unless_overridden() {
        let db = root.appendingPathComponent("brainlayer.db").path
        XCTAssertEqual(BrainBarJobAlerts.url(dbPath: db, environment: [:]), root.appendingPathComponent("job-alerts.json"))
        XCTAssertEqual(
            BrainBarJobAlerts.url(dbPath: db, environment: ["BRAINLAYER_JOB_ALERT_PATH": "/x/alerts.json"]),
            URL(fileURLWithPath: "/x/alerts.json")
        )
    }

    func test_a_schema_invalid_alert_file_is_unknown_not_empty() throws {
        // Macroscope #1062: a null reason must not read as "no alerts" and clear a live one.
        let url = try writeAlerts(#"{"maintenance-light": null}"#)
        XCTAssertNil(BrainBarJobAlerts.read(url: url))
        XCTAssertEqual(
            BrainBarJobAlerts.reconcile(document(errorType: "job_alert:\(alert)"), with: BrainBarJobAlerts.read(url: url))
                .backups.errorType,
            "job_alert:\(alert)"
        )
    }

    func test_show_log_finds_the_job_by_the_raw_reason_not_the_sanitized_text() {
        // Macroscope #1062: the shown text is sanitized; the job key is matched on the raw reason.
        let raw = "BrainLayer light maintenance failed: token sk-ant-api03-\(String(repeating: "x", count: 40))"
        let doc = document(errorType: "job_alert:\(raw)")
        XCTAssertEqual(BrainBarJobAlerts.rawReason(doc), raw)
        XCTAssertEqual(BrainBarJobAlerts(active: ["maintenance-light": raw]).key(for: raw), "maintenance-light")
        XCTAssertNil(BrainBarJobAlerts.rawReason(document(errorType: "drive_credentials_missing")))
    }

    // MARK: a clean run clears the alert

    func test_a_clean_run_clears_a_job_alert_before_the_next_observability_write() {
        let reconciled = BrainBarJobAlerts(active: [:]).reconcile(document(errorType: "job_alert:\(alert)"))
        XCTAssertNil(reconciled.backups.errorType)
        XCTAssertNil(ObservabilityPresentation.backupStatus(for: reconciled.backups).attentionLine)
    }

    func test_reconciling_keeps_a_live_alert_and_never_guesses_from_an_unknown_file() {
        let doc = document(errorType: "job_alert:\(alert)")
        XCTAssertEqual(BrainBarJobAlerts(active: ["maintenance-light": alert]).reconcile(doc).backups.errorType, "job_alert:\(alert)")
        XCTAssertEqual(BrainBarJobAlerts.reconcile(doc, with: nil).backups.errorType, "job_alert:\(alert)")
    }

    func test_reconciling_shows_the_alert_still_active_when_the_shown_one_cleared() {
        let other = "Transcript backup failed; check the backup log"
        let doc = document(errorType: "job_alert:\(alert)")
        XCTAssertEqual(BrainBarJobAlerts(active: ["jsonl-backup": other]).reconcile(doc).backups.errorType, "job_alert:\(other)")
    }

    func test_reconciling_never_touches_a_backup_error_that_is_not_a_job_alert() {
        let doc = document(errorType: "drive_credentials_missing")
        XCTAssertEqual(BrainBarJobAlerts(active: [:]).reconcile(doc).backups.errorType, "drive_credentials_missing")
    }

    func test_the_reader_reconciles_with_the_alert_file_beside_the_database() throws {
        let observability = root.appendingPathComponent("observability.json")
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(document(errorType: "job_alert:\(alert)")).write(to: observability)
        _ = try writeAlerts(#"{"maintenance-light": "\#(alert)"}"#)
        guard case let .readable(live) = ObservabilityReader.readReconciled(url: observability, environment: [:]) else {
            XCTFail("fixture must be readable")
            return
        }
        XCTAssertEqual(live.backups.errorType, "job_alert:\(alert)")
        _ = try writeAlerts("{}")
        guard case let .readable(cleared) = ObservabilityReader.readReconciled(url: observability, environment: [:]) else {
            XCTFail("fixture must be readable")
            return
        }
        XCTAssertNil(cleared.backups.errorType)
    }

    // MARK: Show log

    private var paths: BrainBarBackupSources.Paths {
        .init(
            launchAgents: root, databaseLog: root.appendingPathComponent("logs/backup-daily.log"),
            archiveLog: root.appendingPathComponent("logs/jsonl-backup.log"),
            maintenanceLog: root.appendingPathComponent("logs/maintenance.log"),
            snapshotDirectory: root, archiveDirectory: root
        )
    }

    func test_the_nightly_alert_opens_the_nightly_jobs_own_log() {
        // Macroscope #1062: the nightly LaunchAgent can name its own BRAINLAYER_MAINTENANCE_LOG_PATH.
        var paths = self.paths
        paths.nightlyMaintenanceLog = root.appendingPathComponent("logs/nightly.log")
        XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: "maintenance-light", paths: paths), paths.nightlyMaintenanceLog)
        XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: "maintenance-full", paths: paths), paths.maintenanceLog)
    }

    func test_the_nightly_log_path_comes_from_the_nightly_launch_agent() throws {
        let home = root.appendingPathComponent("home")
        let agents = home.appendingPathComponent("Library/LaunchAgents")
        try FileManager.default.createDirectory(at: agents, withIntermediateDirectories: true)
        let plist: [String: Any] = ["EnvironmentVariables": ["BRAINLAYER_MAINTENANCE_LOG_PATH": "/x/nightly.log"]]
        try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
            .write(to: agents.appendingPathComponent("\(BrainLayerLaunchdJob.maintenanceNightly.launchdLabel).plist"))
        let live = BrainBarBackupSources.Paths.live(databasePath: "/db/brainlayer.db", environment: [:], home: home)
        XCTAssertEqual(live.nightlyMaintenanceLog, URL(fileURLWithPath: "/x/nightly.log"))
        XCTAssertEqual(live.maintenanceLog, home.appendingPathComponent(".local/share/brainlayer/logs/maintenance.log"))
    }

    /// Lead addendum 2026-10-04: Show log opens maintenance.log (which #1061 makes hold every abort
    /// and failure reason), never the LaunchAgent's .out/.err logs.
    func test_show_log_defaults_to_the_shared_maintenance_log_never_the_launch_agent_stdout() throws {
        let home = root.appendingPathComponent("home")
        let agents = home.appendingPathComponent("Library/LaunchAgents")
        try FileManager.default.createDirectory(at: agents, withIntermediateDirectories: true)
        for job in [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly] {
            let plist: [String: Any] = ["StandardOutPath": "/Users/x/Library/Logs/brainlayer/\(job.launchdLabel).out.log"]
            try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
                .write(to: agents.appendingPathComponent("\(job.launchdLabel).plist"))
        }
        let live = BrainBarBackupSources.Paths.live(databasePath: "/db/brainlayer.db", environment: [:], home: home)
        let expected = home.appendingPathComponent(".local/share/brainlayer/logs/maintenance.log")
        for key in ["maintenance-light", "maintenance-full", "maintenance-burn"] {
            XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: key, paths: live), expected, key)
        }
    }

    func test_each_job_alert_key_names_its_own_log() {
        XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: "maintenance-light", paths: paths), paths.maintenanceLog)
        XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: "maintenance-full", paths: paths), paths.maintenanceLog)
        XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: "backup-daily", paths: paths), paths.databaseLog)
        XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: "jsonl-backup", paths: paths), paths.archiveLog)
        XCTAssertNil(BrainBarJobAlerts.logURL(forKey: "drive-consent", paths: paths))
        XCTAssertNil(BrainBarJobAlerts.logURL(forKey: nil, paths: paths))
    }

    func test_show_log_opens_an_existing_log_and_otherwise_reveals_the_logs_folder() throws {
        final class Recorder: BrainBarWorkspaceActing, @unchecked Sendable {
            var opened: [URL] = [], revealed: [URL] = []
            func open(_ url: URL) { opened.append(url) }
            func reveal(_ url: URL) { revealed.append(url) }
            func copy(_ text: String) {}
        }
        let workspace = Recorder()
        BrainBarJobAlerts.showLog(forKey: "maintenance-light", paths: paths, workspace: workspace)
        XCTAssertEqual(workspace.revealed, [root.appendingPathComponent("logs")], "no log yet: show where it will be")
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        BrainBarJobAlerts.showLog(forKey: "maintenance-light", paths: paths, workspace: workspace)
        XCTAssertEqual(workspace.opened, [paths.maintenanceLog])
        BrainBarJobAlerts.showLog(forKey: nil, paths: paths, workspace: workspace)
        XCTAssertEqual(workspace.revealed.last, root.appendingPathComponent("logs"), "an unknown job reveals the logs folder")
    }

    // MARK: the menu

    func test_the_menu_offers_show_log_only_while_a_job_alert_is_active() {
        let on = BadgeStatePresentation(badgeOn: true, reason: alert, activeCodes: ["queue_backed_up", "job_alert_maintenance-light"])
        XCTAssertEqual(BrainBarStatusPopoverController.jobAlertKey(for: on), "maintenance-light")
        XCTAssertEqual(BrainBarStatusPopoverController.menuRowTitles(for: on).prefix(2), ["Needs attention: \(alert)", "Show log"])
        let off = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        XCTAssertNil(BrainBarStatusPopoverController.jobAlertKey(for: off))
        XCTAssertFalse(BrainBarStatusPopoverController.menuRowTitles(for: off).contains("Show log"))
    }
}

/// The Backups page shows a job alert once: one alert card with Show log. The group badge keeps
/// its "Needs attention" verdict without repeating the sentence, and the status rows drop it.
@MainActor
final class BrainBarBackupsJobAlertPageTests: XCTestCase {
    private let alert = "BrainLayer light maintenance failed; check the maintenance log"

    private func viewModel(_ result: ObservabilityReadResult) throws -> BrainBarSettingsViewModel {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-job-alert-page-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let ok = BrainLayerLaunchdJobObservation(loadState: .loaded, runs: 1, lastExitCode: 0,
                                                 lastRunAt: Date(), nextRunAt: Date(), isContinuous: false)
        return BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            initialLaunchdObservations: [.backupDaily: ok, .jsonlBackup: ok],
            refreshStatusOnLoad: false,
            initialObservabilityResult: result
        )
    }

    func test_the_backups_page_shows_a_job_alert_exactly_once() throws {
        let model = try viewModel(BrainBarDashboardFixture.maintenanceAlertObservabilityResult)
        XCTAssertEqual(model.jobAlert, alert)
        let health = model.backupsHealth(drive: nil)
        XCTAssertEqual(health.badge, .attention, "the verdict still says the page needs attention")
        XCTAssertNil(model.backupsBadgeReason(drive: nil), "the alert card already says it")
        XCTAssertFalse(try XCTUnwrap(model.backupStatusLines).contains { $0.text == alert })
        let shown = [model.jobAlert, model.backupsBadgeReason(drive: nil)].compactMap { $0 }
            + (model.backupStatusLines ?? []).map(\.text)
        XCTAssertEqual(shown.filter { $0 == alert }.count, 1)
    }

    func test_after_a_clean_run_the_backups_page_shows_no_alert_at_all() throws {
        let model = try viewModel(BrainBarDashboardFixture.healthyObservabilityResult)
        XCTAssertNil(model.jobAlert)
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .healthy)
        XCTAssertFalse(try XCTUnwrap(model.backupStatusLines).contains { $0.tone == .red })
    }

    func test_any_other_red_reason_still_shows_under_the_badge() throws {
        guard case let .readable(doc) = BrainBarDashboardFixture.healthyObservabilityResult else {
            XCTFail("fixture must be readable")
            return
        }
        let b = doc.backups
        let stale = ObservabilityDocument(
            schemaVersion: 1, generatedAt: doc.generatedAt, dbPath: doc.dbPath, windowHours: 24,
            stores: doc.stores, emitters: doc.emitters, authorUnknown: doc.authorUnknown,
            backups: .init(state: b.state, reason: b.reason, inputs: b.inputs, freshness: "stale",
                           thresholdHours: b.thresholdHours, retentionInvariant: b.retentionInvariant,
                           survivingArchives30D: b.survivingArchives30D, errorType: nil,
                           lastVerifiedUpload: b.lastVerifiedUpload, dbSnapshot: b.dbSnapshot, launchd: b.launchd)
        )
        let model = try viewModel(.readable(stale))
        XCTAssertNil(model.jobAlert)
        XCTAssertEqual(model.backupsBadgeReason(drive: nil), "Backup freshness (DB + transcript): stale (> 36 h)")
    }
}
