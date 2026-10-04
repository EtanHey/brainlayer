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

    func test_the_reader_finds_the_alert_file_the_producers_launch_agent_names() throws {
        // Macroscope #1062 r2: BRAINLAYER_JOB_ALERT_PATH set only in the observability job's own
        // environment (its plist or env file) must be honoured, not just BrainBar's.
        let home = root.appendingPathComponent("home")
        let agents = home.appendingPathComponent("Library/LaunchAgents")
        try FileManager.default.createDirectory(at: agents, withIntermediateDirectories: true)
        let custom = root.appendingPathComponent("elsewhere/alerts.json")
        try FileManager.default.createDirectory(at: custom.deletingLastPathComponent(), withIntermediateDirectories: true)
        let plist: [String: Any] = ["EnvironmentVariables": ["BRAINLAYER_JOB_ALERT_PATH": custom.path]]
        try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
            .write(to: agents.appendingPathComponent("com.brainlayer.observability.plist"))
        let observability = root.appendingPathComponent("observability.json")
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(document(errorType: "job_alert:\(alert)")).write(to: observability)
        _ = try writeAlerts(#"{"maintenance-light": "\#(alert)"}"#) // the default path: must be ignored
        try Data("{}".utf8).write(to: custom)
        guard case let .readable(cleared) = ObservabilityReader.readReconciled(url: observability, environment: [:], home: home) else {
            XCTFail("fixture must be readable")
            return
        }
        XCTAssertNil(cleared.backups.errorType, "the producer's own alert file says the alert cleared")
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

/// Codex #1062 r1 B1: Show log with no log file says so on every surface, never silently reveals
/// a folder. B2: the menu clears exactly like Backups and the Dashboard.
@MainActor
final class BrainBarJobAlertRoundOneTests: XCTestCase {
    private let alert = "BrainLayer light maintenance failed: post-maintenance search latency pathological"
    private let missing = "No maintenance log yet; it's written on the next run."
    private let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent("brainbar-job-alert-r1-\(UUID().uuidString)", isDirectory: true)

    private final class Recorder: BrainBarWorkspaceActing, @unchecked Sendable {
        var opened: [URL] = [], revealed: [URL] = []
        func open(_ url: URL) { opened.append(url) }
        func reveal(_ url: URL) { revealed.append(url) }
        func copy(_ text: String) {}
    }

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("logs"), withIntermediateDirectories: true)
        try Data(#"{"maintenance-light": "\#(alert)"}"#.utf8).write(to: root.appendingPathComponent("job-alerts.json"))
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private var paths: BrainBarBackupSources.Paths {
        .init(launchAgents: root.appendingPathComponent("LA"), databaseLog: root.appendingPathComponent("logs/backup-daily.log"),
              archiveLog: root.appendingPathComponent("logs/jsonl-backup.log"),
              maintenanceLog: root.appendingPathComponent("logs/maintenance.log"),
              snapshotDirectory: root, archiveDirectory: root)
    }

    private func document() -> ObservabilityDocument {
        guard case let .readable(base) = BrainBarDashboardFixture.maintenanceAlertObservabilityResult else {
            fatalError("fixture must be readable")
        }
        let moved = ObservabilityDocument(
            schemaVersion: 1, generatedAt: base.generatedAt, dbPath: root.appendingPathComponent("brainlayer.db").path,
            windowHours: 24, stores: base.stores, emitters: base.emitters, authorUnknown: base.authorUnknown, backups: base.backups
        )
        return moved.replacingBackupsErrorType("job_alert:\(alert)")
    }

    // MARK: B1

    func test_show_log_result_names_a_missing_log_and_opens_an_existing_one() throws {
        let workspace = Recorder()
        XCTAssertEqual(
            BrainBarJobAlerts.showLog(forKey: "maintenance-light", paths: paths, workspace: workspace),
            .missing(message: missing)
        )
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        XCTAssertEqual(
            BrainBarJobAlerts.showLog(forKey: "maintenance-light", paths: paths, workspace: workspace),
            .opened(paths.maintenanceLog)
        )
        XCTAssertEqual(workspace.opened, [paths.maintenanceLog])
        XCTAssertEqual(BrainBarJobAlerts.missingLogMessage(forKey: "backup-daily"), "No database backup log yet; it's written on the next run.")
        XCTAssertEqual(BrainBarJobAlerts.missingLogMessage(forKey: "jsonl-backup"), "No transcript backup log yet; it's written on the next run.")
        XCTAssertEqual(BrainBarJobAlerts.missingLogMessage(forKey: nil), "BrainBar doesn't know this job's log; showing the logs folder.")
    }

    func test_backups_page_shows_the_missing_log_message() throws {
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let workspace = Recorder()
        let model = BrainBarSettingsViewModel(
            store: store, launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false, initialObservabilityResult: .readable(document()),
            backupSources: BrainBarBackupSources(paths: paths, readFile: { _ in nil }, listDirectory: { _ in [] }, isRegularFile: { _ in false }),
            workspace: workspace
        )
        XCTAssertNil(model.jobAlertLogMessage)
        model.showJobAlertLog()
        XCTAssertEqual(model.jobAlertLogMessage, missing)
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        model.showJobAlertLog()
        XCTAssertNil(model.jobAlertLogMessage, "an opened log leaves no message")
        XCTAssertEqual(workspace.opened, [paths.maintenanceLog])
    }

    func test_dashboard_shows_the_missing_log_message() throws {
        let panel = BrainBarDashboardPanelState()
        let workspace = Recorder()
        BrainBarJobAlerts.showLog(for: document(), paths: paths, workspace: workspace, panelState: panel)
        XCTAssertEqual(panel.jobAlertLogMessage(forReason: alert), missing)
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        BrainBarJobAlerts.showLog(for: document(), paths: paths, workspace: workspace, panelState: panel)
        XCTAssertNil(panel.jobAlertLogMessage(forReason: alert))
    }

    func test_menu_says_there_is_no_log_yet_instead_of_offering_show_log() throws {
        let badge = BadgeStatePresentation(badgeOn: true, reason: alert, activeCodes: ["job_alert_maintenance-light"])
        XCTAssertEqual(
            BrainBarStatusPopoverController.showLogItemTitle(for: badge, paths: paths),
            missing
        )
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        XCTAssertEqual(BrainBarStatusPopoverController.showLogItemTitle(for: badge, paths: paths), "Show log")
        XCTAssertNil(BrainBarStatusPopoverController.showLogItemTitle(
            for: .init(badgeOn: false, reason: "", activeCodes: []), paths: paths
        ))
    }

    // MARK: B2

    /// Exactly Codex's fixture: job-alerts.json recovered to {} while badge-state.json still holds
    /// a fresh, failing job_alert_maintenance-light issue.
    func test_the_menu_clears_a_recovered_job_alert_while_the_badge_file_is_still_fresh() throws {
        let now = Date()
        let badgeURL = root.appendingPathComponent("badge-state.json")
        let generated = ISO8601DateFormatter().string(from: now.addingTimeInterval(-30))
        try Data("""
        {"schema_version": 1, "generated_at": "\(generated)", "alerts": {"state": "measured", "reason": "", "inputs": [],
         "badge_on": true, "active": [{"code": "job_alert_maintenance-light", "severity": "critical", "message": "\(alert)"}],
         "suppressed": []}}
        """.utf8).write(to: badgeURL)
        let raw = BadgeStateReader.read(url: badgeURL, now: now, cadence: .known(300))
        XCTAssertTrue(raw.badgeOn, "fixture: the badge file is fresh and failing")
        let alertsURL = root.appendingPathComponent("job-alerts.json")

        let live = raw.reconciled(with: BrainBarJobAlerts.read(url: alertsURL))
        XCTAssertEqual(BrainBarStatusPopoverController.jobAlertKey(for: live), "maintenance-light")

        try Data("{}".utf8).write(to: alertsURL)
        let recovered = raw.reconciled(with: BrainBarJobAlerts.read(url: alertsURL))
        XCTAssertFalse(recovered.badgeOn)
        XCTAssertEqual(recovered.reason, "")
        XCTAssertNil(BrainBarStatusPopoverController.jobAlertKey(for: recovered))
        XCTAssertEqual(BrainBarStatusPopoverController.statusLineTitle(for: recovered), "Nothing needs attention")
        XCTAssertFalse(BrainBarStatusPopoverController.menuRowTitles(for: recovered).contains("Show log"))
    }

    func test_reconciling_the_badge_keeps_other_issues_and_never_guesses_from_an_unknown_file() {
        let raw = BadgeStatePresentation(
            badgeOn: true, reason: "Queue backed up; \(alert)",
            activeCodes: ["queue_backed_up", "job_alert_maintenance-light"],
            activeMessages: ["Queue backed up", alert]
        )
        let cleared = raw.reconciled(with: BrainBarJobAlerts(active: [:]))
        XCTAssertTrue(cleared.badgeOn)
        XCTAssertEqual(cleared.reason, "Queue backed up")
        XCTAssertEqual(cleared.activeCodes, ["queue_backed_up"])
        XCTAssertEqual(raw.reconciled(with: nil), raw, "an unreadable alert file changes nothing")
    }
}

/// Macroscope #1062 (after r1): a Show log note belongs to the alert it was produced for, and the
/// status item follows the menu's reconciled badge.
@MainActor
final class BrainBarJobAlertRoundOneFollowUpTests: XCTestCase {
    private let first = "BrainLayer light maintenance failed: gate A"
    private let second = "BrainLayer full maintenance failed: gate B"
    private let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent("brainbar-job-alert-r1b-\(UUID().uuidString)", isDirectory: true)

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("logs"), withIntermediateDirectories: true)
        try Data(#"{"maintenance-light": "\#(first)", "maintenance-full": "\#(second)"}"#.utf8)
            .write(to: root.appendingPathComponent("job-alerts.json"))
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private var paths: BrainBarBackupSources.Paths {
        .init(launchAgents: root.appendingPathComponent("LA"), databaseLog: root.appendingPathComponent("logs/backup-daily.log"),
              archiveLog: root.appendingPathComponent("logs/jsonl-backup.log"),
              maintenanceLog: root.appendingPathComponent("logs/maintenance.log"),
              snapshotDirectory: root, archiveDirectory: root)
    }

    private func document(_ reason: String) -> ObservabilityDocument {
        guard case let .readable(base) = BrainBarDashboardFixture.maintenanceAlertObservabilityResult else {
            fatalError("fixture must be readable")
        }
        return ObservabilityDocument(
            schemaVersion: 1, generatedAt: base.generatedAt, dbPath: root.appendingPathComponent("brainlayer.db").path,
            windowHours: 24, stores: base.stores, emitters: base.emitters, authorUnknown: base.authorUnknown, backups: base.backups
        ).replacingBackupsErrorType("job_alert:\(reason)")
    }

    func test_the_backups_note_never_outlives_its_alert() async throws {
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let next = document(second)
        let model = BrainBarSettingsViewModel(
            store: store, launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false,
            observabilityURL: root.appendingPathComponent("observability.json"),
            initialObservabilityResult: .readable(document(first)),
            observabilityRead: { _ in .readable(next) },
            backupSources: BrainBarBackupSources(paths: paths, readFile: { _ in nil }, listDirectory: { _ in [] }, isRegularFile: { _ in false })
        )
        // The initial read may already be in flight; pin the first alert, then press Show log.
        model.setObservabilityResultForTesting(.readable(document(first)))
        model.showJobAlertLog()
        XCTAssertEqual(model.jobAlertLogMessage, "No maintenance log yet; it's written on the next run.")
        model.setObservabilityResultForTesting(.readable(next))
        XCTAssertNil(model.jobAlertLogMessage, "a different alert must not inherit the old note")
    }

    func test_the_dashboard_note_never_outlives_its_alert() {
        let panel = BrainBarDashboardPanelState()
        BrainBarJobAlerts.showLog(for: document(first), paths: paths, workspace: BrainBarWorkspaceStub(), panelState: panel)
        XCTAssertEqual(panel.jobAlertLogMessage(forReason: first), "No maintenance log yet; it's written on the next run.")
        XCTAssertNil(panel.jobAlertLogMessage(forReason: second))
    }

    func test_the_status_item_follows_the_menus_reconciled_badge() throws {
        let alertsURL = root.appendingPathComponent("job-alerts.json")
        try Data("{}".utf8).write(to: alertsURL)
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let windowController = BrainBarDashboardPanelController(runtime: runtime)
        let status = BrainBarStatusPopoverController(runtime: runtime, dashboardPanelController: windowController)
        defer { status.stop(); windowController.dismiss() }
        status.setBadgeForTesting(
            .init(badgeOn: true, reason: first, activeCodes: ["job_alert_maintenance-light"], activeMessages: [first]),
            alertsURL: alertsURL
        )
        status.menuNeedsUpdate(status.contextMenuForTesting)
        XCTAssertEqual(status.contextMenuForTesting.items.first?.title, "Nothing needs attention")
        XCTAssertEqual(status.statusItemForTesting.button?.toolTip, "BrainBar", "the icon's tooltip follows the menu")
    }
}

private struct BrainBarWorkspaceStub: BrainBarWorkspaceActing {
    func reveal(_ url: URL) {}
    func copy(_ text: String) {}
}
