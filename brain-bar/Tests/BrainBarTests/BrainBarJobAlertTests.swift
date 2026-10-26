import XCTest
@testable import BrainBar

/// Lead ruling 2026-10-04: Etan saw "BrainLayer light maintenance failed; check the maintenance
/// log" four times. Each alert shows once per screen, carries its own words, offers Show log, and
/// a clean later run clears it everywhere.
@MainActor
final class BrainBarJobAlertTests: XCTestCase {
    private let alert = "BrainLayer light maintenance failed; check the maintenance log"
    private let root = URL(fileURLWithPath: brainBarTestTemporaryPath(), isDirectory: true)
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
        XCTAssertNil(ObservabilityPresentation.backupStatus(for: reconciled.backups, now: reconciled.generatedAt).attentionLine)
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

    func testCalendarStepsHaveLogLinksAndKnownMissingLogMessages() {
        let home = root.appendingPathComponent("home")
        let live = BrainBarBackupSources.Paths.live(databasePath: "/db/brainlayer.db", environment: [:], home: home)
        for key in ["index", "backup-snapshot", "p0-counter", "wal-checkpoint", "decay", "repair-fts"] {
            XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: key, paths: live),
                home.appendingPathComponent("Library/Logs/brainlayer/\(key).err.log"), key)
            XCTAssertFalse(BrainBarJobAlerts.missingLogMessage(forKey: key).contains("doesn't know"), key)
        }
        for key in ["maintenance-nightly", "maintenance-weekly", "maintenance-light", "maintenance-full"] {
            XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: key, paths: live), live.maintenanceLog, key)
        }
        XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: "backup-daily", paths: live), live.databaseLog)
        XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: "jsonl-backup", paths: live), live.archiveLog)
    }

    private var paths: BrainBarBackupSources.Paths {
        .init(
            launchAgents: root, databaseLog: root.appendingPathComponent("logs/backup-daily.log"),
            archiveLog: root.appendingPathComponent("logs/jsonl-backup.log"),
            maintenanceLog: root.appendingPathComponent("logs/maintenance.log"),
            snapshotDirectory: root, archiveDirectory: root
        )
    }

    func testCalendarLogLinksUseConfiguredPlistOrSupervisorArguments() throws {
        let home = root.appendingPathComponent("home")
        let plistURL = home.appendingPathComponent("Library/LaunchAgents/com.brainlayer.maintenance.plist")
        let jobsURL = home.appendingPathComponent(".local/share/brainlayer/supervisor/jobs.json")
        let custom = root.appendingPathComponent("custom calendar logs")
        let plist: [String: Any] = ["ProgramArguments": ["brainlayer", "maintenance-calendar", "--logs-dir", custom.path]]
        let cases: [(URL, Data)] = [
            (plistURL, try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)),
            (jobsURL, try JSONSerialization.data(withJSONObject: ["com.brainlayer.maintenance": ["plist": plist]])),
        ]
        for (source, data) in cases {
            let live = BrainBarBackupSources.Paths.live(databasePath: "/db/brainlayer.db", environment: [:], home: home,
                readFile: { $0 == source ? data : nil })
            for key in ["index", "backup-snapshot", "p0-counter", "wal-checkpoint", "decay", "repair-fts"] {
                let expected = custom.appendingPathComponent("\(key).err.log")
                XCTAssertEqual(BrainBarJobAlerts.logURL(forKey: key, paths: live), expected, source.path)
                let workspace = BrainBarRecordingWorkspace()
                XCTAssertEqual(BrainBarJobAlerts.showLog(forKey: key, paths: live, workspace: workspace,
                    fileExists: { $0 == expected }), .opened(expected))
                XCTAssertEqual(workspace.recorded.map(\.1), [expected.path])
            }
        }
    }

    func testCalendarLogDirectoryFallsBackWithoutAValidArgument() throws {
        let home = root.appendingPathComponent("home")
        for args in [["brainlayer", "maintenance-calendar"], ["brainlayer", "--logs-dir"], ["brainlayer", "--logs-dir", ""]] {
            let data = try PropertyListSerialization.data(fromPropertyList: ["ProgramArguments": args], format: .xml, options: 0)
            let live = BrainBarBackupSources.Paths.live(databasePath: "/db/brainlayer.db", environment: [:], home: home,
                readFile: { $0.lastPathComponent == "com.brainlayer.maintenance.plist" ? data : nil })
            XCTAssertEqual(live.calendarLogDirectory, home.appendingPathComponent("Library/Logs/brainlayer"))
        }
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

    func test_show_log_opens_an_existing_log_or_the_logs_folder() throws {
        final class Recorder: BrainBarWorkspaceActing, @unchecked Sendable {
            var opened: [URL] = [], revealed: [URL] = []
            func open(_ url: URL) { opened.append(url) }
            func openLog(_ url: URL) -> Bool { open(url); return true }
            func reveal(_ url: URL) { revealed.append(url) }
            func copy(_ text: String) {}
        }
        let workspace = Recorder()
        BrainBarJobAlerts.showLog(forKey: "maintenance-light", paths: paths, workspace: workspace)
        XCTAssertEqual(workspace.opened.map(\.path), [root.appendingPathComponent("logs").path], "no log yet: open where it will be")
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        BrainBarJobAlerts.showLog(forKey: "maintenance-light", paths: paths, workspace: workspace)
        XCTAssertEqual(workspace.opened.last, paths.maintenanceLog)
        BrainBarJobAlerts.showLog(forKey: nil, paths: paths, workspace: workspace)
        XCTAssertEqual(workspace.opened.last?.path, root.appendingPathComponent("logs").path, "an unknown job opens the logs folder")
    }

    // MARK: the menu

    func test_the_menu_offers_show_log_only_while_a_job_alert_is_active() {
        let on = BadgeStatePresentation(badgeOn: true, reason: alert, activeCodes: ["queue_backed_up", "job_alert_maintenance-light"])
        XCTAssertEqual(BrainBarStatusPopoverController.jobAlertKey(for: on), "maintenance-light")
        XCTAssertEqual(BrainBarStatusPopoverController.menuRowTitles(for: on).prefix(2), ["Needs attention: BrainLayer light maintenance failed;…", "Show log"])
        let off = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        XCTAssertNil(BrainBarStatusPopoverController.jobAlertKey(for: off))
        XCTAssertFalse(BrainBarStatusPopoverController.menuRowTitles(for: off).contains("Show log"))
    }
    // MARK: S5 — a NEW alert key reaches every surface even when badge-state.json lacks it

    func test_a_new_alert_key_absent_from_the_badge_reaches_icon_menu_dashboard_jobs_and_backups() throws {
        let now = BrainBarStatusSnapshotTests.now
        let inputs = BrainBarStatusSnapshotTests.healthy()
        guard case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else {
            return XCTFail("fixture must be readable")
        }
        let document = base.replacingBackupsForStatusTests(BrainBarStatusSnapshotTests.backups(), generatedAt: now)
        var config = BrainLayerConfig.defaultConfig
        for job in BrainLayerLaunchdJob.allCases { config.launchdJobs[job] = .init(enabled: true, loadState: .loaded) }
        let model = BrainBarSettingsViewModel(
            workspace: BrainBarRecordingWorkspace(),
            store: BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/fixture/config"),
                                         loadDocumentOverride: { BrainLayerEnvDocument(config: config) }),
            initialLaunchdObservations: inputs.observations!, refreshStatusOnLoad: false, now: { now },
            initialObservabilityResult: .readable(document), observabilityCadence: .known(300),
            initialWatcherHealth: .readable(.init(updatedAt: now.addingTimeInterval(-30), pollCount: 50)),
            initialMaintenanceEvidence: inputs.maintenance)
        let reason = "Local DB snapshot invalid; upload abandoned"
        let quietBadge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        model.applyHealthBadge(quietBadge, alerts: BrainBarJobAlerts(active: [:]))
        XCTAssertFalse(model.statusSnapshot(drive: nil).needsAttention, "\(model.statusSnapshot(drive: nil).issues)")

        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        runtime.jobsModel = model
        runtime.driveAuth.setForPreview(status: .init(state: .valid, reason: nil, expiresAt: now.addingTimeInterval(30 * 86_400)))
        let workspace = BrainBarRecordingWorkspace()
        let panel = BrainBarDashboardPanelController(workspace: workspace, runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime,
            dashboardPanelController: panel, statusBar: TestStatusItemHost())
        defer { controller.stop() }
        controller.setBadgeForTesting(quietBadge, alertsURL: try writeAlerts(#"{"backup-upload-abandoned": "\#(reason)"}"#))
        controller.menuNeedsUpdate(controller.contextMenuForTesting)

        let snapshot = model.statusSnapshot(drive: nil)
        XCTAssertTrue(snapshot.needsAttention, "\(snapshot.issues)")
        XCTAssertTrue(controller.statusDotForTesting, "icon")
        XCTAssertEqual(controller.contextMenuForTesting.items.first?.title, "Needs attention: \(reason)", "menu")
        XCTAssertEqual(controller.statusItemForTesting.button?.toolTip, "BrainBar — needs attention: \(reason)")
        XCTAssertEqual(model.groupStatus(.backups).health, .unhealthy, "Jobs")
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .attention, "Backups")
        XCTAssertEqual(snapshot.attention.map(\.text), [reason], "Dashboard")

        // One clean publication removes the same issue everywhere.
        controller.setBadgeForTesting(quietBadge, alertsURL: try writeAlerts("{}"))
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        XCTAssertFalse(model.statusSnapshot(drive: nil).needsAttention)
        XCTAssertFalse(controller.statusDotForTesting)
        XCTAssertEqual(controller.contextMenuForTesting.items.first?.title, "Nothing needs attention")
        XCTAssertNotEqual(model.groupStatus(.backups).health, .unhealthy)
        XCTAssertNotEqual(model.backupsHealth(drive: nil).badge, .attention)
    }

    func test_an_informational_alert_key_is_visible_text_but_not_a_dot() {
        var inputs = BrainBarStatusSnapshotTests.healthy()
        inputs.jobAlerts = BrainBarJobAlerts(active: ["fleet-watchdog-hold": "BrainLayer recovered a stale fleet-watchdog hold; supervisor restored"])
        let (_, surfaces) = BrainBarStatusSnapshotTests.assertAgree(inputs, dot: false, "watchdog recovery")
        XCTAssertEqual(surfaces.menu.hasPrefix("Info:"), true, surfaces.menu)
    }

}

/// The Backups page shows a job alert once: one alert card with Show log. The group badge keeps
/// its "Needs attention" verdict without repeating the sentence, and the status rows drop it.
@MainActor
final class BrainBarBackupsJobAlertPageTests: XCTestCase {
    private let alert = "BrainLayer light maintenance failed; check the maintenance log"

    private func viewModel(_ result: ObservabilityReadResult) throws -> BrainBarSettingsViewModel {
        let now: Date
        switch result {
        case .readable(let document): now = document.generatedAt
        case .unreadable: now = BrainBarDashboardFixture.fetchedAt
        }
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath(), isDirectory: true)
            .appendingPathComponent("brainbar-job-alert-page-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let ok = BrainLayerLaunchdJobObservation(loadState: .loaded, runs: 1, lastExitCode: 0,
                                                 lastRunAt: now, nextRunAt: now, isContinuous: false)
        return BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            initialLaunchdObservations: [.backupDaily: ok, .jsonlBackup: ok],
            refreshStatusOnLoad: false,
            now: { now },
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
                           lastVerifiedUpload: b.lastVerifiedUpload,
                           dbSnapshot: .init(lastAt: doc.generatedAt.addingTimeInterval(-48 * 3600),
                                             destination: "fixture.db.zst", verified: true), launchd: b.launchd)
        )
        let model = try viewModel(.readable(stale))
        XCTAssertNil(model.jobAlert)
        // Lead UX r1 on #1064: the Recovery checks card already says it; the header shows it once.
        XCTAssertEqual(model.backupsHealth(drive: nil).reason, "Database copy: Last offsite DB copy is 2 days old")
        XCTAssertNil(model.backupsBadgeReason(drive: nil))
    }
}

/// Codex #1062 r1 B1: Show log with no log file says so on every surface, never silently reveals
/// a folder. B2: the menu clears exactly like Backups and the Dashboard.
@MainActor
final class BrainBarJobAlertRoundOneTests: XCTestCase {
    private let alert = "BrainLayer light maintenance failed: post-maintenance search latency pathological"
    private let missing = "No maintenance log yet; it's written on the next run. Opened the logs folder."
    private let root = URL(fileURLWithPath: brainBarTestTemporaryPath(), isDirectory: true)
        .appendingPathComponent("brainbar-job-alert-r1-\(UUID().uuidString)", isDirectory: true)

    private final class Recorder: BrainBarWorkspaceActing, @unchecked Sendable {
        var opened: [URL] = [], revealed: [URL] = []
        func open(_ url: URL) { opened.append(url) }
        func reveal(_ url: URL) { revealed.append(url) }
        var acceptsOpen = true
        func openLog(_ url: URL) -> Bool { opened.append(url); return acceptsOpen }
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

    func test_D1_unknown_and_nil_job_open_the_existing_logs_folder() {
        for key: String? in [nil, "unknown-job"] {
            let workspace = Recorder()
            let result = BrainBarJobAlerts.showLog(forKey: key, paths: paths, workspace: workspace)
            XCTAssertEqual(workspace.opened, [paths.maintenanceLog.deletingLastPathComponent()])
            XCTAssertTrue(result.message?.contains("Opened the logs folder") == true)
        }
    }

    func test_D1_missing_folder_never_claims_it_opened() throws {
        try FileManager.default.removeItem(at: paths.maintenanceLog.deletingLastPathComponent())
        let workspace = Recorder()
        let result = BrainBarJobAlerts.showLog(forKey: nil, paths: paths, workspace: workspace)
        XCTAssertTrue(workspace.opened.isEmpty)
        XCTAssertTrue(workspace.revealed.isEmpty)
        XCTAssertTrue(result.message?.contains("logs folder does not exist") == true)
        XCTAssertFalse(result.message?.contains("showing") == true)
    }

    func test_D1_workspace_refusal_is_reported_for_both_file_and_folder() throws {
        for hasLog in [false, true] {
            if hasLog { try Data("log".utf8).write(to: paths.maintenanceLog) }
            let workspace = Recorder()
            workspace.acceptsOpen = false
            let result = BrainBarJobAlerts.showLog(forKey: "maintenance-light", paths: paths, workspace: workspace)
            XCTAssertTrue(result.message?.contains("Could not open") == true)
            XCTAssertNotEqual(result, .opened(paths.maintenanceLog))
        }
    }

    func test_D1_missing_known_log_opens_its_own_parent_directory() throws {
        try FileManager.default.createDirectory(at: paths.calendarLogDirectory, withIntermediateDirectories: true)
        let workspace = Recorder()
        let result = BrainBarJobAlerts.showLog(forKey: "index", paths: paths, workspace: workspace,
            fileExists: { $0 == self.paths.calendarLogDirectory })
        XCTAssertEqual(workspace.opened, [paths.calendarLogDirectory])
        XCTAssertTrue(result.message?.contains("No index log yet") == true)
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
        XCTAssertEqual(workspace.opened, [paths.maintenanceLog.deletingLastPathComponent(), paths.maintenanceLog])
        XCTAssertEqual(BrainBarJobAlerts.missingLogMessage(forKey: "backup-daily"), "No database backup log yet; it's written on the next run.")
        XCTAssertEqual(BrainBarJobAlerts.missingLogMessage(forKey: "jsonl-backup"), "No transcript backup log yet; it's written on the next run.")
        XCTAssertEqual(BrainBarJobAlerts.missingLogMessage(forKey: nil), "BrainBar doesn't know this job's log.")
    }

    func test_backups_page_shows_the_missing_log_message() throws {
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let workspace = Recorder()
        let model = BrainBarSettingsViewModel(workspace: workspace, store: store, launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false, initialObservabilityResult: .readable(document()),
            backupSources: BrainBarBackupSources(paths: paths, readFile: { _ in nil }, listDirectory: { _ in [] }, isRegularFile: { _ in false }))
        XCTAssertNil(model.jobAlertLogMessage)
        model.showJobAlertLog()
        XCTAssertEqual(model.jobAlertLogMessage, missing)
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        model.showJobAlertLog()
        XCTAssertNil(model.jobAlertLogMessage, "an opened log leaves no message")
        XCTAssertEqual(workspace.opened, [paths.maintenanceLog.deletingLastPathComponent(), paths.maintenanceLog])
        workspace.acceptsOpen = false
        model.showJobAlertLog()
        XCTAssertTrue(model.jobAlertLogMessage?.contains("Could not open log") == true)
    }

    func test_dashboard_shows_the_missing_log_message() throws {
        let panel = BrainBarDashboardPanelState()
        let workspace = Recorder()
        BrainBarJobAlerts.showLog(for: document(), paths: paths, workspace: workspace, panelState: panel)
        XCTAssertEqual(panel.jobAlertLogMessage(forReason: alert), missing)
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        BrainBarJobAlerts.showLog(for: document(), paths: paths, workspace: workspace, panelState: panel)
        XCTAssertNil(panel.jobAlertLogMessage(forReason: alert))
        workspace.acceptsOpen = false
        BrainBarJobAlerts.showLog(for: document(), paths: paths, workspace: workspace, panelState: panel)
        XCTAssertTrue(panel.jobAlertLogMessage(forReason: alert)?.contains("Could not open log") == true)
    }

    func test_menu_offers_a_folder_action_until_the_log_exists() throws {
        let badge = BadgeStatePresentation(badgeOn: true, reason: alert, activeCodes: ["job_alert_maintenance-light"])
        XCTAssertEqual(
            BrainBarStatusPopoverController.showLogItemTitle(for: badge, paths: paths),
            "Open logs folder"
        )
        try Data("{}\n".utf8).write(to: paths.maintenanceLog)
        XCTAssertEqual(BrainBarStatusPopoverController.showLogItemTitle(for: badge, paths: paths), "Show log")
        XCTAssertNil(BrainBarStatusPopoverController.showLogItemTitle(
            for: .init(badgeOn: false, reason: "", activeCodes: []), paths: paths
        ))
    }

    func test_transport_job_alert_is_visible_and_clears_on_verified_recovery() {
        let reason = "BrainBar MCP transport: 3 dropped requests"
        let badge = BadgeStatePresentation(badgeOn: true, reason: reason,
            activeCodes: ["job_alert_mcp-transport"], activeMessages: [reason])
        let live = badge.reconciled(with: BrainBarJobAlerts(active: ["mcp-transport": reason]))
        XCTAssertTrue(live.badgeOn)
        XCTAssertEqual(BrainBarStatusPopoverController.jobAlertKey(for: live), "mcp-transport")
        XCTAssertEqual(live.reason, reason)
        XCTAssertTrue(badge.reconciled(with: nil).badgeOn)
        XCTAssertFalse(badge.reconciled(with: BrainBarJobAlerts(active: [:])).badgeOn)
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

    /// Macroscope #1062: a job that re-reports with a new reason (drive-consent does) shows the live
    /// reason, never the badge file's older copy.
    func test_reconciling_the_badge_shows_the_jobs_current_reason() {
        let raw = BadgeStatePresentation(
            badgeOn: true, reason: "Drive consent expires in 2 days",
            activeCodes: ["job_alert_drive-consent"], activeMessages: ["Drive consent expires in 2 days"]
        )
        let live = raw.reconciled(with: BrainBarJobAlerts(active: ["drive-consent": "Drive consent expired; reconnect"]))
        XCTAssertTrue(live.badgeOn)
        XCTAssertEqual(live.reason, "Drive consent expired; reconnect")
        XCTAssertEqual(live.activeMessages, ["Drive consent expired; reconnect"])
        XCTAssertEqual(live.activeCodes, ["job_alert_drive-consent"])
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
    private let root = URL(fileURLWithPath: brainBarTestTemporaryPath(), isDirectory: true)
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
        let workspace = BrainBarRecordingWorkspace()
        let model = BrainBarSettingsViewModel(workspace: workspace, store: store, launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false,
            observabilityURL: root.appendingPathComponent("observability.json"),
            initialObservabilityResult: .readable(document(first)),
            observabilityRead: { _ in .readable(next) },
            backupSources: BrainBarBackupSources(paths: paths, readFile: { _ in nil }, listDirectory: { _ in [] }, isRegularFile: { _ in false })
        )
        // The initial read may already be in flight; pin the first alert, then press Show log.
        model.setObservabilityResultForTesting(.readable(document(first)))
        model.showJobAlertLog()
        XCTAssertEqual(model.jobAlertLogMessage, "No maintenance log yet; it's written on the next run. Opened the logs folder.")
        // Missing logs open the existing folder and return an acknowledged note. Before injection,
        // this test reached Finder; teardown then removed the folder under the open window.
        XCTAssertEqual(workspace.recorded.map(\.0), ["open"])
        XCTAssertEqual(workspace.recorded.map(\.1), [root.appendingPathComponent("logs").path])
        model.setObservabilityResultForTesting(.readable(next))
        XCTAssertNil(model.jobAlertLogMessage, "a different alert must not inherit the old note")
    }

    func test_the_dashboard_note_never_outlives_its_alert() {
        let panel = BrainBarDashboardPanelState()
        BrainBarJobAlerts.showLog(for: document(first), paths: paths, workspace: BrainBarWorkspaceStub(), panelState: panel)
        XCTAssertEqual(panel.jobAlertLogMessage(forReason: first), "No maintenance log yet; it's written on the next run. Opened the logs folder.")
        XCTAssertNil(panel.jobAlertLogMessage(forReason: second))
    }

    func test_the_status_item_follows_the_menus_reconciled_badge() throws {
        let alertsURL = root.appendingPathComponent("job-alerts.json")
        try Data("{}".utf8).write(to: alertsURL)
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let windowController = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let status = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime, dashboardPanelController: windowController, statusBar: TestStatusItemHost()
        )
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
    func openLog(_ url: URL) -> Bool { true }
    func reveal(_ url: URL) {}
    func copy(_ text: String) {}
}
