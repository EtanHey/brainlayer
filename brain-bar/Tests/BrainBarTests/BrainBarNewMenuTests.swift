import AppKit
import XCTest
@testable import BrainBar

/// Never presents a menu, creates a system status item or activates a window.
@MainActor final class BrainBarNewMenuTests: XCTestCase {
    private func defaults() -> UserDefaults {
        let name = "BrainBarNewMenuTests.\(UUID().uuidString)"
        addTeardownBlock { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
        return UserDefaults(suiteName: name)!
    }

    func testPreferenceIsOffAndReReadsChanges() {
        let d = defaults()
        XCTAssertFalse(BrainBarNewMenuPreference.isEnabled(in: d))
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        XCTAssertTrue(BrainBarNewMenuPreference.isEnabled(in: d))
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        XCTAssertFalse(BrainBarNewMenuPreference.isEnabled(in: d))
    }

    func testActionsDismissBeforeCallingTheirTarget() {
        var calls: [String] = []
        let actions = BrainBarLiveMenuActions(settings: { calls.append("settings") }, storage: { calls.append("backups") },
            quit: { calls.append("quit") }, restart: { calls.append("restart") }, reconnect: { calls.append("drive.reconnect") },
            copy: { calls.append("pasteboard") }, dismiss: { calls.append("dismiss") })
        let cases: [(BrainBarMenuAction, String)] = [(.settings, "settings"), (.quit, "quit"), (.restart, "restart"),
            (.row(.connectDrive), "drive.reconnect"), (.row(.reconnectDrive), "drive.reconnect"),
            (.row(.openStorageSettings), "backups"), (.row(.copyReport), "pasteboard")]
        for (action, target) in cases {
            calls = []
            let row: BrainBarMenuPresentation.Row
            switch action {
            case let .row(kind): row = .init(kind: .needsYou, strings: [target], action: .init(kind: kind, label: target))
            case .settings: row = .init(kind: .standardItem, strings: [target], keyEquivalent: "⌘,")
            case .quit, .restart: row = .init(kind: .standardItem, strings: [target], keyEquivalent: "⌘Q", optionAlternate: "Restart")
            }
            let p = BrainBarMenuPresentation(form: .needs, headline: target, count: 1, icon: .marked, rows: [row])
            let builder = BrainBarMenuBuilder(actions: actions, accessibility: .init())
            let menu = builder.build(p)
            menu.performActionForItem(at: action == .restart ? 1 : 0)
            XCTAssertEqual(calls, ["dismiss", target])
        }
    }

    func testLiveFixturesBuildExactlyThePresentation() throws {
        for fixture in try BrainBarMenuPresentationTests.loadCases().prefix(3) {
            let c = try BrainBarMenuCases.capture(fixture["inputs"] as! [String: Any])
            let p = BrainBarMenuPresenter.present(c)
            let builder = BrainBarMenuBuilder(accessibility: .init())
            let menu = builder.build(p)
            assertMenu(menu, equals: p)
            XCTAssertTrue(menu.items.last?.isAlternate == true)
            XCTAssertEqual(menu.items.last?.representedObject as? BrainBarMenuAction, .restart)
        }
    }

    private func assertMenu(_ menu: NSMenu, equals p: BrainBarMenuPresentation) {
        let rows = menu.items.filter { !$0.isSeparatorItem && !$0.isAlternate }
        let expected = p.rows.filter { $0.kind != .separator }
        XCTAssertEqual(rows.count, expected.count)
        for (item, row) in zip(rows, expected) {
            if let host = item.view as? BrainBarMenuRowHost {
                XCTAssertEqual(host.accessibilityLabel(), row.strings.joined(separator: " "))
            } else { XCTAssertEqual(item.title, row.strings.first) }
        }
    }

    func testReportExactShape() {
        let now = Date(timeIntervalSince1970: 0.0)
        let p = BrainBarMenuPresentation(form: .needs, headline: "Needs attention", count: 1, icon: .marked,
            rows: [.init(kind: .headline, strings: ["Needs attention"]), .init(kind: .separator),
                   .init(kind: .needsYou, strings: ["Drive needs you"], action: .init(kind: .reconnectDrive, label: "Reconnect Google Drive"))])
        let v = BrainBarHealthVerdict(dot: .red, checks: [.init(name: "drive_auth", state: .red, note: "expired")], notes: [], evaluatedAt: now)
        XCTAssertEqual(BrainBarMenuReport.text(presentation: p, verdict: v, version: "1.2.3", now: now), """
        BrainBar report
        Version: 1.2.3
        Time: 1970-01-01T00:00:00Z

        Menu:
        Needs attention
        Drive needs you
        Reconnect Google Drive

        Checks:
        drive_auth: red — expired

        """)
    }

    func testModelSamplesAwakeClocksCachedInputsAndDamagedHistory() throws {
        let now = BrainBarStatusSnapshotTests.now
        let evaluated = now.addingTimeInterval(-3620.0)
        let history = BrainBarSleepHistory(uptime: { 0.0 })
        history.sleeping(at: evaluated.addingTimeInterval(10.0))
        history.woke(at: evaluated.addingTimeInterval(3610.0))
        let verdict = BrainBarHealthVerdict(dot: .green,
            checks: [.init(name: "stores", state: .green, note: "ok")], notes: [], evaluatedAt: evaluated)
        for damaged in [false, true] {
            var night = BrainBarStatusSnapshotTests.step()
            night.calendarHistoryDamaged = damaged
            night.calendarNightOutcome = .init(scheduledDate: "fixture", at: now, failed: true,
                abandoned: false, consecutiveFailures: 3)
            let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(),
                store: .init(loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
                initialLaunchdObservations: [.backupDaily: BrainBarStatusSnapshotTests.step(consecutiveFailures: 2),
                    .jsonlBackup: BrainBarStatusSnapshotTests.step(), .maintenanceNightly: night],
                refreshStatusOnLoad: false, now: { now },
                initialWatcherHealth: .readable(.init(updatedAt: evaluated, pollCount: 1, alertReasons: [], dbProbeFailed: false)),
                awakeClock: { 10.0 }, sleepHistory: history, statusDeadlineScheduler: { _, _ in })
            let drive = BrainBarDriveAuthModel(runner: nil)
            drive.setForPreview(status: .init(state: .missing, reason: nil, expiresAt: nil))
            model.attachDriveAuth(drive)
            model.applyHealthBadge(.init(badgeOn: false, reason: "", activeCodes: [], healthVerdict: verdict), alerts: nil)
            let stats = DashboardStats(chunkCount: 100, enrichedChunkCount: 0, pendingEnrichmentCount: 0,
                enrichmentPercent: 0.0, enrichmentRatePerMinute: 0.0, databaseSizeBytes: 0,
                recentActivityBuckets: [], recentEnrichmentBuckets: [], ftsEligibleChunkCount: 100,
                ftsIndexedChunkCount: 75, signalCoverageIsAvailable: true)
            let c = model.menuCapture(stats: stats, timeZone: TimeZone(secondsFromGMT: 7200)!)
            XCTAssertEqual(c.verdict, verdict)
            XCTAssertEqual(c.verdictAwakeAgeS, 20.0)
            XCTAssertEqual(c.watcher?.fresh, true)
            XCTAssertEqual(c.driveAuthModel, .missing)
            XCTAssertEqual(c.keywordCoverage, 0.75)
            XCTAssertFalse(c.backupsMeasured)
            XCTAssertEqual(c.backupDailyFailures, 2)
            XCTAssertEqual(c.jsonlBackupFailures, 0)
            XCTAssertEqual(c.maintenanceNightFailures, damaged ? nil : 3)
            XCTAssertEqual(c.now, now)
            XCTAssertEqual(c.timeZone.secondsFromGMT(), 7200)
            let d = defaults(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
            runtime.jobsModel = model
            d.set(true, forKey: BrainBarNewMenuPreference.key)
            let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
                dashboardPanelController: nil, statusBar: Host(), menuDefaults: d)
            controller.menuNeedsUpdate(controller.contextMenuForTesting)
            assertMenu(controller.contextMenuForTesting, equals: BrainBarMenuPresenter.present(model.menuCapture(stats: nil)))
            controller.stop()
            model.systemDidWake()
            XCTAssertTrue(model.menuCapture(stats: nil).wakeCatchUpPending)
        }
    }

    func testReportRedactsPersonalDiagnosticIdentifiers() {
        let p = BrainBarMenuPresentation(form: .calm, headline: "", count: 0, icon: .resting, rows: [])
        let v = BrainBarHealthVerdict(dot: .red,
            checks: [.init(name: "disk", state: .red, note: "/home/fixture/db a@example.com")], notes: [], evaluatedAt: .distantPast)
        let report = BrainBarMenuReport.text(presentation: p, verdict: v, version: "test", now: Date(timeIntervalSince1970: 0.0))
        XCTAssertTrue(report.contains("disk: red — /home/[user]/db [email]"))
    }

    private final class Item: BrainBarStatusItem {
        let length: CGFloat = NSStatusItem.variableLength
        var menu: NSMenu?
        var button: NSButton? { nil }
    }
    private final class Host: BrainBarStatusItemHosting {
        let item = Item()
        func statusItem(withLength: CGFloat) -> any BrainBarStatusItem { item }
        func removeStatusItem(_ item: any BrainBarStatusItem) {}
    }

    func testSwitchOffBuildsMainsItemsAndRestoresThemAfterOn() {
        let d = defaults(), host = Host(), workspace = BrainBarRecordingWorkspace()
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let controller = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime,
            dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        let menu = controller.contextMenuForTesting
        controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []),
            alertsURL: URL(fileURLWithPath: "/tmp/w81-absent-\(UUID().uuidString).json"))
        controller.menuNeedsUpdate(menu)
        // Item-by-item inventory from main 151a715d configureContextMenu + menuNeedsUpdate, without a model.
        let expected = ["Nothing needs attention", "Show log", "", "", "Open Dashboard", "Settings…", "", "Restart BrainBar", "", "Quit BrainBar"]
        XCTAssertEqual(menu.items.map(\.title), expected)
        XCTAssertEqual(menu.items.map { $0.action.map(NSStringFromSelector) },
            [nil, "showJobAlertLog:", nil, nil, "openDashboard:", "openSettings:", nil, "restartBrainBar:", nil, "quitBrainBar:"])
        XCTAssertEqual(menu.items.map(\.isHidden), [false, true, true, false, false, false, false, false, false, false])
        XCTAssertEqual(menu.items.map(\.isSeparatorItem), [false, false, false, true, false, false, true, false, true, false])
        XCTAssertTrue(menu.items.allSatisfy { $0.keyEquivalent.isEmpty })
        let originals = menu.items
        let shape: (NSMenuItem) -> String = { item in
            "\(item.title)|\(item.isEnabled)|\(item.isHidden)|\(item.isSeparatorItem)|\(item.keyEquivalent)|\(item.action.map(NSStringFromSelector) ?? "")"
        }
        let before = originals.map(shape)
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        controller.menuNeedsUpdate(menu)
        XCTAssertNotEqual(menu.items.map(\.title), expected)
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        controller.menuNeedsUpdate(menu)
        XCTAssertEqual(menu.items.map(shape), before)
        for (actual, original) in zip(menu.items, originals) { XCTAssertTrue(actual === original) }
    }
}
