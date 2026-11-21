import AppKit
import XCTest
@testable import BrainBar

/// Gap 26, lead ruling 21 Nov 2026: Classic keeps main's recorded menu plus this one unmarked item.
@MainActor func brainBarItemsWithoutApprovedDiagnostics(_ menu: NSMenu, file: StaticString = #filePath,
                                                       line: UInt = #line) throws -> [NSMenuItem] {
    let matches = menu.items.indices.filter { menu.items[$0].title == BrainBarMenuPresenter.SurfaceSay.diagnostics }
    XCTAssertEqual(matches.count, 1, file: file, line: line)
    let index = try XCTUnwrap(matches.first, file: file, line: line)
    let settings = try XCTUnwrap(menu.items.firstIndex { $0.title == BrainBarMenuPresenter.Say.settings.text }, file: file, line: line)
    XCTAssertEqual(index, settings + 1, file: file, line: line)
    let item = menu.items[index]
    XCTAssertTrue(item.isEnabled, file: file, line: line)
    XCTAssertFalse(item.isHidden, file: file, line: line)
    XCTAssertEqual(item.keyEquivalent, "", file: file, line: line)
    XCTAssertEqual(item.state, .off, file: file, line: line)
    XCTAssertNil(item.image, file: file, line: line)
    XCTAssertNil(item.view, file: file, line: line)
    return menu.items.enumerated().filter { $0.offset != index }.map(\.element)
}

/// Never presents a menu, creates a system status item or activates a window.
@MainActor final class BrainBarNewMenuTests: XCTestCase {
    private func defaults() -> UserDefaults {
        let name = "BrainBarNewMenuTests.\(UUID().uuidString)"
        addTeardownBlock { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
        return UserDefaults(suiteName: name)!
    }

    func testPreferenceIsOnAndReReadsChanges() {
        let d = defaults()
        XCTAssertTrue(BrainBarNewMenuPreference.isEnabled(in: d))
        XCTAssertNil(d.object(forKey: BrainBarNewMenuPreference.key))
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        XCTAssertTrue(BrainBarNewMenuPreference.isEnabled(in: d))
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        XCTAssertFalse(BrainBarNewMenuPreference.isEnabled(in: d))
    }

    func testActionsDismissBeforeCallingTheirTarget() async {
        // Standard menu-item dispatch needs an application even when no window is created.
        NSApplication.shared.setActivationPolicy(.prohibited)
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
            XCTAssertEqual(calls, ["dismiss"], "No target runs inside tracking teardown")
            await nextMainTurn()
            XCTAssertEqual(calls, ["dismiss", target])
        }
    }

    private func nextMainTurn() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.main.async { continuation.resume() }
        }
    }

    func testAllThreeDropdownsReachTheInjectedDiagnosticsOpener() async throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        let builder = NDMenuBuilder(store: store, perform: { _ in }, openWindow: { _ in })
        let openedNew = expectation(description: "New Diagnostics")
        builder.diagnosticsOpener = { openedNew.fulfill() }
        let newMenu = builder.build()
        _ = try brainBarItemsWithoutApprovedDiagnostics(newMenu)
        newMenu.performActionForItem(at: try XCTUnwrap(newMenu.items.firstIndex { $0.title == "Diagnostics" }))
        await fulfillment(of: [openedNew], timeout: 2)

        let d = defaults()
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(),
            runtime: BrainBarRuntime(launchMode: .menuItemDaemon), dashboardPanelController: nil,
            statusBar: Host(), menuDefaults: d)
        defer { controller.stop() }
        controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []),
            alertsURL: URL(fileURLWithPath: brainBarTestTemporaryPath()).appendingPathComponent("absent-diagnostics.json"))
        let menu = controller.contextMenuForTesting
        for legacy in [true, false] {
            let opened = expectation(description: legacy ? "Classic legacy Diagnostics" : "Classic presenter Diagnostics")
            controller.setDiagnosticsOpenerForTesting { opened.fulfill() }
            if legacy { controller.menuNeedsUpdate(menu) }
            else { controller.populateLiveMenuForTesting(BrainBarMenuPresenter.present(store.inputs.capture)) }
            _ = try brainBarItemsWithoutApprovedDiagnostics(menu)
            menu.performActionForItem(at: try XCTUnwrap(menu.items.firstIndex { $0.title == "Diagnostics" }))
            await fulfillment(of: [opened], timeout: 2)
        }
        // None of the injected open actions construct a Diagnostics window or start New's store in Classic.
        withExtendedLifetime(builder) {}
    }

    private final class Runner: BrainLayerCLIRunning, @unchecked Sendable {
        let lock = NSLock()
        private var recorded: [[String]] = []
        let called: XCTestExpectation
        init(called: XCTestExpectation) { self.called = called }
        var calls: [[String]] { lock.withLock { recorded } }
        func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
            lock.withLock { recorded.append(arguments) }
            called.fulfill()
            return .init(terminationStatus: 0, stdout: #"{"status":"cancelled"}"#)
        }
    }

    func testControllerRealActionWiring() async throws {
        // This XCTest process cannot activate or add a Dock item; no window/status item is created.
        NSApplication.shared.setActivationPolicy(.prohibited)
        let d = defaults(), runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: Host(), menuDefaults: d)
        defer { controller.stop(); BrainBarSettingsActions.installOpenHandler {} }
        let pasteboard = NSPasteboard(name: .init("BrainBarMenuTests." + UUID().uuidString))
        defer { pasteboard.releaseGlobally() }
        var calls: [String] = []
        BrainBarSettingsActions.installOpenHandler { calls.append("settings") }
        controller.setMenuTargetsForTesting(quit: { calls.append("quit") }, restart: { calls.append("restart") },
            storage: { calls.append("section:" + $0.rawValue) }, pasteboard: pasteboard)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        let rowKinds: [BrainBarMenuPresentation.ActionKind] = [.openStorageSettings, .copyReport, .connectDrive, .reconnectDrive]
        let rows: [BrainBarMenuPresentation.Row] = rowKinds.map { .init(kind: .needsYou, strings: [], action: .init(kind: $0, label: $0.rawValue)) }
            + [.init(kind: .standardItem, strings: ["Settings"], keyEquivalent: "⌘,"),
               .init(kind: .standardItem, strings: ["Quit"], keyEquivalent: "⌘Q", optionAlternate: "Restart")]
        controller.populateLiveMenuForTesting(.init(form: .needs, headline: "", count: 4, icon: .marked, rows: rows))
        let menu = controller.contextMenuForTesting
        func click(_ action: BrainBarMenuAction) throws {
            let index = try XCTUnwrap(menu.items.firstIndex { ($0.representedObject as? BrainBarMenuAction) == action })
            menu.performActionForItem(at: index)
        }
        let standing: [(BrainBarMenuAction, String)] = [(.settings, "settings"), (.quit, "quit"),
            (.restart, "restart"), (.row(.openStorageSettings), "section:backups")]
        for (action, expected) in standing {
            calls = []
            try click(action)
            XCTAssertTrue(calls.isEmpty)
            await nextMainTurn()
            XCTAssertEqual(calls, [expected])
        }
        pasteboard.setString("untouched", forType: .string)
        try click(.row(.copyReport))
        XCTAssertEqual(pasteboard.string(forType: .string), "untouched")
        await nextMainTurn()
        XCTAssertTrue(try XCTUnwrap(pasteboard.string(forType: .string)).hasPrefix("BrainBar report\nVersion:"))
        let driveKinds: [BrainBarMenuPresentation.ActionKind] = [.connectDrive, .reconnectDrive]
        for kind in driveKinds {
            let called = expectation(description: kind.rawValue)
            let runner = Runner(called: called)
            runtime.driveAuth.install(runner: runner)
            try click(.row(kind))
            XCTAssertTrue(runner.calls.isEmpty)
            await fulfillment(of: [called], timeout: 5)
            // Let the reconnect Task consume the completed runner result before the next click.
            while runtime.driveAuth.isReconnecting { await Task.yield() }
            XCTAssertEqual(runner.calls, [BrainBarDriveAuthModel.reconnectArguments])
        }
    }

    func testLiveFixturesBuildExactlyThePresentation() throws {
        for fixture in try BrainBarMenuPresentationTests.loadCases().prefix(3) {
            let c = try BrainBarMenuCases.capture(fixture["inputs"] as! [String: Any])
            let p = BrainBarMenuPresenter.present(c)
            let builder = BrainBarMenuBuilder(accessibility: .init())
            let menu = builder.build(p)
            try assertMenu(menu, equals: p)
            XCTAssertTrue(menu.items.last?.isAlternate == true)
            XCTAssertEqual(menu.items.last?.representedObject as? BrainBarMenuAction, .restart)
        }
    }

    private func assertMenu(_ menu: NSMenu, equals p: BrainBarMenuPresentation) throws {
        let items = menu.items.contains { $0.title == BrainBarMenuPresenter.SurfaceSay.diagnostics }
            ? try brainBarItemsWithoutApprovedDiagnostics(menu) : menu.items
        let rows = items.filter { !$0.isSeparatorItem && !$0.isAlternate }
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
            try assertMenu(controller.contextMenuForTesting, equals: BrainBarMenuPresenter.present(model.menuCapture(stats: nil)))
            controller.stop()
            model.systemDidWake()
            XCTAssertTrue(model.menuCapture(stats: nil).wakeCatchUpPending)
        }
    }

    func testReportRedactsPersonalDiagnosticIdentifiers() {
        let p = BrainBarMenuPresentation(form: .calm, headline: "", count: 0, icon: .resting, rows: [])
        let v = BrainBarHealthVerdict(dot: .red,
            checks: [.init(name: "disk", state: .red, note: "/home/fixture/db a@example.com aB3dE6gH9jK2mN5pQ8sT1vW4xY7zA0bC@example.com")], notes: [], evaluatedAt: .distantPast)
        let report = BrainBarMenuReport.text(presentation: p, verdict: v, version: "test", now: Date(timeIntervalSince1970: 0.0))
        XCTAssertEqual(report.components(separatedBy: "\n").first { $0.hasPrefix("disk:") },
                       "disk: red — /home/[user]/db [email] [email]")
    }

    func testReportRemovesUnlabelledMixedClassTokens() {
        let tokens: [String] = ["aB3dE6gH9jK2mN5pQ8sT1vW4xY7zA0bC", "0123456789abcdef0123456789abcdef",
                                "aB3d_6gH-9jK2mN5pQ8sT1vW4xY7zA0bC"]
        let p = BrainBarMenuPresentation(form: .calm, headline: "", count: 0, icon: .resting, rows: [])
        let v = BrainBarHealthVerdict(dot: .red, checks: [.init(name: "stores", state: .red,
            note: "command=worker " + tokens.joined(separator: " "))], notes: [], evaluatedAt: .distantPast)
        let report = BrainBarMenuReport.text(presentation: p, verdict: v, version: "test", now: .distantPast)
        for token in tokens { XCTAssertFalse(report.contains(token)) }
        XCTAssertTrue(report.contains("command=worker [redacted] [redacted] [redacted]"))
    }

    func testModelOffPathRestoresVerdictOverflowAndBackupRows() throws {
        let now = BrainBarStatusSnapshotTests.now
        guard case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else { return XCTFail("Fixture") }
        let document = ObservabilityDocument(schemaVersion: 1, generatedAt: now, dbPath: "/tmp/fixture.db", windowHours: 24,
            stores: base.stores, emitters: base.emitters, authorUnknown: base.authorUnknown, backups: BrainBarStatusSnapshotTests.backups())
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(),
            store: .init(loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
            refreshStatusOnLoad: false, now: { now }, initialObservabilityResult: .readable(document))
        runtime.jobsModel = model
        let checks: [BrainBarHealthVerdict.Check] = ["stores", "search", "disk", "backup_local"].map {
            .init(name: $0, state: .red, note: "fixture failure " + $0)
        }
        model.applyHealthBadge(.init(badgeOn: true, reason: "fixture", activeCodes: [],
            healthVerdict: .init(dot: .red, checks: checks, notes: [], evaluatedAt: now)), alerts: nil)
        let d = defaults()
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        let control = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: Host(), menuDefaults: brainBarLegacyMenuDefaults())
        let switched = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: nil, statusBar: Host(), menuDefaults: d)
        defer { control.stop(); switched.stop() }
        func inventory(_ menu: NSMenu) -> [String] {
            menu.items.flatMap { item in
                ["\(item.title)|\(item.isHidden)|\(item.isEnabled)|\(item.isSeparatorItem)|\(item.keyEquivalent)|\(item.identifier?.rawValue ?? "")"]
                    + (item.submenu.map(inventory) ?? [])
            }
        }
        control.menuNeedsUpdate(control.contextMenuForTesting)
        let expected = inventory(control.contextMenuForTesting)
        XCTAssertTrue(control.contextMenuForTesting.items.contains { $0.identifier?.rawValue.hasPrefix("brainbar.menu.backup.") == true })
        XCTAssertTrue(control.contextMenuForTesting.items.contains { $0.submenu != nil })
        switched.menuNeedsUpdate(switched.contextMenuForTesting)
        XCTAssertEqual(inventory(switched.contextMenuForTesting), expected)
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        switched.menuNeedsUpdate(switched.contextMenuForTesting)
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        switched.menuNeedsUpdate(switched.contextMenuForTesting)
        XCTAssertEqual(inventory(switched.contextMenuForTesting), expected)
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

    func testSwitchOffBuildsMainsItemsAndRestoresThemAfterOn() throws {
        let d = defaults(), host = Host(), workspace = BrainBarRecordingWorkspace()
        d.set(false, forKey: BrainBarNewMenuPreference.key)
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
        // Gap 26, lead ruling 21 Nov: exactly one enabled, unmarked Diagnostics after Settings; golden stays main's.
        let mainItems = try brainBarItemsWithoutApprovedDiagnostics(menu)
        XCTAssertEqual(mainItems.map(\.title), expected)
        XCTAssertEqual(mainItems.map { $0.action.map(NSStringFromSelector) },
            [nil, "showJobAlertLog:", nil, nil, "openDashboard:", "openSettings:", nil, "restartBrainBar:", nil, "quitBrainBar:"])
        XCTAssertEqual(mainItems.map(\.isHidden), [false, true, true, false, false, false, false, false, false, false])
        XCTAssertEqual(mainItems.map(\.isSeparatorItem), [false, false, false, true, false, false, true, false, true, false])
        XCTAssertTrue(menu.items.allSatisfy { $0.keyEquivalent.isEmpty })
        let originals = menu.items
        let shape: (NSMenuItem) -> String = { item in
            "\(item.title)|\(item.isEnabled)|\(item.isHidden)|\(item.isSeparatorItem)|\(item.keyEquivalent)|\(item.action.map(NSStringFromSelector) ?? "")"
        }
        let before = originals.map(shape)
        d.set(true, forKey: BrainBarNewMenuPreference.key)
        controller.menuNeedsUpdate(menu)
        XCTAssertNotEqual(menu.items.map(\.title), originals.map(\.title))
        d.set(false, forKey: BrainBarNewMenuPreference.key)
        controller.menuNeedsUpdate(menu)
        XCTAssertEqual(menu.items.map(shape), before)
        for (actual, original) in zip(menu.items, originals) { XCTAssertTrue(actual === original) }
    }
}
