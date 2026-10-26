import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStatusMenuTrackingTests: XCTestCase {
    func testInitialAsyncBadgeReadDuringTrackingDoesNotChangeButtonUntilClose() throws {
        let fixture = try Fixture()
        defer { fixture.stop() }
        let menu = fixture.controller.contextMenuForTesting
        menu.delegate?.menuWillOpen?(menu)
        let writes = fixture.host.button.imageWrites
        let tooltip = fixture.host.button.toolTip
        // Runtime/collector binding and its real background badge-file read happen here.
        fixture.pump(until: {
            menu.delegate?.menuNeedsUpdate?(menu)
            return fixture.collector.lastDataFetchedAt != nil
                && menu.items.first?.toolTip == "Badge state missing without a pending_first_run marker."
        })
        XCTAssertEqual(fixture.host.button.imageWrites, writes)
        XCTAssertEqual(fixture.host.button.toolTip, tooltip)
        menu.delegate?.menuDidClose?(menu)
        fixture.pump()
        XCTAssertEqual(fixture.host.button.imageWrites, writes + 1)
        XCTAssertEqual(fixture.host.button.toolTip,
            "BrainBar — needs attention: Badge state missing without a pending_first_run marker.")
    }

    func testCollectorAndBadgeUpdatesKeepTrackingAnchorStableAcrossReopens() throws {
        let fixture = try Fixture()
        defer { fixture.stop() }
        let menu = fixture.controller.contextMenuForTesting
        fixture.pump()

        for cycle in 1...3 {
            menu.delegate?.menuNeedsUpdate?(menu)
            menu.delegate?.menuWillOpen?(menu)
            let writes = fixture.host.button.imageWrites
            let tooltip = fixture.host.button.toolTip
            for update in 1...2 {
                fixture.provider.value = cycle * 10 + update
                fixture.collector.refresh(force: true)
                fixture.pump(until: { fixture.collector.stats.chunkCount == cycle * 10 + update })
                fixture.controller.setBadgeForTesting(.failVisible("Tracking update \(cycle)-\(update)"),
                    alertsURL: fixture.root.appendingPathComponent("absent-alerts.json"))
                menu.delegate?.menuNeedsUpdate?(menu)
                fixture.pump()
            }
            XCTAssertEqual(fixture.host.button.imageWrites, writes,
                "Live collector/badge updates must not relayout the status anchor during native tracking")
            XCTAssertEqual(fixture.host.button.toolTip, tooltip)
            menu.delegate?.menuDidClose?(menu)
            XCTAssertEqual(fixture.host.button.imageWrites, writes, "Do not relayout inside menuDidClose")
            fixture.pump()
            XCTAssertEqual(fixture.host.button.imageWrites, writes + 1, "Coalesce to the latest snapshot after close")
            XCTAssertEqual(fixture.host.button.toolTip, "BrainBar — needs attention: Tracking update \(cycle)-2")
            XCTAssertEqual(fixture.host.button.image?.tiffRepresentation,
                BrainBarStatusPopoverController.statusIconImage(stats: fixture.collector.stats, badgeOn: true).tiffRepresentation)
        }
    }

    // MARK: R2-1 — appearance and contrast are part of the icon's identity

    /// The factory output under an app/system appearance, as the reviewer measured it (NSApp
    /// appearance + current drawing appearance); the status button follows that appearance.
    private func factoryIcon(_ stats: BrainDatabase.DashboardStats, badgeOn: Bool, _ name: NSAppearance.Name) -> Data? {
        let saved = NSApplication.shared.appearance
        NSApplication.shared.appearance = NSAppearance(named: name)
        defer { NSApplication.shared.appearance = saved }
        var image: NSImage?
        NSAppearance(named: name)!.performAsCurrentDrawingAppearance {
            image = BrainBarStatusPopoverController.statusIconImage(stats: stats, badgeOn: badgeOn)
        }
        return image?.tiffRepresentation
    }

    func testAppearanceAndContrastChangesRerenderConstantBucketsAndDot() throws {
        let fixture = try Fixture()
        let saved = NSApplication.shared.appearance
        let savedContrast = BrainBarStatusPopoverController.increaseContrast
        var contrast = false
        BrainBarStatusPopoverController.increaseContrast = { contrast }
        defer {
            NSApplication.shared.appearance = saved
            BrainBarStatusPopoverController.increaseContrast = savedContrast
            fixture.stop()
        }
        let menu = fixture.controller.contextMenuForTesting
        fixture.pump()
        // A shaped series, constant through the test: only appearance/contrast change.
        fixture.provider.buckets = [1, 3, 2, 5, 4]
        fixture.provider.value = 7
        fixture.collector.refresh(force: true)
        fixture.pump(until: { fixture.collector.stats.chunkCount == 7 })
        fixture.controller.setBadgeForTesting(.failVisible("Constant dot"),
            alertsURL: fixture.root.appendingPathComponent("absent-alerts.json"))
        menu.delegate?.menuNeedsUpdate?(menu)
        NSApplication.shared.appearance = NSAppearance(named: .aqua)
        fixture.pump()
        let button = fixture.host.button
        let stats = fixture.collector.stats
        // Headless test processes draw the same bitmap for every appearance (measured); the reviewer
        // measured real differences under High Contrast. So this pins the contract: every
        // appearance/contrast change re-renders exactly once, to the factory output for that state.

        // Appearance changes (constant buckets and dot): exactly one render each, to the new factory output.
        for name in [NSAppearance.Name.darkAqua, .aqua, .darkAqua, .aqua] {
            let writes = button.imageWrites
            let expected = factoryIcon(stats, badgeOn: true, name)
            NSApplication.shared.appearance = NSAppearance(named: name)
            fixture.pump()
            XCTAssertEqual(button.imageWrites, writes + 1, "exactly one render for \(name)")
            XCTAssertEqual(button.image?.tiffRepresentation, expected, "\(name)")
        }

        // Increase Contrast is part of the identity: its notification re-renders once per change.
        var writes = button.imageWrites
        contrast = true
        NSWorkspace.shared.notificationCenter.post(name: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil)
        fixture.pump()
        XCTAssertEqual(button.imageWrites, writes + 1, "contrast on re-renders")
        writes = button.imageWrites
        NSWorkspace.shared.notificationCenter.post(name: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil)
        fixture.pump()
        XCTAssertEqual(button.imageWrites, writes, "a notification with nothing changed does not relayout")

        // While the menu tracks, changes wait; exactly one latest-state render after close.
        menu.delegate?.menuNeedsUpdate?(menu)
        menu.delegate?.menuWillOpen?(menu)
        writes = button.imageWrites
        let expected = factoryIcon(stats, badgeOn: true, .darkAqua)
        NSApplication.shared.appearance = NSAppearance(named: .darkAqua)
        contrast = false
        NSWorkspace.shared.notificationCenter.post(name: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil)
        fixture.pump()
        XCTAssertEqual(button.imageWrites, writes, "no relayout during tracking")
        menu.delegate?.menuDidClose?(menu)
        fixture.pump()
        XCTAssertEqual(button.imageWrites, writes + 1, "one latest-state render after close")
        XCTAssertEqual(button.image?.tiffRepresentation, expected)

        // After stop nothing observes.
        let settled = button.imageWrites
        fixture.controller.stop()
        NSApplication.shared.appearance = NSAppearance(named: .aqua)
        contrast = true
        NSWorkspace.shared.notificationCenter.post(name: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification, object: nil)
        fixture.pump()
        XCTAssertEqual(button.imageWrites, settled, "observers are removed on stop")
    }

    func testCloseThenImmediateReopenDoesNotFlushIntoNewTrackingSessionAndStopDiscardsPendingWork() throws {
        let fixture = try Fixture()
        defer { fixture.stop() }
        let menu = fixture.controller.contextMenuForTesting
        fixture.pump()
        menu.delegate?.menuWillOpen?(menu)
        let writes = fixture.host.button.imageWrites
        fixture.provider.value = 31
        fixture.collector.refresh(force: true)
        fixture.pump(until: { fixture.collector.stats.chunkCount == 31 })
        menu.delegate?.menuDidClose?(menu)
        menu.delegate?.menuWillOpen?(menu)
        fixture.pump()
        XCTAssertEqual(fixture.host.button.imageWrites, writes)
        menu.delegate?.menuDidClose?(menu)
        fixture.controller.stop()
        fixture.pump()
        XCTAssertEqual(fixture.host.button.imageWrites, writes)
        XCTAssertNil(menu.delegate)
        XCTAssertNil(fixture.controller.statusItemForTesting.menu)
        XCTAssertEqual(fixture.host.removals, 1)
    }

    func testPendingRenderDoesNotSuppressOrRepeatInsideMenuAction() throws {
        let fixture = try Fixture()
        defer { fixture.stop() }
        fixture.pump()
        let menu = fixture.controller.contextMenuForTesting
        let paths = BrainBarBackupSources.Paths(launchAgents: fixture.root,
            databaseLog: fixture.root.appendingPathComponent("missing.log"),
            archiveLog: fixture.root.appendingPathComponent("archive.log"),
            maintenanceLog: fixture.root.appendingPathComponent("maintenance.log"),
            snapshotDirectory: fixture.root, archiveDirectory: fixture.root)
        fixture.controller.setLogPathsForTesting(paths)
        fixture.controller.setBadgeForTesting(.init(badgeOn: true, reason: "Stopped with exit 143",
            activeCodes: ["job_alert_backupDaily"]), alertsURL: fixture.root.appendingPathComponent("absent-alerts.json"))
        menu.delegate?.menuNeedsUpdate?(menu)
        let index = try XCTUnwrap(menu.items.firstIndex { $0.title == "Open logs folder" })
        menu.delegate?.menuWillOpen?(menu)
        let writes = fixture.host.button.imageWrites
        fixture.provider.value = 7
        fixture.collector.refresh(force: true)
        fixture.pump(until: { fixture.collector.stats.chunkCount == 7 })
        menu.performActionForItem(at: index)
        XCTAssertEqual(fixture.workspace.recorded.map { $0.1 }, [fixture.root.path])
        XCTAssertEqual(fixture.host.button.imageWrites, writes)
        menu.delegate?.menuDidClose?(menu)
        fixture.pump()
        XCTAssertEqual(fixture.workspace.recorded.count, 1)
        XCTAssertEqual(fixture.host.button.imageWrites, writes + 1)
    }

    func testNativeMenuCallbacksFreezeCollectorUpdatesAcrossCancelEscapeReopenAndStop() throws {
        let fixture = try Fixture()
        defer { fixture.stop() }
        fixture.pump()
        let menu = fixture.controller.contextMenuForTesting
        let delegate = TrackingDelegate(forwarding: try XCTUnwrap(menu.delegate), button: fixture.host.button)
        menu.delegate = delegate
        let window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 120, height: 80),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        defer { window.close() }
        let view = try XCTUnwrap(window.contentView)
        let begin = NotificationCenter.default.addObserver(forName: NSMenu.didBeginTrackingNotification,
            object: menu, queue: nil) { _ in
            MainActor.assumeIsolated {
                delegate.beginWrites = fixture.host.button.imageWrites
                delegate.beginTooltipWrites = fixture.host.button.tooltipWrites
            }
        }
        defer { NotificationCenter.default.removeObserver(begin) }
        let observer = NotificationCenter.default.addObserver(forName: NSMenu.didEndTrackingNotification,
            object: menu, queue: nil) { _ in
            MainActor.assumeIsolated {
                delegate.ends += 1
                delegate.nativeWrites.append(fixture.host.button.imageWrites - delegate.beginWrites)
                XCTAssertEqual(fixture.host.button.tooltipWrites, delegate.beginTooltipWrites)
            }
        }
        defer { NotificationCenter.default.removeObserver(observer) }
        for cycle in 1...3 {
            try fixture.prepareJobAlert(currentReason: cycle == 2 ? "Changed failure" : nil)
            let watchdog = Timer(timeInterval: 2, repeats: false) { _ in
                MainActor.assumeIsolated {
                    delegate.watchdogFired = true
                    fixture.controller.contextMenuForTesting.cancelTracking()
                }
            }
            RunLoop.main.add(watchdog, forMode: .eventTracking)
            let timer = Timer(timeInterval: 0.05, repeats: false) { _ in
                MainActor.assumeIsolated {
                    XCTAssertEqual(delegate.opens, cycle, "AppKit must invoke the real menuWillOpen callback")
                    XCTAssertEqual(fixture.controller.contextMenuForTesting.items.first?.title,
                        cycle == 2 ? "Needs attention: Changed failure" : "Nothing needs attention")
                    let writes = fixture.host.button.imageWrites
                    fixture.provider.value = 40 + cycle
                    fixture.collector.refresh(force: true)
                    fixture.pump(until: { fixture.collector.stats.chunkCount == 40 + cycle })
                    delegate.writesDuringTracking = fixture.host.button.imageWrites - writes
                    if cycle == 3 { fixture.controller.stop() }
                    else if cycle == 2 {
                        // App-local posted input, deliberately not described as physical Escape.
                        let escape = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [],
                            timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: 0, context: nil,
                            characters: "\u{1b}", charactersIgnoringModifiers: "\u{1b}", isARepeat: false, keyCode: 53)!
                        NSApp.postEvent(escape, atStart: true)
                    }
                    else { fixture.controller.contextMenuForTesting.cancelTracking() }
                }
            }
            RunLoop.main.add(timer, forMode: .eventTracking)
            _ = menu.popUp(positioning: nil, at: NSPoint(x: 20, y: 40), in: view)
            timer.invalidate()
            watchdog.invalidate()
            XCTAssertFalse(delegate.watchdogFired, "Cancellation must complete without the safety timeout")
            XCTAssertEqual(delegate.opens, cycle)
            XCTAssertEqual(delegate.ends, cycle, "Native tracking must end, including after stop detaches its delegate")
            if cycle < 3 { XCTAssertEqual(delegate.closes, cycle) }
            else {
                // AppKit may finish cancellation after stop has detached the delegate.
                XCTAssertTrue((2...3).contains(delegate.closes))
            }
            XCTAssertEqual(delegate.writesDuringTracking, 0)
            fixture.pump()
        }
        print("PRE_OPEN_WRITES=\(delegate.preOpenWrites) NATIVE_WRITES=\(delegate.nativeWrites)")
        XCTAssertEqual(delegate.preOpenWrites, [0, 0, 0])
        XCTAssertEqual(delegate.nativeWrites, [0, 0, 0])
        XCTAssertEqual(fixture.host.removals, 1)
    }

    func testNontrackingShowLogReconcilesChangedAndRecoveredAlertsAndStoppedNotificationsAreIgnored() throws {
        let fixture = try Fixture()
        defer { fixture.stop() }
        fixture.pump()
        let menu = fixture.controller.contextMenuForTesting
        fixture.controller.setLogPathsForTesting(.init(launchAgents: fixture.root,
            databaseLog: fixture.root.appendingPathComponent("missing.log"), archiveLog: fixture.root,
            maintenanceLog: fixture.root, snapshotDirectory: fixture.root, archiveDirectory: fixture.root))
        for reason in ["Changed failure", "Another failure"] {
            try fixture.prepareJobAlert(currentReason: reason)
            let writes = fixture.host.button.imageWrites
            NotificationCenter.default.post(name: NSMenu.didBeginTrackingNotification, object: NSMenu(title: "Unrelated"))
            menu.delegate?.menuNeedsUpdate?(menu)
            // S5 fix1: icon writes coalesce by content. The changed reason updates the tooltip; the
            // icon (same buckets, dot still on) is the same image, so it is not relaid out again.
            XCTAssertLessThanOrEqual(fixture.host.button.imageWrites, writes + 1)
            XCTAssertEqual(fixture.host.button.image?.tiffRepresentation,
                BrainBarStatusPopoverController.statusIconImage(stats: fixture.collector.stats, badgeOn: true).tiffRepresentation)
            let index = try XCTUnwrap(menu.items.firstIndex { $0.title == "Open logs folder" })
            menu.performActionForItem(at: index)
            XCTAssertEqual(fixture.host.button.toolTip, "BrainBar — needs attention: \(reason)")
        }
        XCTAssertEqual(fixture.workspace.recorded.count, 2)
        try fixture.prepareJobAlert(currentReason: nil)
        let writes = fixture.host.button.imageWrites
        menu.performActionForItem(at: try XCTUnwrap(menu.items.firstIndex { $0.title == "Open logs folder" }))
        XCTAssertEqual(fixture.host.button.imageWrites, writes + 1)
        XCTAssertEqual(fixture.host.button.toolTip, "BrainBar")
        XCTAssertEqual(fixture.workspace.recorded.count, 2, "Recovered action must not open stale logs")
        fixture.controller.stop()
        NotificationCenter.default.post(name: NSMenu.didBeginTrackingNotification, object: menu)
        NotificationCenter.default.post(name: NSMenu.didEndTrackingNotification, object: menu)
        fixture.pump()
        XCTAssertEqual(fixture.host.button.imageWrites, writes + 1)
        XCTAssertNil(menu.delegate)
        XCTAssertEqual(fixture.host.removals, 1)
    }

    @MainActor
    private final class TrackingDelegate: NSObject, NSMenuDelegate {
        let forwarding: any NSMenuDelegate
        var opens = 0
        var closes = 0
        var ends = 0
        var writesDuringTracking = 0
        var watchdogFired = false
        let button: Button
        var beginWrites = 0
        var beginTooltipWrites = 0
        var preOpenWrites: [Int] = []
        var nativeWrites: [Int] = []
        init(forwarding: any NSMenuDelegate, button: Button) { self.forwarding = forwarding; self.button = button }
        func menuNeedsUpdate(_ menu: NSMenu) { forwarding.menuNeedsUpdate?(menu) }
        func menuWillOpen(_ menu: NSMenu) {
            preOpenWrites.append(button.imageWrites - beginWrites)
            opens += 1
            forwarding.menuWillOpen?(menu)
        }
        func menuDidClose(_ menu: NSMenu) { closes += 1; forwarding.menuDidClose?(menu) }
    }

    private final class Button: NSButton {
        var imageWrites = 0
        var tooltipWrites = 0
        override var toolTip: String? { didSet { tooltipWrites += 1 } }
        override var image: NSImage? {
            didSet { imageWrites += 1 }
        }
    }

    private final class Host: BrainBarStatusItemHosting {
        final class Item: BrainBarStatusItem {
            let length = NSStatusItem.variableLength
            var menu: NSMenu?
            let recordingButton = Button(frame: .zero)
            var button: NSButton? { recordingButton }
        }
        let item = Item()
        var button: Button { item.recordingButton }
        var removals = 0
        func statusItem(withLength _: CGFloat) -> any BrainBarStatusItem { item }
        func removeStatusItem(_ item: any BrainBarStatusItem) { XCTAssertTrue(item === self.item); removals += 1 }
    }

    private final class Provider: @unchecked Sendable {
        private let lock = NSLock()
        private var storedValue = 0
        /// Optional shaped buckets for both series (appearance test); nil keeps [0, count] / [count, 0].
        var buckets: [Int]?
        var value: Int {
            get { lock.withLock { storedValue } }
            set { lock.withLock { storedValue = newValue } }
        }
        func fetch() -> DashboardStats {
            let count = value
            return DashboardStats(chunkCount: count, enrichedChunkCount: 0, pendingEnrichmentCount: 0,
                enrichmentPercent: 0, enrichmentRatePerMinute: 0, databaseSizeBytes: 0,
                recentActivityBuckets: [0, count], recentAgentWriteBuckets: buckets ?? [0, count],
                recentWatcherWriteBuckets: buckets?.reversed() ?? [count, 0], recentEnrichmentBuckets: [0, 0])
        }
    }

    @MainActor
    private final class Fixture {
        let root: URL
        let provider = Provider()
        let host = Host()
        let workspace = BrainBarRecordingWorkspace()
        let collector: StatsCollector
        let controller: BrainBarStatusPopoverController
        init() throws {
            root = URL(fileURLWithPath: brainBarTestTemporaryPath()).appendingPathComponent("brainbar-menu-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let provider = self.provider
            collector = StatsCollector(dbPath: root.appendingPathComponent("unused.db").path,
                daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
                agentActivityMonitor: .synthetic, statsRefreshCoalesceInterval: 0,
                observeDatabaseChanges: false, dashboardStatsProvider: provider.fetch,
                signalCoverageProvider: { .init(eligibleChunkCount: 0, vectorIndexedChunkCount: 0,
                    ftsIndexedChunkCount: 0, trigramIndexedChunkCount: 0) },
                databaseOpenConfiguration: .init(readOnly: true))
            let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
            controller = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime,
                dashboardPanelController: BrainBarDashboardPanelController(workspace: workspace, runtime: runtime),
                statusBar: host)
            runtime.install(collector: collector, database: nil, databasePath: root.appendingPathComponent("unused.db").path)
        }
        func pump(until condition: () -> Bool = { true }) {
            let deadline = Date().addingTimeInterval(3)
            repeat { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.01)) }
            while !condition() && Date() < deadline
            XCTAssertTrue(condition(), "Collector must publish within the bounded wait")
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
        }
        func prepareJobAlert(currentReason: String?) throws {
            let alerts = root.appendingPathComponent("job-alerts.json")
            let map: [String: String] = currentReason.map { ["backupDaily": $0] } ?? [:]
            try JSONSerialization.data(withJSONObject: map).write(to: alerts)
            controller.setBadgeForTesting(.init(badgeOn: true, reason: "Stopped with exit 143",
                activeCodes: ["job_alert_backupDaily"], activeMessages: ["Stopped with exit 143"]), alertsURL: alerts)
        }
        func stop() {
            controller.stop()
            collector.stop()
            pump()
            try? FileManager.default.removeItem(at: root)
        }
    }
}
