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
        let delegate = TrackingDelegate(forwarding: try XCTUnwrap(menu.delegate))
        menu.delegate = delegate
        let window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 120, height: 80),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        defer { window.close() }
        let view = try XCTUnwrap(window.contentView)
        let observer = NotificationCenter.default.addObserver(forName: NSMenu.didEndTrackingNotification,
            object: menu, queue: nil) { _ in
            MainActor.assumeIsolated { delegate.ends += 1 }
        }
        defer { NotificationCenter.default.removeObserver(observer) }
        for cycle in 1...3 {
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
        init(forwarding: any NSMenuDelegate) { self.forwarding = forwarding }
        func menuNeedsUpdate(_ menu: NSMenu) { forwarding.menuNeedsUpdate?(menu) }
        func menuWillOpen(_ menu: NSMenu) { opens += 1; forwarding.menuWillOpen?(menu) }
        func menuDidClose(_ menu: NSMenu) { closes += 1; forwarding.menuDidClose?(menu) }
    }

    private final class Button: NSButton {
        var imageWrites = 0
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
        var value: Int {
            get { lock.withLock { storedValue } }
            set { lock.withLock { storedValue = newValue } }
        }
        func fetch() -> DashboardStats {
            let count = value
            return DashboardStats(chunkCount: count, enrichedChunkCount: 0, pendingEnrichmentCount: 0,
                enrichmentPercent: 0, enrichmentRatePerMinute: 0, databaseSizeBytes: 0,
                recentActivityBuckets: [0, count], recentAgentWriteBuckets: [0, count],
                recentWatcherWriteBuckets: [count, 0], recentEnrichmentBuckets: [0, 0])
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
        func stop() {
            controller.stop()
            collector.stop()
            pump()
            try? FileManager.default.removeItem(at: root)
        }
    }
}
