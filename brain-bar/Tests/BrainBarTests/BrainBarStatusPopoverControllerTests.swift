import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStatusPopoverControllerTests: XCTestCase {
    func testLongStatusReasonIsTruncatedAtAWordBoundaryInBothMenuPaths() {
        let reason = String(repeating: "The example status reason keeps growing as neutral words repeat. ", count: 10)
        XCTAssertGreaterThanOrEqual(reason.count, 300)
        let badge = BadgeStatePresentation.failVisible(reason)
        let title = BrainBarStatusPopoverController.statusLineTitle(for: badge)

        XCTAssertLessThanOrEqual(title.count, 60)
        XCTAssertTrue(title.hasSuffix("…"))
        XCTAssertEqual(title, "Needs attention: The example status reason keeps growing…")
        XCTAssertEqual(BrainBarStatusPopoverController.menuRowTitles(for: badge).first, title)
    }

    func testShortAndHealthyStatusTitlesAreUnchanged() {
        XCTAssertEqual(
            BrainBarStatusPopoverController.statusLineTitle(for: .failVisible("Watcher stopped")),
            "Needs attention: Watcher stopped"
        )
        XCTAssertEqual(
            BrainBarStatusPopoverController.statusLineTitle(for: .init(badgeOn: false, reason: "", activeCodes: [])),
            "Nothing needs attention"
        )
    }

    func testStatusTitleLimitCountsVisibleCharactersAndHandlesAnUnbrokenReason() {
        let reason = String(repeating: "👩🏽‍💻", count: 43)
        let atLimit = BrainBarStatusPopoverController.statusLineTitle(for: .failVisible(reason))
        XCTAssertEqual(atLimit, "Needs attention: \(reason)")
        XCTAssertEqual(atLimit.count, 60)

        let overLimit = BrainBarStatusPopoverController.statusLineTitle(for: .failVisible(reason + "👩🏽‍💻"))
        XCTAssertLessThanOrEqual(overLimit.count, 60)
        XCTAssertTrue(overLimit.hasSuffix("…"))
        XCTAssertEqual(overLimit, "Needs attention: \(String(reason.prefix(42)))…")
    }

    func testStatusRowTooltipPreservesTheFullReasonAndClearsOnRecovery() throws {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let panelController = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime, dashboardPanelController: panelController, statusBar: TestStatusItemHost()
        )
        defer { controller.stop() }
        let menu = controller.contextMenuForTesting
        let statusRow = try XCTUnwrap(menu.items.first)
        XCTAssertEqual(statusRow.toolTip, "Badge state has not been read yet.")

        let alertsURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        let reason = String(repeating: "The example status reason keeps growing as neutral words repeat. ", count: 10)
        controller.setBadgeForTesting(.failVisible(reason), alertsURL: alertsURL)
        controller.menuNeedsUpdate(menu)
        XCTAssertLessThanOrEqual(statusRow.title.count, 60)
        XCTAssertTrue(statusRow.title.hasSuffix("…"))
        XCTAssertEqual(statusRow.toolTip, reason)

        let unboundedMenu = NSMenu(title: "Neutral width reference")
        unboundedMenu.addItem(NSMenuItem(title: "Needs attention: \(reason)", action: nil, keyEquivalent: ""))
        XCTAssertLessThan(menu.size.width, unboundedMenu.size.width)
        print("[menu-width] capped=\(menu.size.width) unbounded=\(unboundedMenu.size.width)")

        controller.setBadgeForTesting(.failVisible("Watcher stopped"), alertsURL: alertsURL)
        controller.menuNeedsUpdate(menu)
        XCTAssertEqual(statusRow.title, "Needs attention: Watcher stopped")
        XCTAssertEqual(statusRow.toolTip, "Watcher stopped")

        controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []), alertsURL: alertsURL)
        controller.menuNeedsUpdate(menu)
        XCTAssertEqual(statusRow.title, "Nothing needs attention")
        XCTAssertNil(statusRow.toolTip)
    }

    func testControllerWiresVariableLengthStatusItemToTheWindow() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let panelController = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: panelController,
            statusBar: TestStatusItemHost()
        )
        defer { controller.stop() }

        XCTAssertEqual(controller.statusItemForTesting.length, NSStatusItem.variableLength)
        XCTAssertTrue(controller.statusItemForTesting.menu === controller.contextMenuForTesting)
        XCTAssertEqual(panelController.windowForTesting.contentViewController, panelController.contentViewControllerForTesting)
        XCTAssertEqual(panelController.windowForTesting.contentViewController?.view.frame.size, NSSize(width: 900, height: 640))
        XCTAssertTrue(panelController.windowForTesting.styleMask.contains(.resizable))
    }

    func testStatusItemContextMenuContainsNoLaunchModeSwitchingChoices() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let windowController = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: windowController,
            statusBar: TestStatusItemHost()
        )
        defer { controller.stop() }

        let itemTitles = controller.contextMenuForTesting.items.map(\.title)

        XCTAssertTrue(itemTitles.contains("Open Dashboard"))
        XCTAssertTrue(itemTitles.contains("Settings…"))
        XCTAssertTrue(itemTitles.contains("Restart BrainBar"))
        XCTAssertFalse(itemTitles.contains("Run as App Window"))
        XCTAssertFalse(itemTitles.contains("Run as Menu Item Daemon"))
        XCTAssertTrue(itemTitles.contains("Quit BrainBar"))
    }

    func testStatusItemOwnsOneActionableSettingsMenu() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let windowController = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: windowController,
            statusBar: TestStatusItemHost()
        )
        defer { controller.stop() }

        let menu = controller.contextMenuForTesting
        // The first row is the informational status line; the rest are actions.
        // Show log stays hidden while no job alert is active (lead ruling 2026-10-04).
        let actionableItems = menu.items.filter { !$0.isSeparatorItem && !$0.isHidden }.dropFirst()

        XCTAssertTrue(controller.statusItemForTesting.menu === menu, "every click opens this one menu (#963)")
        XCTAssertFalse(menu === NSApp.mainMenu)
        XCTAssertEqual(actionableItems.filter { $0.title.hasPrefix("Settings") }.count, 1)
        XCTAssertEqual(actionableItems.map(\.title), ["Open Dashboard", "Settings…", "Restart BrainBar", "Quit BrainBar"])
        for item in actionableItems {
            XCTAssertNotNil(item.action, "\(item.title) must have an action")
            XCTAssertTrue(item.target === controller, "\(item.title) must target the status controller")
            if let action = item.action {
                XCTAssertTrue(controller.responds(to: action), "\(item.title) target must respond to its action")
            }
        }
    }

    func testContextMenuOpenDashboardUsesTheStatusPopoverController() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let windowController = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: windowController,
            statusBar: TestStatusItemHost()
        )
        defer { controller.stop() }

        let toggle = controller.contextMenuForTesting.items.first { $0.title == "Open Dashboard" }

        XCTAssertNotNil(toggle?.action)
        XCTAssertTrue(toggle?.target === controller)
        XCTAssertTrue(toggle.map { controller.responds(to: $0.action!) } ?? false)
    }

    func testAppSupportCollectorFactoryWiresBrainBusEvents() {
        let tempDBPath = brainBarTestTemporaryPath() + "brainbar-status-popover-\(UUID().uuidString).db"
        let eventSource = RecordingBrainBusEventSource()
        let collector = BrainBarAppSupport.makeStatsCollector(
            dbPath: tempDBPath,
            targetPID: ProcessInfo.processInfo.processIdentifier,
            agentActivityMonitor: .synthetic,
            brainBusEvents: eventSource,
            // This checks event wiring, not database writes. A detached initial refresh
            // can outlive stop(); it must not recreate a writable fixture after teardown.
            databaseOpenConfiguration: .init(readOnly: true)
        )
        defer {
            collector.stop()
            removeBrainBarTestDatabase(tempDBPath)
            try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
            try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        }

        collector.start()

        XCTAssertEqual(eventSource.streamRequestCount, 1)
    }

    func testBadgeReadsAreCadencedOffTheMainThreadAndIconRenderingUsesTheCache() throws {
        let source = try String(
            contentsOf: packageRoot().appendingPathComponent("Sources/BrainBar/BrainBarStatusPopoverController.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(source.contains("DispatchQueue(label: \"com.brainlayer.brainbar.badge-read\", qos: .utility)"))
        XCTAssertTrue(source.contains("Timer.publish(every: max(cadence.interval, 1)"))
        XCTAssertTrue(source.contains("badgeReadQueue.async"))
        XCTAssertTrue(source.contains("badgePresentation = .failVisible(\"Badge state has not been read yet.\")"))

        let renderStart = try XCTUnwrap(source.range(of: "private func renderStatusIcon"))
        let renderEnd = try XCTUnwrap(source.range(of: "func menuNeedsUpdate", range: renderStart.upperBound..<source.endIndex))
        let render = String(source[renderStart.lowerBound..<renderEnd.lowerBound])
        XCTAssertTrue(render.contains("let badge = badgePresentation"))
        XCTAssertFalse(render.contains("BadgeStateReader.read"), "UI emissions must not perform badge file I/O.")
    }

    func testControllerStatusItemBelongsToTheInMemoryHostAndStopsOnThatHost() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let panelController = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let host = TestStatusItemHost()
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime, dashboardPanelController: panelController, statusBar: host
        )
        defer { controller.stop() }

        XCTAssertTrue(controller.statusItemForTesting === host.createdItem,
                      "test fixtures must never create items in the live system menu bar")
        XCTAssertTrue(controller.statusItemForTesting is TestStatusItemHost.Item)
        controller.stop()
        controller.stop()
        XCTAssertEqual(host.removedItems.count, 1, "stop must remove the item exactly once")
        XCTAssertTrue(host.removedItems.first === controller.statusItemForTesting)
        XCTAssertNil(controller.statusItemForTesting.menu)
        XCTAssertNil(controller.contextMenuForTesting.delegate)
        XCTAssertNil(panelController.statusItemButton)
    }

    func testStoppedPrivateStatusFixturesDoNotSurviveAnAutoreleaseDrain() {
        weak var releasedController: BrainBarStatusPopoverController?
        weak var releasedHost: TestStatusItemHost?
        weak var releasedItem: TestStatusItemHost.Item?
        weak var releasedButton: NSButton?
        autoreleasepool {
            let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
            let panelController = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
            let host = TestStatusItemHost()
            let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime, dashboardPanelController: panelController, statusBar: host
            )
            releasedController = controller
            releasedHost = host
            releasedItem = host.createdItem
            releasedButton = host.createdItem?.button
            controller.stop()
        }
        XCTAssertNil(releasedController)
        XCTAssertNil(releasedHost)
        XCTAssertNil(releasedItem)
        XCTAssertNil(releasedButton)
    }

    private func packageRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}

private final class RecordingBrainBusEventSource: BrainBusEventSource, @unchecked Sendable {
    private let lock = NSLock()
    private var requests = 0

    var streamRequestCount: Int {
        lock.withLock { requests }
    }

    func events() -> AsyncStream<BrainBusEvent> {
        lock.withLock {
            requests += 1
        }
        return AsyncStream { _ in }
    }
}
