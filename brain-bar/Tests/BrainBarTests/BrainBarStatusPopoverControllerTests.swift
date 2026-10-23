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

    func test_D1_missing_log_menu_offers_an_action_without_claiming_it_already_opened() {
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath()).appendingPathComponent(UUID().uuidString)
        let paths = BrainBarBackupSources.Paths(launchAgents: root, databaseLog: root.appendingPathComponent("backup.log"),
            archiveLog: root.appendingPathComponent("archive.log"), maintenanceLog: root.appendingPathComponent("maintenance.log"),
            snapshotDirectory: root, archiveDirectory: root)
        for key in ["unknown-job", "maintenance-light"] {
            let badge = BadgeStatePresentation(badgeOn: true, reason: "Failed", activeCodes: ["job_alert_\(key)"])
            XCTAssertEqual(BrainBarStatusPopoverController.showLogItemTitle(for: badge, paths: paths, fileExists: { _ in false }),
                "Open logs folder")
        }
    }

    func test_D1_native_menu_action_opens_file_or_folder_and_scopes_failure_feedback() throws {
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath()).appendingPathComponent("brainbar-d1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let paths = BrainBarBackupSources.Paths(launchAgents: root, databaseLog: root.appendingPathComponent("backup.log"),
            archiveLog: root.appendingPathComponent("archive.log"), maintenanceLog: root.appendingPathComponent("maintenance.log"),
            snapshotDirectory: root, archiveDirectory: root)
        let workspace = D1Workspace()
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let panel = BrainBarDashboardPanelController(workspace: workspace, runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime,
            dashboardPanelController: panel, statusBar: TestStatusItemHost())
        controller.setLogPathsForTesting(paths)
        defer { controller.stop() }
        let menu = controller.contextMenuForTesting
        let action = try XCTUnwrap(menu.items.first { $0.action == NSSelectorFromString("showJobAlertLog:") })
        XCTAssertTrue(action.target === controller)
        XCTAssertFalse(menu.autoenablesItems)
        func badge(_ key: String, reason: String = "Failed") {
            controller.setBadgeForTesting(.init(badgeOn: true, reason: reason, activeCodes: ["job_alert_\(key)"]),
                alertsURL: root.appendingPathComponent("absent-alerts.json"))
            controller.menuNeedsUpdate(menu)
        }
        for key in ["unknown-job", "maintenance-light"] {
            badge(key)
            XCTAssertFalse(action.isHidden)
            XCTAssertTrue(action.isEnabled, "the fallback must be clickable")
            if action.isEnabled {
                XCTAssertTrue(NSApplication.shared.sendAction(try XCTUnwrap(action.action), to: action.target, from: action))
            }
            XCTAssertEqual(workspace.opened.last?.path, root.path)
        }
        try Data("log".utf8).write(to: paths.maintenanceLog)
        badge("maintenance-light")
        XCTAssertEqual(action.title, "Show log")
        XCTAssertTrue(NSApplication.shared.sendAction(try XCTUnwrap(action.action), to: action.target, from: action))
        XCTAssertEqual(workspace.opened.last, paths.maintenanceLog)

        workspace.acceptsOpen = false
        XCTAssertTrue(NSApplication.shared.sendAction(try XCTUnwrap(action.action), to: action.target, from: action))
        controller.menuNeedsUpdate(menu)
        let note = menu.items.first { !$0.isHidden && $0.title.contains("Could not open") }
        XCTAssertNotNil(note)
        XCTAssertNil(note?.action, "feedback must not pretend to be an opener")
        XCTAssertEqual(note?.isEnabled, false)
        XCTAssertTrue(action.isEnabled, "failure feedback must not disable retry")
        workspace.acceptsOpen = true
        XCTAssertTrue(NSApplication.shared.sendAction(try XCTUnwrap(action.action), to: action.target, from: action))
        XCTAssertFalse(menu.items.contains { !$0.isHidden && $0.title.contains("Could not open") })
        workspace.acceptsOpen = false
        XCTAssertTrue(NSApplication.shared.sendAction(try XCTUnwrap(action.action), to: action.target, from: action))
        badge("maintenance-light", reason: "A different failure")
        XCTAssertFalse(menu.items.contains { !$0.isHidden && $0.title.contains("Could not open") })
        controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []), alertsURL: root)
        controller.menuNeedsUpdate(menu)
        XCTAssertTrue(action.isHidden)
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

private final class D1Workspace: BrainBarWorkspaceActing, @unchecked Sendable {
    var opened: [URL] = []
    var acceptsOpen = true
    func open(_ url: URL) { opened.append(url) }
    func openLog(_ url: URL) -> Bool { opened.append(url); return acceptsOpen }
    func reveal(_ url: URL) {}
    func copy(_ text: String) {}
}
