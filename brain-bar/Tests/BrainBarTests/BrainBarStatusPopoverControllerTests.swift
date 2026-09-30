import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStatusPopoverControllerTests: XCTestCase {
    func testControllerWiresVariableLengthStatusItemToTheWindow() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let panelController = BrainBarDashboardPanelController(runtime: runtime)
        let controller = BrainBarStatusPopoverController(
            runtime: runtime,
            dashboardPanelController: panelController
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
        let windowController = BrainBarDashboardPanelController(runtime: runtime)
        let controller = BrainBarStatusPopoverController(
            runtime: runtime,
            dashboardPanelController: windowController
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
        let windowController = BrainBarDashboardPanelController(runtime: runtime)
        let controller = BrainBarStatusPopoverController(
            runtime: runtime,
            dashboardPanelController: windowController
        )
        defer { controller.stop() }

        let menu = controller.contextMenuForTesting
        // The first row is the informational status line; the rest are actions.
        let actionableItems = menu.items.filter { !$0.isSeparatorItem }.dropFirst()

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
        let windowController = BrainBarDashboardPanelController(runtime: runtime)
        let controller = BrainBarStatusPopoverController(
            runtime: runtime,
            dashboardPanelController: windowController
        )
        defer { controller.stop() }

        let toggle = controller.contextMenuForTesting.items.first { $0.title == "Open Dashboard" }

        XCTAssertNotNil(toggle?.action)
        XCTAssertTrue(toggle?.target === controller)
        XCTAssertTrue(toggle.map { controller.responds(to: $0.action!) } ?? false)
    }

    func testAppSupportCollectorFactoryWiresBrainBusEvents() {
        let tempDBPath = NSTemporaryDirectory() + "brainbar-status-popover-\(UUID().uuidString).db"
        let eventSource = RecordingBrainBusEventSource()
        let collector = BrainBarAppSupport.makeStatsCollector(
            dbPath: tempDBPath,
            targetPID: ProcessInfo.processInfo.processIdentifier,
            agentActivityMonitor: .synthetic,
            brainBusEvents: eventSource
        )
        defer {
            collector.stop()
            try? FileManager.default.removeItem(atPath: tempDBPath)
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
