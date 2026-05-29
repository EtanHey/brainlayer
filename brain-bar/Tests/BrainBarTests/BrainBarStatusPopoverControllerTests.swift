import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStatusPopoverControllerTests: XCTestCase {
    func testControllerWiresVariableLengthStatusItemToUnifiedWindow() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let windowController = BrainBarDashboardPanelController(runtime: runtime)
        let controller = BrainBarStatusPopoverController(
            runtime: runtime,
            dashboardPanelController: windowController
        )
        defer { controller.stop() }

        XCTAssertEqual(controller.statusItemForTesting.length, NSStatusItem.variableLength)
        XCTAssertEqual(controller.statusItemForTesting.button?.target as? BrainBarStatusPopoverController, controller)
        XCTAssertTrue(BrainBarStatusPopoverController.statusItemEventMask.contains(.rightMouseUp))
        XCTAssertNotNil(windowController.panelForTesting.contentViewController)
        XCTAssertEqual(windowController.panelForTesting.minSize, NSSize(width: 760, height: 560))
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

        XCTAssertTrue(itemTitles.contains("Restart BrainBar"))
        XCTAssertFalse(itemTitles.contains("Run as App Window"))
        XCTAssertFalse(itemTitles.contains("Run as Menu Item Daemon"))
        XCTAssertTrue(itemTitles.contains("Quit BrainBar"))
    }

    func testAppSupportCollectorFactoryWiresBrainBusEvents() {
        let tempDBPath = NSTemporaryDirectory() + "brainbar-status-popover-\(UUID().uuidString).db"
        let eventSource = RecordingBrainBusEventSource()
        let collector = BrainBarAppSupport.makeStatsCollector(
            dbPath: tempDBPath,
            targetPID: ProcessInfo.processInfo.processIdentifier,
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
