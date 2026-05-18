import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStatusPopoverControllerTests: XCTestCase {
    func testControllerPrewarmsVariableLengthStatusItemPopover() {
        let runtime = BrainBarRuntime(launchMode: .menuBarWindow)
        let controller = BrainBarStatusPopoverController(runtime: runtime)
        defer { controller.stop() }

        XCTAssertEqual(controller.statusItemForTesting.length, NSStatusItem.variableLength)
        XCTAssertEqual(controller.popoverForTesting.behavior, NSPopover.Behavior.transient)
        XCTAssertNotNil(controller.popoverForTesting.contentViewController)
        XCTAssertTrue(controller.popoverForTesting.contentViewController?.isViewLoaded == true)
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
