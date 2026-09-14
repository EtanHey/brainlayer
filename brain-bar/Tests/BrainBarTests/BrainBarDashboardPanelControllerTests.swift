import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarDashboardPanelControllerTests: XCTestCase {
    func testDashboardPanelUsesResizableMenuBarWindowContract() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let panel = controller.panelForTesting

        XCTAssertEqual(BrainBarDashboardPanelController.defaultSize.width, 900)
        XCTAssertEqual(BrainBarDashboardPanelController.minSize.width, 760)
        XCTAssertLessThan(BrainBarDashboardPanelController.minSize.height, 560)
        XCTAssertGreaterThan(panel.maxSize.width, BrainBarDashboardPanelController.defaultSize.width)
        XCTAssertGreaterThan(panel.maxSize.height, BrainBarDashboardPanelController.defaultSize.height)
        XCTAssertEqual(panel.minSize, BrainBarDashboardPanelController.minSize)
        XCTAssertTrue(panel.styleMask.contains(.resizable))
        XCTAssertFalse(panel.hidesOnDeactivate)
        XCTAssertEqual(panel.contentViewController, controller.contentViewControllerForTesting)
        XCTAssertEqual(controller.contentViewControllerForTesting.view.frame.size, BrainBarDashboardPanelController.defaultSize)
    }

    func testDashboardPanelFitsRestingContentAndGrowsForDetails() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        _ = controller.contentViewControllerForTesting.view
        runMainRunLoop(0.5)

        let restingHeight = controller.panelForTesting.contentLayoutRect.height
        XCTAssertEqual(restingHeight, controller.contentViewControllerForTesting.view.fittingSize.height, accuracy: 2)
        XCTAssertLessThan(restingHeight, 640)

        controller.setDetailsExpandedForTesting(true)
        runMainRunLoop(0.5)
        XCTAssertGreaterThan(controller.panelForTesting.contentLayoutRect.height, restingHeight)
    }

    func testDashboardPanelDoesNotOpenWithoutStatusItemAnchor() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())

        controller.toggle()
        XCTAssertFalse(controller.isShownForTesting)
    }

    func testDashboardLayoutReflowsAtMinFloorAndLargeWindowSizes() {
        let floorLayout = BrainBarDashboardLayout(containerSize: CGSize(width: 760, height: 560))
        XCTAssertEqual(floorLayout.chartColumns, 1)
        XCTAssertEqual(floorLayout.diagnosticColumns, 1)
        XCTAssertTrue(floorLayout.compactCards)

        let largeLayout = BrainBarDashboardLayout(containerSize: CGSize(width: 1_348, height: 1_078))
        XCTAssertEqual(largeLayout.chartColumns, 2)
        XCTAssertEqual(largeLayout.diagnosticColumns, 2)
        XCTAssertFalse(largeLayout.compactCards)
    }

    func testCommandBarBecomesReadyWhenDatabaseWasInstalledWhilePanelWasHidden() {
        BrainBarRetrievalToolsSettings.shared.update(enabled: true)
        defer { BrainBarRetrievalToolsSettings.shared.update(enabled: false) }

        let runtime = BrainBarRuntime()
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        let tempDBPath = NSTemporaryDirectory() + "brainbar-commandbar-ready-\(UUID().uuidString).db"
        let db = BrainDatabase(path: tempDBPath)
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: getpid())
        )
        defer {
            collector.stop()
            db.close()
            controller.dismiss()
            try? FileManager.default.removeItem(atPath: tempDBPath)
            try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
            try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        }

        // Match launch order: AppDelegate creates the hidden popover content before
        // the async database install lands, then the user opens BrainBar later.
        _ = controller.contentViewControllerForTesting.view
        runMainRunLoop()

        runtime.install(collector: collector, database: db)
        runMainRunLoop()

        let anchorWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 32, height: 24),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        let anchorView = NSView(frame: NSRect(x: 0, y: 0, width: 32, height: 24))
        anchorWindow.contentView = anchorView
        anchorWindow.orderFront(nil)
        defer { anchorWindow.orderOut(nil) }

        controller.show(anchoredTo: anchorView)
        runMainRunLoop()

        let field = findSubview(
            ofType: KeyHandlingCommandBarField.self,
            in: controller.contentViewControllerForTesting.view
        )
        XCTAssertNotNil(
            field,
            "Command bar should create its ready text field when runtime.database was installed before the panel became visible."
        )
    }

    private func runMainRunLoop(_ duration: TimeInterval = 0.05) {
        RunLoop.main.run(until: Date().addingTimeInterval(duration))
    }

    private func findSubview<T: NSView>(ofType type: T.Type, in root: NSView) -> T? {
        if let match = root as? T {
            return match
        }
        for subview in root.subviews {
            if let match = findSubview(ofType: type, in: subview) {
                return match
            }
        }
        return nil
    }
}
