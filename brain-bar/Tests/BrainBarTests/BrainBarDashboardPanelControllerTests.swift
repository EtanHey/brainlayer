import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarDashboardPanelControllerTests: XCTestCase {
    override func setUp() {
        super.setUp()
        UserDefaults.standard.removeObject(forKey: "NSWindow Frame BrainBarPanel")
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: "NSWindow Frame BrainBarPanel")
        super.tearDown()
    }

    func testDashboardPanelUsesResizableUtilityWindowContract() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let panel = controller.panelForTesting

        XCTAssertTrue(panel.styleMask.contains(.titled))
        XCTAssertTrue(panel.styleMask.contains(.resizable))
        XCTAssertTrue(panel.styleMask.contains(.closable))
        XCTAssertTrue(panel.styleMask.contains(.nonactivatingPanel))
        XCTAssertEqual(panel.frameAutosaveName, "BrainBarPanel")
        XCTAssertEqual(BrainBarDashboardPanelController.defaultSize, NSSize(width: 900, height: 640))
        XCTAssertEqual(panel.minSize, NSSize(width: 760, height: 560))
        XCTAssertGreaterThanOrEqual(panel.frame.width, panel.minSize.width)
        XCTAssertGreaterThanOrEqual(panel.frame.height, panel.minSize.height)
        XCTAssertTrue(panel.canBecomeKey)
        XCTAssertTrue(panel.canBecomeMain)
    }

    func testDashboardPanelOnlyNeedsInitialPositioningWithoutAutosavedFrame() {
        XCTAssertTrue(
            BrainBarDashboardPanelController.needsInitialPositioning(
                defaults: UserDefaults.standard
            )
        )

        UserDefaults.standard.set("saved frame", forKey: "NSWindow Frame BrainBarPanel")

        XCTAssertFalse(
            BrainBarDashboardPanelController.needsInitialPositioning(
                defaults: UserDefaults.standard
            )
        )
    }

    func testDashboardPanelToggleControlsVisibility() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())

        controller.toggle()
        XCTAssertTrue(controller.panelForTesting.isVisible)
        XCTAssertFalse(controller.needsInitialPositioningForTesting)

        controller.toggle()
        XCTAssertFalse(controller.panelForTesting.isVisible)
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

        // Match launch order: AppDelegate creates the hidden NSPanel before the
        // async database install lands, then the user opens BrainBar later.
        _ = controller.panelForTesting.contentViewController?.view
        runMainRunLoop()

        runtime.install(collector: collector, injectionStore: nil, database: db)
        runMainRunLoop()

        controller.show()
        runMainRunLoop()

        let field = controller.panelForTesting.contentView.flatMap {
            findSubview(ofType: KeyHandlingCommandBarField.self, in: $0)
        }
        XCTAssertNotNil(
            field,
            "Command bar should create its ready text field when runtime.database was installed before the panel became visible."
        )
    }

    private func runMainRunLoop() {
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
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
