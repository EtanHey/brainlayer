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
        XCTAssertEqual(BrainBarDashboardPanelController.defaultSize, NSSize(width: 1_348, height: 1_078))
        XCTAssertEqual(panel.minSize, NSSize(width: 760, height: 560))
        XCTAssertGreaterThanOrEqual(panel.frame.width, panel.minSize.width)
        XCTAssertGreaterThanOrEqual(panel.frame.height, panel.minSize.height)
        XCTAssertTrue(panel.canBecomeKey)
        XCTAssertTrue(panel.canBecomeMain)
    }

    func testDashboardPanelToggleControlsVisibility() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())

        controller.toggle()
        XCTAssertTrue(controller.panelForTesting.isVisible)

        controller.toggle()
        XCTAssertFalse(controller.panelForTesting.isVisible)
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
