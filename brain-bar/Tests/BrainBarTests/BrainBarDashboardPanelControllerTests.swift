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
}
