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
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        let restingHeight = controller.panelForTesting.contentLayoutRect.height
        XCTAssertLessThanOrEqual(controller.measuredContentHeightForTesting, restingHeight + 1)
        XCTAssertLessThan(restingHeight, 640)

        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        let expandedHeight = controller.panelForTesting.contentLayoutRect.height
        XCTAssertGreaterThan(expandedHeight, restingHeight)
        XCTAssertLessThanOrEqual(controller.measuredContentHeightForTesting, expandedHeight + 1)
    }

    func testSearchOverlayDoesNotShrinkExpandedDetailsAndVerticalSizeIsPinned() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        _ = controller.contentViewControllerForTesting.view
        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        let expandedHeight = controller.panelForTesting.contentLayoutRect.height
        controller.setSearchOverlayPresentedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        XCTAssertGreaterThanOrEqual(controller.panelForTesting.contentLayoutRect.height, expandedHeight - 1)
        XCTAssertEqual(
            controller.panelForTesting.contentMinSize.height,
            controller.panelForTesting.contentMaxSize.height,
            accuracy: 1
        )
        XCTAssertLessThan(
            controller.panelForTesting.contentMinSize.width,
            controller.panelForTesting.contentMaxSize.width
        )
    }

    func testResizeDelegatePreservesFittedHeightAndMinimumWidthAcrossRestingTransitions() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        let panel = controller.panelForTesting
        _ = controller.contentViewControllerForTesting.view
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        assertResizeDelegateKeepsCurrentHeightAndMinimumWidth(controller, panel: panel)

        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        controller.setDetailsExpandedForTesting(false)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        assertResizeDelegateKeepsCurrentHeightAndMinimumWidth(controller, panel: panel)

        let widerSize = controller.windowWillResize(
            panel,
            to: NSSize(width: panel.frame.width + 80, height: panel.frame.height + 300)
        )
        panel.setFrame(NSRect(origin: panel.frame.origin, size: widerSize), display: false)
        controller.windowDidResize(Notification(name: NSWindow.didResizeNotification, object: panel))
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        assertResizeDelegateKeepsCurrentHeightAndMinimumWidth(controller, panel: panel)

        let anchorWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 32, height: 24),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        let anchorView = NSView(frame: NSRect(x: 0, y: 0, width: 32, height: 24))
        anchorWindow.contentView = anchorView
        anchorWindow.orderFront(nil)
        defer {
            controller.dismiss()
            anchorWindow.orderOut(nil)
        }

        controller.show(anchoredTo: anchorView)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        assertResizeDelegateKeepsCurrentHeightAndMinimumWidth(controller, panel: panel)
    }

    func testDashboardPanelDoesNotOpenWithoutStatusItemAnchor() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())

        controller.toggle()
        XCTAssertFalse(controller.isShownForTesting)
    }

    func testDevPreviewPanelUsesIdentifyingTitleAndCanOpenWithoutStatusItem() {
        let title = "DEV · wt/badge-state-contract · 17e1ec37"
        let controller = BrainBarDashboardPanelController(
            runtime: BrainBarRuntime(),
            standaloneTitle: title
        )
        defer { controller.dismiss() }

        XCTAssertEqual(controller.panelForTesting.title, title)
        XCTAssertEqual(controller.panelForTesting.titleVisibility, .visible)

        controller.show()
        XCTAssertTrue(controller.isShownForTesting)

        controller.windowDidResignKey(Notification(name: NSWindow.didResignKeyNotification))
        XCTAssertTrue(controller.isShownForTesting, "DEV previews must remain visible side by side")
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

    private func runMainRunLoop() {
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
    }

    private func assertResizeDelegateKeepsCurrentHeightAndMinimumWidth(
        _ controller: BrainBarDashboardPanelController,
        panel: NSPanel,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let fittedFrameHeight = panel.frame.height
        let taller = controller.windowWillResize(
            panel,
            to: NSSize(width: panel.frame.width + 40, height: fittedFrameHeight + 300)
        )
        XCTAssertEqual(taller.height, fittedFrameHeight, accuracy: 1, file: file, line: line)

        let shorter = controller.windowWillResize(
            panel,
            to: NSSize(width: 100, height: max(fittedFrameHeight - 200, 1))
        )
        XCTAssertEqual(shorter.height, fittedFrameHeight, accuracy: 1, file: file, line: line)
        XCTAssertEqual(shorter.width, BrainBarDashboardPanelController.minSize.width, file: file, line: line)
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
