import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarDashboardPanelControllerTests: XCTestCase {
    func testDisclosureAnimationUsesOneTimingInBothDirectionsAndReduceMotionIsInstant() {
        let opening = BrainBarDisclosureAnimation.timing(for: .open, reduceMotion: false)
        let closing = BrainBarDisclosureAnimation.timing(for: .close, reduceMotion: false)

        XCTAssertEqual(opening, closing)
        XCTAssertEqual(opening.duration, 0.25)
        XCTAssertEqual(opening.curve, .easeInOut)
        XCTAssertEqual(
            BrainBarDisclosureAnimation.timing(for: .open, reduceMotion: true).duration,
            0
        )
        XCTAssertEqual(
            BrainBarDisclosureAnimation.timing(for: .close, reduceMotion: true).duration,
            0
        )
    }

    func testDisclosureAnimationCouplesContainerAndWindowAtInteriorProgress() {
        let collapsedContainerHeight: CGFloat = 44
        let expandedContainerHeight: CGFloat = 612
        let chromeAndSurroundingContentHeight: CGFloat = 188
        let samples: [CGFloat] = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]

        let layouts = samples.map {
            BrainBarDisclosureAnimation.layout(
                progress: $0,
                collapsedContainerHeight: collapsedContainerHeight,
                expandedContainerHeight: expandedContainerHeight,
                chromeAndSurroundingContentHeight: chromeAndSurroundingContentHeight
            )
        }

        for layout in layouts {
            XCTAssertEqual(
                layout.windowHeight - chromeAndSurroundingContentHeight,
                layout.containerHeight,
                accuracy: 0.001
            )
        }
        XCTAssertEqual(layouts.first?.containerHeight, collapsedContainerHeight)
        XCTAssertEqual(layouts.last?.containerHeight, expandedContainerHeight)
        XCTAssertTrue(layouts.dropFirst().dropLast().allSatisfy {
            $0.containerHeight > collapsedContainerHeight && $0.containerHeight < expandedContainerHeight
        })
    }

    func testDisclosureChangeResetsAStaleDashboardScrollOffsetToTop() {
        XCTAssertEqual(
            BrainBarDashboardScrollPosition.topOrigin(
                documentBounds: CGRect(x: 0, y: 0, width: 900, height: 1_200),
                viewportHeight: 600,
                documentIsFlipped: true,
                currentX: 12
            ),
            CGPoint(x: 12, y: 0)
        )
        XCTAssertEqual(
            BrainBarDashboardScrollPosition.topOrigin(
                documentBounds: CGRect(x: 0, y: 40, width: 900, height: 1_200),
                viewportHeight: 600,
                documentIsFlipped: false,
                currentX: 4
            ),
            CGPoint(x: 4, y: 640)
        )
    }

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
        XCTAssertLessThanOrEqual(restingHeight, controller.panelForTesting.maxSize.height)

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
