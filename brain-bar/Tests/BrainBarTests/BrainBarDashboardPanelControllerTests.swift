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

    func testEveryDashboardDisclosureUsesTheSharedContainerAndWindowDriver() throws {
        let source = try String(
            contentsOf: packageRoot().appendingPathComponent("Sources/BrainBar/BrainBarWindowRootView.swift"),
            encoding: .utf8
        )
        let details = try XCTUnwrap(source.range(of: "private func diagnostics"))
        let detailsEnd = try XCTUnwrap(source.range(of: "private var daemonSummary", range: details.upperBound..<source.endIndex))
        XCTAssertTrue(String(source[details.lowerBound..<detailsEnd.lowerBound]).contains("BrainBarDisclosureRow("))

        let signal = try XCTUnwrap(source.range(of: "private struct BrainBarSignalCoveragePanel"))
        let signalEnd = try XCTUnwrap(source.range(of: "private struct BrainBarSignalCoverageRow", range: signal.upperBound..<source.endIndex))
        XCTAssertTrue(String(source[signal.lowerBound..<signalEnd.lowerBound]).contains("BrainBarDisclosureRow("))
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

    func testDisclosureExpansionResetsAStaleDashboardScrollOffsetToTopButCollapseDoesNot() {
        let collapsed = BrainBarDashboardDisclosureState(
            detailsExpanded: false,
            signalCoverageExpanded: false
        )
        let detailsExpanded = BrainBarDashboardDisclosureState(
            detailsExpanded: true,
            signalCoverageExpanded: false
        )
        let signalExpanded = BrainBarDashboardDisclosureState(
            detailsExpanded: true,
            signalCoverageExpanded: true
        )

        XCTAssertTrue(detailsExpanded.opensDisclosure(comparedTo: collapsed))
        XCTAssertTrue(signalExpanded.opensDisclosure(comparedTo: detailsExpanded))
        XCTAssertFalse(detailsExpanded.opensDisclosure(comparedTo: signalExpanded))
        XCTAssertFalse(collapsed.opensDisclosure(comparedTo: detailsExpanded))

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
        controller.setVisibleFrameForTesting(CGRect(x: 0, y: 0, width: 1_600, height: 1_200))
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

    func testSignalCoverageUsesTheSameWindowHeightDriverInBothDirections() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        _ = controller.contentViewControllerForTesting.view
        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        let collapsedSignalHeight = controller.panelForTesting.contentLayoutRect.height

        controller.setSignalCoverageExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        let expandedSignalHeight = controller.panelForTesting.contentLayoutRect.height
        XCTAssertGreaterThan(expandedSignalHeight, collapsedSignalHeight)

        controller.setSignalCoverageExpandedForTesting(false)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        XCTAssertEqual(
            controller.panelForTesting.contentLayoutRect.height,
            collapsedSignalHeight,
            accuracy: 1
        )
    }

    func testDisclosureCompletionAppliesOneContentSizeWithoutReentryOrReanchoring() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let panel = controller.panelForTesting
        let visibleFrame = CGRect(x: 0, y: 0, width: 2_000, height: 2_000)
        controller.setVisibleFrameForTesting(visibleFrame)
        let anchorWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 32, height: 24),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        let anchorView = NSView(frame: NSRect(x: 0, y: 0, width: 32, height: 24))
        anchorWindow.contentView = anchorView
        controller.statusItemButton = anchorView

        _ = controller.contentViewControllerForTesting.view
        RunLoop.main.run(until: Date().addingTimeInterval(0.3))
        controller.setDashboardHeightForTesting(400)
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        panel.setFrameOrigin(NSPoint(
            x: visibleFrame.minX + 80,
            y: visibleFrame.maxY - 80 - panel.frame.height
        ))
        let originalTopLeft = NSPoint(x: panel.frame.minX, y: panel.frame.maxY)
        controller.resetGeometryMetricsForTesting()
        for height in stride(from: CGFloat(420), through: 780, by: 20) {
            controller.setDashboardHeightForTesting(height)
            RunLoop.main.run(until: Date().addingTimeInterval(1.0 / 60.0))
        }
        controller.contentSizeDidApplyForTesting = { [weak controller] in
            controller?.windowDidResize(Notification(name: NSWindow.didResizeNotification, object: panel))
        }
        controller.completeDisclosureTransitionForTesting(expanded: true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.6))
        controller.contentSizeDidApplyForTesting = nil

        XCTAssertEqual(controller.contentSizeApplicationCountForTesting, 1)
        XCTAssertEqual(controller.reentrantFitAttemptCountForTesting, 0)
        XCTAssertEqual(panel.frame.minX, originalTopLeft.x, accuracy: 0.5)
        XCTAssertEqual(panel.frame.maxY, originalTopLeft.y, accuracy: 0.5)

        let expandedTopLeft = NSPoint(x: panel.frame.minX, y: panel.frame.maxY)
        controller.resetGeometryMetricsForTesting()
        for height in stride(from: CGFloat(760), through: 360, by: -20) {
            controller.setDashboardHeightForTesting(height)
            RunLoop.main.run(until: Date().addingTimeInterval(1.0 / 60.0))
        }
        controller.contentSizeDidApplyForTesting = { [weak controller] in
            controller?.windowDidResize(Notification(name: NSWindow.didResizeNotification, object: panel))
        }
        controller.completeDisclosureTransitionForTesting(expanded: false)
        RunLoop.main.run(until: Date().addingTimeInterval(0.6))
        controller.contentSizeDidApplyForTesting = nil

        XCTAssertEqual(controller.contentSizeApplicationCountForTesting, 1)
        XCTAssertEqual(controller.reentrantFitAttemptCountForTesting, 0)
        XCTAssertEqual(panel.frame.minX, expandedTopLeft.x, accuracy: 0.5)
        XCTAssertEqual(panel.frame.maxY, expandedTopLeft.y, accuracy: 0.5)
    }

    func testDisclosureGrowthClampsBelowPreservedTopLeftOnConstrainedScreen() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let panel = controller.panelForTesting
        let visibleFrame = CGRect(x: 100, y: 200, width: 900, height: 654)
        controller.setVisibleFrameForTesting(visibleFrame)

        _ = controller.contentViewControllerForTesting.view
        RunLoop.main.run(until: Date().addingTimeInterval(0.4))
        controller.setDashboardHeightForTesting(400)
        controller.completeDisclosureTransitionForTesting(expanded: false)
        RunLoop.main.run(until: Date().addingTimeInterval(0.4))
        panel.setFrameOrigin(NSPoint(
            x: visibleFrame.minX,
            y: visibleFrame.maxY - 80 - panel.frame.height
        ))
        let originalTopLeft = NSPoint(x: panel.frame.minX, y: panel.frame.maxY)

        controller.setDashboardHeightForTesting(900)
        controller.completeDisclosureTransitionForTesting(expanded: true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        XCTAssertTrue(visibleFrame.contains(panel.frame))
        XCTAssertEqual(panel.frame.minX, originalTopLeft.x, accuracy: 0.5)
        XCTAssertEqual(panel.frame.maxY, originalTopLeft.y, accuracy: 0.5)
    }

    func testDisclosureGrowthResolvesRealAnchorScreenAndPanelScreenFallback() throws {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let panel = controller.panelForTesting
        let screen = try XCTUnwrap(NSScreen.main ?? NSScreen.screens.first)
        let visibleFrame = screen.visibleFrame
        let anchorWindow = NSWindow(
            contentRect: NSRect(
                x: visibleFrame.maxX - 40,
                y: visibleFrame.maxY - 28,
                width: 32,
                height: 24
            ),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        let anchorView = NSView(frame: NSRect(x: 0, y: 0, width: 32, height: 24))
        anchorWindow.contentView = anchorView
        controller.statusItemButton = anchorView

        _ = controller.contentViewControllerForTesting.view
        RunLoop.main.run(until: Date().addingTimeInterval(0.4))
        panel.setFrameOrigin(NSPoint(
            x: visibleFrame.minX + 20,
            y: visibleFrame.maxY - 40 - panel.frame.height
        ))
        controller.setDashboardHeightForTesting(visibleFrame.height + 400)
        controller.completeDisclosureTransitionForTesting(expanded: true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        XCTAssertTrue(visibleFrame.contains(panel.frame))

        controller.statusItemButton = nil
        controller.setDashboardHeightForTesting(visibleFrame.height + 500)
        controller.completeDisclosureTransitionForTesting(expanded: true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        XCTAssertTrue(visibleFrame.contains(panel.frame))
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

    private func packageRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
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
