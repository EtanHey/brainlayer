import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarDashboardPanelControllerTests: XCTestCase {
    func testScrollOriginClampExcludesElasticOverscroll() {
        XCTAssertEqual(BrainBarScrollOrigin.clamped(-24, documentHeight: 1_202, viewportHeight: 558), 0)
        XCTAssertEqual(BrainBarScrollOrigin.clamped(644, documentHeight: 1_202, viewportHeight: 558), 644)
        XCTAssertEqual(BrainBarScrollOrigin.clamped(704, documentHeight: 1_202, viewportHeight: 558), 644)
        XCTAssertEqual(BrainBarScrollOrigin.clamped(24, documentHeight: 556, viewportHeight: 558), 0)
    }

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
        XCTAssertTrue(panel.canBecomeKey)
        XCTAssertFalse(panel.canBecomeMain)
        XCTAssertFalse(panel.becomesKeyOnlyIfNeeded)
    }

    func testOpeningSettingsTwiceSelectsTheSameVisiblePanel() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let anchor = NSView(frame: NSRect(x: 0, y: 0, width: 24, height: 24))
        controller.statusItemButton = anchor
        defer { controller.dismiss() }

        controller.showSettings()
        XCTAssertEqual(controller.selectedTabForTesting, .settings)
        XCTAssertTrue(controller.isShownForTesting)
        let panel = controller.panelForTesting
        controller.showSettings()
        XCTAssertTrue(controller.isShownForTesting)
        XCTAssertTrue(controller.panelForTesting === panel)
        XCTAssertFalse(NSApp.windows.contains { $0.title == "BrainLayer Settings" })
    }

    func testDashboardPanelKeepsRestingHeightWhenDetailsExpands() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        _ = controller.contentViewControllerForTesting.view
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        let restingHeight = controller.panelForTesting.contentLayoutRect.height
        XCTAssertLessThanOrEqual(restingHeight, controller.panelForTesting.maxSize.height)

        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        let expandedHeight = controller.panelForTesting.contentLayoutRect.height
        XCTAssertEqual(expandedHeight, restingHeight)
    }

    func testSignalCoverageKeepsFixedWindowHeightInBothDirections() {
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
        XCTAssertEqual(expandedSignalHeight, collapsedSignalHeight)

        controller.setSignalCoverageExpandedForTesting(false)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        XCTAssertEqual(controller.panelForTesting.contentLayoutRect.height, collapsedSignalHeight, accuracy: 1)
    }

    func testSearchOverlayKeepsExpandedDetailsWindowHeight() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        _ = controller.contentViewControllerForTesting.view
        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        let expandedHeight = controller.panelForTesting.contentLayoutRect.height
        controller.setSearchOverlayPresentedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        XCTAssertEqual(controller.panelForTesting.contentLayoutRect.height, expandedHeight, accuracy: 1)
    }

    func testDetailsTransitionsPreserveFixedFrameAndScrollAtSupportedWidths() throws {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        for width: CGFloat in [760, 960, 1_280] {
            let controller = BrainBarDashboardPanelController(runtime: runtime)
            let panel = controller.panelForTesting
            panel.setFrame(NSRect(x: -2_000, y: -2_000, width: width, height: 640), display: false)
            controller.setDetailsExpandedForTesting(true)
            let scroll = try dashboardScroll(in: controller)
            pumpLayout(panel)
            let clip = scroll.contentView
            clip.scroll(to: NSPoint(x: 0, y: 80))
            scroll.reflectScrolledClipView(clip)
            let frame = panel.frame
            let origin = clip.bounds.origin
            XCTAssertGreaterThan(origin.y, 0, "width \(width) needs a real nonzero anchor")
            for expanded in [false, true] {
                controller.setDetailsExpandedForTesting(expanded)
                pumpLayout(panel)
                XCTAssertEqual(panel.frame, frame, "Details transition moved or resized width \(width)")
                XCTAssertEqual(clip.bounds.origin, origin, "Details transition lost scroll at width \(width)")
                if !expanded {
                    let natural = controller.naturalDashboardHeightForTesting
                    XCTAssertLessThanOrEqual(scroll.documentView!.bounds.height, max(natural, origin.y + clip.bounds.height) + 2)
                }
            }
            if width == 760 {
                panel.setFrame(NSRect(x: -2_000, y: -2_000, width: 1_280, height: 640), display: false)
                controller.setDetailsExpandedForTesting(false)
                pumpLayout(panel)
                XCTAssertEqual(clip.bounds.minY, 80, accuracy: 1)
                XCTAssertLessThanOrEqual(scroll.documentView!.bounds.height, max(controller.naturalDashboardHeightForTesting, 80 + clip.bounds.height) + 2)
            }
            controller.setDetailsExpandedForTesting(false)
            clip.scroll(to: .zero)
            scroll.reflectScrolledClipView(clip)
            pumpLayout(panel)
            XCTAssertLessThanOrEqual(scroll.documentView!.bounds.height, max(controller.naturalDashboardHeightForTesting, clip.bounds.height) + 2)
        }
    }

    func testHideAndShowPreserveFrameAndScrollAgainstSameAnchor() throws {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        let panel = controller.panelForTesting
        let visible = NSScreen.screens.first?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1_024, height: 768)
        let anchorWindow = NSWindow(
            contentRect: NSRect(x: visible.maxX - 48, y: visible.maxY - 32, width: 32, height: 24),
            styleMask: [.borderless], backing: .buffered, defer: false
        )
        let anchor = NSView(frame: NSRect(x: 0, y: 0, width: 32, height: 24))
        anchorWindow.contentView = anchor
        anchorWindow.orderFront(nil)
        defer { controller.dismiss(); anchorWindow.orderOut(nil) }
        controller.show(anchoredTo: anchor)
        controller.setDetailsExpandedForTesting(true)
        let scroll = try dashboardScroll(in: controller)
        pumpLayout(panel)
        let clip = scroll.contentView
        clip.scroll(to: NSPoint(x: 0, y: 80))
        scroll.reflectScrolledClipView(clip)
        panel.setFrameOrigin(NSPoint(x: panel.frame.minX - 10, y: panel.frame.minY + 10))
        let frame = panel.frame
        let origin = clip.bounds.origin
        XCTAssertGreaterThan(origin.y, 0)
        controller.dismiss()
        controller.show(anchoredTo: anchor)
        pumpLayout(panel)
        XCTAssertEqual(panel.frame, frame)
        XCTAssertEqual(clip.bounds.origin, origin)
        anchorWindow.setFrameOrigin(NSPoint(x: visible.minX + 80, y: visible.maxY - 32))
        controller.dismiss()
        controller.show(anchoredTo: anchor)
        XCTAssertNotEqual(panel.frame.origin, frame.origin)
        XCTAssertGreaterThanOrEqual(panel.frame.minX, visible.minX)
        XCTAssertLessThanOrEqual(panel.frame.maxX, visible.maxX)
        let second = NSRect(x: 1_440, y: -200, width: 1_280, height: 800)
        let anchorRect = NSRect(x: second.maxX - 24, y: second.maxY - 30, width: 20, height: 20)
        let target = BrainBarDashboardPanelController.anchorOrigin(anchorRect: anchorRect, panelSize: .init(width: 900, height: 640), visibleFrame: second)
        XCTAssertTrue(second.contains(NSRect(origin: target, size: .init(width: 900, height: 640))))
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

    private func pumpLayout(_ panel: NSPanel) {
        for _ in 0..<5 {
            panel.contentView?.layoutSubtreeIfNeeded()
            RunLoop.main.run(until: Date().addingTimeInterval(0.02))
        }
    }

    private func dashboardScroll(in controller: BrainBarDashboardPanelController) throws -> NSScrollView {
        _ = controller.contentViewControllerForTesting.view
        pumpLayout(controller.panelForTesting)
        return try XCTUnwrap(findSubview(ofType: NSScrollView.self, in: controller.contentViewControllerForTesting.view))
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
