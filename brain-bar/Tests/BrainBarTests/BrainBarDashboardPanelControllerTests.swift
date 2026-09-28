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

    func testURLDestinationsShowSamePanelAndSelectRequestedSettingsSection() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let anchor = NSView(frame: NSRect(x: 0, y: 0, width: 24, height: 24))
        controller.statusItemButton = anchor
        defer {
            controller.dismiss()
            BrainBarSettingsActions.installOpenHandler {}
        }
        let panel = controller.panelForTesting

        for section in BrainBarSettingsSection.allCases {
            let action = BrainBarURLAction.parse(url: URL(string: "brainbar://settings/\(section.rawValue)")!)!
            controller.showURLDestination(action)
            XCTAssertTrue(controller.isShownForTesting)
            XCTAssertTrue(controller.panelForTesting === panel)
            XCTAssertEqual(controller.selectedTabForTesting, .settings)
            XCTAssertEqual(controller.selectedSettingsSectionForTesting, section)
        }

        BrainBarSettingsActions.openSettingsWindow(databasePath: nil)
        XCTAssertEqual(controller.selectedSettingsSectionForTesting, .advanced)
        controller.showURLDestination(.dashboard)
        controller.showURLDestination(.dashboard)
        XCTAssertTrue(controller.isShownForTesting)
        XCTAssertTrue(controller.panelForTesting === panel)
        XCTAssertEqual(controller.selectedTabForTesting, .dashboard)
        controller.showURLDestination(BrainBarURLAction.parse(url: URL(string: "brainbar://settings/unknown")!)!)
        controller.showURLDestination(BrainBarURLAction.parse(url: URL(string: "brainbar://settings")!)!)
        XCTAssertTrue(controller.isShownForTesting)
        XCTAssertEqual(controller.selectedSettingsSectionForTesting, .general)
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
            var anchor = origin.y
            for expanded in [false, true] {
                controller.setDetailsExpandedForTesting(expanded)
                pumpLayout(panel)
                XCTAssertEqual(panel.frame, frame, "Details transition moved or resized width \(width)")
                // The anchor survives wherever the reflowed content still reaches it; where it no
                // longer does, the viewport settles on the last real content, never a blank tail (#964).
                let natural = controller.naturalDashboardHeightForTesting
                // Re-expanding keeps a settled anchor; it never jumps back to the pre-collapse offset.
                anchor = min(anchor, max(natural - clip.bounds.height, 0))
                XCTAssertEqual(clip.bounds.minY, anchor, accuracy: 1, "Details transition jumped scroll at width \(width)")
                XCTAssertLessThanOrEqual(scroll.documentView!.bounds.height, max(natural, clip.bounds.height) + 2)
            }
            if width == 760 {
                panel.setFrame(NSRect(x: -2_000, y: -2_000, width: 1_280, height: 640), display: false)
                controller.setDetailsExpandedForTesting(false)
                pumpLayout(panel)
                let natural = controller.naturalDashboardHeightForTesting
                XCTAssertEqual(clip.bounds.minY, min(80, max(natural - clip.bounds.height, 0)), accuracy: 1)
                XCTAssertLessThanOrEqual(scroll.documentView!.bounds.height, max(natural, clip.bounds.height) + 2)
            }
            controller.setDetailsExpandedForTesting(false)
            clip.scroll(to: .zero)
            scroll.reflectScrolledClipView(clip)
            pumpLayout(panel)
            XCTAssertLessThanOrEqual(scroll.documentView!.bounds.height, max(controller.naturalDashboardHeightForTesting, clip.bounds.height) + 2)
        }
    }

    // #964: expand a disclosure, scroll to the bottom, collapse it. The document must shrink back
    // to its natural height and the viewport must end on real content, never on a blank tail.
    func testCollapsingADisclosureAfterScrollingToTheBottomLeavesNoBlankTail() throws {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let disclosures: [(String, (BrainBarDashboardPanelController, Bool) -> Void)] = [
            ("details", { $0.setDetailsExpandedForTesting($1) }),
            ("signal coverage", { controller, expanded in
                controller.setDetailsExpandedForTesting(true)
                controller.setSignalCoverageExpandedForTesting(expanded)
            }),
        ]
        for width: CGFloat in [760, 960, 1_280] {
            for (name, setExpanded) in disclosures {
                let controller = BrainBarDashboardPanelController(runtime: runtime)
                let panel = controller.panelForTesting
                panel.setFrame(NSRect(x: -2_000, y: -2_000, width: width, height: 640), display: false)
                setExpanded(controller, true)
                let scroll = try dashboardScroll(in: controller)
                pumpLayout(panel)
                let clip = scroll.contentView
                let expandedDocument = scroll.documentView!.bounds.height
                clip.scroll(to: NSPoint(x: 0, y: expandedDocument - clip.bounds.height))
                scroll.reflectScrolledClipView(clip)
                pumpLayout(panel)
                XCTAssertGreaterThan(clip.bounds.minY, 0, "\(name) at \(width) must start scrolled")

                setExpanded(controller, false)
                pumpLayout(panel)

                let natural = controller.naturalDashboardHeightForTesting
                let document = scroll.documentView!.bounds.height
                XCTAssertLessThan(natural, expandedDocument - 1, "\(name) at \(width) collapse must shrink content")
                XCTAssertLessThanOrEqual(
                    document, max(natural, clip.bounds.height) + 2,
                    "\(name) at \(width): collapsed document kept a blank tail (\(document) vs natural \(natural))"
                )
                XCTAssertLessThanOrEqual(
                    clip.bounds.maxY, max(natural, clip.bounds.height) + 2,
                    "\(name) at \(width): viewport ends below the content"
                )
            }
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
