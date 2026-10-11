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

    func testDashboardWindowUsesTheResizableWindowContract() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let window = controller.windowForTesting

        XCTAssertEqual(BrainBarDashboardPanelController.defaultSize, NSSize(width: 900, height: 640))
        XCTAssertEqual(BrainBarDashboardPanelController.minSize, NSSize(width: 760, height: 560))
        XCTAssertEqual(window.minSize, BrainBarDashboardPanelController.minSize)
        XCTAssertTrue(window.styleMask.contains(.resizable))
        XCTAssertFalse(window.hidesOnDeactivate)
        XCTAssertEqual(window.contentViewController, controller.contentViewControllerForTesting)
        XCTAssertEqual(controller.contentViewControllerForTesting.view.frame.size, BrainBarDashboardPanelController.defaultSize)
        XCTAssertTrue(window.canBecomeKey)
        XCTAssertTrue(window.canBecomeMain, "#963: a real window, not a panel that never becomes main")
    }

    func testOpeningSettingsTwiceSelectsTheSameVisiblePanel() {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let anchor = NSView(frame: NSRect(x: 0, y: 0, width: 24, height: 24))
        controller.statusItemButton = anchor
        defer { controller.dismiss() }

        controller.showSettings()
        XCTAssertEqual(controller.selectedTabForTesting, .settings)
        XCTAssertTrue(controller.isShownForTesting)
        let panel = controller.windowForTesting
        controller.showSettings()
        XCTAssertTrue(controller.isShownForTesting)
        XCTAssertTrue(controller.windowForTesting === panel)
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
        let panel = controller.windowForTesting

        for section in BrainBarSettingsSection.allCases {
            let action = BrainBarURLAction.parse(url: URL(string: "brainbar://settings/\(section.rawValue)")!)!
            controller.showURLDestination(action)
            XCTAssertTrue(controller.isShownForTesting)
            XCTAssertTrue(controller.windowForTesting === panel)
            XCTAssertEqual(controller.selectedTabForTesting, .settings)
            XCTAssertEqual(controller.selectedSettingsSectionForTesting, section)
        }

        BrainBarSettingsActions.openSettingsWindow(databasePath: nil)
        XCTAssertEqual(controller.selectedSettingsSectionForTesting, .advanced)
        controller.showURLDestination(.dashboard)
        controller.showURLDestination(.dashboard)
        XCTAssertTrue(controller.isShownForTesting)
        XCTAssertTrue(controller.windowForTesting === panel)
        XCTAssertEqual(controller.selectedTabForTesting, .dashboard)
        controller.showURLDestination(BrainBarURLAction.parse(url: URL(string: "brainbar://settings/unknown")!)!)
        controller.showURLDestination(BrainBarURLAction.parse(url: URL(string: "brainbar://settings")!)!)
        XCTAssertTrue(controller.isShownForTesting)
        XCTAssertEqual(controller.selectedSettingsSectionForTesting, .jobs, "a bare or unknown settings route opens the first settings page (#963)")
    }

    func testDashboardPanelKeepsRestingHeightWhenDetailsExpands() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        _ = controller.contentViewControllerForTesting.view
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        let restingHeight = controller.windowForTesting.contentLayoutRect.height
        XCTAssertLessThanOrEqual(restingHeight, controller.windowForTesting.maxSize.height)

        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        let expandedHeight = controller.windowForTesting.contentLayoutRect.height
        XCTAssertEqual(expandedHeight, restingHeight)
    }

    func testSignalCoverageKeepsFixedWindowHeightInBothDirections() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        _ = controller.contentViewControllerForTesting.view
        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        let collapsedSignalHeight = controller.windowForTesting.contentLayoutRect.height

        controller.setSignalCoverageExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        let expandedSignalHeight = controller.windowForTesting.contentLayoutRect.height
        XCTAssertEqual(expandedSignalHeight, collapsedSignalHeight)

        controller.setSignalCoverageExpandedForTesting(false)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        XCTAssertEqual(controller.windowForTesting.contentLayoutRect.height, collapsedSignalHeight, accuracy: 1)
    }



    func testDetailsTransitionsPreserveFixedFrameAndScrollAtSupportedWidths() throws {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        for width: CGFloat in [760, 960, 1_280] {
            let controller = BrainBarDashboardPanelController(runtime: runtime)
            let panel = controller.windowForTesting
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
        // The Attention row renders only when the status strip is amber with items: unmeasured
        // agent activity is one. Details stays open so the dashboard scrolls at every width.
        let attentionRuntime = BrainBarRuntime()
        attentionRuntime.install(
            collector: BrainBarDashboardFixture.makeCollector(agentActivity: .unavailable("fixture agent activity unavailable")),
            database: nil
        )
        let disclosures: [(String, BrainBarRuntime, (BrainBarDashboardPanelController, Bool) -> Void)] = [
            ("details", runtime, { $0.setDetailsExpandedForTesting($1) }),
            ("signal coverage", runtime, { controller, expanded in
                controller.setDetailsExpandedForTesting(true)
                controller.setSignalCoverageExpandedForTesting(expanded)
            }),
            ("attention", attentionRuntime, { controller, expanded in
                controller.setDetailsExpandedForTesting(true)
                controller.setAttentionExpandedForTesting(expanded)
            }),
        ]
        for width: CGFloat in [760, 960, 1_280] {
            for (name, runtime, setExpanded) in disclosures {
                let controller = BrainBarDashboardPanelController(runtime: runtime)
                let panel = controller.windowForTesting
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

    func testHideAndShowPreserveFrameAndScrollWhereverTheIconMoves() throws {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        let window = controller.windowForTesting
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
        pumpLayout(window)
        let clip = scroll.contentView
        clip.scroll(to: NSPoint(x: 0, y: 80))
        scroll.reflectScrolledClipView(clip)
        window.setFrameOrigin(NSPoint(x: window.frame.minX - 10, y: window.frame.minY + 10))
        let frame = window.frame
        let origin = clip.bounds.origin
        XCTAssertGreaterThan(origin.y, 0)
        controller.dismiss()
        controller.show(anchoredTo: anchor)
        pumpLayout(window)
        XCTAssertEqual(window.frame, frame)
        XCTAssertEqual(clip.bounds.origin, origin)
        // #963: a real window stays where the user left it; the menu-bar icon moving does not
        // drag it back under the icon.
        anchorWindow.setFrameOrigin(NSPoint(x: visible.minX + 80, y: visible.maxY - 32))
        controller.dismiss()
        controller.show(anchoredTo: anchor)
        XCTAssertEqual(window.frame, frame)
        // The first placement below the icon still stays on the icon's screen.
        let second = NSRect(x: 1_440, y: -200, width: 1_280, height: 800)
        let anchorRect = NSRect(x: second.maxX - 24, y: second.maxY - 30, width: 20, height: 20)
        let target = BrainBarDashboardPanelController.anchorOrigin(anchorRect: anchorRect, panelSize: .init(width: 900, height: 640), visibleFrame: second)
        XCTAssertTrue(second.contains(NSRect(origin: target, size: .init(width: 900, height: 640))))
    }

    func testResizeDelegateOnlyClampsToTheMinimumAcrossRestingTransitions() {
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
        let controller = BrainBarDashboardPanelController(runtime: runtime)
        let window = controller.windowForTesting
        _ = controller.contentViewControllerForTesting.view
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))

        assertResizeDelegateAllowsBothAxesDownToTheMinimum(controller, window: window)
        controller.setDetailsExpandedForTesting(true)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        controller.setDetailsExpandedForTesting(false)
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
        assertResizeDelegateAllowsBothAxesDownToTheMinimum(controller, window: window)
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

    private func pumpLayout(_ panel: NSWindow) {
        for _ in 0..<5 {
            panel.contentView?.layoutSubtreeIfNeeded()
            RunLoop.main.run(until: Date().addingTimeInterval(0.02))
        }
    }

    private func dashboardScroll(in controller: BrainBarDashboardPanelController) throws -> NSScrollView {
        _ = controller.contentViewControllerForTesting.view
        pumpLayout(controller.windowForTesting)
        return try XCTUnwrap(findSubview(ofType: NSScrollView.self, in: controller.contentViewControllerForTesting.view))
    }

    private func packageRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func assertResizeDelegateAllowsBothAxesDownToTheMinimum(
        _ controller: BrainBarDashboardPanelController,
        window: NSWindow,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let taller = NSSize(width: window.frame.width + 40, height: window.frame.height + 300)
        XCTAssertEqual(controller.windowWillResize(window, to: taller), taller, file: file, line: line)
        let tiny = controller.windowWillResize(window, to: NSSize(width: 100, height: 100))
        XCTAssertEqual(tiny, BrainBarDashboardPanelController.minSize, file: file, line: line)
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
