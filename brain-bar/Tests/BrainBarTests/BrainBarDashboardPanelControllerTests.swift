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

    // #964: expand a disclosure, scroll to the bottom, collapse it. The document must shrink back
    // to its natural height and the viewport must end on real content, never on a blank tail.

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
