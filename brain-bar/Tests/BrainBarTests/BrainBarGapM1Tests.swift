import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Offscreen hosting views only; never attached to a window, menu or status item.
@MainActor
final class BrainBarGapM1Tests: XCTestCase {
    func testCalmOverviewContentFitsViewport() {
        let viewport = NDGrid.windowSize.height - NDGrid.titleBar - NDGrid.footer
        for fixture: BrainBarDesignFixture.Name in [.calm, .twoFaults] {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(fixture), page: .overview)
            let width = NDGrid.windowWidth - 2 * NDGrid.pageSide
            let host = NSHostingController(rootView: NDOverviewPage(store: store).frame(width: width).environment(\.ndPalette, .window))
            let height = host.sizeThatFits(in: CGSize(width: width, height: .greatestFiniteMagnitude)).height + NDGrid.pageTop + 20
            print("M1_LAYOUT \(fixture.rawValue) content=\(height) viewport=\(viewport) scroll=\(max(0, height - viewport))")
            if fixture == .calm { XCTAssertLessThanOrEqual(height, viewport) }
        }
    }

    func testChartAccessibilityTreeExposesCaptionsAndPurposes() throws {
        let i = BrainBarDesignFixture.inputs(.calm)
        let hosting = NSHostingView(rootView: NDPipelineChart(state: NDPipelineState.make(i), memories: "297,412", interactive: false))
        hosting.frame = NSRect(origin: .zero, size: NDPipelineLayout.size)
        hosting.layoutSubtreeIfNeeded()
        func labelLayer(_ view: NSView) -> NDPipelineLabelView? {
            (view as? NDPipelineLabelView) ?? view.subviews.lazy.compactMap { labelLayer($0) }.first
        }
        // Read the native AX subtree actually embedded in the hosted chart, without a window or an AX mock.
        let host = try XCTUnwrap(labelLayer(hosting))
        XCTAssertNil(hosting.window)
        let children = host.accessibilityChildren() as? [NSTextField] ?? []
        XCTAssertEqual(children.count, 4)
        for child in children { XCTAssertEqual(child.accessibilityRole(), .staticText) }
        let words = children.compactMap { $0.accessibilityLabel() }
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        for expected in [Say.sessionBackups, Say.searchAnswers, Say.nightlyPurpose, Say.weeklyPurpose] {
            XCTAssertTrue(words.contains(expected), "Missing \(expected) from \(words)")
        }
    }
}
