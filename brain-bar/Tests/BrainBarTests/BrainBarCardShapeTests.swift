import AppKit
import SwiftUI
import XCTest

@testable import BrainBar

final class BrainBarCardShapeTests: XCTestCase {
    func testEmptyChartRetainsBucketAxesWithoutPlottingAFalseZeroSeries() {
        let chart = SparklineChartPresentation(
            label: "Agent-origin chunks by source time over 1 hour",
            values: Array(repeating: 0, count: 12),
            showsRestingAxes: true,
            plotsSeries: false
        )
        XCTAssertEqual(chart.tightYAxisTicks, [0, 1])
        XCTAssertFalse(chart.shouldPlotSeries(.primary))
        XCTAssertTrue(chart.showsRestingAxes)
        XCTAssertEqual(chart.points.count, 12)
    }

    @MainActor
    func testEveryDashboardCardKeepsItsFrameAcrossEvidenceStates() throws {
        let states: [BrainBarDashboardFixture.OperatorState] = [
            .live, .loadingCards, .stale, .unavailable, .empty, .error,
        ]
        let cards = [
            "summary.backups", "summary.memory", "ingest",
            "ingest.allCommits", "ingest.agentStores", "ingest.jsonlWatcher",
            "coverage.Vector", "coverage.FTS5", "coverage.Trigram",
            "activity", "runtime", "receipt.Last search", "receipt.Last store",
        ]
        for width: CGFloat in [760, 960, 1_280] {
            var sizes: [String: [String: CGSize]] = [:]
            for state in states {
                let panelState = BrainBarDashboardPanelState()
                panelState.detailsExpanded = true
                panelState.signalCoverageExpanded = true
                let view = BrainBarDashboardPreview.make(
                    collector: BrainBarDashboardFixture.makeCollector(state),
                    receiptStore: BrainBarDashboardFixture.makeReceiptStore(state),
                    observabilityResult: state == .unavailable ? .unreadable("Fixture unavailable") : (state == .empty ? BrainBarDashboardFixture.emptyObservabilityResult : BrainBarDashboardFixture.readableObservabilityResult),
                    now: BrainBarDashboardFixture.fetchedAt,
                    panelState: panelState
                )
                let host = NSHostingController(rootView: view)
                host.view.frame = NSRect(x: 0, y: 0, width: width, height: 1_200)
                host.view.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.25))
                host.view.layoutSubtreeIfNeeded()
                sizes[String(describing: state)] = panelState.renderedCardSizes
                XCTAssertEqual(panelState.renderedCardSizes.keys.sorted(), cards.sorted(), "\(state) at \(width)")
            }
            for card in cards {
                let measured = try states.map { state in
                    try XCTUnwrap(sizes[String(describing: state)]?[card], "\(card), \(state) at \(width)")
                }
                let heights = measured.map(\.height)
                let widths = measured.map(\.width)
                XCTAssertEqual(heights.max()! - heights.min()!, 0, accuracy: 0.5, "\(card) heights at \(width): \(heights)")
                XCTAssertEqual(widths.max()! - widths.min()!, 0, accuracy: 0.5, "\(card) widths at \(width): \(widths)")
            }
        }
    }

}
