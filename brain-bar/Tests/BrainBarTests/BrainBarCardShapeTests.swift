import AppKit
import SwiftUI
import XCTest

@testable import BrainBar

final class BrainBarCardShapeTests: XCTestCase {
    @MainActor
    func testSummaryCardsKeepTheirFramesAcrossEvidenceStates() throws {
        let evidence: [(String, ObservabilityReadResult)] = [
            ("loaded", try BrainBarOnePageTestFixture.dashboardResult(indexedToday: 67)),
            ("stale", try BrainBarOnePageTestFixture.staleResult()),
            ("unavailable", .unreadable("Fixture observability unavailable")),
        ]

        for width: CGFloat in [760, 960, 1_280] {
            var measured: [String: [String: CGFloat]] = [:]
            for (name, result) in evidence {
                let panelState = BrainBarDashboardPanelState()
                let view = BrainBarDashboardPreview.make(
                    collector: BrainBarDashboardFixture.makeCollector(),
                    observabilityResult: result,
                    now: BrainBarOnePageTestFixture.now,
                    panelState: panelState
                )
                let host = NSHostingController(rootView: view)
                host.view.frame = NSRect(x: 0, y: 0, width: width, height: 1_200)
                host.view.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.25))
                host.view.layoutSubtreeIfNeeded()
                measured[name] = panelState.renderedSummaryTileHeights
                XCTAssertEqual(measured[name]?.keys.sorted(), ["backups", "memory"], "\(width) \(name)")
            }
            for card in ["backups", "memory"] {
                let heights = try evidence.map { name, _ in
                    try XCTUnwrap(measured[name]?[card])
                }
                XCTAssertEqual(heights.max()! - heights.min()!, 0, accuracy: 0.5, "\(card) at \(width): \(heights)")
            }
        }
    }
}
