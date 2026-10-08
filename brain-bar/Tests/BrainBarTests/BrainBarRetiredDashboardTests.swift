import AppKit
import XCTest
@testable import BrainBar

final class BrainBarRetiredDashboardTests: XCTestCase {
    func testHistoricalMetadataIsAbsentFromFlowAndQueuePresentation() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 300_000, enrichedChunkCount: 100, pendingEnrichmentCount: 274_847,
            enrichmentPercent: 83.3, enrichmentRatePerMinute: 24, databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0], recentEnrichmentBuckets: [0, 1],
            lastEnrichedAt: now.addingTimeInterval(-6)
        )
        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        XCTAssertEqual(PipelineState.derive(daemon: nil, stats: stats, now: now), .idle)
        XCTAssertEqual(PipelineIndicators.derive(daemon: nil, stats: stats, now: now).indexing.status, .idle)
        XCTAssertEqual(PipelineSeries.allCases, [.allCommits, .agentStores, .jsonlWatcher])
        XCTAssertEqual(summary.queue.status, .empty)
        for text in [summary.headline, summary.detail, summary.queue.title, summary.queue.detail] {
            XCTAssertFalse(text.localizedCaseInsensitiveContains("enrich"))
            XCTAssertFalse(text.contains("274"))
        }
    }

    func testFooterHasNoEnrichmentWithUnreadableConfigOrLegacyFlags() {
        for config in [nil, BrainLayerConfig.defaultConfig] {
            let footer = BrainBarSettingsFooterPresentation(config: config, watcher: nil)
            XCTAssertFalse(footer.locality.localizedCaseInsensitiveContains("enrich"))
            XCTAssertTrue(footer.locality.contains("Memory on this Mac"))
        }
    }
}
