import AppKit
import XCTest
@testable import BrainBar

final class BrainBarRetiredDashboardTests: XCTestCase {
    func testHistoricalMetadataDoesNotActivateEnrichmentOrScheduleAQueue() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120, enrichedChunkCount: 100, pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3, enrichmentRatePerMinute: 24, databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0], recentEnrichmentBuckets: [0, 1],
            lastEnrichedAt: now.addingTimeInterval(-6)
        )
        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let indicators = PipelineIndicators.derive(daemon: nil, stats: stats, now: now)
        XCTAssertEqual(PipelineState.derive(daemon: nil, stats: stats, now: now), .idle)
        XCTAssertEqual(indicators.enriching.name, "Enrichment retired")
        XCTAssertEqual(indicators.enriching.status, .idle)
        XCTAssertEqual(summary.enrichment.name, "Enrichment history")
        XCTAssertEqual(summary.enrichment.status, .idle)
        XCTAssertEqual(summary.enrichment.statusText, "Enrichment retired")
        XCTAssertEqual(summary.enrichment.values, stats.recentEnrichmentBuckets)
        XCTAssertEqual(summary.queue.backlogCount, 20)
        XCTAssertEqual(summary.queue.status, .stable)
        XCTAssertFalse(summary.headline.lowercased().contains("draining"))
        XCTAssertTrue(summary.queue.detail.contains("Enrichment retired"))
    }

    func testRetirementPresentationDoesNotDependOnReadableConfigOrLegacyFlags() {
        let footer = BrainBarSettingsFooterPresentation(config: nil, watcher: nil)
        XCTAssertTrue(footer.locality.contains("Enrichment off (retired)"))
        let count = BrainBarQueueDirectionPresentation.derive(.growing, backlogCount: 20)
        XCTAssertEqual(count.label, "Enrichment retired · 20 unenriched")
        XCTAssertEqual(count.tone, .neutral)
        XCTAssertEqual(BrainBarQueueDirectionPresentation.derive(.unavailable, backlogCount: 20).tone, .error)
    }
}
