import AppKit
import SwiftUI
import Vision
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarStoreRowsLabelTests: XCTestCase {
    private func collector() -> StatsCollector {
        let base = BrainBarDashboardFixture.makeCollector()
        let stats = DashboardStats(chunkCount: 152, enrichedChunkCount: 0, pendingEnrichmentCount: 0,
            enrichmentPercent: 0, enrichmentRatePerMinute: 0, databaseSizeBytes: 4096,
            recentActivityBuckets: [0, 0, 0, 152], recentAgentWriteBuckets: [0, 0, 0, 0],
            recentWatcherWriteBuckets: [0, 0, 0, 152], recentEnrichmentBuckets: [0, 0, 0, 0], activityWindowMinutes: 60, bucketCount: 4)
        return StatsCollector.fixture(stats: stats, daemon: nil, agentActivity: base.agentActivity,
            state: base.state, lastDataFetchedAt: BrainBarDashboardFixture.fetchedAt)
    }

    func testSavedMemoriesLaneAndAccessibilityDescribeTheMeasuredRows() {
        let stats = collector().stats
        let lane = DashboardFlowSummary.derive(daemon: nil, stats: stats).lane(for: .agentStores)
        let disclosure = BrainBarDashboardChartDisclosure(series: .agentStores, lane: lane, timeframe: .live)
        XCTAssertEqual(disclosure.emptyMessage, "No saved memories in the last 1h")
        XCTAssertEqual(disclosure.visibleHelp, "Memories saved by agents and hooks. Excludes transcript indexing; does not measure agent activity.")
        XCTAssertEqual(disclosure.subtitle, "Source time · saved memory rows")
        XCTAssertEqual(lane.name, "Saved memories · chunk rows")
        XCTAssertTrue(disclosure.accessibilitySummary.contains("Saved memories"))
        XCTAssertTrue(disclosure.accessibilitySummary.contains("Memories saved by agents and hooks. Excludes transcript indexing; does not measure agent activity."))
        XCTAssertTrue(disclosure.visibleDetail?.contains("MCP, manual, digest, precompact-hook, brain_store, pending, fallback, fallback-replay") == true)
    }

    func testEmptyStoreChartRendersItsTitleExplanationAndWindow() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] else {
            throw XCTSkip("Set BRAINBAR_SNAPSHOT_DIR to render the store chart")
        }
        let root = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for width in [760, 960, 1280] {
            let view = BrainBarDashboardPreview.make(collector: collector(), now: BrainBarDashboardFixture.fetchedAt)
            let host = NSHostingView(rootView: view)
            host.frame = NSRect(x: 0, y: 0, width: width, height: 900)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
            host.layoutSubtreeIfNeeded()
            let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: bitmap)
            let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
            try png.write(to: root.appendingPathComponent("saved-memories-\(width).png"))
            XCTAssertGreaterThan(png.count, 5000)
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            try VNImageRequestHandler(cgImage: XCTUnwrap(bitmap.cgImage)).perform([request])
            let text = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ")
            try text.write(to: root.appendingPathComponent("saved-memories-\(width).txt"), atomically: true, encoding: .utf8)
            XCTAssertTrue(text.contains("Saved memories"), text)
            XCTAssertTrue(text.contains("No saved memories in the last 1h"), text)
            XCTAssertTrue(text.localizedCaseInsensitiveContains("saved by agents and hooks. Excludes transcript"), text)
            XCTAssertTrue(text.localizedCaseInsensitiveContains("indexing; does not measure agent activity."), text)
        }
    }
}
