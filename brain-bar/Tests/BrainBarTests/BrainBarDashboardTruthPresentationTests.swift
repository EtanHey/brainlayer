import Foundation
import XCTest
@testable import BrainBar

final class BrainBarDashboardTruthPresentationTests: XCTestCase {
    func testDashboardShipLabelsMatchTheApprovedMetricAndWatcherTruth() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 90,
            failedEnrichmentCount: 4,
            skippedEnrichmentCount: 6,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 75,
            enrichmentRatePerMinute: 1,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [1, 2, 3, 4],
            recentAgentWriteBuckets: [1, 1, 2, 2],
            recentWatcherWriteBuckets: [0, 1, 0, 2],
            recentEnrichmentBuckets: [1, 1, 1, 1],
            activityWindowMinutes: 60,
            bucketCount: 4,
            lastWriteAt: now.addingTimeInterval(-10),
            lastEnrichedAt: now.addingTimeInterval(-10),
            watcherProcessProbeResult: .running(pid: 42),
            watcherRecentDistinctChunkCount: 2,
            watcherFlowReadability: .readable
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let chunkRows = summary.lane(for: .allCommits)
        let agentOrigin = summary.lane(for: .agentStores)
        let watcherIngested = summary.lane(for: .jsonlWatcher)
        let enriched = summary.lane(for: .enrichment)

        XCTAssertEqual(chunkRows.name, "Chunk rows")
        XCTAssertEqual(chunkRows.sparklineLabel, "Chunk rows by source time over Last 1h")
        XCTAssertFalse(chunkRows.statusText.localizedCaseInsensitiveContains("commit"))
        XCTAssertEqual(agentOrigin.name, "Agent-origin chunks")
        XCTAssertEqual(agentOrigin.sparklineLabel, "Agent-origin chunks by source time over Last 1h")
        XCTAssertFalse(agentOrigin.statusText.localizedCaseInsensitiveContains("MCP"))
        XCTAssertEqual(watcherIngested.name, "Watcher-ingested chunks")
        XCTAssertEqual(watcherIngested.sparklineLabel, "Watcher-ingested chunks by ingest time over Last 1h")
        XCTAssertEqual(watcherIngested.statusText, "FLOWING")
        XCTAssertEqual(enriched.name, "Enriched successfully")
        XCTAssertEqual(enriched.sparklineLabel, "Successful enrichment completions over Last 1h")
    }

    func testDashboardMakesFreshnessPrimaryAndKeepsLastGoodDataVisible() throws {
        let source = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let dashboardRange = try XCTUnwrap(source.range(of: "private struct BrainBarDashboardView"))
        let dashboardSource = String(source[dashboardRange.lowerBound...])
        let freshnessIndex = try XCTUnwrap(dashboardSource.range(of: "freshnessBanner")?.lowerBound)
        let overviewIndex = try XCTUnwrap(dashboardSource.range(of: "overviewCard(layout: layout)")?.lowerBound)

        XCTAssertLessThan(freshnessIndex, overviewIndex, "Freshness must precede last-good dashboard content.")
        XCTAssertTrue(source.contains("private struct BrainBarSnapshotFreshnessBanner"))
        XCTAssertTrue(source.contains("Last good"))
        XCTAssertTrue(source.contains("Data age"))
        XCTAssertTrue(source.contains("brainbar.dashboard.freshness"))
        XCTAssertTrue(source.contains("lastGoodContentOpacity"))
    }

    func testDashboardUsesCompactDensityAtSupportedBreakpoints() {
        let compact = BrainBarDashboardLayout(containerSize: CGSize(width: 760, height: 560))
        let normal = BrainBarDashboardLayout(containerSize: CGSize(width: 960, height: 700))

        XCTAssertLessThanOrEqual(compact.outerPadding, 18)
        XCTAssertLessThanOrEqual(compact.sectionSpacing, 16)
        XCTAssertLessThanOrEqual(compact.cardPadding, 18)
        XCTAssertLessThanOrEqual(compact.sparklineHeight, 112)
        XCTAssertLessThanOrEqual(normal.cardPadding, 24)
        XCTAssertLessThanOrEqual(normal.sparklineHeight, 140)
    }

    func testChartsExposeVisibleTruthTooltipsAndKeyboardAccessIdentifiers() throws {
        let dashboard = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let sparkline = try sourceFile("Sources/BrainBar/Dashboard/SparklineRenderer.swift")
        let commandBar = try sourceFile("Sources/BrainBar/BrainBarCommandBar.swift")

        for forbidden in ["All commits", "ALL COMMITS", "Agent MCP stores", "JSONL watcher"] {
            XCTAssertFalse(dashboard.contains(forbidden), "Dashboard source still presents forbidden label: \(forbidden)")
        }
        for subtitle in [
            "Source time · chunk rows",
            "Source time · chunk rows · documented agent origins",
            "Ingest time · unique chunk IDs first seen in window · not additive with source-time charts",
        ] {
            XCTAssertTrue(dashboard.contains(subtitle), "Missing visible chart disclosure: \(subtitle)")
        }
        for identifier in [
            "brainbar.dashboard.scroll",
            "brainbar.dashboard.chart.chunk-rows",
            "brainbar.dashboard.chart.agent-origin-chunks",
            "brainbar.dashboard.chart.watcher-ingested-chunks",
            "brainbar.dashboard.chart.enriched-successfully",
            "brainbar.dashboard.timeframe",
            "brainbar.dashboard.signal-coverage-disclosure",
            "brainbar.dashboard.runtime-disclosure",
            "brainbar.shell.tabs",
        ] {
            XCTAssertTrue(dashboard.contains(identifier), "Missing stable Dashboard accessibility identifier: \(identifier)")
        }
        XCTAssertTrue(sparkline.contains("metricDisclosure"))
        XCTAssertTrue(sparkline.contains("Text(metricDisclosure)"), "Pointer tooltip must name window, count unit, and clock.")
        XCTAssertTrue(dashboard.contains("accessibilitySummary"), "Charts need a non-pointer semantic summary.")
        XCTAssertTrue(dashboard.contains(".focusable()"), "Dashboard scroll and controls need a keyboard focus path.")
        XCTAssertTrue(commandBar.contains("brainbar.command.mode.capture"))
        XCTAssertTrue(commandBar.contains("brainbar.command.mode.search"))
        XCTAssertTrue(commandBar.contains("brainbar.command.input"))
        XCTAssertFalse(commandBar.contains(".focusable(false)"))
    }

    func testReplayDebtIsDecomposedOneActionFromTheAggregate() throws {
        let source = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")

        XCTAssertTrue(source.contains("Replay debt"))
        XCTAssertTrue(source.contains("Pending stores"))
        XCTAssertTrue(source.contains("Queue entries"))
        XCTAssertTrue(source.contains("Fallback entries"))
        XCTAssertTrue(source.contains("Unreadable inputs"))
        XCTAssertTrue(source.contains("Deduplicated total"))
        XCTAssertTrue(source.contains("brainbar.dashboard.replay-debt-disclosure"))
        XCTAssertTrue(source.contains("replayDebtBreakdown.isPartial"))
    }

    func testPowerActionsRequireExplicitConfirmation() throws {
        let source = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")

        XCTAssertTrue(source.contains("showRestartConfirmation"))
        XCTAssertTrue(source.contains("showQuitConfirmation"))
        XCTAssertTrue(source.contains("confirmationDialog"))
        XCTAssertTrue(source.contains("role: .destructive"))
    }

    func testWideHeroLabelsAndChartStatusColorsRemainSemanticallyLegible() throws {
        let dashboard = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let pipeline = try sourceFile("Sources/BrainBar/Dashboard/PipelineState.swift")
        let overviewRange = try XCTUnwrap(dashboard.range(of: "private struct BrainBarOverviewStat"))
        let overviewSource = String(dashboard[overviewRange.lowerBound...])

        XCTAssertTrue(overviewSource.contains(".lineLimit(2)"), "Wide hero labels must wrap instead of truncating metric truth.")
        XCTAssertTrue(dashboard.contains("lane.status.stateTheme"), "Chart status pills must use semantic state color, not series color.")
        XCTAssertTrue(pipeline.contains("extension DashboardFlowLaneStatus"))
        XCTAssertTrue(pipeline.contains("case .live:\n            return .active"))
    }

    private func sourceFile(_ relativePath: String) throws -> String {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: packageRoot.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }
}
