import Foundation
import XCTest
@testable import BrainBar

final class BrainBarDashboardTruthPresentationTests: XCTestCase {
    deinit {}

    func testMissingAgentBucketsFailClosedInsteadOfCopyingAllChunks() {
        let stats = DashboardStats(
            chunkCount: 1,
            enrichedChunkCount: 0,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 0,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 0,
            recentActivityBuckets: [1, 2, 3],
            recentEnrichmentBuckets: [0, 0, 0]
        )

        XCTAssertEqual(stats.recentAgentWriteBuckets, [0, 0, 0])
        XCTAssertEqual(
            stats.agentWriteReadability,
            .unreadable("agent-origin flow evidence not supplied")
        )
        XCTAssertEqual(
            DashboardFlowSummary.derive(daemon: nil, stats: stats).lane(for: .agentStores).status,
            .unavailable
        )
    }

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

    func testOnePageDoesNotPresentEnrichmentAsAnActiveFlow() throws {
        let dashboard = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let dashboardView = try sourceSlice(
            from: "private struct BrainBarDashboardView",
            throughBefore: "private struct BrainBarSnapshotFreshnessBanner",
            in: dashboard
        )
        let pipeline = try sourceFile("Sources/BrainBar/Dashboard/PipelineState.swift")

        for forbidden in [
            "Enriched successfully",
            "BrainBarSectionLabel(\n                \"Enrichment\"",
            "(\"Enrichment\", flowSummary.enrichment.statusText)",
            "series: .enrichment",
        ] {
            XCTAssertFalse(dashboardView.contains(forbidden), "One-page dashboard still presents \(forbidden)")
        }
        XCTAssertFalse(dashboardView.contains("Enrichment: off (manual batch only)"))
        XCTAssertFalse(pipeline.contains("Enrichment is draining backlog"))
    }

    func testDashboardMakesOneLineStatusPrimaryAndKeepsTechnicalFreshnessInDetails() throws {
        let source = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let dashboardSource = try sourceSlice(
            from: "private struct BrainBarDashboardView",
            throughBefore: "private struct BrainBarSnapshotFreshnessBanner",
            in: source
        )
        let statusIndex = try XCTUnwrap(dashboardSource.range(of: "statusStrip")?.lowerBound)
        let tilesIndex = try XCTUnwrap(dashboardSource.range(of: "summaryTiles(layout: layout)")?.lowerBound)

        XCTAssertLessThan(statusIndex, tilesIndex, "The one-line status must precede the dashboard cards.")
        XCTAssertTrue(source.contains("All good"))
        XCTAssertTrue(source.contains("1 thing needs you"))
        XCTAssertTrue(dashboardSource.contains("brainbar.dashboard.status"))
        XCTAssertFalse(dashboardSource.contains("ObservabilityTechnicalDetailsView"), "Errors must stay on the row they explain.")
        XCTAssertTrue(dashboardSource.contains("BrainBarDefinitionList"))
        XCTAssertFalse(dashboardSource.contains("lastGoodContentOpacity"))
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

        for forbidden in ["All commits", "Agent MCP stores", "JSONL watcher"] {
            XCTAssertFalse(
                dashboard.localizedCaseInsensitiveContains(forbidden),
                "Dashboard source still presents forbidden label: \(forbidden)"
            )
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
            "brainbar.dashboard.timeframe",
            "brainbar.dashboard.timeframe.error",
            "brainbar.dashboard.signal-coverage-disclosure",
            "brainbar.dashboard.runtime-disclosure",
        ] {
            XCTAssertTrue(dashboard.contains(identifier), "Missing stable Dashboard accessibility identifier: \(identifier)")
        }
        XCTAssertFalse(dashboard.contains("brainbar.shell.tabs"), "The one-page Dashboard must not restore section tabs.")
        XCTAssertTrue(dashboard.contains("brainbar.shell.destination"), "The shell still exposes destination navigation.")
        XCTAssertTrue(sparkline.contains("metricDisclosure"))
        XCTAssertTrue(sparkline.contains("Text(metricDisclosure)"), "Pointer tooltip must name window, count unit, and clock.")
        XCTAssertTrue(dashboard.contains("accessibilitySummary"), "Charts need a non-pointer semantic summary.")
        XCTAssertTrue(dashboard.contains(".focusEffectDisabled()"), "Dashboard controls must suppress AppKit's pointer focus ring.")
        XCTAssertTrue(dashboard.contains(".focused("), "Dashboard controls still need a keyboard focus path.")
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

    func testTechnicalProvenanceDetailsGateRetainedPayloadsOnMeasuredState() throws {
        let source = try sourceFile("Sources/BrainBar/Dashboard/ObservabilityView.swift")
        let start = try XCTUnwrap(source.range(of: "struct ObservabilityTechnicalDetailsView"))
        let details = String(source[start.lowerBound...])

        XCTAssertTrue(details.contains("if document.emitters.state == \"measured\""))
        XCTAssertTrue(details.contains("document.emitters.reason.isEmpty"))
        XCTAssertTrue(details.contains("if document.authorUnknown.state == \"measured\""))
        XCTAssertTrue(details.contains("document.authorUnknown.reason.isEmpty"))
    }

    func testOnePageFormatsEveryDisplayedCount() throws {
        XCTAssertEqual(
            DashboardMetricFormatter.integerString(797_727, locale: Locale(identifier: "en_US")),
            "797,727"
        )
        XCTAssertEqual(
            DashboardMetricFormatter.axisTickString(1_260, locale: Locale(identifier: "en_US")),
            "1.3k"
        )
        XCTAssertEqual(
            DashboardMetricFormatter.axisTickString(1_260, locale: Locale(identifier: "fr_FR")),
            "1,3k"
        )

        let dashboard = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let pipeline = try sourceFile("Sources/BrainBar/Dashboard/PipelineState.swift")
        let sparkline = try sourceFile("Sources/BrainBar/Dashboard/SparklineRenderer.swift")
        XCTAssertTrue(dashboard.contains("integerString(total, locale: locale)"))
        XCTAssertTrue(dashboard.contains("integerString(indexedToday, locale: locale)"))
        XCTAssertTrue(dashboard.contains("integerString(lane.values.reduce(0, +), locale: locale)"))
        XCTAssertTrue(sparkline.contains("axisTickString(tick)"))
        for rawInterpolation in [
            "\\(collector.stats.chunkCount)",
            "\\(collector.stats.enrichedChunkCount)",
            "\\(collector.stats.pendingEnrichmentCount)",
            "\\(collector.stats.failedEnrichmentCount)",
            "\\(collector.stats.skippedEnrichmentCount)",
            "\\(debt.deduplicatedTotal)",
            "\\(totalCount)",
            "\\(component.snapshot.depth)",
        ] {
            XCTAssertFalse(dashboard.contains(rawInterpolation), rawInterpolation)
        }
        for rawInterpolation in [
            "\\(stats.recentWriteCount)",
            "\\(stats.recentEnrichmentCount)",
            "\\(backlogCount)",
            "\\(latestBucketCount)",
            "\\(totalEvents)",
            "\\(flushDepth)",
            "\\(replayDebtDepth)",
        ] {
            XCTAssertFalse(pipeline.contains(rawInterpolation), rawInterpolation)
        }
    }

    func testRemovedInjectionsSurfaceLeavesNoOrphanedViewStoreOrTests() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        for relativePath in [
            "Sources/BrainBar/InjectionFeedView.swift",
            "Sources/BrainBar/InjectionPresentation.swift",
            "Sources/BrainBar/InjectionStore.swift",
            "Sources/BrainBar/InjectionSummaryView.swift",
            "Sources/BrainBarDaemon/InjectionFeedView.swift",
            "Sources/BrainBarDaemon/InjectionPresentation.swift",
            "Sources/BrainBarDaemon/InjectionStore.swift",
            "Sources/BrainBarDaemon/InjectionSummaryView.swift",
            "Tests/BrainBarTests/InjectionPresentationTests.swift",
            "Tests/BrainBarTests/InjectionSignalDensityContractTests.swift",
            "Tests/BrainBarTests/InjectionStoreTests.swift",
        ] {
            XCTAssertFalse(FileManager.default.fileExists(atPath: packageRoot.appendingPathComponent(relativePath).path), relativePath)
        }

        let runtime = try sourceFile("Sources/BrainBar/BrainBarRuntime.swift")
        let appSupport = try sourceFile("Sources/BrainBar/BrainBarAppSupport.swift")
        XCTAssertFalse(runtime.contains("injectionStore"))
        XCTAssertFalse(runtime.contains("ensureInjectionStore"))
        XCTAssertFalse(appSupport.contains("InjectionStore"))
    }

    func testPowerActionsRequireExplicitConfirmation() throws {
        let source = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")

        XCTAssertTrue(source.contains("showRestartConfirmation"))
        XCTAssertTrue(source.contains("showQuitConfirmation"))
        XCTAssertTrue(source.contains("confirmationDialog"))
        XCTAssertTrue(source.contains("role: .destructive"))
    }

    func testOnePageTileLabelsAndChartStatusColorsRemainSemanticallyLegible() throws {
        let dashboard = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let pipeline = try sourceFile("Sources/BrainBar/Dashboard/PipelineState.swift")
        let tilesSource = try sourceSlice(
            from: "private func summaryTiles",
            throughBefore: "private var onePageBackupLines",
            in: dashboard
        )

        XCTAssertTrue(tilesSource.contains("title: \"Backups\""))
        XCTAssertTrue(tilesSource.contains("title: \"Today\""))
        XCTAssertTrue(tilesSource.contains("private func ingestBand"))
        XCTAssertFalse(tilesSource.contains("minHeight: height, maxHeight: height"), "Cards must size to their content.")
        XCTAssertTrue(dashboard.contains("Text(\"Evidence unavailable\")"), "Unavailable charts must remain explicit.")
        XCTAssertTrue(pipeline.contains("extension DashboardFlowLaneStatus"))
        XCTAssertTrue(pipeline.contains("case .live:\n            return .active"))
    }

    func testOnePageNeverCallsIndexedChunksMemoriesAndKeepsBrainStoreWritesSeparate() throws {
        let dashboard = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let onePageSource = try sourceSlice(
            from: "struct BrainBarOnePagePresentation",
            throughBefore: "@MainActor\nprivate final class BrainBarCommandBarViewModelProvider",
            in: dashboard
        )
        let dashboardView = try sourceSlice(
            from: "private struct BrainBarDashboardView",
            throughBefore: "private struct BrainBarSnapshotFreshnessBanner",
            in: dashboard
        )

        XCTAssertFalse(onePageSource.localizedCaseInsensitiveContains("memories"))
        XCTAssertFalse(dashboardView.localizedCaseInsensitiveContains("memories total"))
        XCTAssertTrue(dashboardView.contains("totalIndexedChunks"))
        XCTAssertTrue(dashboardView.contains("indexed today"))
        XCTAssertTrue(onePageSource.contains("writes via brain_store"))
        XCTAssertTrue(dashboardView.contains("get: { displayedTimeframe }"))
        XCTAssertTrue(dashboardView.contains("selectedTimeframe = $0"))
        XCTAssertTrue(dashboardView.contains("if selectedTimeframe == $0"))
        XCTAssertTrue(dashboardView.contains("collector.selectTimeframe("))
    }

    func testActivityAndProvenanceRowsPutOneMeasuredWindowInEachValueSlot() throws {
        let dashboard = try sourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let activity = try sourceSlice(
            from: "private func diagnostics",
            throughBefore: "private var daemonSummary",
            in: dashboard
        )
        let observability = try sourceFile("Sources/BrainBar/Dashboard/ObservabilityView.swift")

        XCTAssertFalse(activity.contains("flowSummary.ingress.statusText"))
        XCTAssertFalse(activity.contains("(\"Window\","))
        XCTAssertTrue(activity.contains("flowSummary.allCommits.volumeText"))
        XCTAssertFalse(observability.contains("indexed chunks total · \\(row.inWindow)"))
        XCTAssertTrue(observability.contains("of indexed chunks"))
    }

    private func sourceSlice(from start: String, throughBefore end: String, in source: String) throws -> String {
        let startRange = try XCTUnwrap(source.range(of: start))
        let endRange = try XCTUnwrap(source.range(of: end, range: startRange.upperBound..<source.endIndex))
        return String(source[startRange.lowerBound..<endRange.lowerBound])
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
