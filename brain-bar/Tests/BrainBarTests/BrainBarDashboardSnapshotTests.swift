import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Deterministic, full-dashboard render-verification infra.
///
/// BrainBar is an `LSUIElement` menu-bar app, so `computer-use` reports it
/// "not_installed" and full-screen `screencapture` grabs the wrong window — there
/// is no reliable way to visually verify its UI. These tests render the REAL
/// dashboard views (hero/overview + pipeline + diagnostics) and the settings panel
/// to PNGs that any agent can `Read` to verify the UI, with no live collectors and
/// no live screenshots.
///
/// The renders are deterministic: fixture data (`BrainBarDashboardFixture`) with
/// every relative-time `Date` nil, `accessibilityReduceMotion = true` so SwiftUI
/// animations resolve immediately, and fixed host sizes. Output PNGs land in
/// `brain-bar/docs.local/brainbar-render/` (override with `BRAINBAR_RENDER_DIR`).
///
/// Run just this suite:
///   swift test --filter BrainBarDashboardSnapshotTests
final class BrainBarDashboardSnapshotTests: XCTestCase {
    private var shouldSkipDisplayDependentRenderInCI: Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment["CI"] == "true" && environment["BRAINBAR_RENDER_IN_CI"] != "1"
    }

    /// Layout breakpoints come from `BrainBarDashboardLayout`: compact < 920,
    /// 920 ≤ default < 1040, wide ≥ 1040 (two chart columns).
    private enum Breakpoint: String, CaseIterable {
        case compact
        case `default`
        case wide

        // Match the actual dashboard window height. The resting one-page layout
        // must stay useful without relying on an artificially tall proof frame.
        var size: NSSize {
            switch self {
            case .compact: NSSize(width: 760, height: 640)
            case .default: NSSize(width: 960, height: 640)
            case .wide: NSSize(width: 1_280, height: 640)
            }
        }
    }

    func testOnePageCompositionContract() throws {
        XCTAssertEqual(
            BrainBarOnePageComposition.visibleSectionIDs,
            ["status", "backups", "memory", "ingest", "details"]
        )
        XCTAssertEqual(BrainBarOnePageComposition.primaryTileCount, 3)
        XCTAssertTrue(BrainBarOnePageComposition.primaryTilesHaveEqualHeight)
        XCTAssertFalse(BrainBarOnePageComposition.detailsExpandedByDefault)

        let source = try String(
            contentsOf: packageRoot().appendingPathComponent("Sources/BrainBar/BrainBarWindowRootView.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(source.contains("statusStrip\n                        summaryTiles(layout: layout)"))
        XCTAssertFalse(source.contains("ObservabilityDashboardView("), "The old lower observability grid must not return.")
        XCTAssertFalse(source.contains("Runtime & Details"), "Technical summaries must stay hidden while Details is collapsed.")
    }

    @MainActor
    func testOnePageIngestAreaRendersThreeIndependentlyScaledSmallMultiples() throws {
        let source = try String(
            contentsOf: packageRoot().appendingPathComponent("Sources/BrainBar/BrainBarWindowRootView.swift"),
            encoding: .utf8
        )
        let start = try XCTUnwrap(source.range(of: "private func ingestTile"))
        let end = try XCTUnwrap(source.range(of: "private func summaryTile", range: start.upperBound..<source.endIndex))
        let ingestTile = String(source[start.lowerBound..<end.lowerBound])

        XCTAssertTrue(ingestTile.contains("BrainBarSharedTimeframeSelector"))
        XCTAssertTrue(ingestTile.contains("ingestSeriesRow(.allCommits"))
        XCTAssertTrue(ingestTile.contains("ingestSeriesRow(.agentStores"))
        XCTAssertTrue(ingestTile.contains("ingestSeriesRow(.jsonlWatcher"))

        let summary = DashboardFlowSummary.derive(
            daemon: nil,
            stats: BrainBarDashboardFixture.stats,
            now: BrainBarDashboardFixture.fetchedAt
        )
        let seriesValues = [PipelineSeries.allCommits, .agentStores, .jsonlWatcher].map {
            summary.lane(for: $0).values
        }
        let maxima = [PipelineSeries.allCommits, .agentStores, .jsonlWatcher].map {
            SparklineChartPresentation(
                label: summary.lane(for: $0).sparklineLabel,
                values: summary.lane(for: $0).values,
                activityWindowMinutes: summary.lane(for: $0).activityWindowMinutes,
                latestBucketName: summary.lane(for: $0).latestBucketName,
                fetchedAt: BrainBarDashboardFixture.fetchedAt
            ).maxValue
        }
        XCTAssertEqual(maxima.count, 3)
        XCTAssertNotEqual(seriesValues[0], seriesValues[1])
        XCTAssertNotEqual(seriesValues[0], seriesValues[2])
        XCTAssertNotEqual(seriesValues[1], seriesValues[2])
        XCTAssertTrue(zip(seriesValues[1], seriesValues[0]).allSatisfy { $0.0 <= $0.1 })
        XCTAssertTrue(zip(seriesValues[1], seriesValues[0]).contains { $0.0 < $0.1 })
        XCTAssertNotEqual(maxima[0], maxima[1], "Each small multiple must derive its own y-scale.")
        XCTAssertNotEqual(maxima[0], maxima[2], "Each small multiple must derive its own y-scale.")
        XCTAssertNotEqual(maxima[1], maxima[2], "Each small multiple must derive its own y-scale.")
    }

    @MainActor
    func testDashboardRendersAtAllBreakpoints() throws {
        let indexedInLiveWindow = BrainBarDashboardFixture.stats.recentActivityBuckets.reduce(0, +)
        let observability = try BrainBarOnePageTestFixture.dashboardResult(indexedToday: indexedInLiveWindow)
        guard case let .readable(document) = observability else {
            return XCTFail("dashboard observability fixture must be readable")
        }
        XCTAssertGreaterThanOrEqual(
            document.stores.inWindow?.byHour?.reduce(0) { $0 + $1.count } ?? -1,
            indexedInLiveWindow
        )

        for breakpoint in Breakpoint.allCases {
            let collector = BrainBarDashboardFixture.makeCollector()
            let panelState = BrainBarDashboardPanelState()
            let view = BrainBarDashboardPreview.make(
                collector: collector,
                observabilityResult: observability,
                now: BrainBarOnePageTestFixture.now,
                panelState: panelState
            )
            var size = breakpoint.size
            if breakpoint == .default {
                let host = NSHostingController(rootView: view)
                host.view.frame = NSRect(origin: .zero, size: size)
                host.view.layoutSubtreeIfNeeded()
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
                size.height = panelState.dashboardHeight
                XCTAssertLessThan(size.height, 640)
            }
            let (png, bitmap) = try renderPNG(view, size: size)

            let url = try writePNG(png, name: "dashboard-\(breakpoint.rawValue)")
            XCTAssertGreaterThan(png.count, 5_000, "dashboard-\(breakpoint.rawValue) PNG looks empty")
            XCTAssertGreaterThan(
                distinctSampledColorCount(in: bitmap), 16,
                "dashboard-\(breakpoint.rawValue) render is too flat — likely blank/clipped"
            )
            XCTAssertLessThan(
                warmBrownPixelShare(in: bitmap),
                0.12,
                "dashboard-\(breakpoint.rawValue) regressed to the old brown-card layout"
            )
            if breakpoint == .default { XCTAssertTrue(bottomBandContainsForeground(in: bitmap)) }
            // Surface the path in the test log so an agent knows what to Read.
            print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
        }
    }

    @MainActor
    func testAllGoodStateRendersWithVerifiedBackups() throws {
        let result = try BrainBarOnePageTestFixture.dashboardResult(
            indexedToday: BrainBarDashboardFixture.stats.recentActivityBuckets.reduce(0, +)
        )
        let collector = BrainBarDashboardFixture.makeCollector()
        let view = BrainBarDashboardPreview.make(
            collector: collector,
            observabilityResult: result,
            now: BrainBarOnePageTestFixture.now
        )
        let (png, bitmap) = try renderPNG(view, size: Breakpoint.default.size)
        let url = try writePNG(png, name: "dashboard-all-good")

        let flow = DashboardFlowSummary.derive(
            daemon: collector.daemon,
            stats: collector.stats,
            now: BrainBarOnePageTestFixture.now
        )
        let hero = BrainBarHeroPresentation.derive(
            flow: flow,
            stats: collector.stats,
            backupTruth: BrainBarHeroBackupTruth.derive(
                from: result,
                now: BrainBarOnePageTestFixture.now,
                cadence: .known(300)
            ),
            locale: BrainBarDashboardPreview.goldenLocale
        )
        let presentation = BrainBarOnePagePresentation.derive(
            snapshotFreshness: collector.snapshotFreshnessState,
            hero: hero,
            observability: result,
            stats: collector.stats,
            agentActivity: collector.agentActivity,
            now: BrainBarOnePageTestFixture.now,
            calendar: BrainBarDashboardPreview.goldenCalendar,
            locale: BrainBarDashboardPreview.goldenLocale
        )

        XCTAssertEqual(presentation.status.headline, "All good")
        XCTAssertNil(presentation.status.reason)
        XCTAssertEqual(presentation.backupLines.map(\.tone), [.green, .green])
        XCTAssertGreaterThan(png.count, 5_000, "all-good PNG looks empty")
        XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 16, "all-good render is too flat")
        print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
    }

    @MainActor
    func testDashboardQuietFleetRendersMeasuredQuietState() throws {
        let collector = BrainBarDashboardFixture.makeCollector(agentActivity: .empty)
        let result = try BrainBarOnePageTestFixture.dashboardResult(
            indexedToday: collector.stats.recentActivityBuckets.reduce(0, +)
        )
        let view = BrainBarDashboardPreview.make(
            collector: collector,
            observabilityResult: result,
            now: BrainBarOnePageTestFixture.now
        )
        let (png, bitmap) = try renderPNG(view, size: Breakpoint.default.size)
        let url = try writePNG(png, name: "dashboard-quiet")

        XCTAssertTrue(collector.agentActivity.isMeasured)
        XCTAssertEqual(collector.agentActivity.totalActiveAgents, 0)
        XCTAssertGreaterThan(png.count, 5_000, "quiet-state PNG looks empty")
        XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 16, "quiet-state render is too flat")
        print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
    }

    @MainActor
    func testDashboardStaleObservabilityRendersUnavailableCounts() throws {
        let collector = BrainBarDashboardFixture.makeCollector()
        let result = try BrainBarOnePageTestFixture.staleResult()
        let view = BrainBarDashboardPreview.make(
            collector: collector,
            observabilityResult: result,
            now: BrainBarOnePageTestFixture.now
        )
        let (png, bitmap) = try renderPNG(view, size: Breakpoint.default.size)
        let url = try writePNG(png, name: "dashboard-observability-stale")

        let presentation = try makeOnePagePresentation(
            collector: collector,
            result: result,
            now: BrainBarOnePageTestFixture.now
        )
        XCTAssertEqual(presentation.status.headline, "1 thing needs you")
        XCTAssertNil(presentation.indexedToday)
        XCTAssertEqual(
            presentation.indexedTodayUnavailableText,
            "Indexed today unavailable: observability as of 20:50"
        )
        XCTAssertEqual(
            presentation.agentWritesText,
            "brain_store writes unavailable: observability as of 20:50"
        )
        XCTAssertTrue(presentation.backupLines[0].text.hasSuffix("18:50"))
        XCTAssertTrue(presentation.backupLines[1].text.hasSuffix("19:50"))
        XCTAssertGreaterThan(png.count, 5_000, "stale-observability PNG looks empty")
        XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 16, "stale-observability render is too flat")
        print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
    }

    @MainActor
    func testSettingsRendersDeterministically() throws {
        try XCTSkipIf(
            shouldSkipDisplayDependentRenderInCI,
            "Settings PNG render verification is display-dependent; set BRAINBAR_RENDER_IN_CI=1 to run in CI."
        )

        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-render-settings-\(UUID().uuidString)", isDirectory: true)
        let configURL = tempRoot
            .appendingPathComponent(".config/brainlayer/brainlayer.env", isDirectory: false)
        var config = BrainLayerConfig.defaultConfig
        config.googleAPIKey = .onePasswordReference("op://Private/Google AI/Gemini API key")
        config.enrichmentEnabled = true
        config.enrichmentMode = .remote
        config.enrichmentProvider = .gemini
        config.enrichmentBackend = "gemini"
        config.launchdJobs[.drain]?.enabled = true
        config.launchdJobs[.hotlane]?.enabled = false

        let store = BrainLayerConfigStore(configURL: configURL)
        try store.save(config)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [
                .enrichment: .loaded,
                .hotlane: .unloaded,
                .drain: .running,
            ]),
            initialLaunchdStates: [
                .enrichment: .loaded,
                .hotlane: .unloaded,
                .drain: .running,
            ],
            refreshStatusOnLoad: false
        )
        let view = BrainBarSettingsView(viewModel: viewModel)
            .environment(\.colorScheme, .dark)
            .transaction { $0.disablesAnimations = true }
        let (png, bitmap) = try renderPNG(view, size: NSSize(width: 700, height: 1_080))

        let url = try writePNG(png, name: "settings")
        XCTAssertGreaterThan(png.count, 5_000, "settings PNG looks empty")
        XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 16, "settings render is too flat")
        print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
    }

    @MainActor
    func testDashboardOperatorStatesRenderDeterministically() throws {
        try XCTSkipIf(
            shouldSkipDisplayDependentRenderInCI,
            "Dashboard PNG render verification is display-dependent; set BRAINBAR_RENDER_IN_CI=1 to run in CI."
        )

        for state in BrainBarDashboardFixture.OperatorState.allCases {
            let collector = BrainBarDashboardFixture.makeCollector(state)
            let view = BrainBarDashboardPreview.make(collector: collector)
            let size = state == .loading
                ? NSSize(width: 960, height: 700)
                : NSSize(width: 960, height: 1_820)
            let (png, bitmap) = try renderPNG(view, size: size)
            let name = "dashboard-state-\(String(describing: state))"
            let url = try writePNG(png, name: name)

            XCTAssertGreaterThan(png.count, 5_000, "\(name) PNG looks empty")
            XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 16, "\(name) render is too flat")
            print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
        }
    }

    @MainActor
    func testDashboardWatcherTruthStatesRenderDeterministically() throws {
        try XCTSkipIf(
            shouldSkipDisplayDependentRenderInCI,
            "Dashboard PNG render verification is display-dependent; set BRAINBAR_RENDER_IN_CI=1 to run in CI."
        )

        let states: [(BrainBarDashboardFixture.OperatorState, String)] = [
            (.watcherOffline, "dashboard-watcher-offline"),
            (.watcherUnknown, "dashboard-watcher-unknown"),
            (.watcherRunningNoRecentFlow, "dashboard-watcher-running-no-recent-flow"),
            (.watcherStalledWithPendingWork, "dashboard-watcher-stalled-pending-work"),
        ]

        for (state, name) in states {
            let collector = BrainBarDashboardFixture.makeCollector(state)
            let view = BrainBarDashboardPreview.make(collector: collector)
            let (png, bitmap) = try renderPNG(view, size: NSSize(width: 960, height: 1_820))
            let url = try writePNG(png, name: name)

            XCTAssertGreaterThan(png.count, 5_000, "\(name) PNG looks empty")
            XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 16, "\(name) render is too flat")
            print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
        }
    }

    @MainActor
    func testDashboardChartWindowsAndTooltipSummaryRenderDeterministically() throws {
        try XCTSkipIf(
            shouldSkipDisplayDependentRenderInCI,
            "Dashboard PNG render verification is display-dependent; set BRAINBAR_RENDER_IN_CI=1 to run in CI."
        )

        let windows: [(PipelineTimeframe, String)] = [
            (.threeHour, "dashboard-chart-window-3h"),
            (.day, "dashboard-chart-window-24h"),
        ]
        for (timeframe, name) in windows {
            let windowedStats = BrainBarDashboardFixture.makeStats(
                activityWindowMinutes: timeframe.windowMinutes
            )
            let windowedSummary = DashboardFlowSummary.derive(
                daemon: nil,
                stats: windowedStats,
                now: BrainBarDashboardFixture.fetchedAt
            )
            XCTAssertEqual(
                windowedSummary.allCommits.windowLabel,
                DashboardMetricFormatter.windowLabel(minutes: timeframe.windowMinutes)
            )
            XCTAssertTrue(
                windowedSummary.allCommits.volumeText.hasSuffix("in \(timeframe.windowMinutes == 180 ? "3h" : "24h")"),
                "Chart footer count must follow the selected window instead of retaining a 1h anchor."
            )
            XCTAssertNotEqual(
                windowedSummary.allCommits.rateText,
                "0.0/min",
                "A non-empty wider-window fixture must not render a false zero headline rate."
            )
            let view = BrainBarPipelinePanelPreview.make(
                stats: windowedStats,
                containerSize: CGSize(width: 1_120, height: 1_420),
                fetchedAt: BrainBarDashboardFixture.fetchedAt,
                selectedTimeframe: timeframe
            )
            let (png, bitmap) = try renderPNG(view, size: NSSize(width: 1_120, height: 1_420))
            let url = try writePNG(png, name: name)

            XCTAssertGreaterThan(png.count, 5_000, "\(name) PNG looks empty")
            XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 16, "\(name) render is too flat")
            print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
        }

        let disclosure = "Window: Last 1h · Count: hovered value below · Unit: unique chunk IDs first seen in this window · Clock: ingest time"
        let accessibilitySummary = "WATCHER INGESTED CHUNKS. Window: Last 1h. Count: 14. Unit: unique chunk IDs first seen in this window. Clock: ingest time."
        let presentation = SparklineChartPresentation(
            label: "WATCHER INGESTED CHUNKS",
            values: [1, 0, 2, 1, 0, 3, 1, 2, 0, 1, 2, 1],
            activityWindowMinutes: 60,
            latestBucketName: "latest ingest bucket",
            fetchedAt: BrainBarDashboardFixture.fetchedAt,
            metricDisclosure: disclosure,
            accessibilitySummary: accessibilitySummary
        )
        XCTAssertTrue(presentation.accessibilityValue.contains(accessibilitySummary))
        let tooltipView = SparklineChart(
            presentation: presentation,
            accentColor: BrainBarDesignTokens.Colors.seriesWatcher,
            previewHoveredBucket: 7,
            previewHoverX: 430
        )
        .frame(width: 840, height: 320)
        .padding(20)
        .background(Color.brainBarBackgroundBase)
        .environment(\.colorScheme, .dark)

        let (tooltipPNG, tooltipBitmap) = try renderPNG(
            tooltipView,
            size: NSSize(width: 880, height: 360)
        )
        let tooltipURL = try writePNG(tooltipPNG, name: "dashboard-chart-tooltip-summary")
        XCTAssertGreaterThan(tooltipPNG.count, 5_000, "dashboard tooltip PNG looks empty")
        XCTAssertGreaterThan(distinctSampledColorCount(in: tooltipBitmap), 16, "dashboard tooltip render is too flat")
        print("[brainbar-render] wrote \(tooltipURL.path) (\(tooltipPNG.count) bytes)")
    }

    @MainActor
    func testPipelinePreviewIgnoresAmbientSystemAppearance() throws {
        try XCTSkipIf(
            shouldSkipDisplayDependentRenderInCI,
            "Dashboard PNG render verification is display-dependent; set BRAINBAR_RENDER_IN_CI=1 to run in CI."
        )

        let stats = BrainBarDashboardFixture.makeStats(activityWindowMinutes: 180)
        let size = NSSize(width: 1_120, height: 1_420)
        let darkParent = BrainBarPipelinePanelPreview.make(
            stats: stats,
            containerSize: size,
            fetchedAt: BrainBarDashboardFixture.fetchedAt,
            selectedTimeframe: .threeHour
        )
        .environment(\.colorScheme, .dark)
        let lightParent = BrainBarPipelinePanelPreview.make(
            stats: stats,
            containerSize: size,
            fetchedAt: BrainBarDashboardFixture.fetchedAt,
            selectedTimeframe: .threeHour
        )
        .environment(\.colorScheme, .light)

        let (darkPNG, darkBitmap) = try renderPNG(darkParent, size: size)
        let (lightPNG, _) = try renderPNG(lightParent, size: size)

        XCTAssertGreaterThan(darkPNG.count, 5_000, "pipeline determinism PNG looks empty")
        XCTAssertGreaterThan(distinctSampledColorCount(in: darkBitmap), 16, "pipeline determinism render is too flat")
        XCTAssertEqual(
            lightPNG,
            darkPNG,
            "The deterministic proof seam must pin dark appearance instead of inheriting the host Mac setting."
        )
    }

    @MainActor
    func testDashboardDetailsDisclosureRendersExpandedInteractionState() throws {
        try XCTSkipIf(
            shouldSkipDisplayDependentRenderInCI,
            "Dashboard PNG render verification is display-dependent; set BRAINBAR_RENDER_IN_CI=1 to run in CI."
        )

        let panelState = BrainBarDashboardPanelState()
        var interaction = BrainBarDisclosureInteractionState()
        panelState.detailsExpanded = interaction.activate(isExpanded: false, source: .pointer)
        let view = BrainBarDashboardPreview.make(
            collector: BrainBarDashboardFixture.makeCollector(.partialReplayDebt),
            panelState: panelState
        )
        let (png, bitmap) = try renderPNG(view, size: NSSize(width: 960, height: 1_200))
        let url = try writePNG(png, name: "dashboard-details-expanded-state")

        XCTAssertTrue(panelState.detailsExpanded, "The modeled pointer interaction must expand Details.")
        XCTAssertFalse(interaction.showsKeyboardFocusRing, "Modeled pointer activation must not leave a focus ring.")
        XCTAssertGreaterThan(png.count, 5_000, "expanded Details PNG looks empty")
        XCTAssertGreaterThan(
            distinctSampledColorCount(in: bitmap),
            16,
            "expanded Details render is too flat"
        )
        print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
    }

    @MainActor
    func testDashboardDisclosureKeyboardFocusRingActuallyDraws() throws {
        let size = NSSize(width: 320, height: 64)
        let (keyboardPNG, keyboardBitmap) = try renderPNG(
            BrainBarDisclosureRowPreview.make(focusSource: .keyboard),
            size: size
        )
        let (pointerPNG, pointerBitmap) = try renderPNG(
            BrainBarDisclosureRowPreview.make(focusSource: .pointer),
            size: size
        )
        let url = try writePNG(keyboardPNG, name: "dashboard-disclosure-keyboard-focus-ring")

        XCTAssertNotEqual(keyboardPNG, pointerPNG, "Keyboard focus must change the rendered disclosure pixels.")
        XCTAssertGreaterThan(
            differingPixelCount(keyboardBitmap, pointerBitmap),
            200,
            "The actual disclosure component must draw a visible keyboard focus ring."
        )
        print("[brainbar-render] wrote \(url.path) (\(keyboardPNG.count) bytes)")
    }

    @MainActor
    func testOperatorStateFixturesStayIsolatedFromProductionDatabaseAndProcessServices() throws {
        let collector = BrainBarDashboardFixture.makeCollector()
        let fields = Dictionary(uniqueKeysWithValues: Mirror(reflecting: collector).children.compactMap { child in
            child.label.map { ($0, child.value) }
        })

        XCTAssertEqual(fields["dbPath"] as? String, "/nonexistent/brainbar-fixture.db")
        XCTAssertEqual(fields["isRunning"] as? Bool, false)

        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let fixtureSource = try String(
            contentsOf: packageRoot.appendingPathComponent("Sources/BrainBar/Dashboard/BrainBarDashboardFixture.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(fixtureSource.contains("StatsCollector.fixture("))
        XCTAssertFalse(fixtureSource.contains(".start()"), "Deterministic fixtures must not start collectors or timers.")
        XCTAssertFalse(fixtureSource.contains("launchctl"), "Fixture variants must not probe or mutate process services.")
        XCTAssertFalse(
            fixtureSource.contains(".local/share/brainlayer/brainlayer.db"),
            "Fixture variants must never resolve the canonical production database."
        )
        XCTAssertTrue(fixtureSource.contains("case loading"), "Snapshot fixtures must cover LOADING.")
        XCTAssertTrue(fixtureSource.contains("case live"), "Snapshot fixtures must cover LIVE.")
        XCTAssertTrue(fixtureSource.contains("case stale"), "Snapshot fixtures must cover STALE.")
        XCTAssertTrue(fixtureSource.contains("case error"), "Snapshot fixtures must cover ERROR with last-good data.")
        XCTAssertTrue(
            fixtureSource.contains("case partialReplayDebt"),
            "Snapshot fixtures must cover known replay components plus an unreadable input."
        )

        let loading = BrainBarDashboardFixture.makeCollector(.loading)
        let live = BrainBarDashboardFixture.makeCollector(.live)
        let stale = BrainBarDashboardFixture.makeCollector(.stale)
        let error = BrainBarDashboardFixture.makeCollector(.error)
        let partialReplayDebt = BrainBarDashboardFixture.makeCollector(.partialReplayDebt)
        XCTAssertEqual(loading.snapshotFreshnessState, .loading)
        XCTAssertEqual(live.snapshotFreshnessState, .live(ageSeconds: 0))
        XCTAssertEqual(stale.snapshotFreshnessState, .stale(ageSeconds: 61))
        XCTAssertEqual(
            error.snapshotFreshnessState,
            .error(message: "Fixture fetch failed", lastSuccessAgeSeconds: 15)
        )
        XCTAssertNotNil(error.lastDataFetchedAt)
        XCTAssertEqual(error.lastFetchError, "Fixture fetch failed")
        XCTAssertTrue(partialReplayDebt.stats.replayDebtBreakdown.isPartial)
        XCTAssertGreaterThan(partialReplayDebt.stats.replayDebtBreakdown.pendingStores.snapshot.depth, 0)
        XCTAssertGreaterThan(partialReplayDebt.stats.replayDebtBreakdown.durableQueue.snapshot.depth, 0)
        XCTAssertGreaterThan(partialReplayDebt.stats.replayDebtBreakdown.repositoryFallback.snapshot.depth, 0)
        XCTAssertFalse(partialReplayDebt.stats.replayDebtBreakdown.durableQueue.readability.isReadable)
    }

    // MARK: - Render helpers

    @MainActor
    private func makeOnePagePresentation(
        collector: StatsCollector,
        result: ObservabilityReadResult,
        now: Date
    ) throws -> BrainBarOnePagePresentation {
        let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: now)
        let hero = BrainBarHeroPresentation.derive(
            flow: flow,
            stats: collector.stats,
            backupTruth: BrainBarHeroBackupTruth.derive(from: result, now: now, cadence: .known(300)),
            locale: BrainBarDashboardPreview.goldenLocale
        )
        return BrainBarOnePagePresentation.derive(
            snapshotFreshness: collector.snapshotFreshnessState,
            hero: hero,
            observability: result,
            stats: collector.stats,
            agentActivity: collector.agentActivity,
            now: now,
            calendar: BrainBarDashboardPreview.goldenCalendar,
            locale: BrainBarDashboardPreview.goldenLocale,
            observabilityCadence: .known(300)
        )
    }

    @MainActor
    private func renderPNG(_ view: some View, size: NSSize) throws -> (Data, NSBitmapImageRep) {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(origin: .zero, size: size)
        host.layoutSubtreeIfNeeded()
        // Give SwiftUI onAppear/layout a moment to settle. With reduceMotion the
        // final state is reached immediately; this only flushes the run loop, so
        // the rendered RESULT is deterministic regardless of the wall-clock delay.
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
        host.layoutSubtreeIfNeeded()

        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
            throw RenderError.bitmapUnavailable
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            throw RenderError.encodingFailed
        }
        return (png, bitmap)
    }

    private func writePNG(_ png: Data, name: String) throws -> URL {
        let dir = outputDirectory()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("\(name).png")
        try png.write(to: url)
        return url
    }

    private func outputDirectory() -> URL {
        if let override = ProcessInfo.processInfo.environment["BRAINBAR_RENDER_DIR"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        // #filePath = .../brain-bar/Tests/BrainBarTests/BrainBarDashboardSnapshotTests.swift
        // up 3 → brain-bar/
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("docs.local/brainbar-render", isDirectory: true)
    }

    private func packageRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func distinctSampledColorCount(in bitmap: NSBitmapImageRep) -> Int {
        guard let data = bitmap.bitmapData else { return 0 }
        let bytesPerPixel = max(bitmap.bitsPerPixel / 8, 1)
        let baseStride = max(bitmap.bytesPerRow / 32, bytesPerPixel)
        let sampleStride = baseStride - (baseStride % bytesPerPixel)
        var colors = Set<String>()
        for y in stride(from: 0, to: bitmap.pixelsHigh, by: 24) {
            let rowStart = y * bitmap.bytesPerRow
            for x in stride(from: 0, to: bitmap.bytesPerRow, by: sampleStride) {
                let offset = rowStart + x
                guard offset + 2 < bitmap.bytesPerRow * bitmap.pixelsHigh else { continue }
                colors.insert("\(data[offset])-\(data[offset + 1])-\(data[offset + 2])")
            }
        }
        return colors.count
    }

    private func warmBrownPixelShare(in bitmap: NSBitmapImageRep) -> Double {
        var warm = 0
        var sampled = 0
        for y in stride(from: 0, to: bitmap.pixelsHigh, by: 8) {
            for x in stride(from: 0, to: bitmap.pixelsWide, by: 8) {
                guard let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { continue }
                sampled += 1
                if color.redComponent > 0.16,
                   color.redComponent > color.blueComponent * 1.35,
                   color.greenComponent > color.blueComponent * 1.10,
                   color.redComponent > color.greenComponent * 1.12 {
                    warm += 1
                }
            }
        }
        return sampled == 0 ? 1 : Double(warm) / Double(sampled)
    }

    private func bottomBandContainsForeground(in bitmap: NSBitmapImageRep) -> Bool {
        let bandHeight = max(Int(Double(bitmap.pixelsHigh) * 0.15), 1)
        return (0 ..< bandHeight).contains { y in
            (1 ..< bitmap.pixelsWide).contains { x in
                guard let a = bitmap.colorAt(x: x - 1, y: y), let b = bitmap.colorAt(x: x, y: y) else { return false }
                return abs(a.redComponent - b.redComponent) + abs(a.greenComponent - b.greenComponent)
                    + abs(a.blueComponent - b.blueComponent) > 0.12
            }
        }
    }

    private func differingPixelCount(_ lhs: NSBitmapImageRep, _ rhs: NSBitmapImageRep) -> Int {
        guard lhs.pixelsWide == rhs.pixelsWide, lhs.pixelsHigh == rhs.pixelsHigh else { return .max }
        var count = 0
        for y in 0 ..< lhs.pixelsHigh {
            for x in 0 ..< lhs.pixelsWide {
                guard let a = lhs.colorAt(x: x, y: y)?.usingColorSpace(.sRGB),
                      let b = rhs.colorAt(x: x, y: y)?.usingColorSpace(.sRGB)
                else { continue }
                let delta = abs(a.redComponent - b.redComponent)
                    + abs(a.greenComponent - b.greenComponent)
                    + abs(a.blueComponent - b.blueComponent)
                    + abs(a.alphaComponent - b.alphaComponent)
                if delta > 0.05 { count += 1 }
            }
        }
        return count
    }

    private enum RenderError: Error {
        case bitmapUnavailable
        case encodingFailed
    }
}
