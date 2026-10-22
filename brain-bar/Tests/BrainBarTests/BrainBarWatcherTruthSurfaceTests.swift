import AppKit
import SwiftUI
import Vision
import XCTest
@testable import BrainBar

/// #966: the Dashboard said "Watcher flow needs attention" while Settings said "Watcher running".
/// The hero's `.stalled` fired on BrainBar's replay debt (deferred stores), which is not
/// watcher work, and no reason was shown anywhere.
@MainActor
final class BrainBarWatcherTruthSurfaceTests: XCTestCase {
    private let now = BrainBarDashboardFixture.fetchedAt

    private var renderDirectory: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(".build.noindex/brainbar-render", isDirectory: true)
    }

    private func healthyBackups() -> ObservabilityBackupStatus {
        let green = ObservabilityStatusLine(text: "verified", tone: .green)
        return .init(upload: green, snapshot: green, job: green, freshness: green, retention: green, archives: green, error: nil)
    }

    private func stats(
        replayDebt: Int,
        process: WatcherProcessProbeResult? = .running(pid: 4242),
        health: WatcherHealthFileRead? = BrainBarDashboardFixture.healthyWatcherHealth,
        recentWatcherChunks: Int = 0,
        flowReadable: Bool = true,
        watcherBuckets: [Int] = [0, 0, 0, 0]
    ) -> DashboardStats {
        DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0],
            recentAgentWriteBuckets: [0, 0, 0, 0],
            recentWatcherWriteBuckets: watcherBuckets,
            recentEnrichmentBuckets: [0, 0, 0, 0],
            activityWindowMinutes: 30,
            bucketCount: 4,
            pendingStoreQueueDepth: replayDebt,
            pendingStoreFlushQueueDepth: replayDebt,
            watcherHealth: health,
            watcherProcessProbeResult: process,
            watcherRecentDistinctChunkCount: recentWatcherChunks,
            watcherFlowReadability: flowReadable ? .readable : .unreadable("flow query failed")
        )
    }

    private func degradedHealth() -> WatcherHealthFileRead {
        .readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-70),
            pollCount: 72,
            alertReasons: ["file_ingestion_failure"],
            fileIngestionFailureCount: 2,
            earliestFileIngestionFailureAt: now.addingTimeInterval(-7_200)
        ))
    }

    // MARK: #1014 review round 1

    func testFreshWatcherHeartbeatKeepsUnreadableFlowNeutral() {
        for process in [WatcherProcessProbeResult.running(pid: 42), .failure("launchctl timed out")] {
            let flow = DashboardFlowSummary.derive(daemon: nil,
                stats: stats(replayDebt: 0, process: process, flowReadable: false), now: now)
            let lane = flow.lane(for: .jsonlWatcher)
            XCTAssertTrue(flow.watcherStatus.isRunning)
            XCTAssertEqual(lane.evidenceStatus, .unavailable, "heartbeat does not supply chart counts")
            XCTAssertEqual(lane.status, .running)
            let disclosure = BrainBarDashboardChartDisclosure(series: .jsonlWatcher, lane: lane, timeframe: .live)
            XCTAssertEqual(disclosure.tooltipDisclosure, "Updating…")
            XCTAssertTrue(disclosure.accessibilitySummary.contains("Updating…"))
            XCTAssertFalse(disclosure.accessibilitySummary.contains("Count:"))
            XCTAssertEqual(BrainBarIngestSeriesPresentation(lane: lane).metricText, "Updating…")
        }
    }

    func testUnreadableFlowNeverHidesStoppedOrStaleWatcherEvidence() {
        let cases: [(WatcherProcessProbeResult, WatcherHealthFileRead?, DashboardFlowLaneStatus, String)] = [
            (.absent, BrainBarDashboardFixture.healthyWatcherHealth, .stopped, "Watcher stopped"),
            (.running(pid: 42), degradedHealth(), .attention, "Watcher needs attention"),
            (.running(pid: 42), .readable(WatcherHealthFile(updatedAt: now.addingTimeInterval(-900), pollCount: 72)), .attention, "Watcher needs attention"),
            (.failure("launchctl timed out"), .missing(path: "/fixture/health.json"), .unknown, "Watcher status unknown"),
        ]
        for (process, health, expected, message) in cases {
            let lane = DashboardFlowSummary.derive(daemon: nil,
                stats: stats(replayDebt: 0, process: process, health: health, flowReadable: false), now: now)
                .lane(for: .jsonlWatcher)
            XCTAssertEqual(lane.status, expected)
            let disclosure = BrainBarDashboardChartDisclosure(series: .jsonlWatcher, lane: lane, timeframe: .live)
            XCTAssertEqual(disclosure.tooltipDisclosure, expected == .unknown ? "\(message) · \(lane.lastEventText)" : message)
            XCTAssertFalse(BrainBarIngestSeriesPresentation(lane: lane).showsSparkline)
            if expected == .stopped { XCTAssertEqual(lane.status.stateTheme, .error) }
        }
    }

    func testUnreadableFlowKeepsUnknownMissingAndStaleHealthAmber() {
        let cases: [(WatcherProcessProbeResult?, WatcherHealthFileRead, String)] = [
            (.running(pid: 42), .missing(path: "/fixture/watcher-health.json"), "missing"),
            (nil, .missing(path: "/fixture/watcher-health.json"), "missing"),
            (nil, .unreadable(path: "/fixture/watcher-health.json", reason: "permission denied"), "permission denied"),
            (.failure("launchctl timed out"), .readable(WatcherHealthFile(
                updatedAt: now.addingTimeInterval(-900), pollCount: 72
            )), "15m ago"),
        ]
        for (process, health, reason) in cases {
            let lane = DashboardFlowSummary.derive(daemon: nil,
                stats: stats(replayDebt: 0, process: process, health: health, flowReadable: false), now: now)
                .lane(for: .jsonlWatcher)
            XCTAssertEqual(lane.status, .unknown, "do not invent a process verdict")
            XCTAssertEqual(lane.status.stateTheme, .degraded, "unknown health must retain its amber warning")
            XCTAssertEqual(lane.evidencePlaceholder, "Watcher status unknown")
            let disclosure = BrainBarDashboardChartDisclosure(series: .jsonlWatcher, lane: lane, timeframe: .live)
            XCTAssertTrue(disclosure.tooltipDisclosure.contains(reason), disclosure.tooltipDisclosure)
            XCTAssertTrue(disclosure.accessibilitySummary.contains(reason), disclosure.accessibilitySummary)
            XCTAssertFalse(disclosure.accessibilitySummary.contains("Count:"))
            XCTAssertEqual(BrainBarIngestSeriesPresentation(lane: lane).metricText, "Watcher status unknown")
        }
    }

    func testRenderFreshHeartbeatWithUnreadableFlow() throws {
        let view = BrainBarDashboardPreview.make(
            collector: BrainBarDashboardFixture.makeCollector(stats: stats(replayDebt: 0, flowReadable: false)),
            now: now
        ).frame(width: 960, height: 640)
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: 960, height: 640)
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        let directory = renderDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try png.write(to: directory.appendingPathComponent("watcher-flow-updating.png"))
        XCTAssertGreaterThan(png.count, 5_000)
    }

    func testPreviewHidesUnreadableWatcherCountsAndKeepsUpdatingDisclosure() {
        for readable in [false, true] {
            let sample = stats(replayDebt: 0, flowReadable: readable, watcherBuckets: [9876, 9876, 9876, 9876])
            // Match BrainBarPipelinePanelPreviewView.flowSummary and seriesCard: the raw
            // buckets remain populated even when the query's evidence is unreadable.
            let lane = DashboardFlowSummary.derive(daemon: nil, stats: sample, now: now).lane(for: .jsonlWatcher)
            let presentation = BrainBarIngestSeriesPresentation(lane: lane)
            let disclosure = BrainBarDashboardChartDisclosure(series: .jsonlWatcher, lane: lane, timeframe: .live)
            let presentedText = [
                presentation.metricText, presentation.rateText, presentation.volumeText,
                presentation.latestBucketCaptionText, disclosure.subtitle, disclosure.visibleDetail,
                disclosure.statusLabel, disclosure.tooltipDisclosure, disclosure.accessibilitySummary,
            ].compactMap { $0 }.joined(separator: "\n")

            XCTAssertEqual(presentation.showsSparkline, readable)
            for token in ["39,504", "1316.8/min", "latest:"] {
                XCTAssertEqual(presentedText.contains(token), readable, "\(token): \(presentedText)")
            }
            if readable {
                XCTAssertEqual(presentation.metricText, "39,504 · peak 9.9k")
                XCTAssertEqual(presentation.rateText, "1316.8/min")
                XCTAssertEqual(presentation.volumeText, "39,504 in 30m")
                XCTAssertEqual(presentation.latestBucketCaptionText, "latest: 9.9k")
                XCTAssertEqual(disclosure.accessibilitySummary,
                    "Watcher-ingested chunks. Window: Last 1h. Count: 39,504. Unit: unique chunk IDs first seen in this window. Clock: ingest time.")
                XCTAssertEqual(disclosure.tooltipDisclosure,
                    "Window: Last 1h · Count: hovered value below · Unit: unique chunk IDs first seen in this window · Clock: ingest time")
                XCTAssertFalse(presentedText.contains("Updating…"))
            } else {
                XCTAssertEqual(presentation.metricText, "Updating…")
                XCTAssertEqual(presentation.rateText, "—")
                XCTAssertNil(presentation.volumeText)
                XCTAssertNil(presentation.latestBucketCaptionText)
                XCTAssertEqual(disclosure.tooltipDisclosure, "Updating…")
                XCTAssertEqual(disclosure.accessibilitySummary,
                    "Watcher-ingested chunks. Updating…. Window: Last 1h. Unit: unique chunk IDs first seen in this window. Clock: ingest time.")
                XCTAssertFalse(disclosure.accessibilitySummary.contains("Count:"))
            }
        }
    }

    func testRenderPreviewWatcherCountsForOptionalLocalEvidence() throws {
        guard ProcessInfo.processInfo.environment["BRAINBAR_RENDER_EVIDENCE"] == "1",
              ProcessInfo.processInfo.environment["CI"] == nil else {
            throw XCTSkip("Set BRAINBAR_RENDER_EVIDENCE=1 locally for preview PNGs and diagnostic OCR.")
        }
        for readable in [false, true] {
            let sample = stats(replayDebt: 0, flowReadable: readable, watcherBuckets: [9876, 9876, 9876, 9876])
            let view = BrainBarPipelinePanelPreview.make(
                stats: sample, containerSize: CGSize(width: 960, height: 1400), fetchedAt: now,
                signalCoverageExpanded: false
            ).frame(width: 960, height: 1400)
            let host = NSHostingView(rootView: view)
            host.frame = NSRect(x: 0, y: 0, width: 960, height: 1400)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
            host.layoutSubtreeIfNeeded()
            // OCR is diagnostic only: Vision can misread the correctly rendered 11pt label.
            let bitmap = try XCTUnwrap(NSBitmapImageRep(
                bitmapDataPlanes: nil,
                pixelsWide: Int(host.bounds.width * 3),
                pixelsHigh: Int(host.bounds.height * 3),
                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
            ))
            bitmap.size = host.bounds.size
            host.cacheDisplay(in: host.bounds, to: bitmap)
            let directory = renderDirectory
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
                .write(to: directory.appendingPathComponent(readable ? "watcher-preview-readable.png" : "watcher-preview-updating.png"))
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            try VNImageRequestHandler(cgImage: XCTUnwrap(bitmap.cgImage)).perform([request])
            let text = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
            print("Watcher preview diagnostic OCR (readable=\(readable)):\n\(text)")
        }
    }

    /// B1: the lane's status pill rendered "live" (chunks flowed recently) while its body said
    /// NEEDS ATTENTION. The pill comes from the same `WatcherHealthStatus`.
    func testDegradedWatcherLaneDoesNotDisplayAHealthyFlowPill() {
        let flow = DashboardFlowSummary.derive(
            daemon: nil,
            stats: stats(replayDebt: 0, health: degradedHealth(), recentWatcherChunks: 5),
            now: now
        )
        let lane = flow.lane(for: .jsonlWatcher)
        XCTAssertEqual(lane.statusText, "NEEDS ATTENTION")
        XCTAssertEqual(lane.status.label, "needs attention", "lane pill contradicts its watcher status: \(lane.status.label)")
        XCTAssertEqual(lane.status.stateTheme, .degraded)
    }

    /// B1: over every process, health and flow combination, the pill is a function of the one
    /// watcher state: live/idle/running only while it is Running, and one pill per other state.
    func testWatcherLanePillFollowsTheOneWatcherStateEverywhere() {
        let processes: [WatcherProcessProbeResult] = [.running(pid: 1), .absent, .failure("launchctl failed")]
        let healths: [WatcherHealthFileRead?] = [
            BrainBarDashboardFixture.healthyWatcherHealth, degradedHealth(),
            .missing(path: "/data/watcher-health.json"), nil,
        ]
        for process in processes {
            for health in healths {
                for chunks in [0, 5] {
                    for readable in [true, false] {
                        let flow = DashboardFlowSummary.derive(
                            daemon: nil,
                            stats: stats(replayDebt: 0, process: process, health: health, recentWatcherChunks: chunks, flowReadable: readable),
                            now: now
                        )
                        let pill = flow.lane(for: .jsonlWatcher).status
                        let context = "\(process) \(String(describing: health)) chunks=\(chunks) readable=\(readable): \(pill.label)"
                        switch flow.watcherStatus {
                        case .running:
                            XCTAssertTrue([.live, .idle, .running].contains(pill), context)
                        case .degraded:
                            XCTAssertEqual(pill, .attention, context)
                        case .stopped:
                            XCTAssertEqual(pill, .stopped, context)
                        case .unknown:
                            XCTAssertEqual(pill, .unknown, context)
                        }
                    }
                }
            }
        }
    }

    func testReplayDebtNoLongerImplicatesTheWatcher() {
        let stats = stats(replayDebt: 5)
        XCTAssertGreaterThan(stats.replayDebtBreakdown.deduplicatedTotal, 0, "fixture must carry replay debt")
        let flow = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        XCTAssertNotEqual(flow.lane(for: .jsonlWatcher).statusText, "STALLED")
        let hero = BrainBarHeroPresentation.derive(flow: flow, stats: stats, backupTruth: .measured(healthyBackups()))
        XCTAssertFalse(
            hero.healthReason.localizedCaseInsensitiveContains("watcher flow needs attention"),
            "replay debt is BrainBar's deferred-store queue, not watcher work: \(hero.healthReason)"
        )
        XCTAssertNotEqual(hero.healthVerdict, "Needs attention", hero.healthReason)
        XCTAssertEqual(hero.healthVerdict, "Healthy", "a healthy, idle watcher with replay debt is healthy: \(hero.healthReason)")
        XCTAssertEqual(flow.watcherStatus, .running(heartbeatAt: now.addingTimeInterval(-70)))
    }

    /// Every surface renders the same status with the same reason line (#966 acceptance:
    /// "the footer never contradicts the dashboard").
    func testEverySurfaceShowsTheSameWatcherTruth() {
        let firstFailure = now.addingTimeInterval(-7_200)
        let degraded = WatcherHealthFileRead.readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-70),
            pollCount: 72,
            alertReasons: ["file_ingestion_failure"],
            fileIngestionFailureCount: 2,
            earliestFileIngestionFailureAt: firstFailure
        ))
        let cases: [(String, WatcherProcessProbeResult, WatcherHealthFileRead?, BrainLayerLaunchdLoadState, String)] = [
            ("running", .running(pid: 1), BrainBarDashboardFixture.healthyWatcherHealth, .running, "Watcher running"),
            ("degraded", .running(pid: 1), degraded, .running, "Watcher needs attention"),
            ("stopped", .absent, BrainBarDashboardFixture.healthyWatcherHealth, .unloaded, "Watcher stopped"),
            ("unknown", .running(pid: 1), .missing(path: "/data/watcher-health.json"), .running, "Watcher status unknown"),
        ]
        for (name, process, health, loadState, title) in cases {
            let stats = stats(replayDebt: 0, process: process, health: health)
            let flow = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            let hero = BrainBarHeroPresentation.derive(flow: flow, stats: stats, backupTruth: .measured(healthyBackups()))
            let lane = flow.lane(for: .jsonlWatcher)

            var config = BrainLayerConfig.defaultConfig
            config.launchdJobs[.watch] = BrainLayerLaunchdJobSetting(enabled: true, loadState: loadState)
            let settingsStatus = WatcherHealthStatus.derive(
                launchd: WatcherLaunchdEvidence(setting: config.launchdJobs[.watch], loadState: loadState),
                file: health,
                now: now
            )
            let footer = BrainBarSettingsFooterPresentation(config: config, watcher: settingsStatus, now: now)
            let ingest = BrainLayerLaunchdJobGroup.ingest.status(
                settings: config.launchdJobs,
                observations: [.watch: .stateOnly(loadState), .index: .stateOnly(.loaded)],
                formatDate: { _ in "t" },
                watcher: settingsStatus,
                now: now
            )

            XCTAssertEqual(flow.watcherStatus.title, title, name)
            XCTAssertEqual(footer.state.title, title, "\(name): footer must match the dashboard")
            let reason = flow.watcherStatusReason
            // Settings observes launchd itself (loaded/unloaded/disabled) while the Dashboard probes
            // the process, so a stopped job may carry a more specific launchd detail in Settings.
            // The verdict and every other reason are identical.
            if name == "stopped" {
                XCTAssertTrue(reason?.hasPrefix("Watcher is not running") == true, reason ?? "nil")
                XCTAssertTrue(footer.detail?.hasPrefix("Watcher is not running") == true, footer.detail ?? "nil")
            } else {
                XCTAssertEqual(footer.detail, reason, "\(name): footer reason")
            }
            XCTAssertEqual(ingest.attentionReason, footer.detail, "\(name): Settings Ingest and footer share one reason")
            if let reason {
                XCTAssertEqual(lane.lastEventText, reason, "\(name): lane reason")
                XCTAssertEqual(hero.healthReason, reason, "\(name): hero reason")
            } else {
                XCTAssertEqual(hero.healthVerdict, "Healthy", name)
                XCTAssertEqual(ingest.health, .healthy, name)
            }
        }
    }

    func testConfiguredIngestPreservesRuntimeTruthAcrossDashboardAndSettings() throws {
        let cases: [(WatcherProcessProbeResult, BrainLayerLaunchdLoadState, WatcherHealthFileRead?, String, BrainLayerLaunchdGroupHealth)] = [
            (.running(pid: 42), .running, BrainBarDashboardFixture.healthyWatcherHealth, "Watcher running", .healthy),
            (.running(pid: 42), .running, degradedHealth(), "Watcher needs attention", .unhealthy),
            (.running(pid: 42), .running, .readable(WatcherHealthFile(
                updatedAt: now.addingTimeInterval(-900), pollCount: 72
            )), "Watcher needs attention", .unhealthy),
            (.running(pid: 42), .running, .missing(path: "/fixture/health.json"), "Watcher status unknown", .unknown),
            (.absent, .unloaded, BrainBarDashboardFixture.healthyWatcherHealth, "Watcher stopped", .unhealthy),
            (.absent, .loaded, BrainBarDashboardFixture.healthyWatcherHealth, "Watcher stopped", .unhealthy),
            (.failure("launchctl timed out"), .probeError("launchctl timed out"), nil, "Watcher status unknown", .unknown),
            (.failure("launchd status for com.brainlayer.watch is unknown"), .unknown, nil, "Watcher status unknown", .unknown),
            (.failure("launchctl timed out"), .probeError("launchctl timed out"), BrainBarDashboardFixture.healthyWatcherHealth, "Watcher running", .healthy),
            (.failure("launchctl timed out"), .probeError("launchctl timed out"), degradedHealth(), "Watcher needs attention", .unhealthy),
        ]
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = BrainLayerConfigStore(configURL: directory.appendingPathComponent("brainlayer.env"))
        let configurations: [(Bool, Bool, String?)] = [
            (true, true, nil), (false, false, "Configured OFF: Watcher, Index."),
            (false, true, "Configured OFF: Watcher."), (true, false, "Configured OFF: Index."),
        ]
        for (watchEnabled, indexEnabled, note) in configurations {
            var config = BrainLayerConfig.defaultConfig
            config.launchdJobs[.watch]?.enabled = watchEnabled
            config.launchdJobs[.index]?.enabled = indexEnabled
            try store.save(config)
            for (process, loadState, health, title, groupHealth) in cases {
                let flow = DashboardFlowSummary.derive(
                    daemon: nil, stats: stats(replayDebt: 0, process: process, health: health), now: now
                )
                let viewModel = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
                    launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
                    initialLaunchdStates: [.watch: loadState, .index: .loaded],
                    refreshStatusOnLoad: false,
                    now: { [now] in now },
                    initialWatcherHealth: health
                )
                XCTAssertEqual(viewModel.isGroupEnabled(.ingest), watchEnabled && indexEnabled)
                XCTAssertEqual(viewModel.config.launchdJobs[.watch]?.enabled, watchEnabled)
                XCTAssertEqual(viewModel.config.launchdJobs[.index]?.enabled, indexEnabled)
                XCTAssertEqual(flow.watcherStatus.title, title)
                XCTAssertEqual(viewModel.watcherStatus.title, title)
                XCTAssertEqual(viewModel.footerPresentation.state.title, title)
                let ingest = viewModel.groupStatus(.ingest)
                let expectedHealth: BrainLayerLaunchdGroupHealth = groupHealth == .healthy && !watchEnabled && !indexEnabled
                    ? .skipped : groupHealth
                XCTAssertEqual(ingest.health, expectedHealth, title)
                XCTAssertEqual(ingest.attentionReason, viewModel.footerPresentation.detail, title)
                XCTAssertEqual(ingest.note, note)
                XCTAssertFalse(ingest.nextRunText.contains("Index Continuous"), "loaded scheduled Index is not executing")
                if title != "Watcher stopped" {
                    XCTAssertEqual(viewModel.watcherStatus, flow.watcherStatus)
                }
            }
        }
    }

    func testConfiguredOffKeepsIndexEvidenceAndReenablingRestoresAttention() {
        var settings = BrainLayerConfig.defaultConfig.launchdJobs
        for job in BrainLayerLaunchdJobGroup.ingest.jobs { settings[job]?.enabled = false }
        let observations: [BrainLayerLaunchdJobObservation] = [
            .init(loadState: .loaded, runs: 1, lastExitCode: 1,
                  lastRunAt: now, nextRunAt: nil, isContinuous: false),
            .stateOnly(.unloaded),
        ]
        for (index, reason) in zip(observations, ["Index last run exited 1 at t.", "Index is unloaded."]) {
            let status = BrainLayerLaunchdJobGroup.ingest.status(
                settings: settings, observations: [.watch: .stateOnly(.running), .index: index],
                formatDate: { _ in "t" },
                watcher: .running(heartbeatAt: now.addingTimeInterval(-70)), now: now
            )
            XCTAssertEqual(status.health, .skipped)
            XCTAssertNil(status.attentionReason)
            XCTAssertEqual(status.note, "Configured OFF: Watcher, Index.")
            // Desired OFF changes presentation, never the captured runtime evidence.
            let enabled = BrainLayerLaunchdJobGroup.ingest.status(
                settings: BrainLayerConfig.defaultConfig.launchdJobs,
                observations: [.watch: .stateOnly(.running), .index: index],
                formatDate: { _ in "t" }, watcher: .running(heartbeatAt: now.addingTimeInterval(-70)), now: now
            )
            XCTAssertEqual(enabled.health, .unhealthy)
            XCTAssertEqual(enabled.attentionReason, reason)
        }
    }

    func testIngestToggleChangesConfigurationWithoutChangingObservedRuntime() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = BrainLayerConfigStore(configURL: directory.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let viewModel = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            initialLaunchdStates: [.watch: .running, .index: .loaded],
            refreshStatusOnLoad: false,
            now: { [now] in now },
            initialWatcherHealth: BrainBarDashboardFixture.healthyWatcherHealth
        )
        let runtime = viewModel.watcherStatus
        viewModel.setGroup(.ingest, enabled: false)
        XCTAssertFalse(viewModel.isGroupEnabled(.ingest))
        XCTAssertEqual(try store.loadDocument().config.launchdJobs[.watch]?.enabled, false)
        XCTAssertEqual(viewModel.config.launchdJobs[.watch]?.loadState, .running)
        XCTAssertEqual(viewModel.watcherStatus, runtime)
        XCTAssertEqual(viewModel.groupStatus(.ingest).health, .skipped)
        XCTAssertEqual(viewModel.groupStatus(.ingest).note, "Configured OFF: Watcher, Index.")
        viewModel.setGroup(.ingest, enabled: true)
        XCTAssertTrue(viewModel.isGroupEnabled(.ingest))
        XCTAssertEqual(try store.loadDocument().config.launchdJobs[.watch]?.enabled, true)
        XCTAssertEqual(viewModel.watcherStatus, runtime)
        XCTAssertEqual(viewModel.config.launchdJobs[.index]?.loadState, .loaded)
        XCTAssertEqual(viewModel.groupStatus(.ingest).health, .healthy)
        XCTAssertNil(viewModel.groupStatus(.ingest).note)
    }

    func testDegradedReasonNamesWhatSinceWhenAndWhatToDo() {
        let health = WatcherHealthFileRead.readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-70),
            pollCount: 72,
            alertReasons: ["file_ingestion_failure"],
            fileIngestionFailureCount: 2,
            earliestFileIngestionFailureAt: now.addingTimeInterval(-7_200)
        ))
        let flow = DashboardFlowSummary.derive(daemon: nil, stats: stats(replayDebt: 0, health: health), now: now)
        XCTAssertEqual(flow.lane(for: .jsonlWatcher).statusText, "NEEDS ATTENTION")
        XCTAssertEqual(
            flow.watcherStatusReason,
            "2 transcript files could not be ingested · since 2h ago · See file_ingestion_failures in watcher-health.json"
        )
    }

    func testUnknownIngestIsNotReportedAsAFailure() {
        var config = BrainLayerConfig.defaultConfig
        config.launchdJobs[.watch] = BrainLayerLaunchdJobSetting(enabled: true, loadState: .running)
        let unknown = WatcherHealthStatus.unknown(reason: "Watcher is running, but its health file is missing at /x.")
        let ingest = BrainLayerLaunchdJobGroup.ingest.status(
            settings: config.launchdJobs,
            observations: [.watch: .stateOnly(.running), .index: .stateOnly(.loaded)],
            formatDate: { _ in "t" },
            watcher: unknown,
            now: now
        )
        XCTAssertEqual(ingest.health, .unknown)
        XCTAssertEqual(ingest.health.title, "Status unknown")
        XCTAssertEqual(ingest.attentionReason, "Watcher is running, but its health file is missing at /x.")
    }

    func testSettingsViewModelFooterReadsTheSameHealthFile() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("watcher-health.json")
        let store = BrainLayerConfigStore(configURL: directory.appendingPathComponent("brainlayer.env"))

        // launchd says running, but the health file is missing: honest unknown, never "Watcher running".
        let viewModel = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [.watch: .running]),
            initialLaunchdStates: [.watch: .running],
            refreshStatusOnLoad: false,
            now: { [now] in now },
            watcherHealthURL: url
        )
        viewModel.refreshWatcherHealth()
        for _ in 0..<200 where viewModel.watcherHealth == nil { try await Task.sleep(nanoseconds: 10_000_000) }
        XCTAssertEqual(viewModel.watcherHealth, .missing(path: url.path))
        XCTAssertEqual(viewModel.footerPresentation.state.title, "Watcher status unknown")
        XCTAssertTrue(viewModel.footerPresentation.detail?.contains(url.path) == true)
    }
}

/// Unit tests that reach `BrainDatabase.dashboardStats()` must never read the real watcher
/// LaunchAgent, ~/.config/brainlayer/brainlayer.env or the real watcher-health.json. This points the
/// watcher plist and env file at nonexistent paths and BRAINLAYER_DB at a temp directory, and
/// returns the restore.
func isolateWatcherHealthEnvironment(root: URL) -> () -> Void {
    let keys = ["BRAINLAYER_ENV_FILE", "BRAINLAYER_DB", "BRAINLAYER_WATCHER_HEALTH_PATH", "BRAINBAR_WATCHER_PLIST"]
    let previous = keys.map { ($0, ProcessInfo.processInfo.environment[$0]) }
    setenv("BRAINBAR_WATCHER_PLIST", root.appendingPathComponent("no-watcher.plist").path, 1)
    setenv("BRAINLAYER_ENV_FILE", root.appendingPathComponent("no-brainlayer.env").path, 1)
    setenv("BRAINLAYER_DB", root.appendingPathComponent("watcher-db/brainlayer.db").path, 1)
    unsetenv("BRAINLAYER_WATCHER_HEALTH_PATH")
    return {
        for (key, value) in previous {
            if let value { setenv(key, value, 1) } else { unsetenv(key) }
        }
    }
}
