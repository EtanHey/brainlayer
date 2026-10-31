import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Visual render harness for the BrainBar dashboard redesign (feat/brainbar-dashboard-redesign).
///
/// Renders the PIPELINE + SIGNAL-COVERAGE composition — the writes/enrichment graph
/// cards plus the signal-coverage panel and queue rail — i.e. the slice being
/// redesigned. It drives the real (private) production views via the debug-only
/// `BrainBarPipelinePanelPreview` seam with a realistic mock `DashboardStats`,
/// so no live `StatsCollector`/DB is needed.
///
/// Env-gated by BRAINBAR_SNAPSHOT_DIR (skipped otherwise, like
/// SparklineSnapshotTests). Run with:
///   BRAINBAR_SNAPSHOT_DIR=/path swift test --filter DashboardRedesignSnapshotTests
@MainActor
final class DashboardRedesignSnapshotTests: XCTestCase {
    private var outDir: String? { ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] }

    private func save<V: View>(_ view: V, _ name: String, size: CGSize) {
        guard let outDir else { return }
        let renderer = ImageRenderer(content:
            view
                .frame(width: size.width, height: size.height, alignment: .topLeading)
                .background(Color(nsColor: NSColor(calibratedRed: 0.07, green: 0.08, blue: 0.10, alpha: 1)))
                .environment(\.colorScheme, .dark)
        )
        renderer.scale = 2
        guard let image = renderer.nsImage,
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else {
            XCTFail("render failed for \(name)")
            return
        }
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        let path = "\(outDir)/\(name).png"
        try? png.write(to: URL(fileURLWithPath: path))
        XCTAssertGreaterThan(png.count, 1_000, "expected a non-trivial PNG for \(name)")
        print("SNAPSHOT \(path)")
    }

    /// Realistic mock matching the task brief:
    /// - JSONL-watcher writes high + spiky (~5-9 per 5-min bucket)
    /// - Agent stores low (~0-2)
    /// - enrichment ~0/min with a big backlog
    /// - signal coverage Vector 97% / FTS5 100% / Trigram 100%
    /// - queue backlogged
    private func makeMockStats() -> BrainDatabase.DashboardStats {
        let now = Date()
        // 12 five-minute buckets across the activity window.
        let watcherWrites = [6, 8, 5, 9, 7, 6, 8, 5, 7, 9, 6, 8] // high + spiky
        let agentWrites = [0, 1, 0, 2, 0, 1, 0, 0, 1, 2, 0, 1] // low
        let combinedWrites = zip(watcherWrites, agentWrites).map(+)
        let enrichmentBuckets = Array(repeating: 0, count: 12) // ~0/min

        let totalChunks = 312_540
        let signalEligible = 312_540
        let vectorIndexed = Int(Double(signalEligible) * 0.97) // 97%
        let ftsIndexed = signalEligible // 100%
        let trigramIndexed = signalEligible // 100%
        let backlog = 41_280 // big enrichment backlog
        let enriched = totalChunks - backlog

        return BrainDatabase.DashboardStats(
            chunkCount: totalChunks,
            enrichedChunkCount: enriched,
            pendingEnrichmentCount: backlog,
            enrichmentPercent: (Double(enriched) / Double(totalChunks)) * 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 8_589_934_592, // ~8 GB
            recentActivityBuckets: combinedWrites,
            recentAgentWriteBuckets: agentWrites,
            recentWatcherWriteBuckets: watcherWrites,
            recentEnrichmentBuckets: enrichmentBuckets,
            recentWriteFiveMinuteCount: combinedWrites.suffix(1).reduce(0, +),
            recentEnrichmentFiveMinuteCount: 0,
            activityWindowMinutes: 60,
            bucketCount: 12,
            liveWindowMinutes: 1,
            lastWriteAt: now.addingTimeInterval(-20), // live writes
            lastEnrichedAt: now.addingTimeInterval(-3 * 3600), // stale enrichment (3h ago)
            signalEligibleChunkCount: signalEligible,
            vectorIndexedChunkCount: vectorIndexed,
            ftsIndexedChunkCount: ftsIndexed,
            trigramIndexedChunkCount: trigramIndexed,
            pendingStoreQueueDepth: 27,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-9 * 60),
            pendingStoreFlushRatePerMinute: 0,
            watcherHealth: WatcherHealthFileRead.unreadable(path: "fixture/watcher-health.json", reason: "no parseable updated_at")
        )
    }

    func testRenderPipelineAndSignalCoverage() throws {
        try XCTSkipIf(outDir == nil, "Set BRAINBAR_SNAPSHOT_DIR to render dashboard-redesign snapshots")
        let stats = makeMockStats()
        // Container drives the live layout. Render canvas is taller so the new
        // 3-band stack (two write cards + coverage strip + below-the-fold FLOW
        // panel) is fully captured.
        let containerSize = CGSize(width: 980, height: 780)
        let canvasSize = CGSize(width: 980, height: 1_120)

        // DEFAULT (resting) layout: two separately-scaled write cards stacked,
        // collapsed three-chip SIGNAL COVERAGE strip beneath them, FLOW below.
        save(
            BrainBarPipelinePanelPreview.make(
                stats: stats,
                containerSize: containerSize,
                signalCoverageExpanded: false
            ),
            "pipeline-signal-coverage-collapsed",
            size: canvasSize
        )

        // Expanded SIGNAL COVERAGE (full signal bars + vector detail affordance).
        save(
            BrainBarPipelinePanelPreview.make(
                stats: stats,
                containerSize: containerSize,
                signalCoverageExpanded: true
            ),
            "pipeline-signal-coverage-expanded",
            size: CGSize(width: 980, height: 1_320)
        )

        // REDESIGN: ONE shared selector switches ALL graphs at once. No card
        // expands or collapses. Render with the 24h lens selected so the shared
        // selector chip shows the wider window highlighted (Live / 3h / [24h]).
        save(
            BrainBarPipelinePanelPreview.make(
                stats: stats,
                containerSize: containerSize,
                signalCoverageExpanded: false,
                selectedTimeframe: .day
            ),
            "pipeline-shared-timeframe-24h",
            size: canvasSize
        )
    }

    func testMaintenanceRetryStates() throws {
        guard let outDir else { throw XCTSkip("Set BRAINBAR_SNAPSHOT_DIR to render retry states") }
        let directory = URL(fileURLWithPath: outDir)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var calendar = Calendar(identifier: .iso8601)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let noon = ISO8601DateFormatter().date(from: "2026-10-29T12:00:00Z")!
        let four = ISO8601DateFormatter().date(from: "2026-10-29T04:30:00Z")!
        let states: [(String, String, Bool, String?, Date)] = [
            ("ready", BrainBarMaintenanceRetryTests.ready, false, nil, noon),
            ("running", BrainBarMaintenanceRetryTests.ready, true, nil, noon),
            ("quiet-window", BrainBarMaintenanceRetryTests.ready.replacingOccurrences(of: "2026-10-30", with: "2026-10-29"), false, nil, four),
            ("window-only", #"{"status":"nothing-to-retry","reason":"Everything that can run during the day has already finished.","failures":[{"step":"maintenance-weekly","reason":"Window closed"}],"deferred":[{"step":"maintenance-weekly","reason":"It pauses database writes, so it only runs between 04:00 and 06:00"}]}"#, false, nil, noon),
            ("refused", BrainBarMaintenanceRetryTests.ready.replacingOccurrences(of: "\"status\":\"ready\"", with: "\"status\":\"refused\"").replacingOccurrences(of: "Only daytime-safe failures retry", with: "Maintenance is running right now. Try again when it finishes."), false, "Refused: Maintenance is running right now. Try again when it finishes.", noon),
            ("nothing", #"{"status":"nothing-to-retry","reason":"Everything that can run during the day has already finished."}"#, false, "Nothing to retry: Everything that can run during the day has already finished.", noon),
            ("started", #"{"status":"nothing-to-retry","reason":"Everything that can run during the day has already finished.","failures":[{"step":"maintenance-weekly","reason":"Window closed"}],"deferred":[{"step":"maintenance-weekly","reason":"It pauses database writes, so it only runs between 04:00 and 06:00"}]}"#, false, "Done: Transcript backup retried successfully.", noon),
            ("failed", BrainBarMaintenanceRetryTests.ready, false, "Failed: transcript backup stopped with an error; see its log.", noon),
            ("finished-successfully", #"{"status":"nothing-to-retry","reason":"No failed steps remain. Everything from last night has finished."}"#, false, "Done: Transcript backup retried successfully.", noon)
        ]
        for (name, json, running, outcome, now) in states {
            let model = BrainBarMaintenanceRetryModel(runner: nil, now: { now }, calendar: calendar)
            model.setForPreview(json: json, isRunning: running, outcome: outcome)
            try BrainBarRenderHarness.renderMaintenanceRetryRow(model, name: "maintenance-retry-\(name)", directory: directory)
            try BrainBarRenderHarness.renderMaintenanceRetryRow(model, name: "maintenance-retry-\(name)-390", directory: directory, width: 390)
        }
        let model = BrainBarMaintenanceRetryModel(runner: nil, now: { noon }, calendar: calendar)
        model.setForPreview(json: BrainBarMaintenanceRetryTests.ready)
        try BrainBarRenderHarness.renderMaintenanceRetryDashboard(model, directory: directory)
        // Macroscope r4412051251: one failed night is a notice with no attention list; Retry must still render.
        try BrainBarRenderHarness.renderMaintenanceRetryDashboard(model, directory: directory, attentionReasons: [],
                                                                  name: "maintenance-retry-dashboard-notice-only")
    }
}
