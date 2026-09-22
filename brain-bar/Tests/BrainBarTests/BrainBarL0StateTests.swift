import AppKit
import SwiftUI
import XCTest

@testable import BrainBar

private final class L0CountingCoverageProvider: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0

    var callCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }

    func fetch() -> BrainDatabase.SignalCoverageSnapshot {
        lock.lock()
        calls += 1
        lock.unlock()
        return .init(
            eligibleChunkCount: 100,
            vectorIndexedChunkCount: 71,
            ftsIndexedChunkCount: 83,
            trigramIndexedChunkCount: 96
        )
    }
}

final class BrainBarL0StateTests: XCTestCase {
    @MainActor
    func testDetailsTogglesPreserveLoadedCoverageAndDoNotRestartFillsOrRefetch() async throws {
        let hotStats = BrainDatabase.DashboardStats(
            chunkCount: 100,
            enrichedChunkCount: 80,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 80,
            enrichmentRatePerMinute: 2,
            databaseSizeBytes: 1_024,
            recentActivityBuckets: [1, 2, 3, 4],
            recentAgentWriteBuckets: [1, 1, 2, 2],
            recentWatcherWriteBuckets: [0, 1, 1, 2],
            recentEnrichmentBuckets: [1, 1, 2, 3],
            activityWindowMinutes: 60,
            bucketCount: 4,
            signalEligibleChunkCount: 0,
            signalCoverageIsAvailable: false
        )
        let provider = L0CountingCoverageProvider()
        let collector = StatsCollector(
            dbPath: "/nonexistent/brainbar-l0-state.db",
            daemonMonitor: DaemonHealthMonitor(targetPID: 0),
            dashboardStatsProvider: { hotStats },
            signalCoverageProvider: provider.fetch,
            signalCoverageStartDelay: 0,
            signalCoverageRefreshInterval: 300
        )
        defer { collector.stop() }
        collector.start()

        let loadDeadline = Date().addingTimeInterval(2)
        while !collector.stats.signalCoverageIsAvailable, Date() < loadDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(collector.stats.signalCoverageIsAvailable)
        XCTAssertEqual(provider.callCount, 1)

        var fillStarts: [String] = []
        var presentedLabels: [String: String] = [:]
        BrainBarCoverageLifecycleProbe.fillStarted = { fillStarts.append($0) }
        BrainBarCoverageLifecycleProbe.labelPresented = { presentedLabels[$0] = $1 }
        defer { BrainBarCoverageLifecycleProbe.reset() }

        let panelState = BrainBarDashboardPanelState()
        panelState.detailsExpanded = true
        panelState.signalCoverageExpanded = true
        let view = BrainBarDashboardPreview.make(
            collector: collector,
            panelState: panelState,
            disablesAnimations: false
        )
        let host = NSHostingController(rootView: view)
        host.view.frame = NSRect(x: 0, y: 0, width: 960, height: 1_200)

        try render(host, name: "initial-loaded")
        XCTAssertEqual(fillStarts.sorted(), ["FTS5", "Trigram", "Vector"])
        XCTAssertEqual(presentedLabels, ["Vector": "71%", "FTS5": "83%", "Trigram": "96%"])
        XCTAssertFalse(presentedLabels.values.contains("computing…"))

        for cycle in 1 ... 2 {
            panelState.detailsExpanded = false
            try render(host, name: "collapsed-\(cycle)")
            panelState.detailsExpanded = true
            try render(host, name: "reopened-\(cycle)")
        }

        XCTAssertEqual(provider.callCount, 1, "Presentation-only toggles must not refetch exact coverage.")
        XCTAssertEqual(fillStarts.sorted(), ["FTS5", "Trigram", "Vector"], "Each real bar may start its fill only once.")
        XCTAssertEqual(presentedLabels, ["Vector": "71%", "FTS5": "83%", "Trigram": "96%"])
        XCTAssertFalse(presentedLabels.values.contains("computing…"))
        print(
            "[brainbar-l0-state] provider_calls=\(provider.callCount) "
                + "fill_starts=\(fillStarts.sorted()) labels=\(presentedLabels) details_toggle_cycles=2"
        )
    }

    @MainActor
    private func render(_ host: NSHostingController<AnyView>, name: String) throws {
        host.view.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.55))
        host.view.layoutSubtreeIfNeeded()

        guard let bitmap = host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds) else {
            return XCTFail("Could not allocate bitmap for \(name).")
        }
        host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            return XCTFail("Could not encode \(name).")
        }
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["BRAINBAR_L0_STATE_RENDER_DIR"] ?? "/tmp/brainbar-l0-state")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(name).png")
        try png.write(to: url, options: .atomic)
        print("[brainbar-l0-state] \(name) \(url.path) \(png.count) bytes")
    }
}
