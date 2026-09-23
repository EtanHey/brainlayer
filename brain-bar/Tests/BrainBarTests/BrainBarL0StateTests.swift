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
        var labelEvents: [String] = []
        BrainBarCoverageLifecycleProbe.fillStarted = { fillStarts.append($0) }
        BrainBarCoverageLifecycleProbe.labelPresented = { labelEvents.append("\($0)=\($1)") }
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

        pump(host)
        XCTAssertEqual(fillStarts.sorted(), ["FTS5", "Trigram", "Vector"])
        XCTAssertEqual(labelEvents.sorted(), ["FTS5=83%", "Trigram=96%", "Vector=71%"])

        for cycle in 1 ... 2 {
            panelState.detailsExpanded = false
            pump(host)
            panelState.detailsExpanded = true
            pump(host)
            panelState.signalCoverageExpanded = false
            pump(host)
            panelState.signalCoverageExpanded = true
            pump(host)
        }

        XCTAssertEqual(provider.callCount, 1, "Presentation-only toggles must not refetch exact coverage.")
        XCTAssertEqual(fillStarts.sorted(), ["FTS5", "Trigram", "Vector"], "Each real bar may start its fill only once.")
        XCTAssertFalse(labelEvents.contains { $0.hasSuffix("=computing…") }, "No transient computing label: \(labelEvents)")
        XCTAssertEqual(labelEvents.sorted(), ["FTS5=83%", "Trigram=96%", "Vector=71%"],
                       "Presentation-only toggles must not re-present or change any label.")
        print(
            "[brainbar-l0-state] provider_calls=\(provider.callCount) "
                + "fill_starts=\(fillStarts.sorted()) label_events=\(labelEvents) details_toggle_cycles=2"
        )
        if let directory = ProcessInfo.processInfo.environment["BRAINBAR_L0_STATE_RENDER_DIR"] {
            BrainBarCoverageLifecycleProbe.reset()
            try renderWindowedFrames(host, panelState: panelState, directory: URL(fileURLWithPath: directory))
        }
    }

    @MainActor
    private func pump(_ host: NSHostingController<AnyView>) {
        host.view.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.55))
        host.view.layoutSubtreeIfNeeded()
    }

    @MainActor
    private func renderWindowedFrames(
        _ host: NSHostingController<AnyView>,
        panelState: BrainBarDashboardPanelState,
        directory: URL
    ) throws {
        let window = NSWindow(
            contentRect: NSRect(x: -6_000, y: -6_000, width: 960, height: 1_200),
            styleMask: [.titled], backing: .buffered, defer: false
        )
        window.contentViewController = host
        window.setFrame(NSRect(x: -6_000, y: -6_000, width: 960, height: 1_200), display: false)
        window.orderFront(nil)
        defer { window.orderOut(nil) }
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 1.0))
        try save(host.view, name: "window-initial", directory: directory)
        for cycle in 1 ... 2 {
            panelState.detailsExpanded = false
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.6))
            try save(host.view, name: "window-collapsed-\(cycle)", directory: directory)
            panelState.detailsExpanded = true
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.12))
            try save(host.view, name: "window-reopened-\(cycle)-early", directory: directory)
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.8))
            try save(host.view, name: "window-reopened-\(cycle)-settled", directory: directory)
        }
    }

    @MainActor
    private func save(_ view: NSView, name: String, directory: URL) throws {
        view.layoutSubtreeIfNeeded()
        guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
            return XCTFail("Could not allocate bitmap for \(name).")
        }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            return XCTFail("Could not encode \(name).")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("\(name).png")
        try png.write(to: url, options: .atomic)
        print("[brainbar-l0-state] \(name) \(url.path) \(png.count) bytes")
    }
}
