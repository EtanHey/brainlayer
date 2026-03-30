import Foundation
import SwiftUI

@MainActor
final class StatsCollector: ObservableObject {
    @Published private(set) var stats: DashboardStats
    @Published private(set) var daemon: DaemonHealthSnapshot?
    @Published private(set) var state: PipelineState

    private let database: BrainDatabase
    private let daemonMonitor: DaemonHealthMonitor

    init(
        dbPath: String,
        daemonMonitor: DaemonHealthMonitor
    ) {
        self.database = BrainDatabase(path: dbPath)
        self.daemonMonitor = daemonMonitor
        self.stats = DashboardStats(
            chunkCount: 0,
            enrichedChunkCount: 0,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 0,
            databaseSizeBytes: 0,
            recentActivityBuckets: Array(repeating: 0, count: 12)
        )
        self.state = .offline
    }

    func start() {
        refresh(force: true)
    }

    func stop() {
        database.close()
    }

    func refresh(force: Bool = false) {
        do {
            let nextStats = try database.dashboardStats(activityWindowMinutes: 30, bucketCount: 12)
            let nextDaemon = daemonMonitor.sample()
            stats = nextStats
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: nextStats)
        } catch {
            if force {
                daemon = nil
                state = .offline
            } else {
                NSLog("[StatsCollector] Refresh failed (non-forced): \(error)")
            }
        }
    }
}
