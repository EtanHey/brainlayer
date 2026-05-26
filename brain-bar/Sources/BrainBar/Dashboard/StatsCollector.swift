import Combine
import CoreFoundation
import Foundation

private func statsCollectorDarwinNotificationCallback(
    center: CFNotificationCenter?,
    observer: UnsafeMutableRawPointer?,
    name: CFNotificationName?,
    object: UnsafeRawPointer?,
    userInfo: CFDictionary?
) {
    guard let observer else { return }
    let collector = Unmanaged<StatsCollector>.fromOpaque(observer).takeUnretainedValue()
    Task { @MainActor in
        collector.handleDatabaseMutationNotification()
    }
}

@MainActor
final class StatsCollector: ObservableObject {
    static let defaultActivityWindowMinutes = 60
    static let defaultBucketCount = 12

    @Published private(set) var stats: DashboardStats
    @Published private(set) var daemon: DaemonHealthSnapshot?
    @Published private(set) var agentActivity: AgentActivitySnapshot
    @Published private(set) var state: PipelineState

    private let database: BrainDatabase
    private let daemonMonitor: DaemonHealthMonitor
    private let agentActivityMonitor: AgentActivityMonitor
    private let agentActivitySampleInterval: TimeInterval
    private let statsRefreshCoalesceInterval: TimeInterval
    private let brainBusEvents: BrainBusEventSource?
    private var brainBusTask: Task<Void, Never>?
    private var isRunning = false
    private var lastAgentActivitySampleAt: Date?
    private var lastNonForcedStatsRefreshAt: Date?
    private var pendingStoreQueueDepthSamples: [(date: Date, depth: Int)] = []

    init(
        dbPath: String,
        daemonMonitor: DaemonHealthMonitor,
        agentActivityMonitor: AgentActivityMonitor = AgentActivityMonitor(),
        agentActivitySampleInterval: TimeInterval = 5,
        statsRefreshCoalesceInterval: TimeInterval = 5,
        brainBusEvents: BrainBusEventSource? = nil,
        databaseOpenConfiguration: BrainDatabase.OpenConfiguration = BrainDatabase.OpenConfiguration()
    ) {
        self.database = BrainDatabase(path: dbPath, openConfiguration: databaseOpenConfiguration)
        self.daemonMonitor = daemonMonitor
        self.agentActivityMonitor = agentActivityMonitor
        self.agentActivitySampleInterval = agentActivitySampleInterval
        self.statsRefreshCoalesceInterval = statsRefreshCoalesceInterval
        self.brainBusEvents = brainBusEvents
        self.stats = DashboardStats(
            chunkCount: 0,
            enrichedChunkCount: 0,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 0,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 0,
            recentActivityBuckets: Array(repeating: 0, count: Self.defaultBucketCount),
            recentEnrichmentBuckets: Array(repeating: 0, count: Self.defaultBucketCount),
            activityWindowMinutes: Self.defaultActivityWindowMinutes,
            bucketCount: Self.defaultBucketCount
        )
        self.agentActivity = .empty
        self.state = .degraded
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        installDarwinObserver()
        refresh(force: true)
        if let brainBusEvents {
            let eventStream = brainBusEvents.events()
            brainBusTask = Task { [weak self] in
                for await event in eventStream {
                    guard !Task.isCancelled else { break }
                    await MainActor.run {
                        self?.handleBrainBusEvent(event)
                    }
                }
            }
        }
    }

    func stop() {
        brainBusTask?.cancel()
        brainBusTask = nil
        if isRunning {
            removeDarwinObserver()
        }
        isRunning = false
        database.close()
    }

    func refresh(force: Bool = false) {
        let nextDaemon = daemonMonitor.sample()
        let snapshotTime = Date()
        refreshAgentActivity(force: force, now: snapshotTime)

        if !force, shouldCoalesceStatsRefresh(now: snapshotTime) {
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: stats)
            return
        }

        do {
            database.reopenIfNeeded()
            let nextStats = try database.dashboardStats(
                activityWindowMinutes: Self.defaultActivityWindowMinutes,
                bucketCount: Self.defaultBucketCount
            )
            let queueFlushRate = recordPendingStoreQueueDepth(nextStats.pendingStoreQueueDepth, now: snapshotTime)
            stats = nextStats.withPendingStoreFlushRate(queueFlushRate)
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: stats)
            if !force {
                lastNonForcedStatsRefreshAt = snapshotTime
            }
        } catch {
            daemon = nextDaemon
            if force {
                stats = DashboardStats(
                    chunkCount: 0,
                    enrichedChunkCount: 0,
                    pendingEnrichmentCount: 0,
                    enrichmentPercent: 0,
                    enrichmentRatePerMinute: 0,
                    databaseSizeBytes: 0,
                    recentActivityBuckets: Array(repeating: 0, count: Self.defaultBucketCount),
                    recentEnrichmentBuckets: Array(repeating: 0, count: Self.defaultBucketCount),
                    activityWindowMinutes: Self.defaultActivityWindowMinutes,
                    bucketCount: Self.defaultBucketCount
                )
            }
            state = PipelineState.derive(daemon: nextDaemon, stats: stats)
        }
    }

    private func recordPendingStoreQueueDepth(_ depth: Int, now: Date) -> Double {
        pendingStoreQueueDepthSamples.append((date: now, depth: depth))
        let windowStart = now.addingTimeInterval(-60)
        pendingStoreQueueDepthSamples.removeAll { $0.date < windowStart }

        guard pendingStoreQueueDepthSamples.count > 1 else { return 0 }

        let drained = zip(pendingStoreQueueDepthSamples, pendingStoreQueueDepthSamples.dropFirst())
            .reduce(0) { total, pair in
                let decrease = max(0, pair.0.depth - pair.1.depth)
                return total + decrease
            }
        return Double(drained)
    }

    private func shouldCoalesceStatsRefresh(now: Date) -> Bool {
        guard let lastNonForcedStatsRefreshAt else { return false }
        return now.timeIntervalSince(lastNonForcedStatsRefreshAt) < statsRefreshCoalesceInterval
    }

    private func refreshAgentActivity(force: Bool, now: Date) {
        if !force, let lastAgentActivitySampleAt, now.timeIntervalSince(lastAgentActivitySampleAt) < agentActivitySampleInterval {
            return
        }
        agentActivity = agentActivityMonitor.sample()
        lastAgentActivitySampleAt = now
    }

    fileprivate func handleDatabaseMutationNotification() {
        refresh(force: false)
    }

    private func handleBrainBusEvent(_ event: BrainBusEvent) {
        switch event.type {
        case .healthTick:
            daemon = daemonMonitor.sample()
            refreshAgentActivity(force: false, now: Date())
            state = PipelineState.derive(daemon: daemon, stats: stats)
        case .queueDepth, .enrichStatus, .lastChunkID, .dbBusy:
            refresh(force: false)
        }
    }

    private func installDarwinObserver() {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterAddObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            statsCollectorDarwinNotificationCallback,
            BrainDatabase.dashboardDidChangeNotification as CFString,
            nil,
            .deliverImmediately
        )
    }

    private func removeDarwinObserver() {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterRemoveObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            CFNotificationName(BrainDatabase.dashboardDidChangeNotification as CFString),
            nil
        )
    }
}
