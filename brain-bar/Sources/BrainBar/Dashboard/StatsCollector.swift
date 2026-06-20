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
    @Published private(set) var isRefreshing = false
    @Published private(set) var isManualRefreshInProgress = false
    @Published private(set) var hasPendingStatsRefresh = false
    @Published private(set) var lastDataFetchedAt: Date?
    @Published private(set) var heartbeat: DashboardHeartbeat

    private let dbPath: String
    private let databaseOpenConfiguration: BrainDatabase.OpenConfiguration
    private let daemonMonitor: DaemonHealthMonitor
    private let agentActivityMonitor: AgentActivityMonitor
    private let agentActivitySampleInterval: TimeInterval
    private let statsRefreshCoalesceInterval: TimeInterval
    private let liveStatsRefreshDelay: TimeInterval
    private let autoRefreshInterval: TimeInterval
    private let brainBusEvents: BrainBusEventSource?
    private var brainBusTask: Task<Void, Never>?
    private var autoRefreshTask: Task<Void, Never>?
    private var pendingStatsRefreshTask: Task<Void, Never>?
    private var pendingStatsRefreshFireAt: Date?
    private var pendingStatsRefreshBypassesCoalescing = false
    private var dashboardRefreshTask: Task<Void, Never>?
    private var dashboardRefreshGeneration = 0
    private var isRunning = false
    private var isStopped = false
    private var lastAgentActivitySampleAt: Date?
    private var lastNonForcedStatsRefreshAt: Date?
    private var pendingStoreQueueDepthSamples: [(date: Date, depth: Int)] = []
    private var lastHeartbeatLogKey: String?
    private var lastHeartbeatLogAt: Date?

    init(
        dbPath: String,
        daemonMonitor: DaemonHealthMonitor,
        agentActivityMonitor: AgentActivityMonitor = AgentActivityMonitor(),
        agentActivitySampleInterval: TimeInterval = 5,
        statsRefreshCoalesceInterval: TimeInterval = 5,
        liveStatsRefreshDelay: TimeInterval = 0.2,
        autoRefreshInterval: TimeInterval = 30,
        brainBusEvents: BrainBusEventSource? = nil,
        databaseOpenConfiguration: BrainDatabase.OpenConfiguration = BrainDatabase.OpenConfiguration()
    ) {
        self.dbPath = dbPath
        self.databaseOpenConfiguration = databaseOpenConfiguration
        self.daemonMonitor = daemonMonitor
        self.agentActivityMonitor = agentActivityMonitor
        self.agentActivitySampleInterval = agentActivitySampleInterval
        self.statsRefreshCoalesceInterval = statsRefreshCoalesceInterval
        self.liveStatsRefreshDelay = liveStatsRefreshDelay
        self.autoRefreshInterval = autoRefreshInterval
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
        self.heartbeat = .empty
    }

    var isHeartbeatAheadOfStats: Bool {
        guard hasPendingStatsRefresh, let heartbeatUpdatedAt = heartbeat.updatedAt else { return false }
        guard let lastDataFetchedAt else { return true }
        return heartbeatUpdatedAt > lastDataFetchedAt
    }

    func start() {
        guard !isRunning else { return }
        resetRefreshTimingState()
        isStopped = false
        isRunning = true
        installDarwinObserver()
        requestRefresh(force: true)
        startAutoRefreshLoop()
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
        autoRefreshTask?.cancel()
        autoRefreshTask = nil
        dashboardRefreshTask?.cancel()
        dashboardRefreshTask = nil
        pendingStatsRefreshTask?.cancel()
        pendingStatsRefreshTask = nil
        pendingStatsRefreshFireAt = nil
        pendingStatsRefreshBypassesCoalescing = false
        hasPendingStatsRefresh = pendingStatsRefreshTask != nil
        isRefreshing = false
        isManualRefreshInProgress = false
        if isRunning {
            removeDarwinObserver()
        }
        isRunning = false
        isStopped = true
        resetRefreshTimingState()
    }

    func refresh(force: Bool = false) {
        requestRefresh(force: force, trigger: .auto)
    }

    func manualRefresh() {
        NSLog("[BrainBar] manual refresh requested at %@", ISO8601DateFormatter().string(from: Date()))
        pendingStatsRefreshTask?.cancel()
        pendingStatsRefreshTask = nil
        pendingStatsRefreshFireAt = nil
        pendingStatsRefreshBypassesCoalescing = false
        hasPendingStatsRefresh = false
        requestRefresh(force: true, trigger: .manual)
    }

    func requestRefresh(
        force: Bool = false,
        trigger: DashboardRefreshTrigger = .auto,
        bypassCoalescing: Bool = false
    ) {
        let nextDaemon = daemonMonitor.sample()
        let snapshotTime = Date()
        refreshAgentActivity(force: force, now: snapshotTime)

        if !force, !bypassCoalescing, let coalescedDelay = coalescedStatsRefreshDelay(now: snapshotTime) {
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: stats)
            schedulePendingStatsRefresh(after: coalescedDelay)
            return
        }

        if dashboardRefreshTask != nil, trigger != .manual {
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: stats)
            if !force {
                schedulePendingStatsRefresh(after: statsRefreshCoalesceInterval)
            }
            return
        }

        pendingStatsRefreshTask?.cancel()
        pendingStatsRefreshTask = nil
        pendingStatsRefreshFireAt = nil
        pendingStatsRefreshBypassesCoalescing = false
        hasPendingStatsRefresh = false
        dashboardRefreshTask?.cancel()
        dashboardRefreshGeneration += 1
        let generation = dashboardRefreshGeneration
        let dbPath = self.dbPath
        let openConfiguration = self.databaseOpenConfiguration
        let activityWindowMinutes = Self.defaultActivityWindowMinutes
        let bucketCount = Self.defaultBucketCount
        let startStats = stats
        let startUnix = snapshotTime.timeIntervalSince1970
        isRefreshing = true
        if trigger == .manual {
            isManualRefreshInProgress = true
        }
        daemon = nextDaemon
        state = PipelineState.derive(daemon: nextDaemon, stats: stats)
        logDashboardRefresh(
            timestamp: snapshotTime,
            startUnix: startUnix,
            endUnix: nil,
            rows: startStats.chunkCount,
            writes5m: startStats.recentWriteFiveMinuteCount,
            enrich5m: startStats.recentEnrichmentFiveMinuteCount,
            trigger: trigger
        )

        dashboardRefreshTask = Task.detached(priority: .utility) { [weak self] in
            let result: Result<DashboardStats, Error> = Result {
                let backgroundDatabase = BrainDatabase(path: dbPath, openConfiguration: openConfiguration)
                defer { backgroundDatabase.close() }
                backgroundDatabase.reopenIfNeeded()
                return try backgroundDatabase.dashboardStats(
                    activityWindowMinutes: activityWindowMinutes,
                    bucketCount: bucketCount
                )
            }

            await MainActor.run {
                guard let self, !self.isStopped, generation == self.dashboardRefreshGeneration else { return }
                self.finishRequestedRefresh(
                    result: result,
                    daemon: nextDaemon,
                    snapshotTime: snapshotTime,
                    startUnix: startUnix,
                    force: force,
                    trigger: trigger
                )
            }
        }
    }

    func refresh(force: Bool = false, trigger: DashboardRefreshTrigger) {
        requestRefresh(force: force, trigger: trigger)
    }

    private func finishRequestedRefresh(
        result: Result<DashboardStats, Error>,
        daemon nextDaemon: DaemonHealthSnapshot?,
        snapshotTime: Date,
        startUnix: TimeInterval,
        force: Bool,
        trigger: DashboardRefreshTrigger
    ) {
        let finishDaemon = daemonMonitor.sample() ?? nextDaemon
        switch result {
        case .success(let nextStats):
            let queueFlushRate = recordPendingStoreQueueDepth(nextStats.pendingStoreQueueDepth, now: snapshotTime)
            stats = nextStats.withPendingStoreFlushRate(queueFlushRate)
            daemon = finishDaemon
            state = PipelineState.derive(daemon: finishDaemon, stats: stats)
            lastDataFetchedAt = Date()
            if !force {
                lastNonForcedStatsRefreshAt = snapshotTime
            }
        case .failure:
            daemon = finishDaemon
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
                    recentWriteFiveMinuteCount: 0,
                    recentEnrichmentFiveMinuteCount: 0,
                    activityWindowMinutes: Self.defaultActivityWindowMinutes,
                    bucketCount: Self.defaultBucketCount
                )
            }
            state = PipelineState.derive(daemon: finishDaemon, stats: stats)
        }

        isRefreshing = false
        hasPendingStatsRefresh = pendingStatsRefreshTask != nil
        if trigger == .manual {
            isManualRefreshInProgress = false
        }
        dashboardRefreshTask = nil
        logDashboardRefresh(
            timestamp: snapshotTime,
            startUnix: startUnix,
            endUnix: Date().timeIntervalSince1970,
            rows: stats.chunkCount,
            writes5m: stats.recentWriteFiveMinuteCount,
            enrich5m: stats.recentEnrichmentFiveMinuteCount,
            trigger: trigger
        )
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

    private func coalescedStatsRefreshDelay(now: Date) -> TimeInterval? {
        guard let lastNonForcedStatsRefreshAt else { return nil }
        let elapsed = now.timeIntervalSince(lastNonForcedStatsRefreshAt)
        guard elapsed < statsRefreshCoalesceInterval else { return nil }
        return statsRefreshCoalesceInterval - elapsed
    }

    private func schedulePendingStatsRefresh(after delay: TimeInterval, bypassCoalescing: Bool = false) {
        let fireAt = Date().addingTimeInterval(delay)
        hasPendingStatsRefresh = true
        if pendingStatsRefreshTask != nil {
            let shouldReplacePendingRefresh =
                (bypassCoalescing && !pendingStatsRefreshBypassesCoalescing) ||
                pendingStatsRefreshFireAt.map { fireAt < $0 } ?? false
            guard shouldReplacePendingRefresh else { return }
            pendingStatsRefreshTask?.cancel()
            pendingStatsRefreshTask = nil
        }

        pendingStatsRefreshFireAt = fireAt
        pendingStatsRefreshBypassesCoalescing = bypassCoalescing
        pendingStatsRefreshTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(delay))
            } catch {
                return
            }

            guard !Task.isCancelled else { return }

            await MainActor.run {
                guard let self, !self.isStopped else { return }
                self.pendingStatsRefreshTask = nil
                self.pendingStatsRefreshFireAt = nil
                let shouldBypassCoalescing = self.pendingStatsRefreshBypassesCoalescing
                self.pendingStatsRefreshBypassesCoalescing = false
                self.hasPendingStatsRefresh = false
                self.requestRefresh(force: false, trigger: .auto, bypassCoalescing: shouldBypassCoalescing)
            }
        }
    }

    private func scheduleLiveStatsRefresh() {
        schedulePendingStatsRefresh(
            after: min(statsRefreshCoalesceInterval, liveStatsRefreshDelay),
            bypassCoalescing: true
        )
    }

    private func startAutoRefreshLoop() {
        autoRefreshTask?.cancel()
        let interval = autoRefreshInterval
        autoRefreshTask = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(interval))
                } catch {
                    break
                }

                guard !Task.isCancelled else { break }
                await MainActor.run {
                    guard let self, !self.isStopped else { return }
                    self.pendingStatsRefreshTask?.cancel()
                    self.pendingStatsRefreshTask = nil
                    self.pendingStatsRefreshFireAt = nil
                    self.pendingStatsRefreshBypassesCoalescing = false
                    self.hasPendingStatsRefresh = false
                    self.requestRefresh(force: false, trigger: .auto)
                }
            }
        }
    }

    private func refreshAgentActivity(force: Bool, now: Date) {
        if !force, let lastAgentActivitySampleAt, now.timeIntervalSince(lastAgentActivitySampleAt) < agentActivitySampleInterval {
            return
        }
        agentActivity = agentActivityMonitor.sample()
        lastAgentActivitySampleAt = now
    }

    private func resetRefreshTimingState() {
        lastAgentActivitySampleAt = nil
        lastNonForcedStatsRefreshAt = nil
    }

    fileprivate func handleDatabaseMutationNotification() {
        recordHeartbeat(
            event: nil,
            trigger: "darwin_db_notification",
            timestamp: Date()
        )
        schedulePendingStatsRefresh(after: max(statsRefreshCoalesceInterval, 1.0))
    }

    private func handleBrainBusEvent(_ event: BrainBusEvent) {
        recordHeartbeat(
            event: event,
            trigger: "brain_bus",
            timestamp: event.generatedAt
        )

        switch event.type {
        case .healthTick:
            daemon = daemonMonitor.sample()
            refreshAgentActivity(force: false, now: Date())
            state = PipelineState.derive(daemon: daemon, stats: stats)
        case .queueDepth, .enrichStatus, .lastChunkID, .dbBusy:
            scheduleLiveStatsRefresh()
        }
    }

    private func recordHeartbeat(
        event: BrainBusEvent?,
        trigger: String,
        timestamp: Date
    ) {
        heartbeat = heartbeat.recording(event: event, at: timestamp)
        let eventType = event?.type.rawValue ?? "database_changed"
        let logKey = "\(trigger):\(eventType)"
        if lastHeartbeatLogKey == logKey,
           let lastHeartbeatLogAt,
           timestamp.timeIntervalSince(lastHeartbeatLogAt) < 0.5 {
            return
        }
        lastHeartbeatLogKey = logKey
        lastHeartbeatLogAt = timestamp
        NSLog(
            "[BrainBar] heartbeat: %@ trigger=%@ type=%@ sequence=%d",
            ISO8601DateFormatter().string(from: timestamp),
            trigger,
            eventType,
            event?.sequence ?? 0
        )
    }

    private func logDashboardRefresh(
        timestamp: Date,
        startUnix: TimeInterval,
        endUnix: TimeInterval?,
        rows: Int,
        writes5m: Int,
        enrich5m: Int,
        trigger: DashboardRefreshTrigger
    ) {
        let endText = endUnix.map { String(format: "%.3f", $0) } ?? "ongoing"
        NSLog(
            "[BrainBar] dashboard refresh: %@ start=%.3f end=%@ rows=%d writes_5m=%d enrich_5m=%d trigger=%@",
            ISO8601DateFormatter().string(from: timestamp),
            startUnix,
            endText,
            rows,
            writes5m,
            enrich5m,
            trigger.rawValue
        )
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

enum DashboardRefreshTrigger: String {
    case auto
    case manual
    case tabSwitch = "tab_switch"
}

#if DEBUG
extension StatsCollector {
    /// Builds a `StatsCollector` pre-loaded with fixed snapshot state and NO live
    /// wiring — `start()` is never called, so no database is opened, no Darwin
    /// observer is installed, and no refresh timers run. Used by the deterministic
    /// dashboard render seam (`BrainBarDashboardPreview` + the snapshot tests) so
    /// any agent can render the real dashboard to a PNG without live collectors.
    ///
    /// This lives in the same file as `StatsCollector` because the published
    /// properties are `private(set)`; only same-file code may assign them.
    @MainActor
    static func fixture(
        stats: DashboardStats,
        daemon: DaemonHealthSnapshot?,
        agentActivity: AgentActivitySnapshot,
        state: PipelineState,
        heartbeat: DashboardHeartbeat = .empty,
        lastDataFetchedAt: Date?
    ) -> StatsCollector {
        // targetPID 0 makes the monitor's sample() return nil; it is never used
        // because start()/requestRefresh() are not called on a fixture.
        let collector = StatsCollector(
            dbPath: "/nonexistent/brainbar-fixture.db",
            daemonMonitor: DaemonHealthMonitor(targetPID: 0)
        )
        collector.stats = stats
        collector.daemon = daemon
        collector.agentActivity = agentActivity
        collector.state = state
        collector.heartbeat = heartbeat
        collector.lastDataFetchedAt = lastDataFetchedAt
        return collector
    }
}
#endif
