import Combine
import CoreFoundation
import Darwin
import Foundation

protocol WatcherProcessProbing: Sendable {
    func sample() -> WatcherProcessProbeResult
}

struct StaticWatcherProcessProbe: WatcherProcessProbing {
    let result: WatcherProcessProbeResult

    func sample() -> WatcherProcessProbeResult { result }
}

struct LaunchctlWatcherProcessProbe: WatcherProcessProbing {
    struct CommandResult: Sendable, Equatable {
        let terminationStatus: Int32
        let output: String
    }

    typealias CommandRunner = @Sendable ([String]) -> CommandResult

    private let label: String
    private let commandRunner: CommandRunner
    private let uidProvider: @Sendable () -> uid_t

    init(
        label: String = "com.brainlayer.watch",
        commandRunner: @escaping CommandRunner = { LaunchctlWatcherProcessProbe.run($0) },
        uidProvider: @escaping @Sendable () -> uid_t = getuid
    ) {
        self.label = label
        self.commandRunner = commandRunner
        self.uidProvider = uidProvider
    }

    func sample() -> WatcherProcessProbeResult {
        let target = "gui/\(uidProvider())/\(label)"
        let result = commandRunner(["/bin/launchctl", "print", target])
        if result.terminationStatus == 0 {
            if let pid = Self.parsePID(result.output) {
                return .running(pid: pid)
            }
            return .absent
        }

        let output = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
        if result.terminationStatus == 113 || output.localizedCaseInsensitiveContains("could not find service") {
            return .absent
        }
        return .failure(output.isEmpty ? "launchctl exited \(result.terminationStatus)" : output)
    }

    private static func parsePID(_ output: String) -> pid_t? {
        for line in output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.lowercased().hasPrefix("pid") else { continue }
            let tokens = trimmed.components(separatedBy: CharacterSet.decimalDigits.inverted)
                .filter { !$0.isEmpty }
            for token in tokens {
                guard let value = Int32(token), value > 0 else { continue }
                return pid_t(value)
            }
        }
        return nil
    }

    static func run(_ command: [String], timeout: TimeInterval = 1.0) -> CommandResult {
        guard let executable = command.first else {
            return CommandResult(terminationStatus: 1, output: "missing launchctl executable")
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = Array(command.dropFirst())
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
        } catch {
            return CommandResult(terminationStatus: 1, output: String(describing: error))
        }

        let deadline = Date().addingTimeInterval(max(0.01, timeout))
        while process.isRunning, Date() < deadline, !Task.isCancelled {
            Thread.sleep(forTimeInterval: 0.01)
        }

        if process.isRunning {
            process.terminate()
            let terminationDeadline = Date().addingTimeInterval(0.1)
            while process.isRunning, Date() < terminationDeadline {
                Thread.sleep(forTimeInterval: 0.01)
            }
            if process.isRunning {
                Darwin.kill(process.processIdentifier, SIGKILL)
            }
            process.waitUntilExit()
            _ = pipe.fileHandleForReading.readDataToEndOfFile()
            return CommandResult(
                terminationStatus: 124,
                output: Task.isCancelled ? "launchctl probe cancelled" : "launchctl timed out"
            )
        }

        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return CommandResult(
            terminationStatus: process.terminationStatus,
            output: String(data: data, encoding: .utf8) ?? ""
        )
    }
}

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
    typealias WindowedBucketsProvider = @Sendable (
        _ windowMinutes: Int,
        _ bucketCount: Int
    ) throws -> BrainDatabase.PipelineWindowBuckets

    static let defaultActivityWindowMinutes = 60
    static let defaultBucketCount = 12
    static let snapshotFreshnessThreshold: TimeInterval = 60

    @Published private(set) var stats: DashboardStats
    @Published private(set) var daemon: DaemonHealthSnapshot?
    @Published private(set) var agentActivity: AgentActivitySnapshot
    @Published private(set) var state: PipelineState
    @Published private(set) var isRefreshing = false
    @Published private(set) var isManualRefreshInProgress = false
    @Published private(set) var hasPendingStatsRefresh = false
    @Published private(set) var lastDataFetchedAt: Date?
    @Published private(set) var lastFetchError: String?
    @Published private(set) var snapshotFreshnessState: SnapshotFreshnessState
    @Published private(set) var heartbeat: DashboardHeartbeat
    /// REAL windowed buckets for the shared Live/3h/24h selector. `nil` means
    /// "no wider window selected yet" — the views fall back to the live `stats`
    /// buckets (the resting 1h view). When the selector picks 3h/24h, the view
    /// requests a windowed fetch; this publishes the actual DB data for that
    /// window so the charts re-render with genuine history, not a relabel.
    @Published private(set) var windowedBuckets: BrainDatabase.PipelineWindowBuckets?
    @Published private(set) var windowedBucketsWindowMinutes: Int?
    @Published private(set) var windowedBucketsError: String?
    @Published private(set) var isWindowedBucketsLoading = false
    private var windowedBucketsTask: Task<Void, Never>?
    private var windowedBucketsGeneration = 0

    private let dbPath: String
    private let databaseOpenConfiguration: BrainDatabase.OpenConfiguration
    private let daemonMonitor: DaemonHealthMonitor
    private let watcherProcessProbe: any WatcherProcessProbing
    private let windowedBucketsProvider: WindowedBucketsProvider
    private let agentActivityMonitor: AgentActivityMonitor
    private let agentActivitySampleInterval: TimeInterval
    private let statsRefreshCoalesceInterval: TimeInterval
    private let liveStatsRefreshDelay: TimeInterval
    private let autoRefreshInterval: TimeInterval
    private let freshnessTickerInterval: TimeInterval
    private let nowProvider: () -> Date
    private let brainBusEvents: BrainBusEventSource?
    private var brainBusTask: Task<Void, Never>?
    private var autoRefreshTask: Task<Void, Never>?
    private var freshnessTicker: Task<Void, Never>?
    private var pendingStatsRefreshTask: Task<Void, Never>?
    private var pendingStatsRefreshFireAt: Date?
    private var pendingStatsRefreshBypassesCoalescing = false
    private var dashboardRefreshTask: Task<Void, Never>?
    private var dashboardRefreshGeneration = 0
    private var watcherProcessRefreshTask: Task<Void, Never>?
    private var watcherProcessRefreshGeneration = 0
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
        watcherProcessProbe: any WatcherProcessProbing = StaticWatcherProcessProbe(
            result: .failure("watcher process probe not configured")
        ),
        agentActivityMonitor: AgentActivityMonitor = AgentActivityMonitor(),
        agentActivitySampleInterval: TimeInterval = 5,
        statsRefreshCoalesceInterval: TimeInterval = 5,
        liveStatsRefreshDelay: TimeInterval = 0.2,
        autoRefreshInterval: TimeInterval = 30,
        freshnessTickerInterval: TimeInterval = 1,
        nowProvider: @escaping () -> Date = Date.init,
        brainBusEvents: BrainBusEventSource? = nil,
        windowedBucketsProvider: WindowedBucketsProvider? = nil,
        databaseOpenConfiguration: BrainDatabase.OpenConfiguration = BrainDatabase.OpenConfiguration()
    ) {
        self.dbPath = dbPath
        self.databaseOpenConfiguration = databaseOpenConfiguration
        self.daemonMonitor = daemonMonitor
        self.watcherProcessProbe = watcherProcessProbe
        self.windowedBucketsProvider = windowedBucketsProvider ?? { windowMinutes, bucketCount in
            let backgroundDatabase = BrainDatabase(path: dbPath, openConfiguration: databaseOpenConfiguration)
            defer { backgroundDatabase.close() }
            backgroundDatabase.reopenIfNeeded()
            return try backgroundDatabase.pipelineWindowBuckets(
                activityWindowMinutes: windowMinutes,
                bucketCount: bucketCount
            )
        }
        self.agentActivityMonitor = agentActivityMonitor
        self.agentActivitySampleInterval = agentActivitySampleInterval
        self.statsRefreshCoalesceInterval = statsRefreshCoalesceInterval
        self.liveStatsRefreshDelay = liveStatsRefreshDelay
        self.autoRefreshInterval = autoRefreshInterval
        self.freshnessTickerInterval = freshnessTickerInterval
        self.nowProvider = nowProvider
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
        self.lastFetchError = nil
        self.snapshotFreshnessState = .loading
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
        startFreshnessTicker()
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
        freshnessTicker?.cancel()
        freshnessTicker = nil
        dashboardRefreshTask?.cancel()
        dashboardRefreshTask = nil
        watcherProcessRefreshTask?.cancel()
        watcherProcessRefreshTask = nil
        watcherProcessRefreshGeneration += 1
        windowedBucketsTask?.cancel()
        windowedBucketsTask = nil
        windowedBucketsGeneration += 1
        isWindowedBucketsLoading = false
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
        NSLog("[BrainBar] manual refresh requested at %@", ISO8601DateFormatter().string(from: nowProvider()))
        pendingStatsRefreshTask?.cancel()
        pendingStatsRefreshTask = nil
        pendingStatsRefreshFireAt = nil
        pendingStatsRefreshBypassesCoalescing = false
        hasPendingStatsRefresh = false
        requestRefresh(force: true, trigger: .manual)
    }

    /// Fetch REAL windowed buckets for the shared Live/3h/24h selector.
    ///
    /// The `.live` (1h) lens reads straight off the resting `stats` buckets, so
    /// it clears any wider-window fetch and returns immediately. The 3h/24h
    /// lenses trigger a genuine off-main DB re-fetch over the requested window
    /// (180 / 1440 minutes) and publish `windowedBuckets` so the charts re-render
    /// with actual history — fixing the prior bug where wider lenses only
    /// relabeled the live buckets.
    func selectTimeframe(windowMinutes: Int, isLive: Bool) {
        windowedBucketsTask?.cancel()
        windowedBucketsTask = nil
        windowedBucketsGeneration += 1
        let generation = windowedBucketsGeneration

        if isLive {
            windowedBuckets = nil
            windowedBucketsWindowMinutes = nil
            windowedBucketsError = nil
            isWindowedBucketsLoading = false
            return
        }

        let bucketCount = Self.defaultBucketCount
        let provider = windowedBucketsProvider
        windowedBucketsError = nil
        isWindowedBucketsLoading = true

        windowedBucketsTask = Task.detached(priority: .userInitiated) { [weak self] in
            let result: Result<BrainDatabase.PipelineWindowBuckets, Error> = Result {
                try provider(windowMinutes, bucketCount)
            }

            await self?.finishWindowedBucketsFetch(
                result: result,
                windowMinutes: windowMinutes,
                generation: generation
            )
        }
    }

    func requestRefresh(
        force: Bool = false,
        trigger: DashboardRefreshTrigger = .auto,
        bypassCoalescing: Bool = false
    ) {
        let nextDaemon = daemonMonitor.sample()
        let snapshotTime = nowProvider()
        refreshAgentActivity(force: force, now: snapshotTime)

        if !force, !bypassCoalescing, let coalescedDelay = coalescedStatsRefreshDelay(now: snapshotTime) {
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: stats)
            requestWatcherProcessRefresh()
            schedulePendingStatsRefresh(after: coalescedDelay)
            return
        }

        if dashboardRefreshTask != nil, trigger != .manual {
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: stats)
            requestWatcherProcessRefresh()
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
        let watcherProcessProbe = self.watcherProcessProbe
        let startUnix = snapshotTime.timeIntervalSince1970
        isRefreshing = true
        updateSnapshotFreshness()
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
            let nextWatcherProcess = watcherProcessProbe.sample()
            let result: Result<DashboardStats, Error> = Result {
                let backgroundDatabase = BrainDatabase(path: dbPath, openConfiguration: openConfiguration)
                defer { backgroundDatabase.close() }
                backgroundDatabase.reopenIfNeeded()
                return try backgroundDatabase.dashboardStats(
                    activityWindowMinutes: activityWindowMinutes,
                    bucketCount: bucketCount
                )
            }

            await self?.finishRequestedRefreshIfCurrent(
                result: result,
                daemon: nextDaemon,
                watcherProcess: nextWatcherProcess,
                snapshotTime: snapshotTime,
                startUnix: startUnix,
                force: force,
                trigger: trigger,
                generation: generation
            )
        }
    }

    private func finishWindowedBucketsFetch(
        result: Result<BrainDatabase.PipelineWindowBuckets, Error>,
        windowMinutes: Int,
        generation: Int
    ) {
        guard !isStopped, generation == windowedBucketsGeneration else { return }
        windowedBucketsTask = nil
        isWindowedBucketsLoading = false
        switch result {
        case .success(let buckets):
            windowedBuckets = buckets
            windowedBucketsWindowMinutes = windowMinutes
            windowedBucketsError = nil
        case .failure:
            windowedBucketsError = "Could not load \(DashboardMetricFormatter.windowLabel(minutes: windowMinutes)); showing Last 1h."
        }
    }

    private func requestWatcherProcessRefresh() {
        watcherProcessRefreshTask?.cancel()
        watcherProcessRefreshGeneration += 1
        let generation = watcherProcessRefreshGeneration
        let probe = watcherProcessProbe
        watcherProcessRefreshTask = Task.detached(priority: .utility) { [weak self] in
            let result = probe.sample()
            await self?.finishWatcherProcessRefresh(result, generation: generation)
        }
    }

    private func finishWatcherProcessRefresh(_ result: WatcherProcessProbeResult, generation: Int) {
        guard !isStopped, generation == watcherProcessRefreshGeneration else { return }
        watcherProcessRefreshTask = nil
        stats = stats.withWatcherProcessProbeResult(result)
        state = PipelineState.derive(daemon: daemon, stats: stats)
    }

    private func finishRequestedRefreshIfCurrent(
        result: Result<DashboardStats, Error>,
        daemon nextDaemon: DaemonHealthSnapshot?,
        watcherProcess: WatcherProcessProbeResult,
        snapshotTime: Date,
        startUnix: TimeInterval,
        force: Bool,
        trigger: DashboardRefreshTrigger,
        generation: Int
    ) {
        guard !isStopped, generation == dashboardRefreshGeneration else { return }
        finishRequestedRefresh(
            result: result,
            daemon: nextDaemon,
            watcherProcess: watcherProcess,
            snapshotTime: snapshotTime,
            startUnix: startUnix,
            force: force,
            trigger: trigger
        )
    }

    func refresh(force: Bool = false, trigger: DashboardRefreshTrigger) {
        requestRefresh(force: force, trigger: trigger)
    }

    private func finishRequestedRefresh(
        result: Result<DashboardStats, Error>,
        daemon nextDaemon: DaemonHealthSnapshot?,
        watcherProcess: WatcherProcessProbeResult,
        snapshotTime: Date,
        startUnix: TimeInterval,
        force: Bool,
        trigger: DashboardRefreshTrigger
    ) {
        let finishDaemon = daemonMonitor.sample() ?? nextDaemon
        switch result {
        case .success(let nextStats):
            let queueFlushRate = recordPendingStoreQueueDepth(nextStats.pendingStoreFlushQueueDepth, now: snapshotTime)
            stats = nextStats
                .withPendingStoreFlushRate(queueFlushRate)
                .withWatcherProcessProbeResult(watcherProcess)
            daemon = finishDaemon
            state = PipelineState.derive(daemon: finishDaemon, stats: stats)
            lastDataFetchedAt = nowProvider()
            lastFetchError = nil
            if !force {
                lastNonForcedStatsRefreshAt = snapshotTime
            }
        case .failure(let error):
            daemon = finishDaemon
            stats = stats.withWatcherProcessProbeResult(watcherProcess)
            lastFetchError = String(describing: error)
            state = PipelineState.derive(daemon: finishDaemon, stats: stats)
        }

        isRefreshing = false
        updateSnapshotFreshness()
        hasPendingStatsRefresh = pendingStatsRefreshTask != nil
        if trigger == .manual {
            isManualRefreshInProgress = false
        }
        dashboardRefreshTask = nil
        logDashboardRefresh(
            timestamp: snapshotTime,
            startUnix: startUnix,
            endUnix: nowProvider().timeIntervalSince1970,
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

    private func startFreshnessTicker() {
        freshnessTicker?.cancel()
        let interval = freshnessTickerInterval
        freshnessTicker = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(interval))
                } catch {
                    break
                }
                guard !Task.isCancelled, let self, !self.isStopped else { break }
                self.updateSnapshotFreshness()
            }
        }
    }

    private func updateSnapshotFreshness() {
        snapshotFreshnessState = SnapshotFreshnessState.derive(
            lastSuccessAt: lastDataFetchedAt,
            isRefreshing: isRefreshing,
            lastFetchError: lastFetchError,
            now: nowProvider(),
            snapshotFreshnessThreshold: Self.snapshotFreshnessThreshold
        )
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
            requestWatcherProcessRefresh()
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
        lastDataFetchedAt: Date?,
        lastFetchError: String? = nil,
        snapshotFreshnessState: SnapshotFreshnessState = .live(ageSeconds: 0)
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
        collector.lastFetchError = lastFetchError
        collector.snapshotFreshnessState = snapshotFreshnessState
        return collector
    }
}
#endif
