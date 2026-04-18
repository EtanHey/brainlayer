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
    private var pollTask: Task<Void, Never>?
    private var isRunning = false
    private var lastDataVersion: Int?

    init(
        dbPath: String,
        daemonMonitor: DaemonHealthMonitor,
        agentActivityMonitor: AgentActivityMonitor = AgentActivityMonitor()
    ) {
        self.database = BrainDatabase(path: dbPath)
        self.daemonMonitor = daemonMonitor
        self.agentActivityMonitor = agentActivityMonitor
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
        pollTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard !Task.isCancelled else { break }
                self.pollForChanges()
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
        if isRunning {
            removeDarwinObserver()
        }
        isRunning = false
        database.close()
    }

    func refresh(force: Bool = false) {
        let nextDaemon = daemonMonitor.sample()
        agentActivity = agentActivityMonitor.sample()

        do {
            let currentDataVersion = try database.dataVersion()
            if force || currentDataVersion != lastDataVersion {
                stats = try database.dashboardStats(
                    activityWindowMinutes: Self.defaultActivityWindowMinutes,
                    bucketCount: Self.defaultBucketCount
                )
                lastDataVersion = currentDataVersion
            }
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: stats)
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

    fileprivate func handleDatabaseMutationNotification() {
        refresh(force: false)
    }

    private func pollForChanges() {
        refresh(force: false)
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
