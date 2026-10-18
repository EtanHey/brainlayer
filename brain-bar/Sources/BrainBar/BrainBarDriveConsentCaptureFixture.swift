#if DEBUG && BRAINBAR_UI
import SwiftUI

/// The actual DRIVE capture dependencies, separate from window/process lifetime.
@MainActor
struct BrainBarDriveConsentCaptureFixture {
    let directory: URL
    let now: Date
    let settings: BrainBarSettingsViewModel
    let collector: StatsCollector
    let receipts = BrainBarOperationReceipts()
    let cadence = ObservabilityCadence.known(300)

    init(directory: URL, now: Date, initialEmbeddingProcess: BrainBarEmbeddingProcessState = .unmeasurable,
         didRead: @escaping @Sendable (String) -> Void = { _ in }) {
        self.directory = directory
        self.now = now
        let observability = BrainBarDashboardFixture.healthyObservabilityResult
        let store = BrainLayerConfigStore(
            configURL: directory.appendingPathComponent("unused.env"),
            loadDocumentOverride: { didRead("config"); return BrainLayerEnvDocument(config: .defaultConfig) },
            saveOverride: { _ in }
        )
        settings = BrainBarSettingsViewModel(
            workspace: BrainBarRecordingWorkspace(), store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            runtimeStatusProvider: StaticBrainLayerActiveRuntimeProvider(observation: .unknown("Fixture")),
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: .unmeasurable),
            initialEmbeddingProcess: initialEmbeddingProcess,
            refreshStatusOnLoad: false, now: { now },
            observabilityURL: directory.appendingPathComponent("observability.json"),
            initialObservabilityResult: observability,
            observabilityRead: { _ in didRead("observability"); return observability },
            watcherHealthURL: directory.appendingPathComponent("watcher-health.json"),
            watcherHealthRead: { _ in didRead("watcher"); return .readable(.init(updatedAt: now, pollCount: 72, maxOffsetLagBytes: 0)) },
            backupSources: BrainBarBackupSources(
                paths: .init(launchAgents: directory, databaseLog: directory.appendingPathComponent("db.log"),
                    archiveLog: directory.appendingPathComponent("archive.log"),
                    maintenanceLog: directory.appendingPathComponent("maintenance.log"),
                    snapshotDirectory: directory, archiveDirectory: directory),
                readFile: { _ in didRead("backup-file"); return nil },
                listDirectory: { _ in didRead("backup-list"); return [] }, isRegularFile: { _ in false },
                readStdoutTail: { _ in .unreadable }
            )
        )
        let stats = BrainBarDashboardFixture.stats
        let isolated = StatsCollector(
            dbPath: directory.appendingPathComponent("unused.db").path,
            daemonMonitor: DaemonHealthMonitor(targetPID: 0),
            watcherProcessProbe: StaticWatcherProcessProbe(result: stats.watcherProcessProbeResult ?? .absent),
            agentActivityMonitor: AgentActivityMonitor(snapshotProvider: { didRead("agents"); return nil }, executablePathResolver: { _ in nil }),
            nowProvider: { now }, observeDatabaseChanges: false,
            dashboardStatsProvider: { didRead("stats"); return stats },
            signalCoverageProvider: {
                didRead("coverage")
                guard let coverage = stats.signalCoverageSnapshot else { throw Unavailable.coverage }
                return coverage
            },
            signalCoverageStartDelay: 0,
            windowedBucketsProvider: { _, _ in didRead("window"); throw Unavailable.window }
        )
        collector = StatsCollector.fixture(
            stats: stats, daemon: BrainBarDashboardFixture.daemon,
            agentActivity: BrainBarDashboardFixture.agentActivity,
            state: BrainBarDashboardFixture.state, lastDataFetchedAt: now, collector: isolated
        )
    }

    private enum Unavailable: Error { case coverage, window }

    func dashboard(panelState: BrainBarDashboardPanelState) -> AnyView {
        BrainBarDashboardPreview.make(
            collector: collector, receiptStore: receipts,
            observabilityResult: BrainBarDashboardFixture.healthyObservabilityResult,
            now: now, panelState: panelState, observabilityCadence: cadence
        )
    }
}
#endif
