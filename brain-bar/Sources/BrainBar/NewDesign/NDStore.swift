import AppKit
import Combine
import SwiftUI

/// The new design's one source: it samples what the runtime already holds into `NDInputs` (no file read on the
/// menu-opening path, B6), and carries the actions every surface shares. One store per runtime, so the icon, the
/// dropdown and the window can never disagree (R1).
@MainActor final class NDStore: ObservableObject {
    @Published private(set) var inputs: NDInputs
    @Published private(set) var model: NDModel
    @Published var range: NDRange = .hour {
        didSet { if range != oldValue { refreshWaves() } }
    }
    @Published var page: NDPage = .overview

    let runtime: BrainBarRuntime?
    private let workspace: any BrainBarWorkspaceActing
    private var cancellables: Set<AnyCancellable> = []
    private var collectorCancellables: Set<AnyCancellable> = []
    private var modelCancellables: Set<AnyCancellable> = []
    private var lastWaveFetch: Date?
    private var lastHistoryRead: Date?
    private var waveGeneration = 0
    private var backupPaths: BrainBarBackupSources.Paths?
    /// The report the dropdown and the needs band copy: words and check diagnostics only (`BrainBarMenuReport`).
    private(set) var report = ""
    var onOpenWindow: ((NDPage) -> Void)?
    var onOpenSettings: (() -> Void)?
    var pasteboard: () -> NSPasteboard = { .general }
    /// Disconnecting Google Drive. Nothing in BrainLayer can do it today (no CLI command; deleting the token is a
    /// credentials door), so this stays nil and no Disconnect is drawn (L5).
    var disconnectDrive: (() -> Void)? = nil
    let isFixture: Bool

    private static var stores: [ObjectIdentifier: NDStore] = [:]

    /// The store for this runtime (created on first use).
    static func shared(for runtime: BrainBarRuntime, workspace: any BrainBarWorkspaceActing = BrainBarWorkspace()) -> NDStore {
        if let store = stores[ObjectIdentifier(runtime)] { return store }
        let store: NDStore
#if DEBUG
        if let fixture = BrainBarDesignFixture.requested {
            store = NDStore(fixture: BrainBarDesignFixture.inputs(fixture), page: fixture.page, runtime: runtime)
            stores[ObjectIdentifier(runtime)] = store
            return store
        }
#endif
        store = NDStore(runtime: runtime, workspace: workspace)
        stores[ObjectIdentifier(runtime)] = store
        return store
    }

    init(runtime: BrainBarRuntime, workspace: any BrainBarWorkspaceActing) {
        self.runtime = runtime
        self.workspace = workspace
        isFixture = false
        let first = NDStore.sample(runtime: runtime, previous: nil, now: Date())
        inputs = first
        model = NDModel.make(first)
        bind()
    }

    /// A store that holds one invented moment and reads nothing (headless renders, tests, capture mode).
    init(fixture: NDInputs, page: NDPage = .overview, runtime: BrainBarRuntime? = nil,
         workspace: any BrainBarWorkspaceActing = BrainBarRecordingWorkspace()) {
        self.runtime = runtime
        self.workspace = workspace
        isFixture = true
        inputs = fixture
        model = NDModel.make(fixture)
        self.page = page
        report = BrainBarMenuReport.text(presentation: model.presentation, verdict: fixture.capture.verdict, version: "fixture",
                                         now: fixture.capture.now)
    }

    func setInputsForTesting(_ inputs: NDInputs) {
        self.inputs = inputs
        model = NDModel.make(inputs)
    }

    // MARK: sampling

    private func bind() {
        guard let runtime else { return }
        runtime.$collector.receive(on: RunLoop.main).sink { [weak self] collector in self?.bindCollector(collector) }
            .store(in: &cancellables)
        runtime.$jobsModel.receive(on: RunLoop.main).sink { [weak self] model in
            guard let self else { return }
            self.modelCancellables.removeAll()
            model?.$publishedStatus.receive(on: RunLoop.main).sink { [weak self] _ in self?.refresh() }.store(in: &self.modelCancellables)
            model?.$launchdObservations.receive(on: RunLoop.main).sink { [weak self] _ in self?.refresh() }.store(in: &self.modelCancellables)
            self.refresh()
        }.store(in: &cancellables)
        runtime.driveAuth.$isReconnecting.receive(on: RunLoop.main).sink { [weak self] _ in self?.refresh() }.store(in: &cancellables)
        runtime.driveAuth.$status.receive(on: RunLoop.main).sink { [weak self] _ in self?.refresh() }.store(in: &cancellables)
    }

    private func bindCollector(_ collector: StatsCollector?) {
        collectorCancellables.removeAll()
        guard let collector else { return }
        collector.$stats.receive(on: RunLoop.main).sink { [weak self] _ in
            self?.refresh()
            self?.refreshWaves()
            self?.refreshBackupHistory()
        }.store(in: &collectorCancellables)
        collector.$agentActivity.receive(on: RunLoop.main).sink { [weak self] _ in self?.refresh() }.store(in: &collectorCancellables)
    }

    /// Re-sample the cached inputs (no IO).
    func refresh() {
        guard !isFixture, let runtime else { return }
        let next = Self.sample(runtime: runtime, previous: inputs, now: Date())
        inputs = next
        let model = NDModel.make(next)
        if model != self.model { self.model = model }
        report = BrainBarMenuReport.text(presentation: model.presentation, verdict: next.capture.verdict,
            version: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
            now: next.capture.now)
    }

    static func sample(runtime: BrainBarRuntime, previous: NDInputs?, now: Date) -> NDInputs {
        let collector = runtime.collector
        let stats = collector?.stats
        let model = runtime.jobsModel
        let capture = model?.menuCapture(stats: stats) ?? BrainBarMenuLiveCapture.capture(
            status: .checking, verdictAwakeAgeS: nil, wakeCatchUpPending: false, driveAuth: runtime.driveAuth.status,
            backups: .notRead, backupsNotice: nil, watcher: nil, watcherAwakeAgeS: nil, observations: nil,
            stats: stats, now: now, timeZone: .current)
        var i = NDInputs(capture: capture)
        i.menuWave = previous?.menuWave
        i.heroWaves = previous?.heroWaves ?? [:]
        i.backupHistory = previous?.backupHistory ?? .unread
        i.copyReportFailed = previous?.copyReportFailed ?? false
        i.driveReconnecting = runtime.driveAuth.isReconnecting
        if let model {
            i.extras = NDExtra.derive(snapshot: model.publishedStatus, watchSetting: model.config.launchdJobs[.watch])
        }
        if let collector, let stats, collector.lastDataFetchedAt != nil {
            i.memories = stats.chunkCount
            if stats.signalCoverageIsAvailable, stats.signalEligibleChunkCount > 0 {
                i.searchable = Double(stats.vectorIndexedChunkCount) / Double(stats.signalEligibleChunkCount)
            }
            i.savesWaiting = stats.pendingStoreQueueDepth
            i.oldestSaveQueuedAt = stats.pendingStoreOldestQueuedAt
        }
        if let model, case let .readable(file)? = model.watcherHealth {
            i.watcherHeartbeatAgeS = now.timeIntervalSince(file.updatedAt)
        }
        i.agents = agents(collector?.agentActivity, model?.dashboardObservabilityResult)
        if let model {
            i.jobs = jobs(model)
            i.nightlyLastRun = run(model, .maintenanceNightly)
            i.weeklyLastRun = run(model, .maintenanceWeekly)
            if case let .completed(at) = model.maintenanceEvidence.weeklyCompletion {
                if case let .ran(observed) = i.weeklyLastRun, observed >= at {} else { i.weeklyLastRun = .ran(at) }
            }
            i.canRevealLocalCopy = model.backupSchedules.contains { $0.localCopy != nil }
        }
        if let v = capture.verdict, let age = v.checks.first(where: { $0.name == "backup_local" })?.ageS, age.isFinite, age >= 0 {
            i.newestLocalCopyAt = v.evaluatedAt.addingTimeInterval(-age)
        }
        return i
    }

    /// Live sessions from the agent monitor; saves in 24 h from the observability emitters (the only per-agent count).
    static func agents(_ activity: AgentActivitySnapshot?, _ observability: ObservabilityReadResult?) -> NDAgents {
        var counts: [AgentFamily: Int]? = nil
        if case let .readable(document)? = observability, document.emitters.state == "measured", let rows = document.emitters.byEmitter {
            var c: [AgentFamily: Int] = [:]
            for family in AgentFamily.allCases {
                c[family] = rows.filter { $0.emitter.lowercased().hasPrefix(family.rawValue) }.reduce(0) { $0 + $1.countInWindow }
            }
            counts = c
        }
        let measured = activity?.isMeasured == true
        return NDAgents(agents: AgentFamily.allCases.map { family in
            NDAgents.Agent(family: family,
                           sessions: measured ? activity?.presences.first { $0.family == family }?.count ?? 0 : nil,
                           saves: counts?[family])
        })
    }

    static let jobOrder: [BrainLayerLaunchdJob] = [.watch, .hotlane, .drain, .walCheckpoint, .backupDaily, .jsonlBackup,
                                                    .maintenanceNightly, .maintenanceWeekly, .decay, .repairFTS]

    static func jobs(_ model: BrainBarSettingsViewModel) -> [NDJobInput]? {
        guard model.hasCompletedLaunchdSample else { return nil }
        return jobOrder.map { job in
            let o = model.launchdObservations[job]
            let loaded: Bool? = switch o?.loadState {
            case .running?, .loaded?: true
            case .unloaded?: false
            default: nil
            }
            return NDJobInput(job: job, enabled: model.configReadSucceeded ? model.config.launchdJobs[job]?.enabled : nil,
                              continuous: o?.isContinuous ?? false, lastRun: o?.lastRunAt, nextRun: o?.nextRunAt, loaded: loaded)
        }
    }

    static func run(_ model: BrainBarSettingsViewModel, _ job: BrainLayerLaunchdJob) -> NDRun {
        guard model.hasCompletedLaunchdSample, let o = model.launchdObservations[job] else { return .unread }
        return o.lastRunAt.map { .ran($0) } ?? .never
    }

    /// The waveforms: the dropdown's hour in 24 buckets and the hero's range in 60, read off the main thread at most every
    /// 15 s (and at once on a range switch).
    func refreshWaves(force: Bool = false) {
        guard !isFixture, let collector = runtime?.collector else { return }
        let now = Date()
        if !force, let last = lastWaveFetch, now.timeIntervalSince(last) < 15, inputs.heroWaves[range] != nil { return }
        lastWaveFetch = now
        waveGeneration += 1
        let generation = waveGeneration, range = range, provider = collector.bucketsProvider
        Task { [weak self] in
            async let menu = collector.fetchBuckets(windowMinutes: 60, bucketCount: 24, provider: provider)
            async let hero = collector.fetchBuckets(windowMinutes: range.minutes, bucketCount: 60, provider: provider)
            let (m, h) = await (menu, hero)
            guard let self, generation == self.waveGeneration else { return }
            var next = self.inputs
            next.menuWave = m.map { Self.wave($0, end: now) } ?? NDWave(saves: [], reads: [], readable: false)
            next.heroWaves[range] = h.map { Self.wave($0, end: now) } ?? NDWave(saves: [], reads: [], readable: false)
            self.inputs = next
            self.model = NDModel.make(next)
        }
    }

    static func wave(_ b: BrainDatabase.PipelineWindowBuckets, end: Date) -> NDWave {
        NDWave(saves: b.agentWriteBuckets, reads: b.watcherWriteBuckets,
               readable: b.agentWriteReadability.isReadable && b.watcherFlowReadability.isReadable, endsAt: end)
    }

    /// A week of backup attempts, read off the main thread at most every 5 minutes (L2).
    func refreshBackupHistory(force: Bool = false) {
        guard !isFixture, let runtime else { return }
        let now = Date()
        if !force, let last = lastHistoryRead, now.timeIntervalSince(last) < 300 { return }
        lastHistoryRead = now
        let databasePath = runtime.databasePath ?? BrainBarServer.defaultDBPath()
        let known = backupPaths
        Task { [weak self] in
            let (history, paths) = await Task.detached(priority: .utility) { () -> (NDBackupHistory, BrainBarBackupSources.Paths) in
                let paths = known ?? .live(databasePath: databasePath)
                return (NDBackupHistoryReader.read(databaseLog: paths.databaseLog, conversationLog: paths.archiveLog, now: now), paths)
            }.value
            guard let self else { return }
            self.backupPaths = paths
            var next = self.inputs
            next.backupHistory = history
            self.inputs = next
            self.model = NDModel.make(next)
        }
    }

    // MARK: actions (every control works; none is drawn without one)

    func perform(_ kind: BrainBarMenuPresentation.ActionKind) {
        switch kind {
        case .reconnectDrive, .connectDrive: reconnectDrive()
        case .copyReport: copyReport()
        case .openStorageSettings: openWindow(.backups)
        }
    }

    func reconnectDrive() {
        guard !isFixture, let drive = runtime?.driveAuth, !drive.isReconnecting else { return }
        Task { await drive.reconnect() }
    }

    func copyReport() {
        let board = pasteboard()
        board.clearContents()
        inputs.copyReportFailed = !board.setString(report, forType: .string)
    }

    func setJob(_ job: BrainLayerLaunchdJob, enabled: Bool) {
        guard !isFixture, let model = runtime?.jobsModel else { return }
        model.setJob(job, enabled: enabled)
        refresh()
    }

    func revealLocalCopy() {
        guard !isFixture, let model = runtime?.jobsModel,
              let row = model.backupSchedules.first(where: { $0.localCopy != nil }) else { return }
        model.revealBackup(row)
    }

    func revealSettingsFile() {
        guard !isFixture, let model = runtime?.jobsModel else { return }
        model.revealConfigFile()
    }

    func openWindow(_ page: NDPage) {
        self.page = page
        onOpenWindow?(page)
    }
}
