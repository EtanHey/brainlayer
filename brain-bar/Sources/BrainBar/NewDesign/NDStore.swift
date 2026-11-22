import AppKit
import Combine
import SwiftUI

/// The new design's one source: it samples what the runtime already holds into `NDInputs` (no file read on the
/// menu-opening path, B6), and carries the actions every surface shares. One store per runtime, so the icon, the
/// dropdown and the window can never disagree (R1).
@MainActor final class NDStore: ObservableObject {
    @Published private(set) var inputs: NDInputs
    @Published private(set) var model: NDModel
    /// What round 1 of the design draws (pipeline, headline items, dropdown rows, footer), from the same inputs.
    var r1: NDR1 { NDR1.make(inputs, model) }
    @Published var range: NDRange = .hour {
        didSet { if range != oldValue { refreshWaves(force: true, applyWhileHeld: true) } }
    }
    /// The one "current page": the tabs read it, the window root draws it, the window's height follows it.
    /// D1 (real screen, round 1): the window resize used to run from `$page`, which publishes in willSet, so the resize's
    /// layout pass drew the old page and the old selected segment at the new height. Followers now run here, after the
    /// value is stored, so the page, the segment and the height change together.
    @Published var page: NDPage = .overview {
        didSet {
            guard page != oldValue else { return }
            if let pageDidChange { pageDidChange(page) } else { shownPage = page }
        }
    }
    /// R2 item 3: the page the window root draws. It changes only when the window can hold it: after the window has grown
    /// to a taller page, before it shrinks to a shorter one, so a page is never drawn clipped.
    @Published var shownPage: NDPage = .overview
    /// Called after `page` holds its new value (the window's refit).
    var pageDidChange: (@MainActor (NDPage) -> Void)?
#if DEBUG
    /// The page the window root last drew (its `onAppear`), for the D1 test.
    var drawnPageForTesting: NDPage?
#endif

    let runtime: BrainBarRuntime?
    private let workspace: any BrainBarWorkspaceActing
    private var cancellables: Set<AnyCancellable> = []
    private var collectorCancellables: Set<AnyCancellable> = []
    private var modelCancellables: Set<AnyCancellable> = []
    private var lastWaveFetch: Date?
    var waveNow: () -> Date = Date.init
    var fetchWaveBuckets: (@Sendable (Int, Int) async -> BrainDatabase.PipelineWindowBuckets?)?
    private var lastHistoryRead: Date?
    private var waveGeneration = 0
    private var historyGeneration = 0
    private var waveTask: Task<Void, Never>?
    private typealias WaveBuckets = BrainDatabase.PipelineWindowBuckets?
    private var waveRead: (generation: Int, task: Task<(WaveBuckets, WaveBuckets), Never>)?
    private var historyTask: Task<Void, Never>?
    private var backupPaths: BrainBarBackupSources.Paths?
    /// The report the dropdown and the needs band copy: the shown words (runtime warnings included) and check
    /// diagnostics only (`BrainBarMenuReport`).
    private(set) var report = ""
    var onOpenWindow: ((NDPage) -> Void)?
    var onOpenSettings: (() -> Void)?
    weak var windowActivationPolicy: BrainBarWindowActivationPolicy?
    @Published private(set) var settingsBackupLine: String?
    var pasteboard: () -> NSPasteboard = { .general }
    /// Where the new design's own preferences (the Design choice in its Settings) are read and written. A capture
    /// fixture gives it a separate suite, so a capture never writes the installed app's preferences (F06).
    var preferences: UserDefaults = .standard
    /// L2's read, injectable so tests never touch the real logs.
    var readHistory: @Sendable (BrainBarBackupSources.Paths?, String, Date) -> (NDBackupHistory, BrainBarBackupSources.Paths) = { known, databasePath, now in
        let paths = known ?? .live(databasePath: databasePath)
        return (NDBackupHistoryReader.read(databaseLog: paths.databaseLog, conversationLog: paths.archiveLog, now: now), paths)
    }
    let isFixture: Bool

    // MARK: lifecycle (F04)

    /// The status item and dropdown draw the new design: subscriptions and the dropdown's waveform run.
    private(set) var iconActive = false
    /// The new window is open: the hero waveform and the backup history are read too.
    private(set) var windowActive = false
    private var diagnosticsActive = false
    private var bound = false
    /// Work that is running or subscribed now; zero when neither surface is shown (tests read it).
    var liveWorkCount: Int {
        cancellables.count + collectorCancellables.count + modelCancellables.count
            + (waveTask == nil ? 0 : 1) + (historyTask == nil ? 0 : 1)
    }

    func setIconActive(_ active: Bool) {
        guard active != iconActive else { return }
        iconActive = active
        updateLifecycle()
        if active { refresh(); refreshWaves(force: true) }
    }

    func setWindowActive(_ active: Bool) {
        guard active != windowActive else { return }
        windowActive = active
        if !active {
            historyTask?.cancel(); historyTask = nil
            historyGeneration += 1
        }
        updateLifecycle()
        if active { refresh(); refreshWaves(force: true); refreshBackupHistory(force: true) }
    }

    func setDiagnosticsActive(_ active: Bool) {
        diagnosticsActive = active
        updateLifecycle()
        if active { refresh() }
    }

    /// Closing all surfaces stops subscriptions and cancels waiters. A read already inside SQLite keeps its slot
    /// until it finishes, but its invalidated generation cannot publish.
    private func updateLifecycle() {
        let wanted = iconActive || windowActive || diagnosticsActive
        if wanted, !bound { bind() }
        if !wanted {
            cancellables.removeAll(); collectorCancellables.removeAll(); modelCancellables.removeAll()
            bound = false
            waveTask?.cancel(); waveTask = nil
            historyTask?.cancel(); historyTask = nil
            waveGeneration += 1
            historyGeneration += 1
        }
    }

    private final class WeakStore { weak var store: NDStore? }
    private static var stores: [ObjectIdentifier: WeakStore] = [:]

    /// The store for this runtime, shared by the status item and the window while either holds it. The registry holds
    /// it weakly: once both let go, the store and its runtime can go.
    static func shared(for runtime: BrainBarRuntime, workspace: any BrainBarWorkspaceActing = BrainBarWorkspace()) -> NDStore {
        stores = stores.filter { $0.value.store != nil }
        if let store = stores[ObjectIdentifier(runtime)]?.store, store.runtime === runtime { return store }
        let store: NDStore
#if DEBUG
        if let fixture = BrainBarDesignFixture.requested {
            store = NDStore(fixture: BrainBarDesignFixture.inputs(fixture), page: fixture.page, runtime: runtime)
            store.preferences = BrainBarDesignFixture.capturePreferences
        } else {
            store = NDStore(runtime: runtime, workspace: workspace)
        }
#else
        store = NDStore(runtime: runtime, workspace: workspace)
#endif
        let box = WeakStore()
        box.store = store
        stores[ObjectIdentifier(runtime)] = box
        return store
    }

    static func existing(for runtime: BrainBarRuntime) -> NDStore? {
        stores[ObjectIdentifier(runtime)]?.store.flatMap { $0.runtime === runtime ? $0 : nil }
    }

    init(runtime: BrainBarRuntime, workspace: any BrainBarWorkspaceActing) {
        self.runtime = runtime
        self.workspace = workspace
        isFixture = false
        let first = NDStore.sample(runtime: runtime, previous: nil, now: Date())
        inputs = first
        model = NDModel.make(first)
        report = Self.reportText(model, verdict: first.capture.verdict, extras: first.extras, now: first.capture.now)
    }

    /// A store that holds one invented moment and reads nothing (headless renders, tests, capture mode).
    init(fixture: NDInputs, page: NDPage = .overview, runtime: BrainBarRuntime? = nil,
         workspace: any BrainBarWorkspaceActing = NDStore.inertWorkspace) {
        self.runtime = runtime
        self.workspace = workspace
        isFixture = true
        inputs = fixture
        model = NDModel.make(fixture)
        self.page = page
        shownPage = page
        report = Self.reportText(model, verdict: fixture.capture.verdict, extras: fixture.extras, version: "fixture",
                                 now: fixture.capture.now)
    }

    /// F03: the report says what the surfaces show, runtime warnings included, through the existing scrubber.
    static func reportText(_ model: NDModel, verdict: BrainBarHealthVerdict?, extras: [NDExtra],
                           version: String = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
                           now: Date) -> String {
        BrainBarMenuReport.text(presentation: model.shownPresentation, verdict: verdict, version: version, now: now,
                                extraLines: extras.map { ($0.level == .fault ? "Needs you: " : "Notice: ") + $0.text })
    }

    /// A fixture store performs no action; DEBUG gives it the recording workspace (tests forbid the real one), Release
    /// (where no fixture runs) the real one.
    static var inertWorkspace: any BrainBarWorkspaceActing {
#if DEBUG
        BrainBarRecordingWorkspace()
#else
        BrainBarWorkspace()
#endif
    }

    func setInputsForTesting(_ inputs: NDInputs) {
        self.inputs = inputs
        model = NDModel.make(inputs)
        report = Self.reportText(model, verdict: inputs.capture.verdict, extras: inputs.extras, now: inputs.capture.now)
    }

    func setBackupPathsForTesting(_ paths: BrainBarBackupSources.Paths) { backupPaths = paths }

    // MARK: sampling

    private func bind() {
        guard let runtime, !isFixture else { return }
        bound = true
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
        guard !isFixture, let runtime, iconActive || windowActive || diagnosticsActive else { return }
        let next = Self.sample(runtime: runtime, previous: inputs, now: Date())
        inputs = next
        let model = NDModel.make(next)
        if model != self.model { self.model = model }
        report = Self.reportText(model, verdict: next.capture.verdict, extras: next.extras, now: next.capture.now)
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
            i.backupProgress = model.backupProgress
            i.nightlyLastRun = run(model, .maintenanceNightly)
            i.weeklyLastRun = run(model, .maintenanceWeekly)
            if case let .completed(at) = model.maintenanceEvidence.weeklyCompletion {
                if case let .ran(observed) = i.weeklyLastRun, observed >= at {} else { i.weeklyLastRun = .ran(at) }
            }
            i.canRevealLocalCopy = model.backupSchedules.contains { $0.localCopy != nil }
            // R1.5 change 8: the copies actually there, counted in the newest copy's folder (read only).
            if let newest = model.backupSchedules.compactMap(\.localCopy).first(where: { $0.lastPathComponent.contains(".db") }) {
                let names = (try? FileManager.default.contentsOfDirectory(atPath: newest.deletingLastPathComponent().path)) ?? []
                let kept = BackupLocalFiles.compressedSnapshots(in: names).count
                i.localCopiesKept = kept > 0 ? kept : nil
            }
        }
        if let v = capture.verdict, let age = v.checks.first(where: { $0.name == "backup_local" })?.ageS, age.isFinite, age >= 0 {
            i.newestLocalCopyAt = v.evaluatedAt.addingTimeInterval(-age)
        }
        return i
    }

    /// Live sessions from the agent monitor; memories in 24 h per family from the observability emitters (every chunk the
    /// producer attributes to the family by source, sender or source file; the only per-agent count).
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
                              continuous: o?.isContinuous ?? false, lastRun: o?.lastRunAt, nextRun: o?.nextRunAt, loaded: loaded,
                              lastOutcome: o.map(outcome) ?? .unknown,
                              failureStreak: o?.calendarHistoryDamaged == true ? nil : o?.calendarConsecutiveFailures,
                              scheduleInterval: o?.scheduleInterval, scheduleCalendar: o?.scheduleCalendar)
        }
    }

    /// F02: a run is a success only when a success is recorded: exit status 0, no failure streak, an undamaged
    /// history. A run in progress is running; a deferral, a missing status or a damaged history is unknown.
    static func outcome(_ o: BrainLayerLaunchdJobObservation) -> NDRunOutcome {
        if o.pid != nil || o.loadState == .running { return o.isContinuous ? .unknown : .running }
        if o.calendarHistoryDamaged == true { return .unknown }
        if let failures = o.calendarConsecutiveFailures, failures > 0 { return .failed }
        switch o.lastExitCode {
        case 0?: return .ok
        case BrainLayerMaintenanceExit.deferred?, nil: return .unknown
        default: return .failed
        }
    }

    static func run(_ model: BrainBarSettingsViewModel, _ job: BrainLayerLaunchdJob) -> NDRun {
        guard model.hasCompletedLaunchdSample, let o = model.launchdObservations[job] else { return .unread }
        return o.lastRunAt.map { .ran($0) } ?? .never
    }

    /// Periodic reads use a 15 s throttle. Range changes bypass it; if a read is already running, only the newest
    /// selection starts next. Periodic ticks cannot replace a pending user selection.
    func refreshWaves(force: Bool = false, applyWhileHeld: Bool = false) {
        guard iconActive || windowActive else { return }
        let fetch: @Sendable (Int, Int) async -> BrainDatabase.PipelineWindowBuckets?
        if let injected = fetchWaveBuckets { fetch = injected }
        else {
            guard !isFixture, let collector = runtime?.collector else { return }
            let provider = collector.bucketsProvider
            fetch = { minutes, count in await collector.fetchBuckets(windowMinutes: minutes, bucketCount: count, provider: provider) }
        }
        let now = waveNow()
        if !force, waveTask != nil { return }
        if !force, let last = lastWaveFetch, now.timeIntervalSince(last) < 15 { return }
        lastWaveFetch = now
        waveGeneration += 1
        waveTask?.cancel()
        let generation = waveGeneration, range = range
        let wantsHero = windowActive
        waveTask = Task { [weak self] in
            guard let self else { return }
            // Cancelling a waiter cannot stop SQLite already reading. Keep that read's slot until it finishes;
            // only the newest generation may start the next one, including after a window closes and reopens.
            if let previous = self.waveRead { _ = await previous.task.value }
            guard !Task.isCancelled, generation == self.waveGeneration else { return }
            let sampledAt = self.waveNow()
            self.lastWaveFetch = sampledAt
            let read = Task<(WaveBuckets, WaveBuckets), Never> {
                async let menu = fetch(60, 24)
                async let hero = wantsHero ? fetch(range.minutes, 60) : nil
                return await (menu, hero)
            }
            self.waveRead = (generation, read)
            let (m, h) = await read.value
            if self.waveRead?.generation == generation { self.waveRead = nil }
            guard !Task.isCancelled, generation == self.waveGeneration else { return }
            self.waveTask = nil
            var next = self.inputs
            next.menuWave = m.map { Self.wave($0, end: sampledAt) } ?? NDWave(saves: [], reads: [], readable: false)
            if wantsHero { next.heroWaves[range] = h.map { Self.wave($0, end: sampledAt) } ?? NDWave(saves: [], reads: [], readable: false) }
            self.acceptWaves(next, applyWhileHeld: applyWhileHeld)
        }
    }

#if DEBUG
    private(set) var fixtureTicks = 0

    func waitForWaveFetchForTesting() async { await waveTask?.value }

    /// Capture mode's live waveform (fix 5): one invented bucket per tick, handed over exactly as a real wave sample is.
    func tickFixture() {
        guard isFixture else { return }
        fixtureTicks += 1
        acceptWaves(BrainBarDesignFixture.ticked(inputs, step: fixtureTicks))
    }
#endif

    /// R1.8 item 2: while the pointer is inside the Activity chart, the chart's waves wait here (the newest of each range),
    /// so its bars, its readout's minute and its totals stay as they are; the pointer leaving lets them in. Held at the
    /// data, not in the view, so nothing reaches the chart to redraw it while the pointer rests.
    private(set) var chartHeld = false
    private var heldHeroWaves: [NDRange: NDWave] = [:]

    func holdChart(_ held: Bool) {
        guard held != chartHeld else { return }
        chartHeld = held
        guard !held, !heldHeroWaves.isEmpty else { return }
        var next = inputs
        for (range, wave) in heldHeroWaves { next.heroWaves[range] = wave.linked(after: inputs.heroWaves[range], minutes: range.minutes) }
        heldHeroWaves = [:]
        inputs = next
        model = NDModel.make(next)
    }

    /// The one way a wave sample reaches the surfaces, real or invented. Each wave is linked to the one on screen, so a
    /// one-bucket shift carries its outgoing bucket and starts its slide in the same transaction as the data (B1).
    private func acceptWaves(_ sample: NDInputs, applyWhileHeld: Bool = false) {
        if applyWhileHeld { heldHeroWaves.removeAll() }
        if chartHeld && !applyWhileHeld {
            // The dropdown's hour keeps flowing; the chart's ranges wait for the pointer to leave.
            for (range, wave) in sample.heroWaves { heldHeroWaves[range] = wave }
            var next = inputs
            next.menuWave = sample.menuWave?.linked(after: inputs.menuWave, minutes: 60)
            inputs = next
            return
        }
        var next = sample
        next.menuWave = sample.menuWave?.linked(after: inputs.menuWave, minutes: 60)
        for range in NDRange.allCases {
            next.heroWaves[range] = sample.heroWaves[range]?.linked(after: inputs.heroWaves[range], minutes: range.minutes)
        }
        inputs = next
        model = NDModel.make(next)
    }

    static func wave(_ b: BrainDatabase.PipelineWindowBuckets, end: Date) -> NDWave {
        NDWave(saves: b.agentWriteBuckets, reads: b.watcherWriteBuckets,
               readable: b.agentWriteReadability.isReadable && b.watcherFlowReadability.isReadable, endsAt: end)
    }

    /// A week of backup attempts, read off the main thread at most every 5 minutes (L2).
    func refreshBackupHistory(force: Bool = false) {
        guard !isFixture, windowActive, let runtime else { return }
        let now = Date()
        if !force, let last = lastHistoryRead, now.timeIntervalSince(last) < 300 { return }
        lastHistoryRead = now
        let databasePath = runtime.databasePath ?? BrainBarServer.defaultDBPath()
        let known = backupPaths, read = readHistory
        historyGeneration += 1
        let generation = historyGeneration
        historyTask?.cancel()
        historyTask = Task { [weak self] in
            let reader = Task.detached(priority: .utility) { read(known, databasePath, now) }
            let (history, paths) = await withTaskCancellationHandler { await reader.value } onCancel: { reader.cancel() }
            // A read that finishes after the window closed (or after a newer read began) publishes nothing.
            guard !Task.isCancelled, let self, generation == self.historyGeneration, self.windowActive else { return }
            self.historyTask = nil
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
        // A capture fixture starts no real reconnect; a press shows the waiting line it would show.
        if isFixture { inputs.driveReconnecting = true; return }
        guard let drive = runtime?.driveAuth, !drive.isReconnecting else { return }
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
        settingsBackupLine = model.settingsBackupLine
    }

    var settingsFileURL: URL? { isFixture ? inputs.settingsBackups?.file : runtime?.jobsModel?.configFileURL }
    func refreshSettingsFileSummary() {
        if isFixture {
            settingsBackupLine = inputs.settingsBackups?.line
            return
        }
        guard let model = runtime?.jobsModel else { return }
        model.refreshSettingsFileSummary()
        settingsBackupLine = model.settingsBackupLine
    }

    func showDiagnostics() { NDDiagnosticsWindowController.show(store: self, activationPolicy: windowActivationPolicy) }

    func openWindow(_ page: NDPage) {
        self.page = page
        onOpenWindow?(page)
    }
}

/// L57: one refit after a store transaction, including input-only Jobs and Backups changes.
/// Listening to objectWillChange also covers content that changes without changing the health headline.
@MainActor final class NDPageFitObserver {
    private var subscription: AnyCancellable?
    private var scheduled = false
    private let fit: @MainActor () -> Void

    init(store: NDStore, fit: @escaping @MainActor () -> Void) {
        self.fit = fit
        subscription = store.objectWillChange.sink { [weak self] in
            MainActor.assumeIsolated { self?.schedule() }
        }
    }

    func schedule() {
        guard !scheduled else { return }
        scheduled = true
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.scheduled = false
            self.fit()
        }
    }
}
