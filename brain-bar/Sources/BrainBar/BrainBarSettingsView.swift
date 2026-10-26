import AppKit
import Combine
import SwiftUI

@MainActor
final class BrainBarSettingsViewModel: ObservableObject {
    @Published var config: BrainLayerConfig
    @Published var pendingPlainAPIKey = ""
    @Published var onePasswordReference: String
    @Published var errorMessage: String?
    /// Show log's sentence when the job has no log yet (Codex #1062 r1 B1); nil after a log opens.
    @Published var jobAlertLogNote: BrainBarJobAlertLogNote?
    /// The Backups page's Technical details disclosure (E1); closed by default.
    @Published var backupDetailsExpanded = false
    @Published private(set) var copiedTechnicalDetailID: String?
    @Published var isRefreshingLaunchdStatus = false
    /// A refresh retains its last completed sample; only a first read presents Checking.
    @Published private(set) var hasCompletedLaunchdSample: Bool
    @Published private(set) var hasCompletedObservabilityRead: Bool
    @Published private(set) var isRefreshingObservabilityStatus = false
    @Published private(set) var activeRuntimeObservation: BrainLayerActiveRuntimeObservation
    @Published private(set) var lastSaveReceipt: BrainLayerSettingsSaveReceipt?
    @Published private(set) var observabilityResult: ObservabilityReadResult
    @Published private(set) var observabilityNow: Date
    let observabilityCadence: ObservabilityCadence
    @Published private(set) var launchdObservations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]
    @Published private(set) var configReadSucceeded = true
    @Published private(set) var embeddingProcess: BrainBarEmbeddingProcessState
    /// watcher-health.json beside the resolved DB: the same file the Dashboard reads (#966).
    @Published private(set) var watcherHealth: WatcherHealthFileRead?
    /// The health check's badge as read (job alerts not yet applied); nil until the first read.
    @Published private(set) var healthBadge: BadgeStatePresentation?
    /// The last VERIFIED live job-alert map (R2-3): an unknown read never replaces it.
    @Published private(set) var liveJobAlerts: BrainBarJobAlerts?
    /// How the latest live job-alert read went.
    @Published private(set) var liveJobAlertsState: BrainBarStatusInputs.JobAlertsState = .unread
    /// S5/B5: the ONE published status verdict. Every surface projects this; none re-derives it.
    @Published private(set) var publishedStatus = BrainBarStatusSnapshot.checking
    /// L3: when the app sampler first found observability.json unreadable in the current episode.
    private var observabilityUnreadableSince: Date?
    private let healthBadgeURL: URL?
    private let jobAlertsURL: URL?
    private let badgeReadHistory = BadgeReadHistory()
    let statusSamplingPeriod: TimeInterval
    private weak var driveAuthSource: BrainBarDriveAuthModel?
    private var driveAuthCancellable: AnyCancellable?

    /// The one watcher-health truth, from this view's launchd observation plus the health file.
    var watcherStatus: WatcherHealthStatus {
        WatcherHealthStatus.derive(
            launchd: WatcherLaunchdEvidence(
                setting: config.launchdJobs[.watch],
                loadState: config.launchdJobs[.watch]?.loadState
            ),
            file: watcherHealth,
            now: now()
        )
    }
    /// Each backup's schedule (installed LaunchAgent), last run (its own log), next run and latest
    /// local copy (#968).
    @Published private(set) var backupSchedules: [BrainBarBackupScheduleRow]
    /// The Maintenance card's completion history and skip reasons, read with the backup schedules.
    @Published private(set) var maintenanceEvidence: BrainLayerMaintenanceEvidence

    var footerPresentation: BrainBarSettingsFooterPresentation {
        BrainBarSettingsFooterPresentation(
            config: configReadSucceeded ? config : nil,
            watcher: watcherStatus,
            now: now(),
            isChecking: !hasCompletedLaunchdSample && config.launchdJobs[.watch]?.enabled != false
        )
    }

    var modelResidencyPresentation: BrainBarModelResidencyPresentation {
        BrainBarModelResidencyPresentation(
            modelName: BrainBarEmbeddingModel.configuredName,
            process: embeddingProcess
        )
    }

    private let store: BrainLayerConfigStore
    private let launchdStatusProvider: any BrainLayerLaunchdStatusSampling
    private let runtimeStatusProvider: any BrainLayerActiveRuntimeSampling
    private let embeddingResidencyProbe: any BrainBarEmbeddingResidencySampling
    private let now: @Sendable () -> Date
    private let observabilityURL: URL?
    private let observabilityRead: @Sendable (URL) async -> ObservabilityReadResult
    private let watcherHealthURL: URL?
    private let watcherHealthRead: @Sendable (URL) -> WatcherHealthFileRead
    private var watcherHealthGeneration: UInt64 = 0
    private let backupSources: BrainBarBackupSources?
    private let workspace: any BrainBarWorkspaceActing
    private var backupScheduleGeneration: UInt64 = 0
    private let confirmAPIKeyOverwrite: (() -> Bool)?
    private var previousConfigForLastSaveReceipt: BrainLayerConfig?
    private var observabilityTask: Task<Void, Never>?
    private var embeddingSampleGeneration: UInt64 = 0
    private var launchdSampleGeneration: UInt64 = 0
    private var copyFeedbackGeneration: UInt64 = 0
    private let copyFeedbackScheduler: @MainActor (TimeInterval, @escaping @MainActor @Sendable () -> Void) -> Void

    init(
        workspace: any BrainBarWorkspaceActing,
        store: BrainLayerConfigStore = BrainLayerConfigStore(),
        launchdStatusProvider: any BrainLayerLaunchdStatusSampling = BrainLayerLaunchdStatusProvider(),
        runtimeStatusProvider: any BrainLayerActiveRuntimeSampling = UnknownBrainLayerActiveRuntimeProvider(),
        embeddingResidencyProbe: any BrainBarEmbeddingResidencySampling = HotlaneEmbeddingResidencyProbe(),
        initialEmbeddingProcess: BrainBarEmbeddingProcessState = .unmeasurable,
        initialLaunchdStates: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] = [:],
        initialLaunchdObservations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] = [:],
        refreshStatusOnLoad: Bool = true,
        now: @escaping @Sendable () -> Date = Date.init,
        observabilityURL: URL? = nil,
        initialObservabilityResult: ObservabilityReadResult? = nil,
        observabilityCadence: ObservabilityCadence = ObservabilityReader.installedHealthCheckCadence,
        confirmAPIKeyOverwrite: (() -> Bool)? = nil,
        observabilityRead: @escaping @Sendable (URL) async -> ObservabilityReadResult = { url in
            await Task.detached { ObservabilityReader.readReconciled(url: url) }.value
        },
        watcherHealthURL: URL? = nil,
        initialWatcherHealth: WatcherHealthFileRead? = nil,
        watcherHealthRead: @escaping @Sendable (URL) -> WatcherHealthFileRead = { WatcherHealthReader.read(url: $0) },
        backupSources: BrainBarBackupSources? = nil,
        initialBackupSchedules: [BrainBarBackupScheduleRow] = [],
        initialMaintenanceEvidence: BrainLayerMaintenanceEvidence = .unread,
        copyFeedbackScheduler: @escaping @MainActor (TimeInterval, @escaping @MainActor @Sendable () -> Void) -> Void = { delay, action in
            DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: action)
        },
        healthBadgeURL: URL? = nil,
        jobAlertsURL: URL? = nil,
        statusSamplingPeriod: TimeInterval = 30
    ) {
        self.healthBadgeURL = healthBadgeURL
        self.jobAlertsURL = jobAlertsURL
        self.statusSamplingPeriod = statusSamplingPeriod
        self.store = store
        self.launchdStatusProvider = launchdStatusProvider
        self.runtimeStatusProvider = runtimeStatusProvider
        self.embeddingResidencyProbe = embeddingResidencyProbe
        embeddingProcess = initialEmbeddingProcess
        self.now = now
        self.observabilityURL = observabilityURL
        self.observabilityRead = observabilityRead
        self.observabilityCadence = observabilityCadence
        self.confirmAPIKeyOverwrite = confirmAPIKeyOverwrite
        self.watcherHealthURL = watcherHealthURL
        self.watcherHealthRead = watcherHealthRead
        watcherHealth = initialWatcherHealth
        self.backupSources = backupSources
        self.workspace = workspace
        self.copyFeedbackScheduler = copyFeedbackScheduler
        backupSchedules = initialBackupSchedules
        maintenanceEvidence = initialMaintenanceEvidence
        observabilityResult = initialObservabilityResult ?? .unreadable("Backup status unavailable.")
        observabilityNow = now()
        hasCompletedObservabilityRead = initialObservabilityResult != nil || observabilityURL == nil
        hasCompletedLaunchdSample = !initialLaunchdStates.isEmpty || !initialLaunchdObservations.isEmpty
        launchdObservations = initialLaunchdObservations.isEmpty
            ? initialLaunchdStates.mapValues(BrainLayerLaunchdJobObservation.stateOnly)
            : initialLaunchdObservations
        activeRuntimeObservation = runtimeStatusProvider.sample()
        do {
            let document = try store.loadDocument()
            config = document.config
            onePasswordReference = document.config.googleAPIKey.opReference
        } catch {
            configReadSucceeded = false
            config = .defaultConfig
            onePasswordReference = BrainLayerConfig.defaultConfig.googleAPIKey.opReference
            errorMessage = error.localizedDescription
        }
        applyLaunchdStates(
            initialLaunchdObservations.isEmpty
                ? initialLaunchdStates
                : initialLaunchdObservations.mapValues(\.loadState)
        )
        refreshObservabilityStatus()
        if refreshStatusOnLoad {
            refreshLaunchdStatus()
        }
        republishStatus()
    }

    /// Keep the canonical heartbeat current while the Settings view is mounted, independently
    /// of activation, the supervisor status sweep and database stats. SwiftUI owns cancellation.
    func monitorWatcherHealth(
        waitForNextPoll: @Sendable () async throws -> Void = { try await Task.sleep(for: .seconds(30)) }
    ) async {
        guard watcherHealthURL != nil else { return }
        while !Task.isCancelled {
            await sampleWatcherHealth()
            do { try await waitForNextPoll() } catch { return }
        }
    }

    func refreshWatcherHealth() {
        Task { await sampleWatcherHealth() }
    }

    /// Re-reads watcher-health.json off the main actor; only the newest request publishes.
    private func sampleWatcherHealth() async {
        guard let watcherHealthURL else { return }
        watcherHealthGeneration &+= 1
        let generation = watcherHealthGeneration
        let read = watcherHealthRead
        let result = await Task.detached { read(watcherHealthURL) }.value
        guard !Task.isCancelled, generation == watcherHealthGeneration else { return }
        watcherHealth = result
        republishStatus()
    }

    /// Re-reads the installed plists, the backup logs and the local copies off the main actor;
    /// only the newest request publishes.
    func refreshBackupSchedules() {
        guard let backupSources else { return }
        backupScheduleGeneration &+= 1
        let generation = backupScheduleGeneration
        let now = now()
        let read = Task.detached {
            (
                rows: backupSources.rows(now: now, calendar: .current, formatDate: DashboardMetricFormatter.jobDateTimeString),
                maintenance: backupSources.maintenanceEvidence()
            )
        }
        Task {
            let result = await read.value
            guard generation == backupScheduleGeneration else { return }
            backupSchedules = result.rows
            maintenanceEvidence = result.maintenance
            republishStatus()
        }
    }

    func revealBackup(_ row: BrainBarBackupScheduleRow) { row.reveal(using: workspace) }
    func copyBackupPath(_ row: BrainBarBackupScheduleRow) { row.copyPath(using: workspace) }

    func setSystemEnabled(_ enabled: Bool) {
        updateConfig { $0.systemEnabled = enabled }
    }

    func storePlainAPIKey() {
        let value = pendingPlainAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        if updateConfig({ $0.googleAPIKey = .plain(value) }, beforeUpdate: confirmGoogleAPIKeyOverwriteIfNeeded) {
            pendingPlainAPIKey = ""
        }
    }

    func storeOnePasswordReference() {
        let reference = onePasswordReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !reference.isEmpty else { return }
        let draft = onePasswordReference
        _ = updateConfig({ $0.googleAPIKey = .onePasswordReference(reference) }, beforeUpdate: {
            let accepted = self.config.googleAPIKey == .onePasswordReference(reference)
                || self.confirmGoogleAPIKeyOverwriteIfNeeded()
            if !accepted { self.onePasswordReference = draft }
            return accepted
        })
    }

    func clearGoogleAPIKey() {
        _ = updateConfig { $0.googleAPIKey = .missing }
    }

    func setJob(_ job: BrainLayerLaunchdJob, enabled: Bool) {
        updateConfig {
            $0.launchdJobs[job, default: BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown)].enabled = enabled
        }
    }

    func setGroup(_ group: BrainLayerLaunchdJobGroup, enabled: Bool) {
        updateConfig { config in
            for job in group.jobs {
                config.launchdJobs[job, default: BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown)].enabled = enabled
            }
        }
    }

    func isGroupEnabled(_ group: BrainLayerLaunchdJobGroup) -> Bool {
        group.jobs.allSatisfy { config.launchdJobs[$0]?.enabled == true }
    }

    /// S5: the app-lifetime badge reader publishes here, so every surface sees the same inputs.
    func applyHealthBadge(_ badge: BadgeStatePresentation, alerts: BrainBarJobAlerts?) {
        applyHealthBadge(badge, alertsRead: alerts.map(BrainBarJobAlerts.ReadResult.readable) ?? .absent)
    }

    /// R2-3: a readable map (even empty) is the verified truth and the only recovery; an unavailable
    /// read keeps the last verified map; a file that vanished after reporting alerts is unavailable.
    func applyHealthBadge(_ badge: BadgeStatePresentation, alertsRead: BrainBarJobAlerts.ReadResult) {
        if healthBadge != badge { healthBadge = badge }
        let state: BrainBarStatusInputs.JobAlertsState
        switch alertsRead {
        case let .readable(map):
            if liveJobAlerts != map { liveJobAlerts = map }
            state = .readable
        case .absent:
            if liveJobAlerts?.active.isEmpty == false {
                state = .unavailable("job-alerts.json disappeared after reporting alerts")
            } else {
                if liveJobAlerts != nil { liveJobAlerts = nil }
                state = .absent
            }
        case let .unavailable(reason):
            state = .unavailable(reason)
        }
        if liveJobAlertsState != state { liveJobAlertsState = state }
        republishStatus()
    }

    /// The backups section as the Backups page shows it: a job alert the snapshot rates as neutral
    /// text (a deferred weekly VACUUM) is not shown there as a red backup alert.
    private func displayedBackups(_ document: ObservabilityDocument) -> ObservabilityDocument.Backups {
        guard let reason = BrainBarJobAlerts.rawReason(document),
              publishedStatus.isNoticeAlert(reason: reason) else { return document.backups }
        return document.replacingBackupsErrorType(nil).backups
    }

    var dashboardJobsChecking: Bool {
        BrainLayerLaunchdJobGroup.allCases.contains { groupStatus($0).health == .checking }
    }

    /// S5: the job groups' dot reasons, from the one snapshot; notices are not "things that need you".
    var dashboardJobAttentionReasons: [String] {
        BrainLayerLaunchdJobGroup.allCases.compactMap { group in
            let status = groupStatus(group)
            return status.health == .unhealthy ? status.attentionReason ?? "\(group.title): \(status.health.title)." : nil
        }.reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
    }

    func groupStatus(_ group: BrainLayerLaunchdJobGroup) -> BrainLayerLaunchdGroupStatus {
        // The independent health reader can measure a watcher fault before launchd finishes; the
        // shared snapshot then decides whether it is a dot or neutral text.
        publishedStatus.project(rawGroupStatus(group), group: group)
    }

    /// The group's own launchd/calendar verdict, before the shared snapshot decides its severity.
    private func rawGroupStatus(_ group: BrainLayerLaunchdJobGroup) -> BrainLayerLaunchdGroupStatus {
        guard hasCompletedLaunchdSample || (group == .ingest && watcherStatus.needsAttention) else {
            return .init(health: .checking, attentionReason: nil, lastRunText: "Checking…", nextRunText: "Checking…")
        }
        return group.status(
            settings: config.launchdJobs,
            observations: launchdObservations,
            formatDate: DashboardMetricFormatter.jobDateTimeString,
            watcher: watcherStatus,
            maintenance: maintenanceEvidence,
            now: now()
        )
    }

    /// Samples the hotlane on its own detached task, published independently of the
    /// all-jobs launchd sweep, which has no deadline (#974 review B1). The probe itself is
    /// bounded (1 s launchctl timeout, non-blocking proc_pidinfo). Only the newest request
    /// publishes, so a slow older sample can never overwrite a fresher PID/RSS.
    func refreshEmbeddingResidency() {
        embeddingSampleGeneration &+= 1
        let generation = embeddingSampleGeneration
        let probe = embeddingResidencyProbe
        Task {
            let sample = await Task.detached { probe.sample() }.value
            guard generation == embeddingSampleGeneration else { return }
            embeddingProcess = sample
        }
    }

    /// The Dashboard's refresh cadence (2026-10-21): sample now, then once per tick, for as long as
    /// the calling view's `.task` lives. Owned here, not by a `Timer.publish` built in a view body,
    /// which every re-render replaced before it could fire — so the Dashboard kept the reasons of
    /// whatever sample it took on appear.
    func monitorLaunchdStatus(
        waitForNextTick: @Sendable () async throws -> Void = { try await Task.sleep(for: .seconds(60)) }
    ) async {
        while !Task.isCancelled {
            await refreshLaunchdStatus().value
            do { try await waitForNextTick() } catch { return }
        }
    }

    /// S5: the app-lifetime status cadence: launchd/calendar every 60 s, observability and the
    /// watcher heartbeat every 30 s. Cancelling the calling task stops all three.
    func monitorStatus() async {
        let period = statusSamplingPeriod
        await monitorStatus(launchdEvery: 2) { try await Task.sleep(for: .seconds(period)) }
    }

    /// B6: ONE app-owned cadence. Each tick reads observability, the watcher heartbeat and the
    /// badge/alert files once, off the main actor; launchd/calendar runs on every `launchdEvery` tick.
    func monitorStatus(launchdEvery: Int, waitForNextTick: @Sendable () async throws -> Void) async {
        var tick = 0
        while !Task.isCancelled {
            await sampleStatusTick(includeLaunchd: tick % max(launchdEvery, 1) == 0)
            tick &+= 1
            do { try await waitForNextTick() } catch { return }
        }
    }

    /// One sampler tick; publishes one snapshot when every read it started has landed.
    func sampleStatusTick(includeLaunchd: Bool) async {
        refreshObservabilityStatus()
        await observabilityTask?.value
        await sampleWatcherHealth()
        await sampleHealthBadge()
        if includeLaunchd { await refreshLaunchdStatus(includeWatcher: false).value }
        republishStatus()
    }

    /// Reads badge-state.json and job-alerts.json off the main actor.
    private func sampleHealthBadge() async {
        guard let healthBadgeURL else { return }
        let alertsURL = jobAlertsURL, history = badgeReadHistory, now = now()
        let cadence = observabilityCadence
        let (raw, alerts) = await Task.detached {
            (BadgeStateReader.read(url: healthBadgeURL, now: now, cadence: cadence, history: history),
             alertsURL.map { BrainBarJobAlerts.readResult(url: $0) } ?? .absent)
        }.value
        guard !Task.isCancelled else { return }
        applyHealthBadge(raw, alertsRead: alerts)
    }

    /// The Drive auth state the snapshot reads at publication time; its changes republish.
    func attachDriveAuth(_ auth: BrainBarDriveAuthModel) {
        guard driveAuthSource !== auth else { return }
        driveAuthSource = auth
        driveAuthCancellable = auth.objectWillChange
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.republishStatus() }
        republishStatus()
    }

    /// B5: derive once, with one captured time, and publish; surfaces only read `publishedStatus`.
    func republishStatus() {
        let snapshot = deriveStatus(at: now())
        if snapshot.issues != publishedStatus.issues || snapshot.isChecking != publishedStatus.isChecking {
            publishedStatus = snapshot
        }
    }

    private func deriveStatus(at time: Date) -> BrainBarStatusSnapshot {
        let backups: BrainBarStatusInputs.Backups
        if !hasCompletedObservabilityRead {
            backups = .notRead
        } else if case let .readable(document) = observabilityResult, document.backups.state == "measured" {
            backups = .measured(document.backups)
        } else {
            backups = .unavailable(backupStatusReason ?? "Backup status is unmeasurable.")
        }
        let file: WatcherHealthFile? = if case let .readable(value) = watcherHealth { value } else { nil }
        let watcher = WatcherHealthStatus.derive(
            launchd: WatcherLaunchdEvidence(setting: config.launchdJobs[.watch], loadState: config.launchdJobs[.watch]?.loadState),
            file: watcherHealth, now: time)
        let drive = driveAuthSource.map {
            DriveAuthPresentation.derive(status: $0.status, isReconnecting: $0.isReconnecting, lastOutcome: $0.lastOutcome,
                                         now: time, formatDate: BrainBarDriveAuthFormat.date)
        }
        return BrainBarStatusSnapshot.derive(BrainBarStatusInputs(
            now: time, badge: healthBadge, jobAlerts: liveJobAlerts, jobAlertsState: liveJobAlertsState,
            watcher: watcher, watcherHeartbeatAt: file?.updatedAt, watcherFile: file, settings: config.launchdJobs,
            observations: hasCompletedLaunchdSample ? launchdObservations : nil,
            backups: backups, backupsNotice: hasCompletedObservabilityRead ? backupTruth.statusNotice : nil,
            drive: drive, maintenance: maintenanceEvidence,
            observabilityUnreadableFor: observabilityUnreadableSince.map { time.timeIntervalSince($0) },
            samplingPeriod: statusSamplingPeriod
        ))
    }

    /// One launchd/calendar sample, off the main actor; only the newest request publishes.
    @discardableResult
    func refreshLaunchdStatus(includeWatcher: Bool = true) -> Task<Void, Never> {
        launchdSampleGeneration &+= 1
        let generation = launchdSampleGeneration
        isRefreshingLaunchdStatus = true
        refreshEmbeddingResidency()
        if includeWatcher { refreshWatcherHealth() }
        refreshBackupSchedules()
        let provider = launchdStatusProvider
        return Task {
            let observations = await Task.detached {
                provider.sampleActivity()
            }.value
            guard generation == launchdSampleGeneration else { return }
            launchdObservations = observations
            applyLaunchdStates(Dictionary(uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map {
                ($0, observations[$0]?.loadState ?? .unknown)
            }))
            hasCompletedLaunchdSample = true
            activeRuntimeObservation = runtimeStatusProvider.sample()
            refreshLastSaveReceiptActiveState()
            isRefreshingLaunchdStatus = false
            republishStatus()
        }
    }

    /// Advanced keeps its last observation on refresh, just like the group cards.
    func launchdStatusTitle(_ job: BrainLayerLaunchdJob) -> String {
        guard hasCompletedLaunchdSample else { return "Checking…" }
        if let observation = launchdObservations[job], observation.ownership == .supervisor,
           observation.loadState == .loaded, observation.lastExitCode == 75 {
            return observation.backupScheduleLine?.text ?? "Needs attention"
        }
        let title = config.launchdJobs[job]?.loadState.title ?? "Unknown"
        return title + (launchdObservations[job]?.ownerNote.map { " · " + $0 } ?? "")
    }

    var presentedBackupSchedules: [BrainBarBackupScheduleRow] {
        let jobs: [(String, BrainLayerLaunchdJob)] = [("Database", .backupDaily),
            ("Transcripts", .jsonlBackup), ("Nightly maintenance", .maintenanceNightly),
            ("Weekly maintenance", .maintenanceWeekly)]
        guard let owner = launchdObservations[.jsonlBackup]?.ownership, owner != .legacy else { return backupSchedules }
        return jobs.map { title, job in
            let observation = launchdObservations[job]
            let cadence: String = switch owner {
            case .supervisor: observation?.ownerNote ?? "Maintenance calendar status unavailable"
            case let .unknown(reason): reason
            case .legacy: ""
            }
            return BrainBarBackupScheduleRow(title: title, cadence: cadence,
                lastRun: observation?.lastRunText(formatDate: DashboardMetricFormatter.jobDateTimeString,
                                                  never: observation?.lastRunNeverText ?? "Last run unavailable") ?? "Last run unavailable",
                nextRun: observation?.nextRunAt.map(DashboardMetricFormatter.jobDateTimeString) ?? "Unavailable",
                localCopy: backupSchedules.first { $0.title == title }?.localCopy)
        }
    }

    var backupStatus: ObservabilityBackupStatus? {
        guard case let .readable(document) = observabilityResult,
              document.backups.state == "measured" else { return nil }
        return ObservabilityPresentation.backupStatus(for: displayedBackups(document), jobStatus: launchdObservations[.jsonlBackup]?.backupScheduleLine, now: observabilityNow)
    }

    /// The Backups page's checks in plain words (E1), from the same document as `backupStatus`.
    var backupChecks: BrainBarBackupChecks? {
        guard case let .readable(document) = observabilityResult,
              document.backups.state == "measured" else { return nil }
        return BrainBarBackupChecks.derive(displayedBackups(document), now: observabilityNow,
            jobStatus: launchdObservations[.jsonlBackup]?.backupScheduleLine, statusNotice: backupTruth.statusNotice)
    }

    /// The Backups page's one verdict (#1029 review B1), from the same Drive presentation the card
    /// shows, the Backups job group and the backup checks, in the checks' own words.
    func backupsHealth(drive: DriveAuthPresentation?) -> BrainBarBackupsHealth {
        // The page's own wording comes from the raw group; the snapshot then decides the severity.
        let verdict = BrainBarBackupsHealth.derive(
            job: rawGroupStatus(.backups),
            drive: drive,
            status: backupStatus,
            statusUnavailableReason: backupStatusReason ?? "Backup status is unmeasurable.",
            attentionSentence: backupChecks?.attentionSentence,
            isCheckingStatus: !hasCompletedObservabilityRead
        )
        let projected = publishedStatus.project(verdict)
        if projected.badge == .attention || !hasCompletedObservabilityRead || projected != verdict { return projected }
        if let reason = backupTruth.statusNotice { return .init(badge: .unknown, reason: reason) }
        return projected
    }

    /// A job alert (#1031) in its own words, shown once on the Backups page as its own card — also
    /// when the snapshot rates it a neutral notice (then the card is neutral, and the page's verdict
    /// does not count it).
    var jobAlert: String? {
        guard case let .readable(document) = observabilityResult, document.backups.state == "measured" else { return nil }
        let status = ObservabilityPresentation.backupStatus(for: document.backups, now: observabilityNow)
        guard status.errorIsJobAlert, status.error?.tone == .red else { return nil }
        return status.error?.text
    }

    /// The job-alert card is neutral text: the shared snapshot rates its alert a notice.
    var jobAlertIsNotice: Bool {
        guard case let .readable(document) = observabilityResult, let reason = BrainBarJobAlerts.rawReason(document) else { return false }
        return publishedStatus.isNoticeAlert(reason: reason)
    }

    /// The backup status rows without the job alert, which the alert card already shows.
    var backupStatusLines: [ObservabilityStatusLine]? {
        guard let status = backupStatus else { return nil }
        return status.lines.filter { line in !(status.errorIsJobAlert && line == status.error) }
    }

    /// The line under the Backups badge: the verdict's reason, unless it is the job alert the
    /// alert card above it already shows.
    func backupsBadgeReason(drive: DriveAuthPresentation?) -> String? {
        // Each failure once per screen (lead UX r1 on #1064): the job-alert card, the Google Drive
        // card and the Recovery checks card each say their own; the header keeps only the badge.
        let reason = backupsHealth(drive: drive).reason
        let shownElsewhere = [jobAlert, drive?.line, backupChecks?.attentionSentence, backupChecks?.statusNotice].compactMap { $0 }
        // A copy-age issue is already the red Recovery check below; the header keeps only the badge.
        let copyIssue = publishedStatus.attention.contains { [.database, .transcripts].contains($0.component) && $0.text == reason }
        if copyIssue, backupChecks?.checks.contains(where: { $0.tone == .red }) == true { return nil }
        return reason.flatMap { shownElsewhere.contains($0) ? nil : $0 }
    }

    /// Show log for the job alert: the failing job's log, named by the live job-alert state.
    func showJobAlertLog() {
        guard jobAlert != nil, case let .readable(document) = observabilityResult else { return }
        let paths = backupSources?.paths ?? .live(databasePath: document.dbPath)
        let message = BrainBarJobAlerts.showLog(for: document, paths: paths, workspace: workspace).message
        jobAlertLogNote = BrainBarJobAlertLogNote(reason: BrainBarJobAlerts.rawReason(document), message: message)
    }

    func showBackupLog(_ target: BrainBarBackupLogTarget) -> BrainBarJobAlertLogResult {
        guard BrainBarBackupLogTarget.failures(in: config).contains(target) else {
            return .failed(message: "Backup failure changed; check its current status.")
        }
        let dbPath: String
        if case let .readable(document) = observabilityResult { dbPath = document.dbPath }
        else { dbPath = BrainBarServer.defaultDBPath() }
        return BrainBarJobAlerts.showLog(forKey: target.key,
            paths: backupSources?.paths ?? .live(databasePath: dbPath), workspace: workspace)
    }

    /// The note's sentence, only while its alert is still the current one (Macroscope #1062).
    var jobAlertLogMessage: String? {
        guard let note = jobAlertLogNote, case let .readable(document) = observabilityResult,
              BrainBarJobAlerts.rawReason(document) == note.reason else { return nil }
        return note.message
    }

#if DEBUG
    func setObservabilityResultForTesting(_ result: ObservabilityReadResult) {
        observabilityNow = now()
        observabilityResult = result
        hasCompletedObservabilityRead = true
        republishStatus()
    }
#endif

    /// The one Config file row (E5): the file this model reads and saves.
    var configFileURL: URL { store.configURL }
    func copyConfigFilePath() { workspace.copy(store.configURL.path) }
    func revealConfigFile() { workspace.reveal(store.configURL) }
    func copyText(_ text: String) { workspace.copy(text) }

    func copyTechnicalDetail(_ detail: BrainBarBackupChecks.Detail) {
        workspace.copy(detail.value)
        copiedTechnicalDetailID = detail.id
        copyFeedbackGeneration &+= 1
        let generation = copyFeedbackGeneration
        copyFeedbackScheduler(1.5) { [weak self] in
            guard let self, self.copyFeedbackGeneration == generation else { return }
            self.copiedTechnicalDetailID = nil
        }
    }

    var backupStatusReason: String? {
        guard hasCompletedObservabilityRead else { return nil }
        return switch observabilityResult {
        case let .unreadable(value):
            value
        case let .readable(document):
            if document.backups.state == "measured" { backupTruth.statusNotice }
            else if document.backups.reason.localizedCaseInsensitiveContains("no backup or maintenance LaunchAgents"),
                    let observation = launchdObservations[.jsonlBackup], observation.ownership != .legacy {
                (observation.backupScheduleLine?.text ?? "Job owner unknown") + "; backup history has not been measured."
            }
            else if document.backups.reason.isEmpty { "Backup status is unmeasurable." }
            else { "Backup status is unmeasurable — \(document.backups.reason)" }
        }
    }

    func refreshAllStatus() {
        _ = reloadConfigFromDisk()
        refreshLaunchdStatus()
        refreshObservabilityStatus()
    }

    func reloadConfigFromDisk(preservingDrafts: Bool = false) -> Bool {
        do {
            let loaded = try store.loadDocument().config
            let previous = config
            config = loaded
            configReadSucceeded = true
            if !preservingDrafts || onePasswordReference == previous.googleAPIKey.opReference {
                onePasswordReference = loaded.googleAPIKey.opReference
            }
            applyLaunchdStates(launchdObservations.mapValues(\.loadState))
            errorMessage = nil
            republishStatus()
            if !preservingDrafts { lastSaveReceipt = nil }
            return true
        } catch {
            configReadSucceeded = false
            errorMessage = error.localizedDescription
            return false
        }
    }

    func refreshObservabilityStatus() {
        guard let observabilityURL else { return }
        observabilityTask?.cancel()
        isRefreshingObservabilityStatus = true
        let read = observabilityRead
        observabilityTask = Task { [weak self] in
            let result = await read(observabilityURL)
            guard !Task.isCancelled else { return }
            guard let self else { return }
            let readAt = now()
            observabilityNow = readAt
            observabilityResult = result
            hasCompletedObservabilityRead = true
            isRefreshingObservabilityStatus = false
            // L3: an unreadable file starts (or continues) an episode; any readable result ends it.
            if case .unreadable = result { observabilityUnreadableSince = observabilityUnreadableSince ?? readAt }
            else { observabilityUnreadableSince = nil }
            republishStatus()
        }
    }

    /// The unified window keeps one published receipt snapshot for Dashboard and Backups.
    /// SwiftUI owns this loop's cancellation; the fallback standalone dashboard has its own reader.
    func monitorObservabilityStatus(
        waitForNextTick: @Sendable () async throws -> Void = { try await Task.sleep(for: .seconds(30)) }
    ) async {
        guard observabilityURL != nil else { return }
        while !Task.isCancelled {
            refreshObservabilityStatus()
            await observabilityTask?.value
            do { try await waitForNextTick() } catch { return }
        }
    }

    var dashboardObservabilityResult: ObservabilityReadResult {
        hasCompletedObservabilityRead ? observabilityResult : .unreadable("Loading observability data.")
    }

    var backupTruth: BrainBarHeroBackupTruth {
        .derive(from: dashboardObservabilityResult, now: observabilityNow, cadence: observabilityCadence)
    }

    var backupReceiptLines: [ObservabilityStatusLine]? {
        guard case let .readable(document) = observabilityResult,
              document.backups.state == "measured" else { return nil }
        return ObservabilityPresentation.backupReceiptLines(for: document.backups, now: observabilityNow)
    }

    private func applyLaunchdStates(_ states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]) {
        for (job, state) in states {
            config.launchdJobs[job, default: BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown)].loadState = state
        }
    }

    @discardableResult
    private func updateConfig(
        _ apply: (inout BrainLayerConfig) -> Void,
        beforeUpdate: (() -> Bool)? = nil
    ) -> Bool {
        let confirmedReferenceDraft = beforeUpdate == nil ? nil : onePasswordReference
        guard reloadConfigFromDisk() else { return false }
        let apiKeyBeforeConfirmation = config.googleAPIKey
        guard beforeUpdate?() ?? true else { return false }
        if beforeUpdate != nil {
            guard reloadConfigFromDisk() else {
                if let confirmedReferenceDraft { onePasswordReference = confirmedReferenceDraft }
                return false
            }
            guard config.googleAPIKey == apiKeyBeforeConfirmation else {
                if let confirmedReferenceDraft { onePasswordReference = confirmedReferenceDraft }
                recordValidationFailure(
                    "API key changed while overwrite confirmation was open. Review it and try again."
                )
                return false
            }
        }
        let previousConfig = config
        var nextConfig = config
        apply(&nextConfig)
        let restartRequirements = Self.restartRequirements(from: previousConfig, to: nextConfig)
        var fileUpdated = false
        do {
            try store.save(nextConfig)
            fileUpdated = true
            let persistedConfig = try store.loadDocument().config
            guard persistedConfig.persistedValuesEqual(to: nextConfig) else {
                recordPostWriteValidationFailure(
                    "Saved configuration did not validate on reload.",
                    previousConfig: previousConfig,
                    persistedConfig: persistedConfig
                )
                return false
            }
            activeRuntimeObservation = runtimeStatusProvider.sample()
            config = nextConfig
            defer { republishStatus() }
            onePasswordReference = nextConfig.googleAPIKey.opReference
            errorMessage = nil
            previousConfigForLastSaveReceipt = previousConfig
            lastSaveReceipt = BrainLayerSettingsSaveReceipt(
                configURL: store.configURL,
                savedAt: now(),
                fileUpdated: true,
                validation: .passed,
                servicesRequiringRestart: restartRequirements,
                activeRuntimeState: activeRuntimeState(
                    from: previousConfig,
                    configured: nextConfig,
                    requirements: restartRequirements
                )
            )
            return true
        } catch {
            errorMessage = error.localizedDescription
            previousConfigForLastSaveReceipt = nil
            lastSaveReceipt = BrainLayerSettingsSaveReceipt(
                configURL: store.configURL,
                savedAt: now(),
                fileUpdated: fileUpdated,
                validation: fileUpdated ? .failed("Saved configuration could not be reloaded.") : .passed,
                servicesRequiringRestart: fileUpdated ? restartRequirements : [],
                activeRuntimeState: .unknown(
                    fileUpdated
                        ? "Configuration file changed, but reload failed."
                        : "Configuration file was not updated."
                )
            )
            return false
        }
    }

    private func recordValidationFailure(_ message: String) {
        errorMessage = nil
        previousConfigForLastSaveReceipt = nil
        lastSaveReceipt = BrainLayerSettingsSaveReceipt(
            configURL: store.configURL,
            savedAt: now(),
            fileUpdated: false,
            validation: .failed(message),
            servicesRequiringRestart: [],
            activeRuntimeState: .unknown("Configuration was not written.")
        )
    }

    private func recordPostWriteValidationFailure(
        _ message: String,
        previousConfig: BrainLayerConfig,
        persistedConfig: BrainLayerConfig
    ) {
        config = persistedConfig
        onePasswordReference = persistedConfig.googleAPIKey.opReference
        errorMessage = nil
        activeRuntimeObservation = runtimeStatusProvider.sample()
        previousConfigForLastSaveReceipt = nil
        lastSaveReceipt = BrainLayerSettingsSaveReceipt(
            configURL: store.configURL,
            savedAt: now(),
            fileUpdated: true,
            validation: .failed(message),
            servicesRequiringRestart: Self.restartRequirements(from: previousConfig, to: persistedConfig),
            activeRuntimeState: .unknown("Configuration file changed, but reload validation failed.")
        )
    }

    private func refreshLastSaveReceiptActiveState() {
        guard let receipt = lastSaveReceipt,
              receipt.fileUpdated,
              receipt.validation == .passed,
              let previousConfig = previousConfigForLastSaveReceipt else {
            return
        }
        lastSaveReceipt = BrainLayerSettingsSaveReceipt(
            configURL: receipt.configURL,
            savedAt: receipt.savedAt,
            fileUpdated: receipt.fileUpdated,
            validation: receipt.validation,
            servicesRequiringRestart: receipt.servicesRequiringRestart,
            activeRuntimeState: activeRuntimeState(
                from: previousConfig,
                configured: config,
                requirements: receipt.servicesRequiringRestart
            )
        )
    }

    private static func restartRequirements(
        from previous: BrainLayerConfig,
        to configured: BrainLayerConfig
    ) -> [BrainLayerSettingsService] {
        var requirements: [BrainLayerSettingsService] = []
        if previous.systemEnabled != configured.systemEnabled {
            requirements.append(.systemJobs)
        }
        for job in BrainLayerLaunchdJob.allCases
        where previous.launchdJobs[job]?.enabled != configured.launchdJobs[job]?.enabled {
            requirements.append(.launchdJob(job))
        }
        return requirements
    }

    private func activeRuntimeState(
        from previous: BrainLayerConfig,
        configured: BrainLayerConfig,
        requirements: [BrainLayerSettingsService]
    ) -> BrainLayerActiveRuntimeReceiptState {
        if previous.googleAPIKey != configured.googleAPIKey {
            return .unknown("Secret reload state is not observable.")
        }

        for requirement in requirements {
            guard case let .launchdJob(job) = requirement else { continue }
            let setting = configured.launchdJobs[job] ?? BrainLayerLaunchdJobSetting(
                enabled: true,
                loadState: .unknown
            )
            switch setting.loadState {
            case .unknown:
                return .unknown("\(job.title) active state is unknown.")
            case let .probeError(reason):
                return .unknown("\(job.title) probe failed: \(reason)")
            case .running where setting.enabled, .loaded where setting.enabled:
                continue
            case .unloaded where !setting.enabled:
                continue
            default:
                return .notObserved
            }
        }

        let hasRuntimeConfigRequirement = requirements.contains(.systemJobs)
        guard hasRuntimeConfigRequirement else { return .observed }
        switch activeRuntimeObservation {
        case let .observed(values):
            return values.matches(configured) ? .observed : .notObserved
        case let .unknown(reason):
            return .unknown(reason)
        }
    }

    private func confirmGoogleAPIKeyOverwriteIfNeeded() -> Bool {
        guard config.googleAPIKey.kind != .missing else { return true }
        if let confirmAPIKeyOverwrite { return confirmAPIKeyOverwrite() }
        let alert = NSAlert()
        alert.messageText = "Replace existing Gemini API key?"
        alert.informativeText = "BrainBar will update the BrainLayer config file without displaying the current value."
        alert.addButton(withTitle: "Replace")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
    }
}

enum BrainBarSettingsFooterState: Equatable {
    case checking
    case watcher(WatcherHealthStatus)
    case systemOff
    case unavailable

    var title: String {
        switch self {
        case .checking: "Checking…"
        case let .watcher(status): status.title
        case .systemOff: "System off"
        case .unavailable: "Status unavailable"
        }
    }
}

struct BrainBarSettingsFooterPresentation {
    let state: BrainBarSettingsFooterState
    /// The watcher's reason line (what · since · what to do), identical to the Dashboard's.
    let detail: String?
    let locality: String
    let showsLock: Bool
    let symbol: String

    init(config: BrainLayerConfig?, watcher: WatcherHealthStatus?, now: Date = Date(), isChecking: Bool = false) {
        guard let config else {
            state = .unavailable
            detail = nil
            locality = "Memory on this Mac · Backups unknown"
            showsLock = false
            symbol = "questionmark.circle"
            return
        }

        if !config.systemEnabled {
            state = .systemOff
            detail = nil
        } else if isChecking, watcher?.needsAttention != true {
            state = .checking
            detail = nil
        } else if let watcher {
            state = .watcher(watcher)
            detail = watcher.reasonText(now: now)
        } else {
            state = .unavailable
            detail = nil
        }

        let driveJobs: [BrainLayerLaunchdJob] = [.backupDaily, .jsonlBackup, .maintenanceWeekly]
        let driveStates = driveJobs.map { config.launchdJobs[$0]?.enabled }
        let backups: String
        let driveConfigured: Bool
        let backupsOff = driveStates.allSatisfy { $0 == false }
        if driveStates.contains(where: { $0 == true }) {
            backups = "Backups → Drive"
            driveConfigured = true
        } else if backupsOff {
            backups = "Backups off"
            driveConfigured = false
        } else {
            backups = "Backups unknown"
            driveConfigured = false
        }
        locality = "Memory on this Mac · \(backups)"
        showsLock = backupsOff
        symbol = showsLock ? "lock" : (driveConfigured ? "icloud" : "questionmark.circle")
    }
}

/// The settings pages. The old General page only pointed at Dashboard, Jobs and Backups; it is
/// absorbed into Dashboard, which is now the window's first sidebar item (#963).
enum BrainBarSettingsSection: String, CaseIterable, Identifiable {
    case jobs, backups, advanced

    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var symbol: String {
        switch self {
        case .jobs: "clock.arrow.circlepath"
        case .backups: "externaldrive"
        case .advanced: "gearshape.2"
        }
    }
    var groups: [BrainLayerLaunchdJobGroup] {
        switch self {
        case .jobs: [.ingest, .maintenance]
        case .backups: [.backups]
        case .advanced: []
        }
    }
    var advancedJobs: [BrainLayerLaunchdJob] { self == .advanced ? BrainLayerLaunchdJobGroup.advancedJobs : [] }
}

/// The one window's sidebar (#963): Dashboard first, then each settings page.
enum BrainBarSidebarItem: String, CaseIterable, Identifiable {
    case dashboard, jobs, backups, advanced

    init(section: BrainBarSettingsSection) {
        switch section {
        case .jobs: self = .jobs
        case .backups: self = .backups
        case .advanced: self = .advanced
        }
    }

    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var symbol: String { settingsSection?.symbol ?? "gauge" }

    /// The settings page this item shows, or nil for Dashboard.
    var settingsSection: BrainBarSettingsSection? {
        switch self {
        case .dashboard: nil
        case .jobs: .jobs
        case .backups: .backups
        case .advanced: .advanced
        }
    }
}

@MainActor
final class BrainBarSettingsNavigation: ObservableObject {
    @Published private(set) var selected: BrainBarSettingsSection = .jobs
    init(selected: BrainBarSettingsSection = .jobs) { self.selected = selected }
    func select(_ section: BrainBarSettingsSection) { selected = section }
}

struct BrainBarSettingsView: View {
    @StateObject var viewModel: BrainBarSettingsViewModel
    @StateObject private var navigation = BrainBarSettingsNavigation()
    @Environment(\.brainBarDriveAuth) private var driveAuth
    private let activationRevision: Int
    /// False inside the one BrainBar window, whose own sidebar lists these pages (#963).
    private let showsSidebar: Bool

    static func observabilityURL(
        databasePath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        ObservabilityReader.url(dbPath: databasePath, environment: environment)
    }

    init(workspace: any BrainBarWorkspaceActing, databasePath: String, activationRevision: Int = 0,
         navigation: BrainBarSettingsNavigation = BrainBarSettingsNavigation(), showsSidebar: Bool = true) {
        self.activationRevision = activationRevision
        self.showsSidebar = showsSidebar
        _navigation = StateObject(wrappedValue: navigation)
        _viewModel = StateObject(wrappedValue: BrainBarSettingsViewModel(
            workspace: workspace,
            observabilityURL: Self.observabilityURL(databasePath: databasePath),
            watcherHealthURL: WatcherHealthReader.resolvedURL(),
            backupSources: .live(databasePath: databasePath)
        ))
    }

    init(viewModel: BrainBarSettingsViewModel, initialSection: BrainBarSettingsSection = .jobs) {
        activationRevision = 0
        showsSidebar = true
        _viewModel = StateObject(wrappedValue: viewModel)
        _navigation = StateObject(wrappedValue: BrainBarSettingsNavigation(selected: initialSection))
    }

    init(viewModel: BrainBarSettingsViewModel, navigation: BrainBarSettingsNavigation, showsSidebar: Bool, activationRevision: Int = 0) {
        self.activationRevision = activationRevision
        self.showsSidebar = showsSidebar
        _viewModel = StateObject(wrappedValue: viewModel)
        _navigation = StateObject(wrappedValue: navigation)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                if showsSidebar {
                    sidebar
                        .frame(width: 214)
                        .frame(maxHeight: .infinity, alignment: .top)
                        .background(Color.brainBarGlassSecondary)
                    Rectangle().fill(Color.brainBarBorderSoft).frame(width: 1)
                }
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        header
                        if let errorMessage = viewModel.errorMessage { errorBanner(errorMessage) }
                        if let receipt = viewModel.lastSaveReceipt {
                            VStack(alignment: .leading, spacing: 8) {
                                sectionHeading("Last save receipt")
                                saveReceipt(receipt)
                            }
                        }
                        sectionContent
                    }
                    .padding(.horizontal, 28)
                    .padding(.vertical, 25)
                    .frame(maxWidth: 720, alignment: .leading)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                }
            }
            .frame(maxHeight: .infinity)
            footer
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.brainBarBackgroundBase)
        .foregroundStyle(Color.brainBarTextPrimary)
        .environment(\.colorScheme, .dark)
        .task { await viewModel.monitorWatcherHealth() }
        .onChange(of: activationRevision, initial: true) { _, _ in
            _ = viewModel.reloadConfigFromDisk(preservingDrafts: true)
            viewModel.refreshLaunchdStatus()
            viewModel.refreshObservabilityStatus()
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(BrainBarSettingsSection.allCases) { section in
                Button {
                    navigation.select(section)
                } label: {
                    Label(section.title, systemImage: section.symbol)
                        .font(.system(size: 13, weight: navigation.selected == section ? .semibold : .medium))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 9)
                        .background(navigation.selected == section ? Color.brainBarGlassPrimary : .clear)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(navigation.selected == section ? .isSelected : [])
            }
            Spacer(minLength: 0)
        }
        .padding(15)
    }

    private var footer: some View {
        let presentation = viewModel.footerPresentation
        return HStack(spacing: 12) {
            HStack(spacing: 7) {
                Circle()
                    .fill(footerDotColor(presentation.state))
                    .frame(width: 6, height: 6)
                Text(presentation.state.title)
                    .font(.system(size: 11, weight: .semibold))
                if let detail = presentation.detail {
                    Text(detail)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Color.brainBarTextMuted)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                        .help(detail)
                }
            }
            Rectangle().fill(Color.brainBarBorderSoft).frame(width: 1, height: 14)
            HStack(spacing: 6) {
                Image(systemName: presentation.symbol)
                Text(presentation.locality)
                    .lineLimit(1)
            }
            .font(.system(size: 10, weight: .medium))
            .foregroundStyle(Color.brainBarTextMuted)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.brainBarGlassSecondary)
        .overlay(alignment: .top) { Color.brainBarBorderSoft.frame(height: 1) }
        .accessibilityElement(children: .combine)
    }

    private func footerDotColor(_ state: BrainBarSettingsFooterState) -> Color {
        guard case let .watcher(status) = state else { return Color.brainBarTextMuted }
        switch status {
        case .running: return BrainBarStateTheme.active.theme.swiftUIColor
        case .degraded, .stopped: return BrainBarStateTheme.degraded.theme.swiftUIColor
        case .unknown: return Color.brainBarTextMuted
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            // E5 (2026-10-01): no config path under every title; Advanced has the one Config file row.
            Text(navigation.selected.title)
                .font(.system(size: 25, weight: .semibold))
            Spacer()
            Button {
                viewModel.refreshAllStatus()
            } label: {
                Label(viewModel.isRefreshingLaunchdStatus || viewModel.isRefreshingObservabilityStatus ? "Refreshing…" : "Refresh",
                      systemImage: "arrow.clockwise")
            }
            .controlSize(.small)
            .disabled(viewModel.isRefreshingLaunchdStatus || viewModel.isRefreshingObservabilityStatus)
        }
    }

    @ViewBuilder
    private var sectionContent: some View {
        switch navigation.selected {
        case .jobs:
            VStack(alignment: .leading, spacing: 16) {
                BrainBarBackupFailureLogs(model: viewModel)
                jobsGrid
            }
        case .backups:
            VStack(alignment: .leading, spacing: 16) {
                BrainBarBackupFailureLogs(model: viewModel)
                if let alert = viewModel.jobAlert {
                    BrainBarJobAlertCard(alert: alert, logMessage: viewModel.jobAlertLogMessage,
                                         isNotice: viewModel.jobAlertIsNotice) { viewModel.showJobAlertLog() }
                }
                if let driveAuth {
                    BrainBarDriveAwareBackupsGroup(viewModel: viewModel, driveAuth: driveAuth)
                } else {
                    BrainBarJobGroupCard(
                        group: .backups, viewModel: viewModel,
                        backupsHealth: viewModel.backupsHealth(drive: nil), backupsReason: viewModel.backupsBadgeReason(drive: nil)
                    )
                }
                Divider()
                backupSchedule
                backupStatus
            }
        case .advanced:
            VStack(alignment: .leading, spacing: 16) {
                sectionHeading("Embedding model")
                ForEach(viewModel.modelResidencyPresentation.rows) { row in
                    settingsTruthRow(label: row.label, value: row.value)
                }
                Divider()
                ForEach(navigation.selected.advancedJobs) { job in
                    BrainBarJobToggle(job: job, viewModel: viewModel)
                    Divider()
                }
                configFileRow
            }
        }
    }

    /// E5: the one place the config file appears, with what it is and Copy / Reveal.
    private var configFileRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionHeading("Config file")
            Text("Where BrainBar saves these settings. BrainLayer's background jobs read the same file.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextMuted)
                .fixedSize(horizontal: false, vertical: true)
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(viewModel.configFileURL.path)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.brainBarTextSecondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                Button("Copy path") { viewModel.copyConfigFilePath() }
                Button("Reveal in Finder") { viewModel.revealConfigFile() }
            }
            .controlSize(.small)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("brainbar.settings.config-file")
    }

    private func sectionHeading(_ title: String) -> some View {
        Text(title).font(.system(size: 14, weight: .semibold))
    }

    private func saveReceipt(_ receipt: BrainLayerSettingsSaveReceipt) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            settingsTruthRow(label: "File", value: receipt.fileUpdated ? "Updated" : "Not updated")
            settingsTruthRow(label: "Validation", value: receipt.validation.title)
            settingsTruthRow(
                label: "Restart",
                value: receipt.servicesRequiringRestart.isEmpty
                    ? "Not required"
                    : receipt.servicesRequiringRestart.map(\.title).joined(separator: ", ")
            )
            settingsTruthRow(label: "Active runtime", value: receipt.activeRuntimeState.title)
        }
    }

    private func settingsTruthRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(label)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Color.brainBarTextMuted)
                .frame(width: 110, alignment: .leading)
            Spacer(minLength: 12)
            Text(value)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextSecondary)
                .textSelection(.enabled)
        }
        .padding(.vertical, 5)
    }

    private var jobsGrid: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(navigation.selected.groups) { group in
                BrainBarJobGroupCard(group: group, viewModel: viewModel)
                Divider()
            }
        }
    }

    @ViewBuilder
    private var backupSchedule: some View {
        if let lines = viewModel.backupReceiptLines {
            VStack(alignment: .leading, spacing: 10) {
                sectionHeading("Verified copies")
                ObservabilityStatusRows(lines: lines, textColor: .brainBarTextPrimary)
                Text("Copy dates from observability.json: database snapshot creation and transcript upload receipt. Local files below are on this Mac.")
                    .font(.system(size: 11))
                    .foregroundStyle(Color.brainBarTextMuted)
            }
        }
        if !viewModel.presentedBackupSchedules.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                sectionHeading("Schedule")
                Text("Job attempt times from the maintenance calendar or each job's log.")
                    .font(.system(size: 11))
                    .foregroundStyle(Color.brainBarTextMuted)
                ForEach(viewModel.presentedBackupSchedules) { row in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(alignment: .firstTextBaseline) {
                            Text(row.title).font(.system(size: 12, weight: .semibold))
                            Spacer(minLength: 12)
                            Text(row.cadence)
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(Color.brainBarTextSecondary)
                                .multilineTextAlignment(.trailing)
                        }
                        Text("\(row.lastRun) · \(row.nextRun)")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(Color.brainBarTextMuted)
                        if let localCopy = row.localCopy {
                            HStack(spacing: 8) {
                                Text(localCopy.lastPathComponent)
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundStyle(Color.brainBarTextMuted)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                    .textSelection(.enabled)
                                    .help(localCopy.path)
                                Spacer(minLength: 8)
                                Button("Reveal in Finder") { viewModel.revealBackup(row) }
                                Button("Copy path") { viewModel.copyBackupPath(row) }
                            }
                            .controlSize(.small)
                        }
                    }
                    .accessibilityElement(children: .contain)
                    .accessibilityIdentifier("brainbar.settings.backup-schedule.\(row.title)")
                }
                Divider()
            }
        }
    }

    @ViewBuilder
    private var backupStatus: some View {
        if let checks = viewModel.backupChecks {
            BrainBarBackupChecksCard(
                checks: checks,
                // The job alert has its own card at the top of the page; say it once.
                hiddenAlert: viewModel.jobAlert,
                detailsExpanded: $viewModel.backupDetailsExpanded,
                copiedDetailID: viewModel.copiedTechnicalDetailID,
                copy: { viewModel.copyTechnicalDetail($0) }
            )
        } else if !viewModel.hasCompletedObservabilityRead {
            Label("Checking backup status…", systemImage: "clock")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextMuted)
        } else {
            Label(
                viewModel.backupStatusReason ?? "Backup status is unmeasurable.",
                systemImage: "exclamationmark.triangle"
            )
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(Color(nsColor: BrainBarStateTheme.error.theme.color))
        }
    }

    private func errorBanner(_ message: String) -> some View {
        Label(message, systemImage: "exclamationmark.triangle")
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(Color(nsColor: BrainBarStateTheme.error.theme.color))
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: BrainBarStateTheme.error.theme.glow))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

}

/// A job alert (#1031), once per page: the failed job's own sentence and Show log.
private struct BrainBarJobAlertCard: View {
    let alert: String
    /// Show log's sentence when the job has no log yet.
    var logMessage: String?
    /// S5: the shared snapshot rates this alert neutral text (e.g. a deferred maintenance run).
    var isNotice = false
    let showLog: () -> Void

    var body: some View {
        let tone = isNotice ? Color.brainBarTextMuted : Color(nsColor: BrainBarDesignTokens.Colors.statusError)
        VStack(alignment: .leading, spacing: 6) {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(tone)
            Text(alert)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Color.brainBarTextPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            Spacer(minLength: 8)
            Button("Show log", action: showLog)
                .controlSize(.small)
                .help("Opens the failed job's log. If it has no log yet, shows the logs folder.")
        }
            if let logMessage {
                Text(logMessage)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.brainBarTextSecondary)
                    .padding(.leading, 26)
                    .accessibilityIdentifier("brainbar.backups.job-alert.log-message")
            }
        }
        .font(.system(size: 12, weight: .semibold))
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(tone.opacity(0.10)))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(tone.opacity(0.35), lineWidth: 1))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("brainbar.backups.job-alert")
    }
}

/// The Drive card and the Backups group card, observing both models so the badge follows Drive.
private struct BrainBarDriveAwareBackupsGroup: View {
    @ObservedObject var viewModel: BrainBarSettingsViewModel
    @ObservedObject var driveAuth: BrainBarDriveAuthModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            BrainBarDriveAuthCard(model: driveAuth)
            Divider()
            let drive = driveAuth.presentation(formatDate: BrainBarDriveAuthFormat.date)
            BrainBarJobGroupCard(
                group: .backups,
                viewModel: viewModel,
                backupsHealth: viewModel.backupsHealth(drive: drive),
                backupsReason: viewModel.backupsBadgeReason(drive: drive)
            )
        }
    }
}

// Internal so isolated native tests can render the actual Jobs card without launching the app.
struct BrainBarJobGroupCard: View {
    let group: BrainLayerLaunchdJobGroup
    @ObservedObject var viewModel: BrainBarSettingsViewModel
    /// The Backups page passes its combined verdict; other groups show launchd health alone.
    var backupsHealth: BrainBarBackupsHealth?
    /// The Backups verdict's reason, minus anything another card on the page already shows.
    var backupsReason: String?

    static func symbol(_ health: BrainLayerLaunchdGroupHealth) -> String {
        switch health {
        case .checking: "clock"
        case .healthy: "checkmark.circle.fill"
        case .awaitingRun, .deferred: "clock.fill"
        case .unhealthy: "exclamationmark.triangle.fill"
        case .unknown: "questionmark.circle.fill"
        case .notice: "info.circle.fill"
        case .skipped: "forward.end.circle.fill"
        }
    }

    static func color(_ health: BrainLayerLaunchdGroupHealth) -> Color {
        switch health {
        case .healthy: BrainBarStateTheme.active.theme.swiftUIColor
        case .awaitingRun: BrainBarStateTheme.loading.theme.swiftUIColor
        case .unhealthy: BrainBarStateTheme.error.theme.swiftUIColor
        case .deferred: Color(nsColor: BrainBarDesignTokens.Colors.statusAttention)
        case .checking, .unknown, .skipped, .notice: Color.brainBarTextMuted
        }
    }

    /// The badge and its reason: the Backups page's combined verdict, else launchd health.
    private struct Badge {
        let title: String, symbol: String, color: Color
        let reason: String?, reasonColor: Color
    }

    private func badge(_ status: BrainLayerLaunchdGroupStatus) -> Badge {
        guard let backupsHealth else {
            return Badge(
                title: status.health.title, symbol: Self.symbol(status.health), color: Self.color(status.health),
                reason: status.attentionReason,
                reasonColor: status.health == .unhealthy ? BrainBarStateTheme.error.theme.swiftUIColor : Color.brainBarTextMuted
            )
        }
        let (symbol, color): (String, Color) = switch backupsHealth.badge {
        case .checking: (Self.symbol(.checking), Self.color(.checking))
        case .healthy: (Self.symbol(.healthy), Self.color(.healthy))
        case .idle, .queued, .uploading, .notice: ("info.circle.fill", Color.brainBarTextMuted)
        case .awaitingRun: (Self.symbol(.awaitingRun), Self.color(.awaitingRun))
        case .expiring: ("clock.badge.exclamationmark.fill", Color(nsColor: BrainBarDesignTokens.Colors.statusAttention))
        case .attention: (Self.symbol(.unhealthy), Self.color(.unhealthy))
        case .unknown: (Self.symbol(.unknown), Self.color(.unknown))
        case .skipped: (Self.symbol(.skipped), Self.color(.skipped))
        case .deferred: (Self.symbol(.deferred), Self.color(.deferred))
        }
        return Badge(
            title: backupsHealth.badge.title, symbol: symbol, color: color,
            reason: backupsReason,
            reasonColor: backupsHealth.badge == .unknown ? Color.brainBarTextMuted : color
        )
    }

    var body: some View {
        let status = viewModel.groupStatus(group)
        let badge = badge(status)
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(group.title).font(.system(size: 13, weight: .semibold))
                Spacer()
                Label(badge.title, systemImage: badge.symbol)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(badge.color)
                Toggle(group.title, isOn: Binding(
                    get: { viewModel.isGroupEnabled(group) },
                    set: { viewModel.setGroup(group, enabled: $0) }
                )).labelsHidden().toggleStyle(.switch).controlSize(.small)
            }
            Text(group.summary)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextMuted)
            if let reason = badge.reason {
                Text(reason)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(badge.reasonColor)
            }
            // N1: a run that succeeded with warnings is a quiet amber note; the badge stays as is.
            // An informational note (owner, "last result: ok", covered by weekly) is plain text.
            if let note = status.note {
                switch status.noteTone {
                case .warning:
                    Label(note, systemImage: "exclamationmark.circle")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Color(nsColor: BrainBarDesignTokens.Colors.statusAttention))
                        .accessibilityIdentifier("brainbar.jobs.\(group.rawValue).note")
                case .info:
                    Text(note)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Color.brainBarTextSecondary)
                        .accessibilityIdentifier("brainbar.jobs.\(group.rawValue).note")
                }
            }
            // The Backups page's Schedule section is its one source of last/next runs (lead UX r1).
            if backupsHealth == nil {
                groupTiming(label: "LAST RUN", value: status.lastRunText)
                groupTiming(label: "NEXT RUN", value: status.nextRunText)
            }
        }
        .padding(.vertical, 6)
    }

    private func groupTiming(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .font(.system(size: 9, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(Color.brainBarTextMuted)
                .frame(width: 72, alignment: .leading)
            Text(value)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(Color.brainBarTextSecondary)
            Spacer(minLength: 0)
        }
    }
}

private struct BrainBarJobToggle: View {
    let job: BrainLayerLaunchdJob
    @ObservedObject var viewModel: BrainBarSettingsViewModel

    var body: some View {
        let setting = viewModel.config.launchdJobs[job] ?? BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown)
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(job.title).font(.system(size: 12, weight: .semibold))
                Spacer()
                Toggle(job.title, isOn: Binding(
                    get: { viewModel.config.launchdJobs[job]?.enabled ?? true },
                    set: { viewModel.setJob(job, enabled: $0) }
                )).labelsHidden().toggleStyle(.switch).controlSize(.small)
            }
            HStack(spacing: 6) {
                Circle()
                    .fill(viewModel.hasCompletedLaunchdSample ? loadStateColor(setting.loadState) : Color.brainBarTextMuted)
                    .frame(width: 7, height: 7)
                Text(viewModel.launchdStatusTitle(job))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.brainBarTextMuted)
            }
            Text("Configured: \(setting.enabled ? "Enabled" : "Disabled")")
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(Color.brainBarTextMuted)
        }
        .padding(.vertical, 7)
    }

    private func loadStateColor(_ state: BrainLayerLaunchdLoadState) -> Color {
        switch state {
        case .running: BrainBarStateTheme.active.theme.swiftUIColor
        case .loaded: BrainBarStateTheme.loading.theme.swiftUIColor
        case .unloaded: BrainBarStateTheme.idle.theme.swiftUIColor
        case .unknown: BrainBarStateTheme.degraded.theme.swiftUIColor
        case .probeError, .ownerUnknown, .paused, .failed: BrainBarStateTheme.degraded.theme.swiftUIColor
        }
    }
}
