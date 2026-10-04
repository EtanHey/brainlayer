import AppKit
import SwiftUI

@MainActor
final class BrainBarSettingsViewModel: ObservableObject {
    @Published var config: BrainLayerConfig
    @Published var pendingPlainAPIKey = ""
    @Published var onePasswordReference: String
    @Published var backendDraft: String
    @Published var errorMessage: String?
    @Published var isRefreshingLaunchdStatus = false
    @Published private(set) var activeRuntimeObservation: BrainLayerActiveRuntimeObservation
    @Published private(set) var lastSaveReceipt: BrainLayerSettingsSaveReceipt?
    @Published private(set) var observabilityResult: ObservabilityReadResult
    @Published private(set) var launchdObservations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]
    @Published private(set) var configReadSucceeded = true
    @Published private(set) var embeddingProcess: BrainBarEmbeddingProcessState
    /// watcher-health.json beside the resolved DB: the same file the Dashboard reads (#966).
    @Published private(set) var watcherHealth: WatcherHealthFileRead?

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
            now: now()
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

    init(
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
        initialObservabilityResult: ObservabilityReadResult = .unreadable("Backup status unavailable."),
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
        workspace: any BrainBarWorkspaceActing = BrainBarWorkspace()
    ) {
        self.store = store
        self.launchdStatusProvider = launchdStatusProvider
        self.runtimeStatusProvider = runtimeStatusProvider
        self.embeddingResidencyProbe = embeddingResidencyProbe
        embeddingProcess = initialEmbeddingProcess
        self.now = now
        self.observabilityURL = observabilityURL
        self.observabilityRead = observabilityRead
        self.confirmAPIKeyOverwrite = confirmAPIKeyOverwrite
        self.watcherHealthURL = watcherHealthURL
        self.watcherHealthRead = watcherHealthRead
        watcherHealth = initialWatcherHealth
        self.backupSources = backupSources
        self.workspace = workspace
        backupSchedules = initialBackupSchedules
        maintenanceEvidence = initialMaintenanceEvidence
        observabilityResult = initialObservabilityResult
        launchdObservations = initialLaunchdObservations.isEmpty
            ? initialLaunchdStates.mapValues(BrainLayerLaunchdJobObservation.stateOnly)
            : initialLaunchdObservations
        activeRuntimeObservation = runtimeStatusProvider.sample()
        do {
            let document = try store.loadDocument()
            config = document.config
            onePasswordReference = document.config.googleAPIKey.opReference
            backendDraft = document.config.enrichmentBackend
        } catch {
            configReadSucceeded = false
            config = .defaultConfig
            onePasswordReference = BrainLayerConfig.defaultConfig.googleAPIKey.opReference
            backendDraft = BrainLayerConfig.defaultConfig.enrichmentBackend
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
    }

    /// Re-reads watcher-health.json off the main actor; only the newest request publishes.
    func refreshWatcherHealth() {
        guard let watcherHealthURL else { return }
        watcherHealthGeneration &+= 1
        let generation = watcherHealthGeneration
        let read = watcherHealthRead
        Task {
            let result = await Task.detached { read(watcherHealthURL) }.value
            guard generation == watcherHealthGeneration else { return }
            watcherHealth = result
        }
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
        }
    }

    func revealBackup(_ row: BrainBarBackupScheduleRow) { row.reveal(using: workspace) }
    func copyBackupPath(_ row: BrainBarBackupScheduleRow) { row.copyPath(using: workspace) }

    func setEnrichmentEnabled(_ enabled: Bool) {
        updateConfig { $0.enrichmentEnabled = enabled }
    }

    func setSystemEnabled(_ enabled: Bool) {
        updateConfig { $0.systemEnabled = enabled }
    }

    func setEnrichmentMode(_ mode: BrainLayerEnrichmentMode) {
        updateConfig { $0.enrichmentMode = mode }
    }

    func setEnrichmentProvider(_ provider: BrainLayerEnrichmentProvider) {
        guard provider.isWiredToday else {
            recordValidationFailure(
                "\(provider.title) cannot be activated because its runtime integration is unavailable."
            )
            return
        }
        updateConfig { nextConfig in
            nextConfig.enrichmentProvider = provider
            if provider == .gemini,
               nextConfig.enrichmentBackend.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                nextConfig.enrichmentBackend = "gemini"
            }
        }
    }

    func commitBackendDraft() {
        let backend = backendDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard backend != config.enrichmentBackend else {
            backendDraft = config.enrichmentBackend
            return
        }
        updateConfig { $0.enrichmentBackend = backend }
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

    func groupStatus(_ group: BrainLayerLaunchdJobGroup) -> BrainLayerLaunchdGroupStatus {
        group.status(
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

    func refreshLaunchdStatus() {
        isRefreshingLaunchdStatus = true
        refreshEmbeddingResidency()
        refreshWatcherHealth()
        refreshBackupSchedules()
        let provider = launchdStatusProvider
        Task {
            let observations = await Task.detached {
                provider.sampleActivity()
            }.value
            launchdObservations = observations
            applyLaunchdStates(observations.mapValues(\.loadState))
            activeRuntimeObservation = runtimeStatusProvider.sample()
            refreshLastSaveReceiptActiveState()
            isRefreshingLaunchdStatus = false
        }
    }

    var backupStatus: ObservabilityBackupStatus? {
        guard case let .readable(document) = observabilityResult,
              document.backups.state == "measured" else { return nil }
        return ObservabilityPresentation.backupStatus(for: document.backups)
    }

    /// The Backups page's one verdict (#1029 review B1), from the same Drive presentation the card
    /// shows, the Backups job group and the backup status lines.
    func backupsHealth(drive: DriveAuthPresentation?) -> BrainBarBackupsHealth {
        BrainBarBackupsHealth.derive(
            job: groupStatus(.backups),
            drive: drive,
            status: backupStatus,
            statusUnavailableReason: backupStatusReason ?? "Backup status is unmeasurable."
        )
    }

    /// A job alert (#1031) in its own words, shown once on the Backups page as its own card.
    var jobAlert: String? {
        guard let status = backupStatus, status.errorIsJobAlert, status.error?.tone == .red else { return nil }
        return status.error?.text
    }

    /// The backup status rows without the job alert, which the alert card already shows.
    var backupStatusLines: [ObservabilityStatusLine]? {
        guard let status = backupStatus else { return nil }
        return status.lines.filter { line in !(status.errorIsJobAlert && line == status.error) }
    }

    /// The line under the Backups badge: the verdict's reason, unless it is the job alert the
    /// alert card above it already shows.
    func backupsBadgeReason(drive: DriveAuthPresentation?) -> String? {
        let reason = backupsHealth(drive: drive).reason
        return reason == jobAlert ? nil : reason
    }

    /// Show log for the job alert: the failing job's log, named by the live job-alert state.
    func showJobAlertLog() {
        guard let alert = jobAlert, case let .readable(document) = observabilityResult else { return }
        let key = BrainBarJobAlerts.read(url: BrainBarJobAlerts.url(dbPath: document.dbPath))?.key(for: alert)
        let paths = backupSources?.paths ?? .live(databasePath: document.dbPath)
        BrainBarJobAlerts.showLog(forKey: key, paths: paths, workspace: workspace)
    }

    var backupStatusReason: String? {
        switch observabilityResult {
        case let .unreadable(value):
            value
        case let .readable(document):
            if document.backups.state == "measured" { nil }
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
            if !preservingDrafts || backendDraft == previous.enrichmentBackend {
                backendDraft = loaded.enrichmentBackend
            }
            applyLaunchdStates(launchdObservations.mapValues(\.loadState))
            errorMessage = nil
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
        let read = observabilityRead
        observabilityTask = Task { [weak self] in
            let result = await read(observabilityURL)
            guard !Task.isCancelled else { return }
            guard let self else { return }
            observabilityResult = result
        }
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
        let validation = BrainLayerConfigValidator.validate(nextConfig, previousConfig: previousConfig)
        guard validation == .passed else {
            if case let .failed(message) = validation {
                recordValidationFailure(message)
            }
            return false
        }

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
            onePasswordReference = nextConfig.googleAPIKey.opReference
            backendDraft = nextConfig.enrichmentBackend
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
        backendDraft = persistedConfig.enrichmentBackend
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
        if previous.googleAPIKey != configured.googleAPIKey ||
            previous.enrichmentEnabled != configured.enrichmentEnabled ||
            previous.enrichmentMode != configured.enrichmentMode ||
            previous.enrichmentProvider != configured.enrichmentProvider ||
            previous.enrichmentBackend != configured.enrichmentBackend ||
            previous.tuningValues != configured.tuningValues {
            requirements.append(.enrichment)
        }
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
        if previous.googleAPIKey != configured.googleAPIKey || previous.tuningValues != configured.tuningValues {
            return .unknown("Secret and tuning reload state is not observable.")
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

        let hasRuntimeConfigRequirement = requirements.contains(.enrichment) || requirements.contains(.systemJobs)
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
    case watcher(WatcherHealthStatus)
    case systemOff
    case unavailable

    var title: String {
        switch self {
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

    init(config: BrainLayerConfig?, watcher: WatcherHealthStatus?, now: Date = Date()) {
        guard let config else {
            state = .unavailable
            detail = nil
            locality = "Memory on this Mac · Enrichment unknown · Backups unknown"
            showsLock = false
            symbol = "questionmark.circle"
            return
        }

        if !config.systemEnabled {
            state = .systemOff
            detail = nil
        } else if let watcher {
            state = .watcher(watcher)
            detail = watcher.reasonText(now: now)
        } else {
            state = .unavailable
            detail = nil
        }

        let enrichment: String
        let enrichmentCloud: Bool
        let enrichmentOff = config.enrichmentIsOff
        if enrichmentOff {
            enrichment = "Enrichment off"
            enrichmentCloud = false
        } else if config.launchdJobs[.enrichment]?.enabled == true {
            // The realtime enrichment launchd job invokes enrich_realtime,
            // which uses Gemini regardless of BACKEND/MODE in the env file.
            enrichment = "Enrichment → Gemini"
            enrichmentCloud = true
        } else {
            enrichment = "Enrichment unknown"
            enrichmentCloud = false
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
        locality = "Memory on this Mac · \(enrichment) · \(backups)"
        showsLock = enrichmentOff && backupsOff
        symbol = showsLock ? "lock" : (enrichmentCloud || driveConfigured ? "icloud" : "questionmark.circle")
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

    init(databasePath: String, activationRevision: Int = 0,
         navigation: BrainBarSettingsNavigation = BrainBarSettingsNavigation(), showsSidebar: Bool = true) {
        self.activationRevision = activationRevision
        self.showsSidebar = showsSidebar
        _navigation = StateObject(wrappedValue: navigation)
        _viewModel = StateObject(wrappedValue: BrainBarSettingsViewModel(
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

    init(viewModel: BrainBarSettingsViewModel, navigation: BrainBarSettingsNavigation, showsSidebar: Bool) {
        activationRevision = 0
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
        .onChange(of: activationRevision) { _, _ in
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
            VStack(alignment: .leading, spacing: 3) {
                Text(navigation.selected.title)
                    .font(.system(size: 25, weight: .semibold))
                Text(BrainLayerConfigStore.defaultConfigURL().path)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.brainBarTextMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            Button {
                viewModel.refreshAllStatus()
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .controlSize(.small)
            .disabled(viewModel.isRefreshingLaunchdStatus)
        }
    }

    @ViewBuilder
    private var sectionContent: some View {
        switch navigation.selected {
        case .jobs:
            VStack(alignment: .leading, spacing: 16) {
                jobsGrid
            }
        case .backups:
            VStack(alignment: .leading, spacing: 16) {
                if let alert = viewModel.jobAlert {
                    BrainBarJobAlertCard(alert: alert) { viewModel.showJobAlertLog() }
                }
                if let driveAuth {
                    BrainBarDriveAwareBackupsGroup(viewModel: viewModel, driveAuth: driveAuth)
                } else {
                    BrainBarJobGroupCard(group: .backups, viewModel: viewModel, backupsHealth: viewModel.backupsHealth(drive: nil))
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
            }
        }
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
        if !viewModel.backupSchedules.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                sectionHeading("Schedule")
                ForEach(viewModel.backupSchedules) { row in
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
        if let lines = viewModel.backupStatusLines {
            VStack(alignment: .leading, spacing: 8) {
                ObservabilityStatusRows(lines: lines, textColor: Color.brainBarTextSecondary)
                    .font(.system(size: 11, weight: .medium))
            }
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
    let showLog: () -> Void

    var body: some View {
        let tone = Color(nsColor: BrainBarDesignTokens.Colors.statusError)
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
            BrainBarJobGroupCard(
                group: .backups,
                viewModel: viewModel,
                backupsHealth: viewModel.backupsHealth(drive: driveAuth.presentation(formatDate: BrainBarDriveAuthFormat.date))
            )
        }
    }
}

private struct BrainBarJobGroupCard: View {
    let group: BrainLayerLaunchdJobGroup
    @ObservedObject var viewModel: BrainBarSettingsViewModel
    /// The Backups page passes its combined verdict; other groups show launchd health alone.
    var backupsHealth: BrainBarBackupsHealth?

    static func symbol(_ health: BrainLayerLaunchdGroupHealth) -> String {
        switch health {
        case .healthy: "checkmark.circle.fill"
        case .awaitingRun: "clock.fill"
        case .unhealthy: "exclamationmark.triangle.fill"
        case .unknown: "questionmark.circle.fill"
        case .skipped: "forward.end.circle.fill"
        }
    }

    static func color(_ health: BrainLayerLaunchdGroupHealth) -> Color {
        switch health {
        case .healthy: BrainBarStateTheme.active.theme.swiftUIColor
        case .awaitingRun: BrainBarStateTheme.loading.theme.swiftUIColor
        case .unhealthy: BrainBarStateTheme.error.theme.swiftUIColor
        case .unknown, .skipped: Color.brainBarTextMuted
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
        case .healthy: (Self.symbol(.healthy), Self.color(.healthy))
        case .awaitingRun: (Self.symbol(.awaitingRun), Self.color(.awaitingRun))
        case .expiring: ("clock.badge.exclamationmark.fill", Color(nsColor: BrainBarDesignTokens.Colors.statusAttention))
        case .attention: (Self.symbol(.unhealthy), Self.color(.unhealthy))
        case .unknown: (Self.symbol(.unknown), Self.color(.unknown))
        }
        return Badge(
            title: backupsHealth.badge.title, symbol: symbol, color: color,
            // The job alert has its own card on this page; say it once (lead ruling 2026-10-04).
            reason: backupsHealth.reason == viewModel.jobAlert ? nil : backupsHealth.reason,
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
            groupTiming(label: "LAST RUN", value: status.lastRunText)
            groupTiming(label: "NEXT RUN", value: status.nextRunText)
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
                    .fill(loadStateColor(setting.loadState))
                    .frame(width: 7, height: 7)
                Text(setting.loadState.title)
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
        case .probeError: BrainBarStateTheme.degraded.theme.swiftUIColor
        }
    }
}
