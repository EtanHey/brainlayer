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

    var footerPresentation: BrainBarSettingsFooterPresentation {
        BrainBarSettingsFooterPresentation(
            config: configReadSucceeded ? config : nil,
            watcher: config.launchdJobs[.watch]?.loadState
        )
    }

    static var modelResidencyPresentation: BrainBarModelResidencyPresentation { .unavailable }

    private let store: BrainLayerConfigStore
    private let launchdStatusProvider: any BrainLayerLaunchdStatusSampling
    private let runtimeStatusProvider: any BrainLayerActiveRuntimeSampling
    private let now: @Sendable () -> Date
    private let observabilityURL: URL?
    private let observabilityRead: @Sendable (URL) async -> ObservabilityReadResult
    private let confirmAPIKeyOverwrite: (() -> Bool)?
    private var previousConfigForLastSaveReceipt: BrainLayerConfig?
    private var observabilityTask: Task<Void, Never>?

    init(
        store: BrainLayerConfigStore = BrainLayerConfigStore(),
        launchdStatusProvider: any BrainLayerLaunchdStatusSampling = BrainLayerLaunchdStatusProvider(),
        runtimeStatusProvider: any BrainLayerActiveRuntimeSampling = UnknownBrainLayerActiveRuntimeProvider(),
        initialLaunchdStates: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] = [:],
        initialLaunchdObservations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] = [:],
        refreshStatusOnLoad: Bool = true,
        now: @escaping @Sendable () -> Date = Date.init,
        observabilityURL: URL? = nil,
        initialObservabilityResult: ObservabilityReadResult = .unreadable("Backup status unavailable."),
        confirmAPIKeyOverwrite: (() -> Bool)? = nil,
        observabilityRead: @escaping @Sendable (URL) async -> ObservabilityReadResult = { url in
            await Task.detached { ObservabilityReader.read(url: url) }.value
        }
    ) {
        self.store = store
        self.launchdStatusProvider = launchdStatusProvider
        self.runtimeStatusProvider = runtimeStatusProvider
        self.now = now
        self.observabilityURL = observabilityURL
        self.observabilityRead = observabilityRead
        self.confirmAPIKeyOverwrite = confirmAPIKeyOverwrite
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
            formatDate: DashboardMetricFormatter.jobDateTimeString
        )
    }

    func refreshLaunchdStatus() {
        isRefreshingLaunchdStatus = true
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
    case watcherRunning, systemOff, unavailable

    var title: String {
        switch self {
        case .watcherRunning: "Watcher running"
        case .systemOff: "System off"
        case .unavailable: "Status unavailable"
        }
    }
}

struct BrainBarSettingsFooterPresentation {
    let state: BrainBarSettingsFooterState
    let locality: String
    let showsLock: Bool
    let symbol: String

    init(config: BrainLayerConfig?, watcher: BrainLayerLaunchdLoadState?) {
        guard let config else {
            state = .unavailable
            locality = "Memory on this Mac · Enrichment unknown · Backups unknown"
            showsLock = false
            symbol = "questionmark.circle"
            return
        }

        if !config.systemEnabled {
            state = .systemOff
        } else if watcher == .running {
            state = .watcherRunning
        } else {
            state = .unavailable
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

struct BrainBarModelResidencyPresentation {
    let modelName: String
    let status: String
    let memory: String

    // No model-specific loaded state or resident bytes are exposed to BrainBar.
    // The daemon RSS measures the entire process, not the embedding model.
    static let unavailable = Self(
        modelName: "Name unavailable",
        status: "Residency unavailable",
        memory: "Unavailable"
    )
}

enum BrainBarSettingsSection: String, CaseIterable, Identifiable {
    case general, jobs, backups, advanced

    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var symbol: String {
        switch self {
        case .general: "slider.horizontal.3"
        case .jobs: "clock.arrow.circlepath"
        case .backups: "externaldrive"
        case .advanced: "gearshape.2"
        }
    }
    var groups: [BrainLayerLaunchdJobGroup] {
        switch self {
        case .jobs: [.ingest, .maintenance]
        case .backups: [.backups]
        case .general, .advanced: []
        }
    }
    var advancedJobs: [BrainLayerLaunchdJob] { self == .advanced ? BrainLayerLaunchdJobGroup.advancedJobs : [] }
}

@MainActor
final class BrainBarSettingsNavigation: ObservableObject {
    @Published private(set) var selected: BrainBarSettingsSection = .general
    init(selected: BrainBarSettingsSection = .general) { self.selected = selected }
    func select(_ section: BrainBarSettingsSection) { selected = section }
}

struct BrainBarSettingsView: View {
    @StateObject var viewModel: BrainBarSettingsViewModel
    @StateObject private var navigation = BrainBarSettingsNavigation()
    private let activationRevision: Int

    static func observabilityURL(
        databasePath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        ObservabilityReader.url(dbPath: databasePath, environment: environment)
    }

    init(databasePath: String, activationRevision: Int = 0,
         navigation: BrainBarSettingsNavigation = BrainBarSettingsNavigation()) {
        self.activationRevision = activationRevision
        _navigation = StateObject(wrappedValue: navigation)
        _viewModel = StateObject(wrappedValue: BrainBarSettingsViewModel(
            observabilityURL: Self.observabilityURL(databasePath: databasePath)
        ))
    }

    init(viewModel: BrainBarSettingsViewModel, initialSection: BrainBarSettingsSection = .general) {
        activationRevision = 0
        _viewModel = StateObject(wrappedValue: viewModel)
        _navigation = StateObject(wrappedValue: BrainBarSettingsNavigation(selected: initialSection))
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                sidebar
                    .frame(width: 214)
                    .frame(maxHeight: .infinity, alignment: .top)
                    .background(Color.brainBarGlassSecondary)
                Rectangle().fill(Color.brainBarBorderSoft).frame(width: 1)
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
                    .fill(presentation.state == .watcherRunning
                        ? BrainBarStateTheme.active.theme.swiftUIColor : Color.brainBarTextMuted)
                    .frame(width: 6, height: 6)
                Text(presentation.state.title)
                    .font(.system(size: 11, weight: .semibold))
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
        case .general:
            VStack(alignment: .leading, spacing: 12) {
                sectionHeading("BrainBar")
                Text("Monitor memory activity on Dashboard. Manage services in Jobs and Backups.")
                    .foregroundStyle(Color.brainBarTextMuted)
            }
        case .jobs:
            VStack(alignment: .leading, spacing: 16) {
                jobsGrid
            }
        case .backups:
            VStack(alignment: .leading, spacing: 16) {
                BrainBarJobGroupCard(group: .backups, viewModel: viewModel)
                Divider()
                backupStatus
            }
        case .advanced:
            VStack(alignment: .leading, spacing: 16) {
                sectionHeading("Embedding model")
                let residency = BrainBarSettingsViewModel.modelResidencyPresentation
                settingsTruthRow(label: "Model", value: residency.modelName)
                settingsTruthRow(label: "Status", value: residency.status)
                settingsTruthRow(label: "Resident memory", value: residency.memory)
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
    private var backupStatus: some View {
        if let status = viewModel.backupStatus {
            VStack(alignment: .leading, spacing: 8) {
                ObservabilityStatusRows(lines: status.lines, textColor: Color.brainBarTextSecondary)
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

private struct BrainBarJobGroupCard: View {
    let group: BrainLayerLaunchdJobGroup
    @ObservedObject var viewModel: BrainBarSettingsViewModel

    var body: some View {
        let status = viewModel.groupStatus(group)
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(group.title).font(.system(size: 13, weight: .semibold))
                Spacer()
                Label(
                    status.health.title,
                    systemImage: status.health == .healthy ? "checkmark.circle.fill" :
                        status.health == .awaitingRun ? "clock.fill" : "exclamationmark.triangle.fill"
                )
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(
                        status.health == .healthy
                            ? BrainBarStateTheme.active.theme.swiftUIColor
                            : status.health == .awaitingRun
                                ? BrainBarStateTheme.loading.theme.swiftUIColor
                                : BrainBarStateTheme.error.theme.swiftUIColor
                    )
                Toggle(group.title, isOn: Binding(
                    get: { viewModel.isGroupEnabled(group) },
                    set: { viewModel.setGroup(group, enabled: $0) }
                )).labelsHidden().toggleStyle(.switch).controlSize(.small)
            }
            Text(group.summary)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextMuted)
            if let reason = status.attentionReason {
                Text(reason)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(BrainBarStateTheme.error.theme.swiftUIColor)
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
