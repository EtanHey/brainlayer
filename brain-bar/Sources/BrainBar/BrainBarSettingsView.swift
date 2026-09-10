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
    @Published private(set) var backupRunStatus: BrainLayerBackupCoordinator.Status = .idle

    private let store: BrainLayerConfigStore
    private let launchdStatusProvider: any BrainLayerLaunchdStatusSampling
    private let runtimeStatusProvider: any BrainLayerActiveRuntimeSampling
    private let backupCoordinator: BrainLayerBackupCoordinator
    private let now: @Sendable () -> Date
    private var previousConfigForLastSaveReceipt: BrainLayerConfig?

    init(
        store: BrainLayerConfigStore = BrainLayerConfigStore(),
        launchdStatusProvider: any BrainLayerLaunchdStatusSampling = BrainLayerLaunchdStatusProvider(),
        runtimeStatusProvider: any BrainLayerActiveRuntimeSampling = UnknownBrainLayerActiveRuntimeProvider(),
        backupCoordinator: BrainLayerBackupCoordinator = BrainLayerBackupCoordinator(),
        initialLaunchdStates: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] = [:],
        refreshStatusOnLoad: Bool = true,
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.store = store
        self.launchdStatusProvider = launchdStatusProvider
        self.runtimeStatusProvider = runtimeStatusProvider
        self.backupCoordinator = backupCoordinator
        self.now = now
        activeRuntimeObservation = runtimeStatusProvider.sample()
        do {
            let document = try store.loadDocument()
            config = document.config
            onePasswordReference = document.config.googleAPIKey.opReference
            backendDraft = document.config.enrichmentBackend
        } catch {
            config = .defaultConfig
            onePasswordReference = BrainLayerConfig.defaultConfig.googleAPIKey.opReference
            backendDraft = BrainLayerConfig.defaultConfig.enrichmentBackend
            errorMessage = error.localizedDescription
        }
        applyLaunchdStates(initialLaunchdStates)
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
        guard confirmGoogleAPIKeyOverwriteIfNeeded() else { return }
        if updateConfig({ $0.googleAPIKey = .plain(value) }) {
            pendingPlainAPIKey = ""
        }
    }

    func storeOnePasswordReference() {
        let reference = onePasswordReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !reference.isEmpty else { return }
        if config.googleAPIKey != .onePasswordReference(reference) {
            guard confirmGoogleAPIKeyOverwriteIfNeeded() else { return }
        }
        _ = updateConfig { $0.googleAPIKey = .onePasswordReference(reference) }
    }

    func clearGoogleAPIKey() {
        _ = updateConfig { $0.googleAPIKey = .missing }
    }

    func setJob(_ job: BrainLayerLaunchdJob, enabled: Bool) {
        updateConfig {
            $0.launchdJobs[job, default: BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown)].enabled = enabled
        }
    }

    func setBackupEnabled(_ enabled: Bool) {
        guard updateConfig({ $0.backupEnabled = enabled }) else { return }
        if enabled {
            startBackupIfEnabledAfterSave()
        } else {
            backupCoordinator.cancel()
            backupRunStatus = .idle
        }
    }

    func chooseBackupDestination() {
        let panel = NSOpenPanel()
        panel.title = "Choose Backup Folder"
        panel.message = "Choose an iCloud or mounted-drive folder. Backup status is unavailable until the backup service is connected."
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let identity = BrainLayerBackupDestinationIdentity(path: url.path) else {
            errorMessage = "The selected backup folder could not be identified."
            return
        }
        let knownRoots = BrainLayerBackupKnownRoots.paths()
        var includedRoots: [String] = []
        var missingRoots: [String] = []
        for root in knownRoots {
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue {
                includedRoots.append(root.standardizedFileURL.path)
            } else {
                missingRoots.append(root.standardizedFileURL.path)
            }
        }
        guard !includedRoots.isEmpty else {
            errorMessage = "No known transcript source folder is available."
            return
        }
        guard updateConfig({ config in
            config.backupDestinationPath = url.path
            config.backupDestinationIdentity = identity
            config.backupTranscriptRoots = includedRoots
            config.backupMissingOptionalRoots = missingRoots
        }) else { return }
        startBackupIfEnabledAfterSave()
    }

    func clearBackupDestination() {
        guard updateConfig({ config in
            config.backupDestinationPath = nil
            config.backupDestinationIdentity = nil
            config.backupEnabled = false
        }) else { return }
        backupCoordinator.cancel()
        backupRunStatus = .idle
    }

    private func startBackupIfEnabledAfterSave() {
        guard config.backupEnabled else { return }
        backupCoordinator.startIfConfigured(config: config) { [weak self] status in
            self?.backupRunStatus = status
        }
    }

    func refreshLaunchdStatus() {
        isRefreshingLaunchdStatus = true
        let provider = launchdStatusProvider
        Task {
            let states = await Task.detached {
                provider.sample()
            }.value
            applyLaunchdStates(states)
            activeRuntimeObservation = runtimeStatusProvider.sample()
            refreshLastSaveReceiptActiveState()
            isRefreshingLaunchdStatus = false
        }
    }

    private func applyLaunchdStates(_ states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]) {
        for (job, state) in states {
            config.launchdJobs[job, default: BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown)].loadState = state
        }
    }

    @discardableResult
    private func updateConfig(_ apply: (inout BrainLayerConfig) -> Void) -> Bool {
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
        let alert = NSAlert()
        alert.messageText = "Replace existing Gemini API key?"
        alert.informativeText = "BrainBar will update the BrainLayer config file without displaying the current value."
        alert.addButton(withTitle: "Replace")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
    }
}

struct BrainBarSettingsView: View {
    @StateObject var viewModel: BrainBarSettingsViewModel
    @FocusState private var focusedField: Field?

    enum Field {
        case plainKey
        case opReference
        case backend
    }

    init(viewModel: BrainBarSettingsViewModel = BrainBarSettingsViewModel()) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                if let errorMessage = viewModel.errorMessage {
                    errorBanner(errorMessage)
                }
                BrainBarSettingsPanel(title: "Enrichment") {
                    enrichmentControls
                }
                BrainBarSettingsPanel(title: "Backups") {
                    backupControls
                }
                if let receipt = viewModel.lastSaveReceipt {
                    BrainBarSettingsPanel(title: "Last save receipt") {
                        saveReceipt(receipt)
                    }
                }
                BrainBarSettingsPanel(title: "Gemini API Key") {
                    secretControls
                }
                BrainBarSettingsPanel(title: "System Jobs") {
                    jobsGrid
                }
            }
            .padding(22)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .frame(width: 700)
        .frame(minHeight: 640)
        .background(Color.brainBarBackgroundBase)
        .foregroundStyle(Color.brainBarTextPrimary)
        .environment(\.colorScheme, .dark)
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: "gearshape.2")
                .font(.system(size: 28, weight: .semibold))
                .foregroundStyle(Color.brainBarAccentBright)
            VStack(alignment: .leading, spacing: 3) {
                Text("BrainLayer Settings")
                    .font(.system(size: 22, weight: .semibold))
                Text(BrainLayerConfigStore.defaultConfigURL().path)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.brainBarTextMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            Button {
                viewModel.refreshLaunchdStatus()
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .controlSize(.small)
            .disabled(viewModel.isRefreshingLaunchdStatus)
        }
    }

    private var enrichmentControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(
                "Enable automatic background enrichment",
                isOn: Binding(
                    get: { viewModel.config.enrichmentEnabled },
                    set: { viewModel.setEnrichmentEnabled($0) }
                )
            )
            Toggle(
                "Enable BrainLayer jobs",
                isOn: Binding(
                    get: { viewModel.config.systemEnabled },
                    set: { viewModel.setSystemEnabled($0) }
                )
            )

            Picker(
                "Mode",
                selection: Binding(
                    get: { viewModel.config.enrichmentMode },
                    set: { viewModel.setEnrichmentMode($0) }
                )
            ) {
                ForEach(BrainLayerEnrichmentMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Picker(
                "Provider",
                selection: Binding(
                    get: { viewModel.config.enrichmentProvider },
                    set: { viewModel.setEnrichmentProvider($0) }
                )
            ) {
                ForEach(BrainLayerEnrichmentProvider.allCases) { provider in
                    Text(provider.isWiredToday ? provider.title : "\(provider.title) — Unavailable")
                        .tag(provider)
                        .disabled(!provider.isWiredToday)
                }
            }

            if let reason = viewModel.config.enrichmentProvider.unavailableReason {
                Label(
                    "Configured provider unavailable: \(reason)",
                    systemImage: "exclamationmark.triangle"
                )
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color(nsColor: BrainBarStateTheme.error.theme.color))
            } else {
                Text("OpenAI and Anthropic are unavailable because this build has no runtime integration for them.")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color.brainBarTextMuted)
            }

            HStack {
                Text("Backend")
                    .foregroundStyle(Color.brainBarTextSecondary)
                    .frame(width: 110, alignment: .leading)
                TextField(
                    "gemini",
                    text: $viewModel.backendDraft
                )
                .focused($focusedField, equals: .backend)
                .textFieldStyle(.roundedBorder)
                .onSubmit {
                    viewModel.commitBackendDraft()
                }
                Button("Save") {
                    viewModel.commitBackendDraft()
                }
                .disabled(
                    viewModel.backendDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
                        viewModel.backendDraft.trimmingCharacters(in: .whitespacesAndNewlines) == viewModel.config.enrichmentBackend
                )
            }

            Divider()
                .overlay(Color.brainBarBorderSoft)

            settingsTruthRow(
                label: "Configured",
                value: BrainLayerActiveRuntimeValues(config: viewModel.config).summary
            )
            settingsTruthRow(label: "Active", value: viewModel.activeRuntimeObservation.summary)
        }
    }

    private var backupControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(
                "Enable backup configuration",
                isOn: Binding(
                    get: { viewModel.config.backupEnabled },
                    set: { viewModel.setBackupEnabled($0) }
                )
            )
            HStack {
                Text("Destination")
                    .foregroundStyle(Color.brainBarTextSecondary)
                    .frame(width: 110, alignment: .leading)
                Text(viewModel.config.backupDestinationPath ?? "Not selected")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.brainBarTextSecondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
                Spacer()
                Button("Choose Folder") {
                    viewModel.chooseBackupDestination()
                }
                Button("Clear") {
                    viewModel.clearBackupDestination()
                }
                .disabled(viewModel.config.backupDestinationPath == nil)
            }
            settingsTruthRow(label: "Configuration", value: viewModel.config.backupConfigurationState.title)
            settingsTruthRow(label: "Run", value: viewModel.backupRunStatus.title)
            settingsTruthRow(
                label: "Unavailable optional sources",
                value: String(viewModel.config.backupMissingOptionalRoots.count)
            )
            Text("Backup status is unavailable until the backup service is connected; this setting does not claim cloud protection.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextMuted)
        }
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
            Text(value)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.brainBarTextSecondary)
                .textSelection(.enabled)
        }
    }

    private var secretControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(viewModel.config.googleAPIKey.displayText, systemImage: "key")
                    .foregroundStyle(Color.brainBarTextSecondary)
                Spacer()
                Button("Clear") {
                    viewModel.clearGoogleAPIKey()
                }
                .disabled(viewModel.config.googleAPIKey.kind == .missing)
            }

            HStack {
                Text("1Password")
                    .foregroundStyle(Color.brainBarTextSecondary)
                    .frame(width: 110, alignment: .leading)
                TextField("op://Private/Google AI/Gemini API key", text: $viewModel.onePasswordReference)
                    .focused($focusedField, equals: .opReference)
                    .textFieldStyle(.roundedBorder)
                Button("Use") {
                    viewModel.storeOnePasswordReference()
                }
            }

            HStack {
                Text("Plain key")
                    .foregroundStyle(Color.brainBarTextSecondary)
                    .frame(width: 110, alignment: .leading)
                SecureField("Paste new key", text: $viewModel.pendingPlainAPIKey)
                    .focused($focusedField, equals: .plainKey)
                    .textFieldStyle(.roundedBorder)
                Button("Store") {
                    viewModel.storePlainAPIKey()
                }
                .disabled(viewModel.pendingPlainAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    private var jobsGrid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 205), spacing: 12)], alignment: .leading, spacing: 12) {
            ForEach(BrainLayerLaunchdJob.allCases) { job in
                BrainBarJobToggle(job: job, viewModel: viewModel)
            }
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

private struct BrainBarSettingsPanel<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(Color.brainBarTextPrimary)
            VStack(alignment: .leading, spacing: 12) {
                content
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color.brainBarGlassPrimary)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color.brainBarBorderEdge, lineWidth: 1)
        )
    }
}

private struct BrainBarJobToggle: View {
    let job: BrainLayerLaunchdJob
    @ObservedObject var viewModel: BrainBarSettingsViewModel

    var body: some View {
        let setting = viewModel.config.launchdJobs[job] ?? BrainLayerLaunchdJobSetting(enabled: true, loadState: .unknown)
        VStack(alignment: .leading, spacing: 7) {
            Toggle(
                job.title,
                isOn: Binding(
                    get: { viewModel.config.launchdJobs[job]?.enabled ?? true },
                    set: { viewModel.setJob(job, enabled: $0) }
                )
            )
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
        .padding(12)
        .background(Color.brainBarGlassSecondary)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color.brainBarBorderSoft, lineWidth: 1)
        )
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
