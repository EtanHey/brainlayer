import XCTest
@testable import BrainBar

final class BrainBarSettingsViewModelTests: XCTestCase {
    private let fixedNow = Date(timeIntervalSince1970: 1_784_466_000)

    func testFooterQualifiesCloudAndLocalConfiguration() {
        var config = BrainLayerConfig.defaultConfig
        config.enrichmentMode = .local
        config.enrichmentBackend = "mlx"
        let cloud = BrainBarSettingsFooterPresentation(config: config, watcher: .running)
        XCTAssertEqual(cloud.state, .watcherRunning)
        XCTAssertTrue(cloud.locality.contains("Memory on this Mac"))
        XCTAssertTrue(cloud.locality.contains("Enrichment → Gemini"))
        XCTAssertFalse(cloud.locality.contains("Local enrichment"))
        XCTAssertFalse(cloud.showsLock)
        XCTAssertEqual(cloud.symbol, "icloud")

        config.launchdJobs[.backupDaily]?.enabled = false
        config.launchdJobs[.jsonlBackup]?.enabled = false
        let weeklyOnly = BrainBarSettingsFooterPresentation(config: config, watcher: .unknown)
        XCTAssertTrue(weeklyOnly.locality.contains("Backups → Drive"))
        XCTAssertFalse(weeklyOnly.showsLock)

        config.enrichmentEnabled = false
        config.launchdJobs[.enrichment]?.enabled = false
        config.launchdJobs[.maintenanceWeekly]?.enabled = false
        let local = BrainBarSettingsFooterPresentation(config: config, watcher: .unknown)
        XCTAssertTrue(local.locality.contains("Memory on this Mac"))
        XCTAssertTrue(local.locality.contains("Enrichment off"))
        XCTAssertTrue(local.locality.contains("Backups off"))
        XCTAssertTrue(local.showsLock)
        XCTAssertEqual(local.symbol, "lock")
        XCTAssertEqual(local.state, .unavailable)

        let unreadable = BrainBarSettingsFooterPresentation(config: nil, watcher: nil)
        XCTAssertTrue(unreadable.locality.contains("Enrichment unknown"))
        XCTAssertTrue(unreadable.locality.contains("Backups unknown"))
        XCTAssertFalse(unreadable.showsLock)
        XCTAssertEqual(unreadable.symbol, "questionmark.circle")
    }

    @MainActor
    func testModelResidencyDoesNotInferLoadedStateFromConfigOrDaemonMemory() {
        let residency = BrainBarSettingsViewModel.modelResidencyPresentation
        XCTAssertEqual(residency.modelName, "Name unavailable")
        XCTAssertEqual(residency.status, "Residency unavailable")
        XCTAssertEqual(residency.memory, "Unavailable")
    }

    @MainActor
    func testUnreadableConfigKeepsFooterUnknown() {
        let store = BrainLayerConfigStore(
            configURL: URL(fileURLWithPath: "/nonexistent/settings.env"),
            loadDocumentOverride: { throw CocoaError(.fileReadNoPermission) }
        )
        let viewModel = BrainBarSettingsViewModel(store: store, refreshStatusOnLoad: false)
        XCTAssertFalse(viewModel.configReadSucceeded)
        XCTAssertFalse(viewModel.reloadConfigFromDisk())
        XCTAssertFalse(viewModel.configReadSucceeded)
        let footer = viewModel.footerPresentation
        XCTAssertTrue(footer.locality.contains("Enrichment unknown"))
        XCTAssertTrue(footer.locality.contains("Backups unknown"))
        XCTAssertEqual(footer.state, .unavailable)
    }

    @MainActor
    func testSettingsReadsBackupTruthFromObservabilityDocument() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "observability-main-58849a70", withExtension: "json", subdirectory: "Fixtures"
        ))
        let viewModel = BrainBarSettingsViewModel(
            store: fixture.store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false,
            observabilityURL: url
        )

        let status = try await waitForBackupStatus(viewModel)
        XCTAssertEqual(status.upload.text, "No verified transcript upload on record")
        XCTAssertTrue(status.snapshot.text.contains("→ 2026-09-13.db.gz"))
        XCTAssertEqual(
            status.job.text,
            "Transcript backup (com.brainlayer.jsonl-backup): NOT loaded — parked in .disabled-retention-P0"
        )
        XCTAssertEqual(status.job.tone, .red)
    }

    @MainActor
    func testSettingsPreservesReadableUnmeasurableBackupReason() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("tests/fixtures/observability/golden/missing-launchd-dev.json")
        let viewModel = BrainBarSettingsViewModel(
            store: fixture.store,
            refreshStatusOnLoad: false,
            observabilityURL: url
        )
        let reason = try await waitForBackupReason(viewModel)
        XCTAssertEqual(
            reason,
            "Backup status is unmeasurable — launchd output is empty: launchd/missing-launchd-dev.txt"
        )
    }

    @MainActor
    func testNewestObservabilityRefreshWinsWhenOlderReadFinishesLast() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let reads = SequencedObservabilityReads()
        let url = fixture.root.appendingPathComponent("observability.json")
        let viewModel = BrainBarSettingsViewModel(
            store: fixture.store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false,
            observabilityURL: url,
            observabilityRead: { url in reads.read(url: url) }
        )

        try await reads.waitForCallCount(1)
        viewModel.refreshObservabilityStatus()
        try await reads.waitForCallCount(2)
        reads.release(call: 2)
        try await waitForBackupReason(viewModel, equalTo: "Newest status")
        reads.release(call: 1)
        try await Task.sleep(for: .milliseconds(50))

        XCTAssertEqual(viewModel.backupStatusReason, "Newest status")
    }

    @MainActor
    func testRetrievalToolsSettingPersistsEnabledState() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        defer { BrainBarRetrievalToolsSettings.shared.update(enabled: false) }

        XCTAssertFalse(fixture.viewModel.config.showRetrievalTools)
        fixture.viewModel.setShowRetrievalTools(true)

        XCTAssertTrue(fixture.viewModel.config.showRetrievalTools)
        XCTAssertTrue(try fixture.store.loadDocument().config.showRetrievalTools)
    }

    @MainActor
    func testSettingsReloadPreservesExternalEditBeforeSavingAnotherSetting() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        var externalConfig = try fixture.store.loadDocument().config
        externalConfig.enrichmentBackend = "mlx"
        try fixture.store.save(externalConfig)
        _ = fixture.viewModel.reloadConfigFromDisk()
        fixture.viewModel.setShowRetrievalTools(true)
        let persisted = try fixture.store.loadDocument().config
        XCTAssertEqual(persisted.enrichmentBackend, "mlx")
        XCTAssertTrue(persisted.showRetrievalTools)
    }

    @MainActor
    func testSidebarSelectionDoesNotWriteAndReachesEveryJobGroup() throws {
        let (root, store, _) = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let before = try Data(contentsOf: store.configURL)
        let navigation = BrainBarSettingsNavigation()
        XCTAssertEqual(navigation.selected, .general)
        for section in BrainBarSettingsSection.allCases {
            navigation.select(section)
            XCTAssertEqual(navigation.selected, section)
            XCTAssertEqual(try Data(contentsOf: store.configURL), before)
        }
        XCTAssertEqual(
            Set(BrainBarSettingsSection.jobs.groups + BrainBarSettingsSection.backups.groups),
            Set(BrainLayerLaunchdJobGroup.allCases)
        )
        XCTAssertEqual(BrainBarSettingsSection.advanced.advancedJobs, BrainLayerLaunchdJobGroup.advancedJobs)
    }

    @MainActor
    func testReshowReloadKeepsUncommittedSettingsDrafts() throws {
        let (root, store, viewModel) = try makeFixture()
        defer { try? FileManager.default.removeItem(at: root) }
        viewModel.backendDraft = "draft-backend"
        viewModel.onePasswordReference = "op://draft/reference"
        viewModel.pendingPlainAPIKey = "draft-secret"
        var external = try store.loadDocument().config
        external.showRetrievalTools = true
        try store.save(external)

        XCTAssertTrue(viewModel.reloadConfigFromDisk(preservingDrafts: true))
        XCTAssertTrue(viewModel.config.showRetrievalTools)
        XCTAssertEqual(viewModel.backendDraft, "draft-backend")
        XCTAssertEqual(viewModel.onePasswordReference, "op://draft/reference")
        XCTAssertEqual(viewModel.pendingPlainAPIKey, "draft-secret")
    }

    @MainActor
    func testFailedSaveLeavesDisplayedConfigAtLastPersistedValue() throws {
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-settings-model-\(UUID().uuidString)", isDirectory: false)
        try "not a directory".write(to: tempRoot, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        let store = BrainLayerConfigStore(configURL: tempRoot.appendingPathComponent("brainlayer.env"))
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false
        )

        XCTAssertTrue(viewModel.config.enrichmentEnabled)
        viewModel.setEnrichmentEnabled(false)

        XCTAssertTrue(viewModel.config.enrichmentEnabled)
        XCTAssertNotNil(viewModel.errorMessage)
    }

    @MainActor
    func testBackendDraftDoesNotPersistUntilCommitted() throws {
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-settings-model-\(UUID().uuidString)", isDirectory: true)
        let configURL = tempRoot.appendingPathComponent("brainlayer.env")
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        let store = BrainLayerConfigStore(configURL: configURL)
        try store.save(BrainLayerConfig.defaultConfig)
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false
        )

        viewModel.backendDraft = "mlx"
        var document = try store.loadDocument()
        XCTAssertEqual(document.config.enrichmentBackend, "gemini")

        viewModel.commitBackendDraft()
        document = try store.loadDocument()
        XCTAssertEqual(document.config.enrichmentBackend, "mlx")
    }

    @MainActor
    func testUnwiredProviderCannotReplaceConfiguredProvider() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        fixture.viewModel.setEnrichmentProvider(.openai)

        XCTAssertEqual(fixture.viewModel.config.enrichmentProvider, .gemini)
        XCTAssertEqual(
            fixture.viewModel.lastSaveReceipt?.validation,
            .failed("OpenAI cannot be activated because its runtime integration is unavailable.")
        )
        XCTAssertEqual(try fixture.store.loadDocument().config.enrichmentProvider, .gemini)
    }

    @MainActor
    func testExistingUnavailableProviderDoesNotTrapSafeDisableOrUnrelatedEdits() throws {
        var config = BrainLayerConfig.defaultConfig
        config.enrichmentProvider = .openai
        config.enrichmentBackend = "openai"
        let fixture = try makeFixture(config: config)
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        fixture.viewModel.setSystemEnabled(false)
        fixture.viewModel.setEnrichmentEnabled(false)

        let persisted = try fixture.store.loadDocument().config
        XCTAssertFalse(persisted.systemEnabled)
        XCTAssertFalse(persisted.enrichmentEnabled)
        XCTAssertEqual(persisted.enrichmentProvider, .openai)
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.validation, .passed)
    }

    @MainActor
    func testSelectingGeminiRepairsWhitespaceOnlyBackend() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        fixture.viewModel.config.enrichmentProvider = .openai
        fixture.viewModel.config.enrichmentBackend = "   "

        fixture.viewModel.setEnrichmentProvider(.gemini)

        XCTAssertEqual(fixture.viewModel.config.enrichmentProvider, .gemini)
        XCTAssertEqual(fixture.viewModel.config.enrichmentBackend, "gemini")
        XCTAssertEqual(try fixture.store.loadDocument().config.enrichmentBackend, "gemini")
    }

    @MainActor
    func testSaveKeepsConfiguredAndActiveValuesSeparateUntilRuntimeObservesChange() throws {
        let initial = BrainLayerConfig.defaultConfig
        let fixture = try makeFixture(
            runtimeObservation: .observed(BrainLayerActiveRuntimeValues(config: initial))
        )
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        fixture.viewModel.backendDraft = "mlx"
        fixture.viewModel.commitBackendDraft()

        XCTAssertEqual(fixture.viewModel.config.enrichmentBackend, "mlx")
        XCTAssertEqual(
            fixture.viewModel.activeRuntimeObservation,
            .observed(BrainLayerActiveRuntimeValues(config: initial))
        )
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.fileUpdated, true)
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.validation, .passed)
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.servicesRequiringRestart, [.enrichment])
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.activeRuntimeState, .notObserved)
        XCTAssertEqual(try fixture.store.loadDocument().config.enrichmentBackend, "mlx")
    }

    @MainActor
    func testSaveReceiptReportsWhenActiveRuntimeAlreadyMatchesConfiguredValue() throws {
        var active = BrainLayerConfig.defaultConfig
        active.enrichmentBackend = "mlx"
        let fixture = try makeFixture(
            runtimeObservation: .observed(BrainLayerActiveRuntimeValues(config: active))
        )
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        fixture.viewModel.backendDraft = "mlx"
        fixture.viewModel.commitBackendDraft()

        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.activeRuntimeState, .observed)
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.savedAt, fixedNow)
    }

    @MainActor
    func testValidationErrorDoesNotOverwritePersistedConfig() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        fixture.viewModel.backendDraft = "   "
        fixture.viewModel.commitBackendDraft()

        XCTAssertEqual(fixture.viewModel.config.enrichmentBackend, "gemini")
        XCTAssertEqual(try fixture.store.loadDocument().config.enrichmentBackend, "gemini")
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.fileUpdated, false)
        XCTAssertEqual(
            fixture.viewModel.lastSaveReceipt?.validation,
            .failed("Enrichment backend is required.")
        )
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.servicesRequiringRestart, [])
    }

    @MainActor
    func testPostWriteReloadMismatchStillReportsThatFileChanged() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        fixture.viewModel.backendDraft = "MLX"
        fixture.viewModel.commitBackendDraft()

        XCTAssertEqual(try fixture.store.loadDocument().config.enrichmentBackend, "mlx")
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.fileUpdated, true)
        XCTAssertEqual(
            fixture.viewModel.lastSaveReceipt?.validation,
            .failed("Saved configuration did not validate on reload.")
        )
        XCTAssertEqual(
            fixture.viewModel.lastSaveReceipt?.activeRuntimeState,
            .unknown("Configuration file changed, but reload validation failed.")
        )
    }

    @MainActor
    func testReloadErrorAfterSuccessfulWriteReportsFileUpdatedTruthfully() throws {
        let configURL = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-settings-reload-error-\(UUID().uuidString).env")
        var persisted = BrainLayerConfig.defaultConfig
        var loadCount = 0
        let now = fixedNow
        let store = BrainLayerConfigStore(
            configURL: configURL,
            loadDocumentOverride: {
                loadCount += 1
                if loadCount > 2 {
                    throw CocoaError(.fileReadUnknown)
                }
                return BrainLayerEnvDocument(config: persisted)
            },
            saveOverride: { persisted = $0 }
        )
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false,
            now: { now }
        )

        viewModel.setSystemEnabled(false)

        XCTAssertFalse(persisted.systemEnabled)
        XCTAssertEqual(viewModel.lastSaveReceipt?.fileUpdated, true)
        XCTAssertEqual(
            viewModel.lastSaveReceipt?.validation,
            .failed("Saved configuration could not be reloaded.")
        )
        XCTAssertEqual(
            viewModel.lastSaveReceipt?.activeRuntimeState,
            .unknown("Configuration file changed, but reload failed.")
        )
    }

    @MainActor
    func testJobSaveSeparatesConfiguredIntentFromActiveLaunchdState() throws {
        let fixture = try makeFixture(initialLaunchdStates: [.drain: .running])
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        fixture.viewModel.setJob(.drain, enabled: false)

        XCTAssertEqual(fixture.viewModel.config.launchdJobs[.drain]?.enabled, false)
        XCTAssertEqual(fixture.viewModel.config.launchdJobs[.drain]?.loadState, .running)
        XCTAssertEqual(
            fixture.viewModel.lastSaveReceipt?.servicesRequiringRestart,
            [.launchdJob(.drain)]
        )
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.activeRuntimeState, .notObserved)
    }

    @MainActor
    func testRefreshUpdatesSaveReceiptAfterLaunchdObservesConfiguredState() async throws {
        let provider = MutableBrainLayerLaunchdStatusProvider(states: [.drain: .running])
        let fixture = try makeFixture(
            launchdStatusProvider: provider,
            initialLaunchdStates: [.drain: .running]
        )
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        fixture.viewModel.setJob(.drain, enabled: false)
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.activeRuntimeState, .notObserved)

        provider.replaceStates(with: [.drain: .unloaded])
        fixture.viewModel.refreshLaunchdStatus()
        let deadline = Date().addingTimeInterval(2)
        while fixture.viewModel.isRefreshingLaunchdStatus {
            if Date() > deadline {
                XCTFail("Timed out waiting for launchd refresh to complete")
                break
            }
            await Task.yield()
        }

        XCTAssertEqual(fixture.viewModel.config.launchdJobs[.drain]?.loadState, .unloaded)
        XCTAssertEqual(fixture.viewModel.lastSaveReceipt?.activeRuntimeState, .observed)
    }

    @MainActor
    func testSecretSaveReceiptAndDebugOutputNeverContainSecret() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let secret = "settings-secret-fixture-value"

        fixture.viewModel.pendingPlainAPIKey = secret
        fixture.viewModel.storePlainAPIKey()

        XCTAssertEqual(fixture.viewModel.config.googleAPIKey.kind, .plainPresent)
        XCTAssertNotNil(fixture.viewModel.lastSaveReceipt)
        XCTAssertFalse(String(reflecting: fixture.viewModel.lastSaveReceipt).contains(secret))
        XCTAssertFalse(String(reflecting: fixture.viewModel.config.googleAPIKey).contains(secret))
    }

    @MainActor
    private func waitForBackupStatus(
        _ viewModel: BrainBarSettingsViewModel
    ) async throws -> ObservabilityBackupStatus {
        for _ in 0 ..< 100 {
            if let status = viewModel.backupStatus { return status }
            try await Task.sleep(for: .milliseconds(10))
        }
        return try XCTUnwrap(viewModel.backupStatus)
    }

    @MainActor
    private func waitForBackupReason(_ viewModel: BrainBarSettingsViewModel) async throws -> String {
        for _ in 0 ..< 100 {
            if let reason = viewModel.backupStatusReason,
               reason != "Backup status unavailable." {
                return reason
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        return try XCTUnwrap(viewModel.backupStatusReason)
    }

    @MainActor
    private func waitForBackupReason(
        _ viewModel: BrainBarSettingsViewModel,
        equalTo expected: String
    ) async throws {
        for _ in 0 ..< 100 {
            if viewModel.backupStatusReason == expected { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(viewModel.backupStatusReason, expected)
    }

    @MainActor
    private func makeFixture(
        config: BrainLayerConfig = .defaultConfig,
        runtimeObservation: BrainLayerActiveRuntimeObservation = .unknown(
            "Active runtime configuration is not observable."
        ),
        launchdStatusProvider: (any BrainLayerLaunchdStatusSampling)? = nil,
        initialLaunchdStates: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] = [:]
    ) throws -> (
        root: URL,
        store: BrainLayerConfigStore,
        viewModel: BrainBarSettingsViewModel
    ) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-settings-model-\(UUID().uuidString)", isDirectory: true)
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(config)
        let now = fixedNow
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: launchdStatusProvider ??
                StaticBrainLayerLaunchdStatusProvider(states: initialLaunchdStates),
            runtimeStatusProvider: StaticBrainLayerActiveRuntimeProvider(observation: runtimeObservation),
            initialLaunchdStates: initialLaunchdStates,
            refreshStatusOnLoad: false,
            now: { now }
        )
        return (root, store, viewModel)
    }
}

private final class SequencedObservabilityReads: @unchecked Sendable {
    private let lock = NSLock()
    private var callCount = 0
    private let gates = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]

    func read(url _: URL) -> ObservabilityReadResult {
        let call = lock.withLock {
            callCount += 1
            return callCount
        }
        gates[call - 1].wait()
        return .unreadable(call == 1 ? "Stale status" : "Newest status")
    }

    func waitForCallCount(_ expected: Int) async throws {
        for _ in 0 ..< 100 {
            if lock.withLock({ callCount >= expected }) { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Timed out waiting for observability read \(expected)")
    }

    func release(call: Int) {
        gates[call - 1].signal()
    }
}

private final class MutableBrainLayerLaunchdStatusProvider: BrainLayerLaunchdStatusSampling, @unchecked Sendable {
    private let lock = NSLock()
    private var states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]

    init(states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]) {
        self.states = states
    }

    func replaceStates(with states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]) {
        lock.withLock {
            self.states = states
        }
    }

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        lock.withLock { states }
    }
}
