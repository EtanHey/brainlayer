import XCTest
@testable import BrainBar

final class BrainBarSettingsViewModelTests: XCTestCase {
    private let fixedNow = Date(timeIntervalSince1970: 1_784_466_000)

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
                if loadCount > 1 {
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
