import XCTest
@testable import BrainBar

final class BrainBarSettingsViewModelTests: XCTestCase {
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
}
