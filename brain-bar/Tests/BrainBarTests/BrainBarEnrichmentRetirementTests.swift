import Foundation
import XCTest
@testable import BrainBar

final class BrainBarEnrichmentRetirementTests: XCTestCase {
    func testSettingsHasNoEnrichmentActionsOrBackendDraft() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("Sources/BrainBar/BrainBarSettingsView.swift"), encoding: .utf8)
        for retired in ["func setEnrichmentEnabled(", "func setEnrichmentMode(",
                        "func setEnrichmentProvider(", "func commitBackendDraft(", "var backendDraft:"] {
            XCTAssertFalse(source.contains(retired), retired)
        }
    }

    @MainActor
    func testGoogleKeyEditDoesNotRequestRetiredServiceRestart() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = BrainLayerConfigStore(configURL: directory.appendingPathComponent("fixture.env"))
        try store.save(.defaultConfig)
        let model = BrainBarSettingsViewModel(
            store: store, launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false
        )
        model.pendingPlainAPIKey = "synthetic-google-key"
        model.storePlainAPIKey()
        XCTAssertEqual(model.lastSaveReceipt?.validation, .passed)
        XCTAssertEqual(model.lastSaveReceipt?.servicesRequiringRestart, [])
        XCTAssertEqual(try store.loadDocument().config.googleAPIKey, .plain("synthetic-google-key"))
    }
}
