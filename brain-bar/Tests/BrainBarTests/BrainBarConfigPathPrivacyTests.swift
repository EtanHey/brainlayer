import XCTest
@testable import BrainBar

final class BrainBarConfigPathPrivacyTests: XCTestCase {
    @MainActor
    func testHiddenRevealedRehiddenAndExplicitActions() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("private-person/brainlayer.env")
        let store = BrainLayerConfigStore(configURL: url)
        try store.save(.defaultConfig)
        let workspace = ConfigPathWorkspace()
        let model = BrainBarSettingsViewModel(store: store, refreshStatusOnLoad: false, workspace: workspace)
        XCTAssertFalse(model.configPathRevealed)
        XCTAssertEqual(model.configFilePathLabel, "brainlayer.env")
        XCTAssertTrue(workspace.copies.isEmpty)
        XCTAssertTrue(workspace.reveals.isEmpty)
        model.toggleConfigPathVisibility()
        XCTAssertTrue(model.configPathRevealed)
        XCTAssertEqual(model.configFilePathLabel, url.path)
        model.toggleConfigPathVisibility()
        XCTAssertEqual(model.configFilePathLabel, "brainlayer.env")
        model.copyConfigFilePath()
        model.revealConfigFile()
        XCTAssertEqual(workspace.copies, [url.path])
        XCTAssertEqual(workspace.reveals, [url])
        XCTAssertFalse(model.configPathRevealed, "Copy/Finder must not implicitly reveal the label")
        model.toggleConfigPathVisibility()
        model.hideConfigPath()
        model.hideConfigPath()
        XCTAssertFalse(model.configPathRevealed)
        XCTAssertEqual(model.configFilePathLabel, "brainlayer.env")
    }
}

private final class ConfigPathWorkspace: BrainBarWorkspaceActing, @unchecked Sendable {
    var copies: [String] = []
    var reveals: [URL] = []
    func copy(_ text: String) { copies.append(text) }
    func reveal(_ url: URL) { reveals.append(url) }
}
