import Foundation
import XCTest
@testable import BrainBar

/// Headless only: no window, menu, status item, Finder or production database.
final class BrainBarExplainDiagnosticsTests: XCTestCase {
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    func testExplanationUsesTheKnownModelAndDescribesBothSearchPaths() {
        let lines = BrainBarSettingsExplanation.model("fixture/model")
        XCTAssertEqual(lines.count, 2)
        XCTAssertTrue(lines[0].contains("fixture/model"))
        XCTAssertTrue(lines[0].contains("on this Mac"))
        XCTAssertTrue(lines[1].contains("Search by meaning uses it"))
        XCTAssertTrue(lines[1].contains("Keyword search does not need it"))
        XCTAssertTrue(BrainBarSettingsExplanation.model(BrainBarEmbeddingModel.configuredName)[0]
            .contains(BrainBarEmbeddingModel.configuredName))
        XCTAssertEqual(BrainBarSettingsExplanation.model("fixture/model", namesModel: false)[0],
                       "This model turns memories into searchable meaning on this Mac.")
    }

    func testUnknownModelHasNoInventedExplanation() {
        let names: [String?] = [nil, "", "  \n"]
        for name in names {
            XCTAssertEqual(BrainBarSettingsExplanation.model(name), [])
        }
    }

    @MainActor func testFullReportIncludesEveryCurrentCheckAndRuntimeWarning() {
        for name in BrainBarDesignFixture.Name.allCases {
            let inputs = BrainBarDesignFixture.inputs(name)
            let store = NDStore(fixture: inputs)
            for check in inputs.capture.verdict?.checks ?? [] {
                XCTAssertTrue(store.report.contains("\(check.name): \(check.state.rawValue) — \(check.note)"), name.rawValue)
            }
            for extra in inputs.extras { XCTAssertTrue(store.report.contains(extra.text), name.rawValue) }
            let list = NDDiagnosticsPresentation(inputs).lines
            XCTAssertEqual(list.count, (inputs.capture.verdict?.checks.count ?? 1) + inputs.extras.count)
            for extra in inputs.extras { XCTAssertTrue(list.contains(extra.text)) }
        }
    }

    func testUnreadChecksAreExplicit() {
        var inputs = BrainBarDesignFixture.inputs(.calm)
        inputs.capture.verdict = nil
        XCTAssertEqual(NDDiagnosticsPresentation(inputs).lines, [BrainBarMenuPresenter.SurfaceSay.checksUnreadable])
        inputs.capture.verdict = .init(dot: .red, checks: [
            .init(name: "a", state: .green, note: ""), .init(name: "b", state: .amber, note: "late"),
            .init(name: "c", state: .red, note: ""), .init(name: "d", state: .unknown, note: "")
        ], notes: [], evaluatedAt: inputs.capture.now)
        let presentation = NDDiagnosticsPresentation(inputs)
        XCTAssertEqual(presentation.lines, ["a: OK", "b: Warning — late", "c: Problem", "d: Not read"])
        XCTAssertEqual(presentation.rows[2].state, .red)
    }

    @MainActor func testRevealSelectsCurrentFileAndLeavesEveryBackupUntouched() throws {
        let directory = URL(fileURLWithPath: brainBarTestTemporaryPath()).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent("brainlayer.env")
        let configStore = BrainLayerConfigStore(configURL: file)
        try configStore.save(.defaultConfig)
        let names = ["brainlayer.env.bak-2026-09-05", "brainlayer.env.20261015.ruling.bak",
                     "brainlayer.env.backup-2026", "unrelated.env.bak", "brainlayer.env.example"]
        for name in names { try Data(name.utf8).write(to: directory.appendingPathComponent(name)) }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("brainlayer.env.bak-folder"),
                                                withIntermediateDirectories: true)
        let before = try Dictionary(uniqueKeysWithValues: (["brainlayer.env"] + names).map {
            ($0, try Data(contentsOf: directory.appendingPathComponent($0)))
        })
        let workspace = BrainBarRecordingWorkspace()
        let model = BrainBarSettingsViewModel(workspace: workspace, store: configStore,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]), refreshStatusOnLoad: false)
        XCTAssertNil(model.settingsBackupLine, "No folder read before Settings opens or Reveal is pressed")
        model.revealConfigFile()
        XCTAssertEqual(workspace.recorded.count, 1)
        XCTAssertEqual(workspace.recorded.first?.0, "reveal")
        XCTAssertEqual(workspace.recorded.first?.1, file.path)
        XCTAssertTrue(try XCTUnwrap(model.settingsBackupLine).hasPrefix("3 settings backups in "))
        XCTAssertTrue(try XCTUnwrap(model.settingsBackupLine).contains((directory.path as NSString).abbreviatingWithTildeInPath))
        try Data("new fixture backup".utf8).write(to: directory.appendingPathComponent("brainlayer.env.bak-new"))
        XCTAssertTrue(try XCTUnwrap(model.settingsBackupLine).hasPrefix("3 settings backups in "), "Redraws use the cached count")
        model.refreshSettingsFileSummary()
        XCTAssertTrue(try XCTUnwrap(model.settingsBackupLine).hasPrefix("4 settings backups in "))
        for (name, data) in before {
            XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent(name)), data)
        }
        let adapter = try String(contentsOf: root.appendingPathComponent("brain-bar/Sources/BrainBar/BrainBarWorkspace.swift"), encoding: .utf8)
        XCTAssertTrue(adapter.contains("activateFileViewerSelecting([url])"))
        let settings = try String(contentsOf: root.appendingPathComponent("brain-bar/Sources/BrainBar/BrainBarSettingsView.swift"), encoding: .utf8)
        XCTAssertTrue(settings.contains("workspace.reveal(store.configURL)"))
    }

    func testNoBackupsHasNoCountAndUnreadableFolderIsExplicit() throws {
        let directory = URL(fileURLWithPath: brainBarTestTemporaryPath()).appendingPathComponent(UUID().uuidString)
        let file = directory.appendingPathComponent("brainlayer.env")
        XCTAssertEqual(BrainBarSettingsExplanation.backupLine(for: file), BrainBarMenuPresenter.SurfaceSay.settingsBackupsUnreadable)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        XCTAssertNil(BrainBarSettingsExplanation.backupLine(for: file))
    }
}
