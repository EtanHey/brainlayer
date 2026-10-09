import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Opt-in only: the Apple/native owner runs these and inspects every image for clipping,
/// wrapping and legibility. PNG creation alone is not visual acceptance or installed UX proof.
final class BrainBarUIPolishRenderFixturesTests: XCTestCase {
    @MainActor
    func testNarrowMediumWideHiddenRevealedRehiddenAndLogStates() throws {
        guard let directory = ProcessInfo.processInfo.environment["BRAINBAR_UI_POLISH_RENDER_DIR"] else {
            throw XCTSkip("Native owner must authorize rendering and provide a private output directory")
        }
        let output = URL(fileURLWithPath: directory)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("synthetic-private-long-prefix/brainlayer.env"))
        try store.save(.defaultConfig)
        let model = BrainBarSettingsViewModel(store: store, refreshStatusOnLoad: false)
        let paths = BrainBarBackupSources.Paths(launchAgents: root, databaseLog: root, archiveLog: root,
                                               maintenanceLog: root, snapshotDirectory: root, archiveDirectory: root)
        for width in [460, 700, 1_000] {
            for state in ["hidden", "revealed", "rehidden"] {
                if state == "revealed" { model.toggleConfigPathVisibility() } else { model.hideConfigPath() }
                try render(BrainBarSettingsView(viewModel: model, initialSection: .advanced), width: width,
                           url: output.appendingPathComponent("config-\(state)-\(width).png"))
            }
            let states: [BrainBarBackupLogPreview] = [.empty, .unavailable("Log changed while reading. Refresh to try again."),
                .entries(String(repeating: "PermissionError · upload failed · [private path] · verified: 0\n", count: 20), truncated: true)]
            for (index, state) in states.enumerated() {
                let log = BrainBarBackupLogPreviewModel(paths: paths, initialPreview: state, read: { _ in state })
                try render(BrainBarBackupLogPreviewView(model: log).padding(28), width: width,
                           url: output.appendingPathComponent("log-\(index)-\(width).png"))
            }
        }
    }

    @MainActor
    private func render<V: View>(_ view: V, width: Int, url: URL) throws {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: width, height: 900)
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let data = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        XCTAssertGreaterThan(data.count, 1_000)
        try data.write(to: url)
    }
}
