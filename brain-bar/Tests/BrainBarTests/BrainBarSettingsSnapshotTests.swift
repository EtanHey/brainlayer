import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

final class BrainBarSettingsSnapshotTests: XCTestCase {
    @MainActor
    func testSettingsViewRendersConfigBackedPanelScreenshot() throws {
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-settings-\(UUID().uuidString)", isDirectory: true)
        let configURL = tempRoot
            .appendingPathComponent(".config", isDirectory: true)
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("brainlayer.env", isDirectory: false)
        var config = BrainLayerConfig.defaultConfig
        config.googleAPIKey = .onePasswordReference("op://Private/Google AI/Gemini API key")
        config.enrichmentEnabled = true
        config.enrichmentMode = .remote
        config.enrichmentProvider = .gemini
        config.enrichmentBackend = "gemini"
        config.launchdJobs[.drain]?.enabled = true
        config.launchdJobs[.hotlane]?.enabled = false

        let store = BrainLayerConfigStore(configURL: configURL)
        try store.save(config)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [
                .enrichment: .loaded,
                .hotlane: .unloaded,
                .drain: .running,
            ])
        )
        let view = NSHostingView(rootView: BrainBarSettingsView(viewModel: viewModel))
        view.frame = NSRect(x: 0, y: 0, width: 700, height: 1_080)
        view.layoutSubtreeIfNeeded()

        guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
            XCTFail("Expected hosting view to produce a bitmap")
            return
        }
        view.cacheDisplay(in: view.bounds, to: bitmap)

        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            XCTFail("Expected renderer to produce a PNG")
            return
        }

        let url = screenshotURL()
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try png.write(to: url)
        XCTAssertGreaterThan(png.count, 1_000)
        XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 8)
    }

    private func screenshotURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["BRAINBAR_SETTINGS_SCREENSHOT_PATH"],
           !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("docs.local/brainbar-settings-ui-phase2/brainbar-settings.png")
    }

    private func distinctSampledColorCount(in bitmap: NSBitmapImageRep) -> Int {
        guard let data = bitmap.bitmapData else { return 0 }
        let bytesPerPixel = max(bitmap.bitsPerPixel / 8, 1)
        let sampleStride = max(bitmap.bytesPerRow / 32, bytesPerPixel)
        var colors = Set<String>()
        for y in stride(from: 0, to: bitmap.pixelsHigh, by: 24) {
            let rowStart = y * bitmap.bytesPerRow
            for x in stride(from: 0, to: bitmap.bytesPerRow, by: sampleStride) {
                let offset = rowStart + x
                guard offset + 2 < bitmap.bytesPerRow * bitmap.pixelsHigh else { continue }
                colors.insert("\(data[offset])-\(data[offset + 1])-\(data[offset + 2])")
            }
        }
        return colors.count
    }
}
