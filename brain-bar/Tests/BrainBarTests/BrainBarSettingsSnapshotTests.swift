import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

final class BrainBarSettingsSnapshotTests: XCTestCase {
    @MainActor
    func testSettingsViewRendersProviderJobsAndSaveTruthScenarios() throws {
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-settings-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        var providerConfig = BrainLayerConfig.defaultConfig
        providerConfig.googleAPIKey = .onePasswordReference("op://Private/Google AI/Gemini API key")
        providerConfig.enrichmentProvider = .openai
        providerConfig.launchdJobs[.hotlane]?.enabled = false
        let providerViewModel = try makeViewModel(
            root: tempRoot,
            name: "provider",
            config: providerConfig
        )
        try render(viewModel: providerViewModel, named: "provider-and-jobs")

        let savedViewModel = try makeViewModel(
            root: tempRoot,
            name: "saved",
            config: .defaultConfig
        )
        savedViewModel.backendDraft = "mlx"
        savedViewModel.commitBackendDraft()
        try render(viewModel: savedViewModel, named: "saved-restart-needed")

        let validationViewModel = try makeViewModel(
            root: tempRoot,
            name: "validation",
            config: .defaultConfig
        )
        validationViewModel.backendDraft = "   "
        validationViewModel.commitBackendDraft()
        try render(viewModel: validationViewModel, named: "validation-error")
    }

    @MainActor
    private func makeViewModel(
        root: URL,
        name: String,
        config: BrainLayerConfig
    ) throws -> BrainBarSettingsViewModel {
        let configURL = root.appendingPathComponent("\(name)-brainlayer.env")
        let store = BrainLayerConfigStore(configURL: configURL)
        try store.save(config)
        let states: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] = [
            .enrichment: .loaded,
            .hotlane: .unloaded,
            .drain: .running,
            .watch: .probeError("launchctl exited 1"),
        ]
        return BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: states),
            runtimeStatusProvider: StaticBrainLayerActiveRuntimeProvider(
                observation: .unknown("Active runtime configuration is not observable.")
            ),
            initialLaunchdStates: states,
            refreshStatusOnLoad: false
        )
    }

    @MainActor
    private func render(viewModel: BrainBarSettingsViewModel, named name: String) throws {
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

        let url = screenshotURL(named: name)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try png.write(to: url)
        XCTAssertGreaterThan(png.count, 1_000)
        XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 8)
    }

    private func screenshotURL(named name: String) -> URL {
        if let override = ProcessInfo.processInfo.environment["BRAINBAR_SETTINGS_RENDER_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
                .appendingPathComponent("settings-\(name).png")
        }
        return URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(
                "brainbar-p5-settings-renders-\(ProcessInfo.processInfo.processIdentifier)",
                isDirectory: true
            )
            .appendingPathComponent("settings-\(name).png")
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
