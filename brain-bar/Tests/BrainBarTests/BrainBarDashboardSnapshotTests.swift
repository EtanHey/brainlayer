import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Deterministic, full-dashboard render-verification infra.
///
/// BrainBar is an `LSUIElement` menu-bar app, so `computer-use` reports it
/// "not_installed" and full-screen `screencapture` grabs the wrong window — there
/// is no reliable way to visually verify its UI. These tests render the REAL
/// dashboard views (hero/overview + pipeline + diagnostics) and the settings panel
/// to PNGs that any agent can `Read` to verify the UI, with no live collectors and
/// no live screenshots.
///
/// The renders are deterministic: fixture data (`BrainBarDashboardFixture`) with
/// every relative-time `Date` nil, `accessibilityReduceMotion = true` so SwiftUI
/// animations resolve immediately, and fixed host sizes. Output PNGs land in
/// `brain-bar/docs.local/brainbar-render/` (override with `BRAINBAR_RENDER_DIR`).
///
/// Run just this suite:
///   swift test --filter BrainBarDashboardSnapshotTests
final class BrainBarDashboardSnapshotTests: XCTestCase {
    /// Layout breakpoints come from `BrainBarDashboardLayout`: compact < 920,
    /// 920 ≤ default < 1040, wide ≥ 1040 (two chart columns).
    private enum Breakpoint: String, CaseIterable {
        case compact
        case `default`
        case wide

        // Heights are intentionally generous: the dashboard is a ScrollView, so a
        // too-short frame clips the lower cards (queue rail, agent presence,
        // diagnostics) while a too-tall frame only adds dark background below the
        // content. Narrower widths stack everything vertically and run tallest.
        var size: NSSize {
            switch self {
            case .compact: NSSize(width: 760, height: 1_900)
            case .default: NSSize(width: 960, height: 1_820)
            case .wide: NSSize(width: 1_280, height: 1_500)
            }
        }
    }

    @MainActor
    func testDashboardRendersAtAllBreakpoints() throws {
        for breakpoint in Breakpoint.allCases {
            let collector = BrainBarDashboardFixture.makeCollector()
            let view = BrainBarDashboardPreview.make(collector: collector)
            let (png, bitmap) = try renderPNG(view, size: breakpoint.size)

            let url = try writePNG(png, name: "dashboard-\(breakpoint.rawValue)")
            XCTAssertGreaterThan(png.count, 5_000, "dashboard-\(breakpoint.rawValue) PNG looks empty")
            XCTAssertGreaterThan(
                distinctSampledColorCount(in: bitmap), 16,
                "dashboard-\(breakpoint.rawValue) render is too flat — likely blank/clipped"
            )
            // Surface the path in the test log so an agent knows what to Read.
            print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
        }
    }

    @MainActor
    func testSettingsRendersDeterministically() throws {
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-render-settings-\(UUID().uuidString)", isDirectory: true)
        let configURL = tempRoot
            .appendingPathComponent(".config/brainlayer/brainlayer.env", isDirectory: false)
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
            ]),
            initialLaunchdStates: [
                .enrichment: .loaded,
                .hotlane: .unloaded,
                .drain: .running,
            ],
            refreshStatusOnLoad: false
        )
        let view = BrainBarSettingsView(viewModel: viewModel)
            .environment(\.colorScheme, .dark)
            .transaction { $0.disablesAnimations = true }
        let (png, bitmap) = try renderPNG(view, size: NSSize(width: 700, height: 1_080))

        let url = try writePNG(png, name: "settings")
        XCTAssertGreaterThan(png.count, 5_000, "settings PNG looks empty")
        XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 16, "settings render is too flat")
        print("[brainbar-render] wrote \(url.path) (\(png.count) bytes)")
    }

    // MARK: - Render helpers

    @MainActor
    private func renderPNG(_ view: some View, size: NSSize) throws -> (Data, NSBitmapImageRep) {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(origin: .zero, size: size)
        host.layoutSubtreeIfNeeded()
        // Give SwiftUI onAppear/layout a moment to settle. With reduceMotion the
        // final state is reached immediately; this only flushes the run loop, so
        // the rendered RESULT is deterministic regardless of the wall-clock delay.
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
        host.layoutSubtreeIfNeeded()

        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
            throw RenderError.bitmapUnavailable
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            throw RenderError.encodingFailed
        }
        return (png, bitmap)
    }

    private func writePNG(_ png: Data, name: String) throws -> URL {
        let dir = outputDirectory()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("\(name).png")
        try png.write(to: url)
        return url
    }

    private func outputDirectory() -> URL {
        if let override = ProcessInfo.processInfo.environment["BRAINBAR_RENDER_DIR"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        // #filePath = .../brain-bar/Tests/BrainBarTests/BrainBarDashboardSnapshotTests.swift
        // up 3 → brain-bar/
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("docs.local/brainbar-render", isDirectory: true)
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

    private enum RenderError: Error {
        case bitmapUnavailable
        case encodingFailed
    }
}
