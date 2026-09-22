#if DEBUG
import AppKit
import Darwin
import SwiftUI

@MainActor
enum BrainBarRenderHarness {
    private static let environmentVariable = "BRAINBAR_RENDER_ONLY"
    private static let breakpoints: [(name: String, width: CGFloat)] = [
        ("compact", 760), ("default", 960), ("wide", 1_280),
    ]

    @MainActor
    private enum Scenario: String, CaseIterable {
        case readable
        case unreadable
        case loading
        case queueDraining
        case queueBacklogged

        var detailsStates: [Bool] {
            switch self {
            case .loading, .queueDraining, .queueBacklogged:
                [false]
            case .readable, .unreadable:
                [false, true]
            }
        }

        var breakpoints: [(name: String, width: CGFloat)] {
            switch self {
            case .queueDraining, .queueBacklogged:
                [("default", 960)]
            case .readable, .unreadable, .loading:
                BrainBarRenderHarness.breakpoints
            }
        }

        var collectorState: BrainBarDashboardFixture.OperatorState {
            switch self {
            case .loading:
                .loading
            case .queueDraining:
                .queueDraining
            case .queueBacklogged:
                .queueBacklogged
            case .readable, .unreadable:
                .live
            }
        }

        var observabilityResult: ObservabilityReadResult {
            switch self {
            case .readable, .loading, .queueDraining, .queueBacklogged:
                BrainBarDashboardFixture.readableObservabilityResult
            case .unreadable:
                .unreadable("Database path unavailable.")
            }
        }
    }

    private struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    static func runIfRequested() {
        guard let path = ProcessInfo.processInfo.environment[environmentVariable] else { return }
        do {
            guard !path.isEmpty, NSString(string: path).isAbsolutePath else {
                throw Failure("\(environmentVariable) must name an absolute output directory; got \(path.debugDescription)")
            }

            // This runs before App.main(): no AppDelegate, status UI, DB, socket,
            // collector, or timer exists. Prohibited apps cannot be activated.
            NSApplication.shared.setActivationPolicy(.prohibited)
            let outputDirectory = URL(fileURLWithPath: path, isDirectory: true)
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
            try verifyDirectionalStateCoverage()
            for scenario in Scenario.allCases {
                for breakpoint in scenario.breakpoints {
                    for detailsExpanded in scenario.detailsStates {
                        let artifact = try render(
                            breakpoint: breakpoint,
                            scenario: scenario,
                            detailsExpanded: detailsExpanded,
                            outputDirectory: outputDirectory
                        )
                        print("[brainbar-render] \(artifact)")
                    }
                }
            }
            try verifyDirectionalStatesDiffer(in: outputDirectory)
            Darwin.exit(EXIT_SUCCESS)
        } catch {
            FileHandle.standardError.write(Data("[brainbar-render] ERROR: \(error.localizedDescription)\n".utf8))
            Darwin.exit(EXIT_FAILURE)
        }
    }

    private static func verifyDirectionalStateCoverage() throws {
        let expectations: [(DashboardQueueStatus, BrainBarQueueDirectionPresentation)] = [
            (.empty, .init(label: "Queue empty", symbol: "arrow.left.and.right", tone: .neutral)),
            (.stable, .init(label: "Queue stable", symbol: "arrow.left.and.right", tone: .neutral)),
            (.draining, .init(label: "Queue draining", symbol: "arrow.down.right", tone: .active)),
            (.growing, .init(label: "Queue growing", symbol: "arrow.up.right", tone: .warning)),
            (.backlogged, .init(label: "Queue backlogged", symbol: "arrow.up.right", tone: .warning)),
            (.unavailable, .init(label: "Queue offline", symbol: "exclamationmark.triangle", tone: .error)),
        ]
        for (state, expected) in expectations {
            let actual = BrainBarQueueDirectionPresentation.derive(state)
            guard actual == expected else {
                throw Failure("directional-state probe failed for \(state.rawValue): \(actual)")
            }
        }
        print("[brainbar-render] directional-state coverage PASS: empty, stable, draining, growing, backlogged, unavailable")
    }

    private static func verifyDirectionalStatesDiffer(in outputDirectory: URL) throws {
        let draining = outputDirectory.appendingPathComponent("dashboard-cli-default-queueDraining.png")
        let backlogged = outputDirectory.appendingPathComponent("dashboard-cli-default-queueBacklogged.png")
        guard try Data(contentsOf: draining) != Data(contentsOf: backlogged) else {
            throw Failure("directional-state probe collapsed: draining and backlogged rendered byte-identically")
        }
        print("[brainbar-render] directional-state probe PASS: draining and backlogged differ")
    }

    private static func render(
        breakpoint: (name: String, width: CGFloat),
        scenario: Scenario,
        detailsExpanded: Bool,
        outputDirectory: URL
    ) throws -> String {
        let panelState = BrainBarDashboardPanelState()
        panelState.detailsExpanded = detailsExpanded
        let view = BrainBarDashboardPreview.make(
            collector: BrainBarDashboardFixture.makeCollector(scenario.collectorState),
            observabilityResult: scenario.observabilityResult,
            now: BrainBarDashboardFixture.fetchedAt,
            panelState: panelState
        )
        let suffix = detailsExpanded ? "-details-expanded" : ""
        // The XCTest renderer owns dashboard-<breakpoint>.png. Include the CLI
        // state in every filename so the two producers cannot overwrite each other.
        let name = "dashboard-cli-\(breakpoint.name)-\(scenario.rawValue)\(suffix)"

        let measuringHost = NSHostingView(rootView: view)
        measuringHost.frame = NSRect(x: 0, y: 0, width: breakpoint.width, height: 10_000)
        settle(measuringHost)
        let height = ceil(panelState.fittingHeight)
        guard height.isFinite, height > 0 else {
            throw Failure("\(name): dashboard reported invalid fitting height \(height)")
        }

        let size = NSSize(width: breakpoint.width, height: height)
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(origin: .zero, size: size)
        settle(host)
        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
            throw Failure("\(name): AppKit could not allocate an off-screen bitmap")
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            throw Failure("\(name): AppKit could not encode PNG data")
        }

        let url = outputDirectory.appendingPathComponent("\(name).png")
        try png.write(to: url, options: .atomic)
        let emittedPNG = try Data(contentsOf: url)
        guard let emittedBitmap = NSBitmapImageRep(data: emittedPNG) else {
            try? FileManager.default.removeItem(at: url)
            throw Failure("\(name): emitted PNG could not be decoded for pixel verification")
        }
        let colors = distinctSampledColorCount(in: emittedBitmap)
        guard emittedPNG.count > 5_000, colors > 16 else {
            try? FileManager.default.removeItem(at: url)
            throw Failure("\(name): refusing a blank or trivial render (\(emittedPNG.count) PNG bytes, \(colors) sampled colors)")
        }

        let cardHeights = panelState.renderedSummaryTileHeights
        let cardReceipt = if let backups = cardHeights["backups"], let today = cardHeights["memory"] {
            "; cards Backups=\(Int(backups.rounded()))pt Today=\(Int(today.rounded()))pt"
        } else {
            ""
        }
        return "\(name) \(Int(size.width))×\(Int(size.height))\(cardReceipt); wrote \(url.path) "
            + "(\(emittedPNG.count) bytes, \(colors) sampled colors)"
    }

    private static func settle(_ host: NSView) {
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
        host.layoutSubtreeIfNeeded()
    }

    private static func distinctSampledColorCount(in bitmap: NSBitmapImageRep) -> Int {
        guard let data = bitmap.bitmapData else { return 0 }
        let bytesPerPixel = max(bitmap.bitsPerPixel / 8, 1)
        let baseStride = max(bitmap.bytesPerRow / 32, bytesPerPixel)
        let sampleStride = baseStride - (baseStride % bytesPerPixel)
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
#endif
