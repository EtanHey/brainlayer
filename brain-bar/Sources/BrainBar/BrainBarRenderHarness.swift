#if DEBUG
import AppKit
import Darwin
import SwiftUI
import Vision

@MainActor
enum BrainBarRenderHarness {
    private static let environmentVariable = "BRAINBAR_RENDER_ONLY"
    private static let sampleReceipts = BrainBarOperationReceipts()
    private static let breakpoints: [(name: String, width: CGFloat)] = [
        ("compact", 760), ("default", 960), ("wide", 1_280),
    ]

    @MainActor
    private enum Scenario: String, CaseIterable {
        case readable
        case attentionCollapsed = "readable-attention"
        case attentionExpanded = "readable-attention-expanded"
        case unreadable
        case stale
        case loading
        case queueDraining
        case queueBacklogged
        case receiptUnavailable
        case receiptFailed
        case vectorAt100 = "vector-at-100"

        var detailsStates: [Bool] {
            switch self {
            case .loading, .stale, .attentionCollapsed, .attentionExpanded, .queueDraining, .queueBacklogged,
                 .receiptUnavailable, .receiptFailed:
                [false]
            case .vectorAt100:
                [true]
            case .readable, .unreadable:
                [false, true]
            }
        }

        var breakpoints: [(name: String, width: CGFloat)] {
            switch self {
            case .queueDraining, .queueBacklogged, .receiptUnavailable, .receiptFailed, .vectorAt100:
                [("default", 960)]
            case .readable, .attentionCollapsed, .attentionExpanded, .unreadable, .stale, .loading:
                BrainBarRenderHarness.breakpoints
            }
        }

        var collectorState: BrainBarDashboardFixture.OperatorState {
            switch self {
            case .loading:
                .loading
            case .stale:
                .stale
            case .attentionCollapsed, .attentionExpanded:
                .live
            case .queueDraining:
                .queueDraining
            case .queueBacklogged:
                .queueBacklogged
            case .readable, .unreadable, .receiptUnavailable, .receiptFailed:
                .live
            case .vectorAt100:
                .live
            }
        }

        var observabilityResult: ObservabilityReadResult {
            switch self {
            case .readable, .stale, .attentionCollapsed, .attentionExpanded, .loading, .queueDraining,
                 .queueBacklogged, .receiptUnavailable, .receiptFailed, .vectorAt100:
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
            try verifyReadableChartMarkerContract()
            try renderUnifiedSettings(in: outputDirectory)
            sampleReceipts.record(BrainBarOperationReceipt(
                kind: .search, durationMillis: 142, count: 10, recordedAt: BrainBarDashboardFixture.fetchedAt
            ))
            sampleReceipts.record(BrainBarOperationReceipt(
                kind: .ingest, durationMillis: 1_200, count: 1, recordedAt: BrainBarDashboardFixture.fetchedAt
            ))
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
            for breakpoint in breakpoints {
                for (expanded, afterCollapse) in [(false, false), (true, false), (false, true)] {
                    let artifact = try render(
                        breakpoint: breakpoint, scenario: .readable, detailsExpanded: expanded,
                        outputDirectory: outputDirectory, fixedHeight: 640, afterCollapse: afterCollapse
                    )
                    print("[brainbar-render] \(artifact)")
                }
            }
            try verifyDirectionalStatesDiffer(in: outputDirectory)
            try verifyAttentionDisclosureChangesPixels(in: outputDirectory)
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

    private static func verifyReadableChartMarkerContract() throws {
        let stats = BrainBarDashboardFixture.stats
        let charts = [
            ("All chunks", stats.recentActivityBuckets),
            ("Agent", stats.recentAgentWriteBuckets),
            ("Watcher", stats.recentWatcherWriteBuckets),
        ]

        for (name, values) in charts {
            let presentation = SparklineChartPresentation(
                label: name,
                values: values,
                lastBucketIsPartial: true
            )
            let markerIndices = presentation.visiblePointMarkers(for: .primary, compact: false).map(\.bucket)
            guard markerIndices.count == 1 else {
                throw Failure("\(name) chart rendered \(markerIndices.count) point markers; expected exactly one")
            }
            guard markerIndices[0] == values.count - 2 else {
                throw Failure(
                    "\(name) chart partial bucket \(values.count - 1) has 2 marks (series + point marker); "
                        + "marker landed on bucket \(markerIndices[0]), expected latest complete bucket \(values.count - 2)"
                )
            }
        }
        print("[brainbar-render] chart-marker contract PASS: All chunks, Agent, Watcher each have one marker on the latest complete bucket")
    }

    private static func renderUnifiedSettings(in outputDirectory: URL) throws {
        let fixtureDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-settings-render-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: fixtureDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDirectory) }
        let configURL = fixtureDirectory.appendingPathComponent("settings-render-fixture.env")
        let store = BrainLayerConfigStore(configURL: configURL)
        try store.save(.defaultConfig)

        let settingsScenarios: [(section: BrainBarSettingsSection, receipt: Bool)] =
            BrainBarSettingsSection.allCases.map { ($0, false) } + [(.general, true)]
        for scenario in settingsScenarios {
          for breakpoint in breakpoints {
            if scenario.receipt { try store.save(.defaultConfig) }
            let viewModel = BrainBarSettingsViewModel(
                store: store,
                launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
                runtimeStatusProvider: StaticBrainLayerActiveRuntimeProvider(
                    observation: .unknown("Fixture runtime state unavailable.")
                ),
                refreshStatusOnLoad: false,
                initialObservabilityResult: .unreadable("Fixture backup status unavailable.")
            )
            if scenario.receipt {
                viewModel.backendDraft = "mlx"
                viewModel.commitBackendDraft()
                guard viewModel.lastSaveReceipt != nil else {
                    throw Failure("Settings receipt fixture did not produce a save receipt")
                }
            }
            let panelState = BrainBarDashboardPanelState()
            let view = BrainBarUnifiedWindowPreview.make(
                collector: BrainBarDashboardFixture.makeCollector(),
                settingsViewModel: viewModel,
                panelState: panelState,
                section: scenario.section
            )
            let size = NSSize(width: breakpoint.width, height: panelState.fittingHeight)
            let name = "unified-settings-\(scenario.receipt ? "receipt" : scenario.section.rawValue)-\(breakpoint.name)"
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
            let colors = distinctSampledColorCount(in: bitmap)
            guard png.count > 5_000, colors > 16 else {
                throw Failure("\(name): refusing blank render (\(png.count) bytes, \(colors) colors)")
            }
            print("[brainbar-render] \(name) \(Int(size.width))×\(Int(size.height)); wrote \(url.path) (\(png.count) bytes, \(colors) sampled colors)")
          }
        }
    }

    private static func verifyDirectionalStatesDiffer(in outputDirectory: URL) throws {
        let draining = outputDirectory.appendingPathComponent("dashboard-cli-default-queueDraining.png")
        let backlogged = outputDirectory.appendingPathComponent("dashboard-cli-default-queueBacklogged.png")
        guard try Data(contentsOf: draining) != Data(contentsOf: backlogged) else {
            throw Failure("directional-state probe collapsed: draining and backlogged rendered byte-identically")
        }
        print("[brainbar-render] directional-state probe PASS: draining and backlogged differ")
    }

    private static func scrollViews(in view: NSView) -> [NSScrollView] {
        ((view as? NSScrollView).map { [$0] } ?? []) + view.subviews.flatMap { scrollViews(in: $0) }
    }

    private static func render(
        breakpoint: (name: String, width: CGFloat),
        scenario: Scenario,
        detailsExpanded: Bool,
        outputDirectory: URL,
        fixedHeight: CGFloat? = nil,
        afterCollapse: Bool = false
    ) throws -> String {
        let panelState = BrainBarDashboardPanelState()
        panelState.detailsExpanded = detailsExpanded || afterCollapse
        panelState.attentionExpanded = scenario == .attentionExpanded || scenario == .stale
        panelState.signalCoverageExpanded = scenario == .vectorAt100
        let collector: StatsCollector
        if scenario == .vectorAt100 {
            collector = BrainBarDashboardFixture.makeCollector(stats: BrainBarDashboardFixture.vectorAt100Stats)
        } else if scenario == .attentionCollapsed || scenario == .attentionExpanded {
            collector = BrainBarDashboardFixture.makeCollector(
                scenario.collectorState,
                agentActivity: .unavailable("fixture agent activity unavailable")
            )
        } else {
            collector = BrainBarDashboardFixture.makeCollector(scenario.collectorState)
        }
        let receipts: BrainBarOperationReceipts
        if scenario == .receiptUnavailable {
            receipts = BrainBarOperationReceipts()
        } else if scenario == .receiptFailed {
            receipts = BrainBarOperationReceipts()
            receipts.record(BrainBarOperationReceipt(
                kind: .search, durationMillis: 42, count: nil, failed: true,
                recordedAt: BrainBarDashboardFixture.fetchedAt.addingTimeInterval(-180)
            ))
        } else {
            receipts = sampleReceipts
        }
        let view = BrainBarDashboardPreview.make(
            collector: collector,
            receiptStore: receipts,
            observabilityResult: scenario.observabilityResult,
            now: BrainBarDashboardFixture.fetchedAt,
            panelState: panelState
        )
        // The XCTest renderer owns dashboard-<breakpoint>.png. Include the CLI
        // state in every filename so the two producers cannot overwrite each other.
        let suffix = detailsExpanded ? "-details-expanded" : ""
        let name = fixedHeight == nil ? "dashboard-cli-\(breakpoint.name)-\(scenario.rawValue)\(suffix)"
            : "dashboard-fixed-\(breakpoint.name)-\(afterCollapse ? "after-collapse" : detailsExpanded ? "expanded" : "collapsed")"

        let measuringHost = NSHostingView(rootView: view)
        measuringHost.frame = NSRect(x: 0, y: 0, width: breakpoint.width, height: 10_000)
        settle(measuringHost)
        let height = fixedHeight ?? ceil(panelState.fittingHeight)
        guard height.isFinite, height > 0 else {
            throw Failure("\(name): dashboard reported invalid fitting height \(height)")
        }

        let size = NSSize(width: breakpoint.width, height: height)
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(origin: .zero, size: size)
        var window: NSPanel?
        defer { window?.orderOut(nil) }
        if fixedHeight != nil {
            window = NSPanel(contentRect: NSRect(origin: NSPoint(x: -2_000, y: -2_000), size: size), styleMask: [.borderless], backing: .buffered, defer: false)
            window?.alphaValue = 0
            window?.contentView = host
            window?.orderFront(nil)
        }
        settle(host)
        if fixedHeight != nil {
            guard let scroll = scrollViews(in: host).first(where: { ($0.documentView?.bounds.height ?? 0) > 0 }),
                  let document = scroll.documentView else { throw Failure("\(name): dashboard scroll view is missing") }
            let clip = scroll.contentView
            if afterCollapse {
                clip.scroll(to: NSPoint(x: 0, y: max(document.bounds.maxY - clip.bounds.height, 0)))
                scroll.reflectScrolledClipView(clip)
                panelState.detailsExpanded = false
                settle(host)
                clip.scroll(to: .zero)
                scroll.reflectScrolledClipView(clip)
                settle(host)
            }
            let bottom = document.isFlipped ? max(document.bounds.maxY - clip.bounds.height, document.bounds.minY) : document.bounds.minY
            clip.scroll(to: NSPoint(x: 0, y: bottom))
            scroll.reflectScrolledClipView(clip)
            settle(host)
        }
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
        if fixedHeight != nil {
            guard emittedBitmap.pixelsWide * Int(height) == emittedBitmap.pixelsHigh * Int(breakpoint.width), let image = emittedBitmap.cgImage else { throw Failure("\(name): invalid fixed-height bitmap") }
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            try VNImageRequestHandler(cgImage: image).perform([request])
            for label in ["Details", detailsExpanded ? "Last seen" : "Details"] {
                guard request.results?.contains(where: {
                    $0.topCandidates(1).first?.string.localizedCaseInsensitiveContains(label) == true && $0.boundingBox.minY > 0.025
                }) == true else { throw Failure("\(name): \(label) or bottom padding clipped") }
            }
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

    private static func verifyAttentionDisclosureChangesPixels(in outputDirectory: URL) throws {
        for breakpoint in breakpoints {
            let collapsed = outputDirectory.appendingPathComponent(
                "dashboard-cli-\(breakpoint.name)-readable-attention.png"
            )
            let expanded = outputDirectory.appendingPathComponent(
                "dashboard-cli-\(breakpoint.name)-readable-attention-expanded.png"
            )
            guard try Data(contentsOf: collapsed) != Data(contentsOf: expanded) else {
                throw Failure(
                    "attention disclosure probe collapsed at \(breakpoint.width)pt: "
                        + "collapsed and expanded rendered byte-identically"
                )
            }
        }
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
