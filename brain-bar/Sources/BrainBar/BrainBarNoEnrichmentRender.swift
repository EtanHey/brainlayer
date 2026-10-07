#if DEBUG
import AppKit
import Darwin
import SwiftUI
import Vision

/// Executes before App.main. Only fixture collectors and config stores exist;
/// no daemon, status item, live database, socket, or refresh timer is started.
@MainActor
enum BrainBarNoEnrichmentRender {
    private struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    static func runIfRequested() {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_NO_ENRICHMENT_RENDER"] else { return }
        do {
            guard NSString(string: path).isAbsolutePath else { throw Failure("absolute render directory required") }
            NSApplication.shared.setActivationPolicy(.prohibited)
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let store = BrainLayerConfigStore(
                configURL: URL(fileURLWithPath: "/redacted/config.env"),
                loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) },
                saveOverride: { _ in }
            )
            let fixtureNow = BrainBarDashboardFixture.fetchedAt
            let states: [(String, BrainBarDashboardFixture.OperatorState)] = [
                ("historical-274847", .live), ("empty", .empty), ("loading", .loading),
                ("stale", .stale), ("error", .error), ("unavailable", .unavailable),
                ("partial-replay", .partialReplayDebt), ("draining", .queueDraining),
                ("backlogged", .queueBacklogged),
            ]
            var captures: [[String: Any]] = []
            var violations: [String] = []
            for (name, state) in states {
                let collector = name == "historical-274847"
                    ? BrainBarDashboardFixture.makeHistoricalBacklogCollector()
                    : BrainBarDashboardFixture.makeCollector(state)
                let panel = BrainBarDashboardPanelState()
                panel.detailsExpanded = true
                let dashboard = BrainBarDashboardPreview.make(
                    collector: collector,
                    observabilityResult: state == .unavailable
                        ? .unreadable("Synthetic unavailable") : BrainBarDashboardFixture.readableObservabilityResult,
                    now: BrainBarDashboardFixture.fetchedAt, panelState: panel
                )
                let measuring = NSHostingView(rootView: dashboard)
                measuring.frame = NSRect(x: 0, y: 0, width: 960, height: 10000)
                settle(measuring)
                captures.append(try capture(dashboard, name: "dashboard-\(name)", height: max(640, ceil(panel.fittingHeight)),
                    directory: directory, required: state == .loading ? ["Loading"] : ["Details"], violations: &violations))
            }
            for configAvailable in [true, false] {
                let pageStore = configAvailable ? store : BrainLayerConfigStore(
                    configURL: URL(fileURLWithPath: "/redacted/config.env"),
                    loadDocumentOverride: { throw Failure("Synthetic unavailable config") }, saveOverride: { _ in }
                )
                for section in BrainBarSettingsSection.allCases {
                    let model = BrainBarSettingsViewModel(
                        store: pageStore,
                        launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
                        runtimeStatusProvider: StaticBrainLayerActiveRuntimeProvider(observation: .unknown("Synthetic runtime")),
                        refreshStatusOnLoad: false,
                        now: { fixtureNow },
                        initialObservabilityResult: BrainBarDashboardFixture.readableObservabilityResult
                    )
                    let page = BrainBarUnifiedWindowPreview.make(
                        collector: BrainBarDashboardFixture.makeHistoricalBacklogCollector(),
                        settingsViewModel: model, panelState: BrainBarDashboardPanelState(), section: section
                    )
                    captures.append(try capture(page, name: "\(section.rawValue)-\(configAvailable ? "known" : "unknown")",
                        height: 1500, directory: directory, required: ["Memory on this Mac"], violations: &violations))
                }
            }
            captures.append(try capture(
                BrainBarPendingStoreQueuePreview.make(stats: BrainBarDashboardFixture.makeHistoricalBacklogCollector().stats),
                name: "pending-store-replay", height: 500, directory: directory,
                required: ["Pending stores", "Replay debt"], violations: &violations
            ))
            // Positive control: fail closed if the OCR cannot see either forbidden shape.
            let control = AnyView(VStack(spacing: 20) {
                Text("Enrichment paused · 274,847 queued")
                Text("Enrichment retired")
            }.font(.system(size: 24)).padding(30).background(Color.white).foregroundStyle(Color.black))
            let controlCapture = try capture(control, name: "ocr-positive-control", height: 220,
                directory: directory, required: ["Enrichment"], violations: nil)
            guard let controlText = controlCapture["text"] as? String,
                  controlText.lowercased().contains("enrichment") else { throw Failure("forbidden-text control unreadable") }
            let active = captures.first { ($0["name"] as? String) == "pending-store-replay" }?["text"] as? String ?? ""
            guard active.lowercased().contains("pending stores"), active.lowercased().contains("replay") else {
                throw Failure("active pending-store/replay UI was not rendered")
            }
            let report: [String: Any] = [
                "schema_version": 1, "row": "BrainBar renders no enrichment status",
                "status": violations.isEmpty ? "PASS" : "FAIL", "historical_backlog": 274847,
                "captures": captures, "violations": violations, "positive_control": controlCapture,
                "active_pending_store_visible": true, "mode": "synthetic-source-build",
                "measured_sha": ProcessInfo.processInfo.environment["BRAINBAR_RENDER_SHA"] ?? "unknown",
            ]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent("render-report.json"), options: .atomic)
            guard violations.isEmpty else { throw Failure(violations.joined(separator: "; ")) }
            print("BrainBar renders no enrichment status: PASS (\(captures.count) actual-view captures)")
            Darwin.exit(EXIT_SUCCESS)
        } catch {
            fputs("BrainBar renders no enrichment status: FAIL \(error.localizedDescription)\n", stderr)
            Darwin.exit(EXIT_FAILURE)
        }
    }

    private static func settle(_ host: NSView) {
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
        host.layoutSubtreeIfNeeded()
    }

    private static func capture(_ view: AnyView, name: String, height: CGFloat, directory: URL,
                                required: [String], violations: inout [String]) throws -> [String: Any] {
        let capture = try capture(view, name: name, height: height, directory: directory, required: required, violations: nil)
        let text = capture["text"] as? String ?? ""
        if text.lowercased().contains("enrich") || text.contains("274,847") || text.contains("274847") {
            violations.append("\(name): forbidden enrichment UI: \(text)")
        }
        return capture
    }

    private static func capture(_ view: AnyView, name: String, height: CGFloat, directory: URL,
                                required: [String], violations: Never?) throws -> [String: Any] {
        let host = NSHostingView(rootView: view)
        host.appearance = NSAppearance(named: .darkAqua)
        host.frame = NSRect(x: 0, y: 0, width: 960, height: height)
        settle(host)
        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { throw Failure("\(name): bitmap unavailable") }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        guard let image = bitmap.cgImage, let png = bitmap.representation(using: .png, properties: [:]), png.count > 5000 else {
            throw Failure("\(name): missing/blank render")
        }
        try png.write(to: directory.appendingPathComponent("\(name).png"), options: .atomic)
        var lines = Set<String>()
        for _ in 0..<3 {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            try VNImageRequestHandler(cgImage: image).perform([request])
            for observation in request.results ?? [] {
                if let text = observation.topCandidates(1).first?.string { lines.insert(text) }
            }
        }
        let text = lines.sorted().joined(separator: "\n")
        for label in required where !text.localizedCaseInsensitiveContains(label) { throw Failure("\(name): OCR missing \(label)") }
        print("[no-enrichment-render] \(name): \(png.count) bytes, \(lines.count) OCR lines")
        return ["name": name, "png": "\(name).png", "text": text, "width": 960, "height": Int(height)]
    }
}
#endif
