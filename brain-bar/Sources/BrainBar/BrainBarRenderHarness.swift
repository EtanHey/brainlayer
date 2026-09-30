#if DEBUG
import AppKit
import BrainBarLifecycle
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
        case staleDocument
        case pausedGrowing
        case runningGrowing
        case loading
        case queueDraining
        case queueBacklogged
        case receiptUnavailable
        case receiptFailed
        case vectorAt100 = "vector-at-100"

        var detailsStates: [Bool] {
            switch self {
            case .loading, .stale, .pausedGrowing, .runningGrowing,
                 .attentionCollapsed, .attentionExpanded, .queueDraining, .queueBacklogged,
                 .receiptUnavailable, .receiptFailed:
                [false]
            case .vectorAt100:
                [true]
            case .readable, .unreadable, .staleDocument:
                [false, true]
            }
        }

        var breakpoints: [(name: String, width: CGFloat)] {
            switch self {
            case .queueDraining, .queueBacklogged, .receiptUnavailable, .receiptFailed, .vectorAt100:
                [("default", 960)]
            case .readable, .attentionCollapsed, .attentionExpanded, .unreadable, .stale,
                 .staleDocument, .pausedGrowing, .runningGrowing, .loading:
                BrainBarRenderHarness.breakpoints
            }
        }

        var collectorState: BrainBarDashboardFixture.OperatorState {
            switch self {
            case .loading:
                .loading
            case .stale:
                .stale
            case .attentionCollapsed, .attentionExpanded, .staleDocument, .pausedGrowing, .runningGrowing:
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
            case .staleDocument:
                BrainBarDashboardFixture.staleObservabilityResult
            case .readable, .stale, .pausedGrowing, .runningGrowing,
                 .attentionCollapsed, .attentionExpanded, .loading, .queueDraining,
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
            try renderWatcherTruth(in: outputDirectory)
            try renderMainWindowShell(in: outputDirectory)
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
                let inPlace = try render(
                    breakpoint: breakpoint, scenario: .readable, detailsExpanded: true,
                    outputDirectory: outputDirectory, fixedHeight: 640, collapseInPlace: true
                )
                print("[brainbar-render] \(inPlace)")
            }
            try verifyCollapseInPlaceLeavesNoBlankTail(in: outputDirectory)
            try verifyDirectionalStatesDiffer(in: outputDirectory)
            try verifyAttentionDisclosureChangesPixels(in: outputDirectory)
            Darwin.exit(EXIT_SUCCESS)
        } catch {
            BrainBarSignalSafety.write(
                Data("[brainbar-render] ERROR: \(error.localizedDescription)\n".utf8),
                to: .standardError,
                context: "render error"
            )
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

    /// #966: each watcher-health state rendered on both surfaces that show it: the Dashboard
    /// (hero + watcher lane) and Settings → Jobs (Ingest card + footer). One status, one reason.
    private static func renderWatcherTruth(in outputDirectory: URL) throws {
        let now = BrainBarDashboardFixture.fetchedAt
        let degradedFile = WatcherHealthFileRead.readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-70),
            pollCount: 72,
            alertReasons: ["file_ingestion_failure"],
            fileIngestionFailureCount: 2,
            earliestFileIngestionFailureAt: now.addingTimeInterval(-7_200)
        ))
        let missingFile = WatcherHealthFileRead.missing(path: "~/.local/share/brainlayer/watcher-health.json")
        let states: [(name: String, dashboard: BrainBarDashboardFixture.OperatorState,
                      launchd: BrainLayerLaunchdLoadState, file: WatcherHealthFileRead, title: String)] = [
            ("running", .live, .running, BrainBarDashboardFixture.healthyWatcherHealth, "Watcher running"),
            ("idle-replay-debt", .watcherIdleWithReplayDebt, .running, BrainBarDashboardFixture.healthyWatcherHealth, "Watcher running"),
            ("degraded", .watcherDegraded, .running, degradedFile, "Watcher needs attention"),
            ("stopped", .watcherOffline, .unloaded, BrainBarDashboardFixture.healthyWatcherHealth, "Watcher stopped"),
            ("unknown", .watcherHealthMissing, .running, missingFile, "Watcher status unknown"),
        ]
        let fixtureDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-watcher-render-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: fixtureDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDirectory) }
        let store = BrainLayerConfigStore(configURL: fixtureDirectory.appendingPathComponent("watcher-render.env"))
        try store.save(.defaultConfig)

        for state in states {
            // #1014 R1 B1: the watcher lane card's pill comes from the same state as the headline.
            // Render it on its own (the Dashboard sheets show Details collapsed) and refuse a
            // contradicting pill.
            let laneFlow = DashboardFlowSummary.derive(
                daemon: nil, stats: BrainBarDashboardFixture.makeCollector(state.dashboard).stats, now: now
            )
            let lane = laneFlow.lane(for: .jsonlWatcher)
            let expectedPills: [DashboardFlowLaneStatus] = switch laneFlow.watcherStatus {
            case .running: [.live, .idle, .running]
            case .degraded: [.attention]
            case .stopped: [.stopped]
            case .unknown: [.unknown]
            }
            guard expectedPills.contains(lane.status) else {
                throw Failure("watcher-\(state.name): lane pill \(lane.status.label) contradicts \(laneFlow.watcherStatus.title)")
            }
            try writeWatcherRender(
                BrainBarFlowLaneCardPreview.make(lane: lane, fetchedAt: now),
                size: NSSize(width: 380, height: 320),
                name: "watcher-\(state.name)-lane-card",
                in: outputDirectory
            )

            for breakpoint in breakpoints {
                let collector = BrainBarDashboardFixture.makeCollector(state.dashboard)
                let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: now)
                guard flow.watcherStatus.title == state.title else {
                    throw Failure("watcher-\(state.name): dashboard status \(flow.watcherStatus.title), expected \(state.title)")
                }
                let dashboardState = BrainBarDashboardPanelState()
                dashboardState.attentionExpanded = true
                let dashboard = BrainBarDashboardPreview.make(
                    collector: collector,
                    receiptStore: sampleReceipts,
                    observabilityResult: BrainBarDashboardFixture.readableObservabilityResult,
                    now: now,
                    panelState: dashboardState
                )
                let measuring = NSHostingView(rootView: dashboard)
                measuring.frame = NSRect(x: 0, y: 0, width: breakpoint.width, height: 10_000)
                settle(measuring)
                try writeWatcherRender(
                    dashboard,
                    size: NSSize(width: breakpoint.width, height: ceil(dashboardState.fittingHeight)),
                    name: "watcher-\(state.name)-dashboard-\(breakpoint.name)",
                    in: outputDirectory
                )

                let viewModel = BrainBarSettingsViewModel(
                    store: store,
                    launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
                    runtimeStatusProvider: StaticBrainLayerActiveRuntimeProvider(
                        observation: .unknown("Fixture runtime state unavailable.")
                    ),
                    initialLaunchdObservations: [
                        .watch: .init(loadState: state.launchd, runs: 7, lastExitCode: 0,
                                      lastRunAt: now.addingTimeInterval(-3_600), nextRunAt: nil, isContinuous: true),
                        .index: .init(loadState: .loaded, runs: 4, lastExitCode: 0,
                                      lastRunAt: now.addingTimeInterval(-1_800),
                                      nextRunAt: now.addingTimeInterval(1_800), isContinuous: false),
                        .maintenanceNightly: .init(loadState: .loaded, runs: 3, lastExitCode: 0,
                                                   lastRunAt: now.addingTimeInterval(-43_200),
                                                   nextRunAt: now.addingTimeInterval(43_200), isContinuous: false),
                        .maintenanceWeekly: .init(loadState: .loaded, runs: 1, lastExitCode: 0,
                                                  lastRunAt: now.addingTimeInterval(-259_200),
                                                  nextRunAt: now.addingTimeInterval(345_600), isContinuous: false),
                    ],
                    refreshStatusOnLoad: false,
                    now: { now },
                    initialObservabilityResult: .unreadable("Fixture backup status unavailable."),
                    initialWatcherHealth: state.file
                )
                guard viewModel.footerPresentation.state.title == state.title else {
                    throw Failure("watcher-\(state.name): footer \(viewModel.footerPresentation.state.title), expected \(state.title)")
                }
                let settingsState = BrainBarDashboardPanelState()
                let settings = BrainBarUnifiedWindowPreview.make(
                    collector: collector,
                    settingsViewModel: viewModel,
                    panelState: settingsState,
                    section: .jobs
                )
                try writeWatcherRender(
                    settings,
                    size: NSSize(width: breakpoint.width, height: max(settingsState.fittingHeight, 640)),
                    name: "watcher-\(state.name)-settings-\(breakpoint.name)",
                    in: outputDirectory
                )
            }
        }
    }

    private static func writeWatcherRender(_ view: some View, size: NSSize, name: String, in outputDirectory: URL) throws {
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
        guard png.count > 5_000, distinctSampledColorCount(in: bitmap) > 16 else {
            throw Failure("\(name): refusing a blank or trivial render (\(png.count) PNG bytes)")
        }
        let url = outputDirectory.appendingPathComponent("\(name).png")
        try png.write(to: url, options: .atomic)
        print("[brainbar-render] \(name) \(Int(size.width))×\(Int(size.height)); wrote \(url.path)")
    }

    private static func renderUnifiedSettings(in outputDirectory: URL) throws {
        let fixtureDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-settings-render-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: fixtureDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDirectory) }
        let configURL = fixtureDirectory.appendingPathComponent("settings-render-fixture.env")
        let store = BrainLayerConfigStore(configURL: configURL)
        try store.save(.defaultConfig)

        // #968: the Backups section with every schedule known, and with the transcript archive's
        // LaunchAgent missing (honest unknown, no local copy, so no Reveal/Copy).
        // Fixture times are built in the local calendar, like the LaunchAgent schedules they describe.
        func at(_ day: Int, _ hour: Int, _ minute: Int, month: Int = 9) -> Date {
            Calendar.current.date(from: DateComponents(year: 2026, month: month, day: day, hour: hour, minute: minute))!
        }
        let show = DashboardMetricFormatter.jobDateTimeString
        let backupRows: [BrainBarBackupScheduleRow] = [
            .init(title: "Database", cadence: "daily at 03:17",
                  lastRun: "Last run \(show(at(29, 3, 17))) · verified", nextRun: "Next run \(show(at(30, 3, 17)))",
                  localCopy: URL(fileURLWithPath: "/Users/fixture/.local/share/brainlayer/backups/2026-09-29.db.gz")),
            .init(title: "Transcripts", cadence: "daily at 05:00",
                  lastRun: "Last run \(show(at(29, 5, 1))) · verified", nextRun: "Next run \(show(at(30, 5, 0)))",
                  localCopy: URL(fileURLWithPath: "/Users/fixture/.local/share/brainlayer/jsonl-backups/claude-jsonl-2026-09-29.tar.gz")),
            .init(title: "Weekly maintenance", cadence: "weekly on Sunday at 04:00",
                  lastRun: "Last run \(show(at(27, 4, 31)))", nextRun: "Next run \(show(at(4, 4, 0, month: 10)))", localCopy: nil),
        ]
        var unknownRows = backupRows
        unknownRows[1] = .init(
            title: "Transcripts",
            cadence: "Schedule unknown — no LaunchAgent installed at ~/Library/LaunchAgents/com.brainlayer.jsonl-backup.plist",
            lastRun: "No run recorded in jsonl-backup.log", nextRun: "Next run unknown", localCopy: nil
        )
        let settingsScenarios: [(section: BrainBarSettingsSection, receipt: Bool, backups: [BrainBarBackupScheduleRow], suffix: String)] =
            BrainBarSettingsSection.allCases.map { ($0, false, $0 == .backups ? backupRows : [], "") }
            + [(.general, true, [], ""), (.backups, false, unknownRows, "-unknown")]
        for scenario in settingsScenarios {
          for breakpoint in breakpoints {
            if scenario.receipt { try store.save(.defaultConfig) }
            let viewModel = BrainBarSettingsViewModel(
                store: store,
                launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
                runtimeStatusProvider: StaticBrainLayerActiveRuntimeProvider(
                    observation: .unknown("Fixture runtime state unavailable.")
                ),
                initialLaunchdObservations: scenario.section == .jobs ? [
                    .watch: .init(loadState: .running, runs: 7, lastExitCode: 0,
                                  lastRunAt: Date(timeIntervalSince1970: 1_790_164_800), nextRunAt: nil,
                                  isContinuous: true),
                    .index: .init(loadState: .loaded, runs: 4, lastExitCode: 1,
                                  lastRunAt: Date(timeIntervalSince1970: 1_790_161_200),
                                  nextRunAt: Date(timeIntervalSince1970: 1_790_208_900), isContinuous: false),
                    .maintenanceNightly: .init(loadState: .loaded, runs: 0, lastExitCode: nil,
                                               lastRunAt: nil, nextRunAt: Date(timeIntervalSince1970: 1_790_211_600),
                                               isContinuous: false),
                    .maintenanceWeekly: .init(loadState: .loaded, runs: 2, lastExitCode: 0,
                                              lastRunAt: Date(timeIntervalSince1970: 1_789_866_900),
                                              nextRunAt: Date(timeIntervalSince1970: 1_790_471_700), isContinuous: false),
                ] : scenario.section == .backups ? [
                    .backupDaily: .init(loadState: .loaded, runs: 3, lastExitCode: 0,
                                        lastRunAt: at(29, 3, 17), nextRunAt: at(30, 3, 17), isContinuous: false),
                    .jsonlBackup: .init(loadState: .loaded, runs: 3, lastExitCode: 0,
                                        lastRunAt: at(29, 5, 1), nextRunAt: at(30, 5, 0), isContinuous: false),
                ] : [:],
                refreshStatusOnLoad: false,
                initialObservabilityResult: .unreadable("Fixture backup status unavailable."),
                initialBackupSchedules: scenario.backups
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
            let name = "unified-settings-\(scenario.receipt ? "receipt" : scenario.section.rawValue)\(scenario.suffix)-\(breakpoint.name)"
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
        afterCollapse: Bool = false,
        collapseInPlace: Bool = false
    ) throws -> String {
        let panelState = BrainBarDashboardPanelState()
        panelState.detailsExpanded = detailsExpanded || afterCollapse
        panelState.attentionExpanded = scenario == .attentionExpanded || scenario == .stale
        panelState.signalCoverageExpanded = scenario == .vectorAt100 || (scenario == .readable && detailsExpanded)
        let collector: StatsCollector
        if scenario == .pausedGrowing || scenario == .runningGrowing {
            collector = BrainBarDashboardFixture.makeCollector(stats: BrainBarDashboardFixture.growingQueueStats)
        } else if scenario == .vectorAt100 {
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
            panelState: panelState,
            enrichmentPausedOverride: scenario == .pausedGrowing ? true : false
        )
        // The XCTest renderer owns dashboard-<breakpoint>.png. Include the CLI
        // state in every filename so the two producers cannot overwrite each other.
        let suffix = detailsExpanded ? "-details-expanded" : ""
        let name = fixedHeight == nil ? "dashboard-cli-\(breakpoint.name)-\(scenario.rawValue)\(suffix)"
            : "dashboard-fixed-\(breakpoint.name)-\(collapseInPlace ? "collapsed-in-place" : afterCollapse ? "after-collapse" : detailsExpanded ? "expanded" : "collapsed")"

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
            if collapseInPlace {
                // #964: expand, scroll to the bottom, collapse, and capture exactly where the
                // viewport settles. No manual re-scroll, so a pinned document height shows up
                // as a blank tail in the pixels.
                clip.scroll(to: NSPoint(x: 0, y: max(document.bounds.maxY - clip.bounds.height, 0)))
                scroll.reflectScrolledClipView(clip)
                settle(host)
                panelState.detailsExpanded = false
                settle(host)
            } else if afterCollapse {
                clip.scroll(to: NSPoint(x: 0, y: max(document.bounds.maxY - clip.bounds.height, 0)))
                scroll.reflectScrolledClipView(clip)
                panelState.detailsExpanded = false
                settle(host)
                clip.scroll(to: .zero)
                scroll.reflectScrolledClipView(clip)
                settle(host)
            }
            if !collapseInPlace {
                let bottom = document.isFlipped ? max(document.bounds.maxY - clip.bounds.height, document.bounds.minY) : document.bounds.minY
                clip.scroll(to: NSPoint(x: 0, y: bottom))
                scroll.reflectScrolledClipView(clip)
                settle(host)
            }
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
            for label in ["Details", detailsExpanded && !collapseInPlace ? "Last seen" : "Details"] {
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

    /// #964: collapsing Details while scrolled to the bottom must settle on the same pixels as
    /// the collapsed dashboard scrolled to its bottom. A pinned document height leaves a blank
    /// tail in the viewport and fails this comparison.
    private static func verifyCollapseInPlaceLeavesNoBlankTail(in outputDirectory: URL) throws {
        for breakpoint in breakpoints {
            let collapsed = outputDirectory.appendingPathComponent("dashboard-fixed-\(breakpoint.name)-collapsed.png")
            let inPlace = outputDirectory.appendingPathComponent("dashboard-fixed-\(breakpoint.name)-collapsed-in-place.png")
            guard try Data(contentsOf: collapsed) == Data(contentsOf: inPlace) else {
                throw Failure(
                    "collapse-in-place probe at \(breakpoint.width)pt: collapsing Details at the bottom "
                        + "did not settle on the collapsed dashboard (blank tail or lost reflow)"
                )
            }
        }
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

    private final class RenderDefaults: BrainBarKeyValueStoring {
        var values: [String: String] = [:]
        func string(forKey defaultName: String) -> String? { values[defaultName] }
        func setString(_ value: String?, forKey defaultName: String) { values[defaultName] = value }
    }

    /// #963 PR 1: the real `BrainBarMainWindow`, captured with its title bar, at each width,
    /// and the status item's icon. The window is built by the production controller from a
    /// fixture runtime; nothing is shown on screen and no frame is saved to user defaults.
    private static func renderMainWindowShell(in outputDirectory: URL) throws {
        for breakpoint in breakpoints {
            let runtime = BrainBarRuntime()
            runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil)
            let controller = BrainBarDashboardPanelController(
                runtime: runtime,
                frameStore: BrainBarWindowFrameStore(defaults: RenderDefaults(), key: "render")
            )
            let window = controller.windowForTesting
            window.setFrame(NSRect(x: 0, y: 0, width: breakpoint.width, height: 640), display: false)
            guard let frameView = window.contentView?.superview else {
                throw Failure("window-\(breakpoint.name): the window has no frame view")
            }
            settle(frameView)
            try writeBitmap(of: frameView, name: "window-dashboard-\(breakpoint.name)", in: outputDirectory)
        }

        let stats = BrainBarDashboardFixture.makeCollector().stats
        for badgeOn in [false, true] {
            let icon = SparklineRenderer.renderStatusBarIcon(
                agent: stats.recentAgentWriteBuckets,
                watcher: stats.recentWatcherWriteBuckets,
                enrichment: stats.recentEnrichmentBuckets,
                badgeOn: badgeOn,
                size: NSSize(width: 26, height: 14)
            )
            // Shown 8x on a menu-bar-dark strip so the 26x14 pt icon is inspectable.
            let strip = NSImage(size: NSSize(width: 26 * 8 + 32, height: 14 * 8 + 32), flipped: false) { rect in
                NSColor(calibratedWhite: 0.13, alpha: 1).setFill()
                rect.fill()
                icon.draw(in: NSRect(x: 16, y: 16, width: 26 * 8, height: 14 * 8))
                return true
            }
            guard let tiff = strip.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff),
                  let png = bitmap.representation(using: .png, properties: [:]) else {
                throw Failure("status-item icon: AppKit could not encode PNG data")
            }
            let name = "status-item-icon\(badgeOn ? "-attention" : "")"
            let url = outputDirectory.appendingPathComponent("\(name).png")
            try png.write(to: url, options: .atomic)
            print("[brainbar-render] \(name); wrote \(url.path)")
        }
    }

    private static func writeBitmap(of view: NSView, name: String, in outputDirectory: URL) throws {
        guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
            throw Failure("\(name): AppKit could not allocate an off-screen bitmap")
        }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            throw Failure("\(name): AppKit could not encode PNG data")
        }
        let colors = distinctSampledColorCount(in: bitmap)
        guard png.count > 5_000, colors > 16 else {
            throw Failure("\(name): refusing blank render (\(png.count) bytes, \(colors) colors)")
        }
        let url = outputDirectory.appendingPathComponent("\(name).png")
        try png.write(to: url, options: .atomic)
        print("[brainbar-render] \(name) \(Int(view.bounds.width))×\(Int(view.bounds.height)); wrote \(url.path)")
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
