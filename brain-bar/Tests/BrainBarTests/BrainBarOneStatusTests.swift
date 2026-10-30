import AppKit
import Darwin
import SwiftUI
import XCTest
@testable import BrainBar

/// One status truth (Etan, 2026-10-29: the M1 showed "skipped" in Jobs beside "All good"; the M4
/// showed a failed night that a later retry had recovered). Every state below is driven through
/// the real app model, the real menu-bar controller and the real Dashboard derivation, and every
/// surface must tell the same story. A new `Surface` or `Fixture` case does not compile until it
/// joins the table.
@MainActor
final class BrainBarOneStatusTests: XCTestCase {
    typealias T = BrainBarStatusSnapshotTests

    /// What a surface says, reduced to the one question they must agree on.
    enum Tone: Equatable { case healthy, notice, attention, checking }

    enum Surface: CaseIterable {
        case icon, menuLine, maintenanceRow, panel, headline
    }

    enum Fixture: CaseIterable {
        case healthy, skippedWithReason, failedNight, recoveredNight, unreadableSource, noticeOnly, vectorScan

        /// The tone every surface must show, and how the Jobs → Maintenance row must read.
        var expected: (tone: Tone, row: BrainLayerLaunchdGroupHealth) {
            switch self {
            case .healthy, .recoveredNight, .vectorScan: (.healthy, .healthy)
            case .skippedWithReason: (.notice, .skipped)
            case .failedNight: (.attention, .unhealthy)
            case .unreadableSource: (.notice, .unknown)
            case .noticeOnly: (.notice, .healthy)
            }
        }
    }

    struct Reading {
        let tone: Tone?
        /// The words the surface shows for this state; empty when it shows none.
        let texts: [String]
    }

    // MARK: fixtures

    private static let scheduledDate = "2026-10-26"

    private static func ts(_ date: String, _ time: String) -> Double {
        ISO8601DateFormatter().date(from: "\(date)T\(time)Z")!.timeIntervalSince1970
    }

    private static func night(_ date: String, failing: [String] = [], abandoned: Bool = false) -> [[String: Any]] {
        var rows: [[String: Any]] = [["event": "runner", "kind": "nightly", "scheduled_date": date,
                                      "steps": ["maintenance-nightly", "jsonl-backup"], "ts": ts(date, "01:15:00")]]
        if abandoned {
            rows.append(["event": "abandoned", "kind": "nightly", "scheduled_date": date,
                         "incomplete_steps": ["maintenance-nightly"], "reason": "runner-vanished", "ts": ts(date, "03:00:00")])
        } else {
            for step in ["maintenance-nightly", "jsonl-backup"] {
                let failed = failing.contains(step)
                rows.append(["event": "finished", "step": step, "status": failed ? "failed" : "ok",
                             "returncode": failed ? 1 : 0, "ts": ts(date, "01:20:00")])
            }
        }
        rows.append(["event": "summary", "kind": "nightly", "scheduled_date": date,
                     "failed": abandoned ? 1 : failing.count, "ts": ts(date, "03:00:00")])
        return rows
    }

    private static func recovery(_ date: String, step: String) -> [[String: Any]] {
        [["event": "transcript-recovery", "kind": "recovery", "scheduled_date": date, "steps": [step], "ts": ts(date, "07:40:00")],
         ["event": "finished", "step": step, "status": "ok", "returncode": 0, "ts": ts(date, "07:42:00")],
         ["event": "summary", "kind": "recovery", "scheduled_date": date, "failed": 0, "ts": ts(date, "07:44:00")]]
    }

    /// The maintenance observations the real launchd/calendar provider reads from this journal.
    private func calendarObservations(_ rows: [[String: Any]]) throws -> [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] {
        let canonical = try XCTUnwrap(realpath(brainBarTestTemporaryDirectory().path, nil))
        defer { free(canonical) }
        let directory = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var data = Data()
        for row in rows { data.append(try JSONSerialization.data(withJSONObject: row)); data.append(10) }
        try data.write(to: directory.appendingPathComponent("runs.jsonl"))
        let history = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertNil(history.error)
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { _ in .init(terminationStatus: 0, output: "runs = 0") },
            now: { T.now }, ownership: { .supervisor }, calendarHistory: { history },
            calendarSchedule: { DateComponents(hour: 3, minute: 15) }, calendarScheduleProblem: { nil })
        // The weekly keeps its own healthy history; the journal is about the nightly plan.
        return [.maintenanceNightly: try XCTUnwrap(provider.sampleActivity()[.maintenanceNightly])]
    }

    private func inputs(_ fixture: Fixture) throws -> BrainBarStatusInputs {
        var inputs = T.healthy()
        func setMaintenance(_ observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]) {
            for (job, observation) in observations { inputs.observations?[job] = observation }
        }
        switch fixture {
        case .healthy:
            break
        case .skippedWithReason:
            // The M1 shape: a verified, deliberate gate deferral (exit 75) of last night's nightly.
            let started = T.hoursAgo(11)
            var nightly = T.step(code: 75, at: started)
            nightly.calendarStartedAt = started
            nightly.calendarDeferralRecord = .aborted(
                reason: "unexpected writer holds BrainLayer DB: pid=812 command='python3 -m brainlayer.enrich' fd=7u",
                writtenAt: started.addingTimeInterval(1))
            setMaintenance([.maintenanceNightly: nightly])
        case .failedNight:
            // The M4 shape: two abandoned nights, then a failed transcript backup nobody retried.
            setMaintenance(try calendarObservations(Self.night("2026-10-24", abandoned: true)
                + Self.night("2026-10-25", abandoned: true) + Self.night(Self.scheduledDate, failing: ["jsonl-backup"])))
        case .recoveredNight:
            // #1550: the same nights, but a later retry completed the failed step.
            setMaintenance(try calendarObservations(Self.night("2026-10-24", abandoned: true)
                + Self.night("2026-10-25", abandoned: true) + Self.night(Self.scheduledDate, failing: ["jsonl-backup"])
                + Self.recovery(Self.scheduledDate, step: "jsonl-backup")))
        case .unreadableSource:
            let unreadable = BrainLayerLaunchdJobObservation.stateOnly(.probeError("Maintenance calendar status unavailable"))
            setMaintenance([.maintenanceNightly: unreadable, .maintenanceWeekly: unreadable])
        case .vectorScan:
            inputs.badge = .init(badgeOn: false, reason: "old partial scan notice", activeCodes: ["missing_embeddings_scan_incomplete"],
                activeMessages: ["missing embeddings partial: scanned 1864 rows (instruction_limit); not an exact count"],
                activeSeverities: ["warning"], vectorScan: .init(completeness: "partial", scanned: 1864, missing: 0, reason: "instruction_limit"))
        case .noticeOnly:
            inputs.jobAlerts = BrainBarJobAlerts(active: [
                "fleet-watchdog-hold": "BrainLayer recovered a stale fleet-watchdog hold; supervisor restored"])
        }
        return inputs
    }

    // MARK: the real surfaces

    private struct Rig {
        let model: BrainBarSettingsViewModel
        let controller: BrainBarStatusPopoverController
        let page: BrainBarOnePagePresentation
    }

    private func collector(_ inputs: BrainBarStatusInputs) -> StatsCollector {
        let base = BrainBarDashboardFixture.makeCollector(watcherHeartbeatAt: inputs.now)
        guard inputs.badge?.vectorScan != nil else { return base }
        return .fixture(stats: base.stats.withSignalCoverage(.init(eligibleChunkCount: 100,
            vectorIndexedChunkCount: 95, ftsIndexedChunkCount: 100, trigramIndexedChunkCount: 100)),
            daemon: base.daemon, agentActivity: base.agentActivity, state: base.state, lastDataFetchedAt: inputs.now)
    }

    private func rig(_ inputs: BrainBarStatusInputs) throws -> Rig {
        guard case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else {
            throw XCTSkip("fixture must be readable")
        }
        let backups: ObservabilityDocument.Backups = if case let .measured(value) = inputs.backups { value } else { T.backups() }
        let document = base.replacingBackupsForStatusTests(backups, generatedAt: inputs.now)
        var config = BrainLayerConfig.defaultConfig
        for job in BrainLayerLaunchdJob.allCases {
            config.launchdJobs[job] = .init(enabled: true, loadState: job == .watch ? .running : .loaded)
        }
        let model = BrainBarSettingsViewModel(
            workspace: BrainBarRecordingWorkspace(),
            store: BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/fixture/config"),
                                         loadDocumentOverride: { BrainLayerEnvDocument(config: config) }),
            initialLaunchdObservations: inputs.observations!, refreshStatusOnLoad: false, now: { inputs.now },
            initialObservabilityResult: .readable(document), observabilityCadence: .known(300),
            initialWatcherHealth: .readable(.init(updatedAt: inputs.now.addingTimeInterval(-30), pollCount: 50)),
            initialMaintenanceEvidence: inputs.maintenance)
        let badge = inputs.badge ?? .init(badgeOn: false, reason: "", activeCodes: [])
        model.applyHealthBadge(badge, alerts: inputs.jobAlerts ?? BrainBarJobAlerts(active: [:]))

        let collector = collector(inputs)
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        if inputs.badge?.vectorScan != nil { runtime.install(collector: collector, database: nil) }
        runtime.jobsModel = model
        runtime.driveAuth.setForPreview(status: .init(state: .valid, reason: nil, expiresAt: inputs.now.addingTimeInterval(30 * 86_400)))
        let workspace = BrainBarRecordingWorkspace()
        let controller = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime,
            dashboardPanelController: BrainBarDashboardPanelController(workspace: workspace, runtime: runtime),
            statusBar: TestStatusItemHost())
        controller.menuNeedsUpdate(controller.contextMenuForTesting)

        // Exercise the production renderer without starting collector refreshes or file readers.
        controller.renderStatusIconForTesting(stats: collector.stats, state: collector.state)
        _ = try XCTUnwrap(controller.renderedStatusDotForTesting, "fixture icon must actually render")
        XCTAssertNotNil(controller.statusItemForTesting.button?.image)

        let page = BrainBarDashboardPreview.jobsPresentation(collector: collector, model: model,
                                                            workspace: workspace, now: inputs.now)
        return Rig(model: model, controller: controller, page: page)
    }

    private func read(_ surface: Surface, _ rig: Rig) -> Reading {
        switch surface {
        case .icon:
            return .init(tone: rig.controller.renderedStatusDotForTesting == true ? .attention : nil, texts: [])
        case .menuLine:
            let title = rig.controller.contextMenuForTesting.items.first?.title ?? ""
            let tone: Tone = title.hasPrefix("Needs attention:") ? .attention : title.hasPrefix("Info:") ? .notice
                : title == "Checking status…" ? .checking : .healthy
            let tooltip = rig.controller.contextMenuForTesting.items.first?.toolTip
            return .init(tone: tone, texts: tooltip.map { [$0] } ?? [])
        case .maintenanceRow:
            let row = rig.model.groupStatus(.maintenance)
            let tone: Tone = switch row.health {
            case .healthy: .healthy
            case .checking: .checking
            case .unhealthy: .attention
            case .awaitingRun, .unknown, .skipped, .deferred, .notice: .notice
            }
            return .init(tone: tone, texts: row.attentionReason.map { [$0] } ?? [])
        case .panel:
            return .init(tone: rig.page.attentionItems.isEmpty ? nil : .attention, texts: rig.page.attentionItems)
        case .headline:
            let status = rig.page.status
            let tone: Tone = switch status.tone {
            case .amber: .attention
            case .green: .healthy
            case .neutral: status.headline == "Checking…" ? .checking : .notice
            }
            return .init(tone: tone, texts: [status.headline] + (status.reason.map { [$0] } ?? []))
        }
    }

    // MARK: the table

    func testEveryStateReadsTheSameOnEverySurface() throws {
        for fixture in Fixture.allCases {
            let name = "\(fixture)"
            let rig = try rig(try inputs(fixture))
            defer { rig.controller.stop() }
            let (tone, rowHealth) = fixture.expected
            let readings = Dictionary(uniqueKeysWithValues: Surface.allCases.map { ($0, read($0, rig)) })
            let verdict = rig.model.publishedStatus.verdict
            let shared = verdict.attention + verdict.notices

            for surface in Surface.allCases {
                let reading = readings[surface]!
                switch surface {
                case .icon, .panel:
                    // Two-state surfaces: they light up for a dot and stay quiet otherwise.
                    XCTAssertEqual(reading.tone == .attention, tone == .attention, "\(name): \(surface) \(reading)")
                case .maintenanceRow:
                    XCTAssertEqual(rig.model.groupStatus(.maintenance).health, rowHealth, "\(name): \(surface) \(reading)")
                    if reading.tone == .attention { XCTAssertEqual(tone, .attention, "\(name): a red row needs the dot") }
                case .menuLine, .headline:
                    XCTAssertEqual(reading.tone, tone, "\(name): \(surface) \(reading)")
                }
                // Whatever a surface says comes from the one shared verdict.
                for text in reading.texts where surface != .headline {
                    XCTAssertTrue(shared.contains(text) || text == shared.joined(separator: "; "),
                                  "\(name): \(surface) says \(text.debugDescription), not in the shared verdict \(shared)")
                }
                // A plain reason: no exit codes, process ids or file descriptors.
                for text in reading.texts {
                    XCTAssertNil(text.range(of: #"\bexit \d|pid=|fd=|returncode"#, options: .regularExpression),
                                 "\(name): \(surface) shows a raw id: \(text)")
                }
            }

            // Observe the rendered headline/reason and panel items after both forwarding points.
            XCTAssertEqual(rig.page.status, verdict.dashboardStatus, name)
            // The panel count is the headline's count, and its items are the verdict's dots.
            XCTAssertEqual(rig.page.attentionItems, verdict.attention, name)
            if !verdict.attention.isEmpty {
                let count = verdict.attention.count
                XCTAssertEqual(rig.page.status.headline, count == 1 ? "1 thing needs you" : "\(count) things need you", name)
            }
            // A non-green row's reason is what the menu and Dashboard say about it.
            let row = rig.model.groupStatus(.maintenance)
            if row.health != .healthy {
                let reason = try XCTUnwrap(row.attentionReason, "\(name): a non-green row must say why")
                XCTAssertTrue(shared.contains(reason), "\(name): row reason \(reason.debugDescription) missing from \(shared)")
                XCTAssertTrue((rig.page.status.reason ?? "").contains(reason) || rig.page.attentionItems.contains(reason),
                              "\(name): Dashboard \(rig.page.status) omits the row's reason")
            }
            // "All good" is only possible when every Jobs row is green.
            if rig.page.status.headline == "All good" {
                for group in BrainLayerLaunchdJobGroup.allCases {
                    XCTAssertEqual(rig.model.groupStatus(group).health, .healthy, "\(name): All good beside \(group)")
                }
            }
        }
    }

    /// `BRAINBAR_SNAPSHOT_DIR=<dir> swift test --filter BrainBarOneStatusTests`: the menu (its icon
    /// and the controller's real rows), the Dashboard and Jobs, per fixture, to compare by eye.
    func testRenderMenuDashboardAndJobsPerFixture() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] else {
            throw XCTSkip("Set BRAINBAR_SNAPSHOT_DIR to render the one-status fixtures")
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        func write(_ view: AnyView, width: CGFloat, height: CGFloat, name: String) throws {
            let host = NSHostingView(rootView: view)
            host.appearance = NSAppearance(named: .darkAqua)
            host.frame = NSRect(x: 0, y: 0, width: width, height: height)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
            host.layoutSubtreeIfNeeded()
            let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: bitmap)
            let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
            XCTAssertGreaterThan(png.count, 5_000, name)
            try png.write(to: directory.appendingPathComponent("\(name).png"), options: .atomic)
        }
        for fixture in [Fixture.healthy, .skippedWithReason, .failedNight, .recoveredNight, .vectorScan] {
            let inputs = try inputs(fixture)
            let rig = try rig(inputs)
            defer { rig.controller.stop() }
            let collector = collector(inputs)
            if fixture == .vectorScan {
                try write(BrainBarVectorDetailPreview.make(check: .init(count: 5, checkedAt: inputs.now, error: nil),
                    scan: inputs.badge!.vectorScan!, now: inputs.now), width: 560, height: 240, name: "vector-details-fresh")
                try write(BrainBarVectorDetailPreview.make(check: .init(count: 5, checkedAt: inputs.now.addingTimeInterval(-601), error: "Coverage query failed"),
                    scan: inputs.badge!.vectorScan!, now: inputs.now), width: 560, height: 260, name: "vector-details-stale")
                for (name, check) in [("starting", BrainBarVectorCheck(count: nil, checkedAt: nil, error: nil)),
                                      ("unavailable", .init(count: nil, checkedAt: nil, error: "Coverage query failed")),
                                      ("singular", .init(count: 1, checkedAt: inputs.now, error: nil))] {
                    for width: CGFloat in [390, 560] {
                        try write(BrainBarVectorDetailPreview.make(check: check, scan: inputs.badge!.vectorScan!, now: inputs.now),
                                  width: width, height: 260, name: "vector-details-\(name)-\(Int(width))")
                    }
                }
            }
            let icon = BrainBarStatusPopoverController.statusIconImage(stats: collector.stats,
                                                                       badgeOn: try XCTUnwrap(rig.controller.renderedStatusDotForTesting))
            let rows = rig.controller.contextMenuForTesting.items.filter { !$0.isHidden }
            let menu = AnyView(VStack(alignment: .leading, spacing: 0) {
                Image(nsImage: icon).padding(.horizontal, 14).padding(.vertical, 6)
                ForEach(Array(rows.enumerated()), id: \.offset) { _, item in
                    if item.isSeparatorItem { Divider().padding(.vertical, 4) }
                    else { Text(item.title).font(.system(size: 13)).padding(.horizontal, 14).padding(.vertical, 3) }
                }
            }.padding(.vertical, 6).frame(width: 560, alignment: .leading)
                .background(Color(nsColor: .windowBackgroundColor)))
            try write(menu, width: 560, height: 60 + CGFloat(rows.count) * 22, name: "one-status-\(fixture)-menu")
            guard case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else { return }
            let document = base.replacingBackupsForStatusTests(T.backups(), generatedAt: inputs.now)
            try write(BrainBarDashboardPreview.make(collector: collector, observabilityResult: .readable(document),
                                                    now: inputs.now, calendar: .current,
                                                    statusVerdict: rig.model.publishedStatus.verdict),
                      width: 980, height: 900, name: "one-status-\(fixture)-dashboard")
            try write(BrainBarUnifiedWindowPreview.make(collector: collector, settingsViewModel: rig.model,
                                                        panelState: BrainBarDashboardPanelState(), section: .jobs),
                      width: 980, height: 900, name: "one-status-\(fixture)-jobs")
        }
    }

    func testSkippedNightSaysWhyInPlainWords() throws {
        let rig = try rig(try inputs(.skippedWithReason))
        defer { rig.controller.stop() }
        let reason = try XCTUnwrap(rig.model.groupStatus(.maintenance).attentionReason)
        XCTAssertTrue(reason.hasPrefix("Nightly skipped at "), reason)
        XCTAssertTrue(reason.hasSuffix(": another program was writing to the database."), reason)
        XCTAssertEqual(rig.page.status.headline, "Status notice")
        XCTAssertEqual(rig.page.status.reason, reason)
    }
}
