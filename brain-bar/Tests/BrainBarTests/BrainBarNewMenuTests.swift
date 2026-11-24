import AppKit
import XCTest
@testable import BrainBar

/// Shared standing-item contract: one enabled Diagnostics action follows Settings.
@MainActor func brainBarItemsWithoutApprovedDiagnostics(_ menu: NSMenu, file: StaticString = #filePath,
                                                       line: UInt = #line) throws -> [NSMenuItem] {
    let matches = menu.items.indices.filter { menu.items[$0].title == BrainBarMenuPresenter.SurfaceSay.diagnostics }
    XCTAssertEqual(matches.count, 1, file: file, line: line)
    let index = try XCTUnwrap(matches.first, file: file, line: line)
    let settings = try XCTUnwrap(menu.items.firstIndex { $0.title == BrainBarMenuPresenter.Say.settings.text }, file: file, line: line)
    XCTAssertEqual(index, settings + 1, file: file, line: line)
    let item = menu.items[index]
    XCTAssertTrue(item.isEnabled, file: file, line: line)
    XCTAssertFalse(item.isHidden, file: file, line: line)
    XCTAssertEqual(item.keyEquivalent, "", file: file, line: line)
    XCTAssertEqual(item.state, .off, file: file, line: line)
    XCTAssertNil(item.image, file: file, line: line)
    XCTAssertNil(item.view, file: file, line: line)
    return menu.items.enumerated().filter { $0.offset != index }.map(\.element)
}

/// Never presents a menu, creates a system status item or activates a window.
@MainActor final class BrainBarNewMenuTests: XCTestCase {
    func testNewDropdownReachesTheInjectedDiagnosticsOpener() async throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        let builder = NDMenuBuilder(store: store, perform: { _ in }, openWindow: { _ in })
        let openedNew = expectation(description: "New Diagnostics")
        builder.diagnosticsOpener = { openedNew.fulfill() }
        let newMenu = builder.build()
        _ = try brainBarItemsWithoutApprovedDiagnostics(newMenu)
        newMenu.performActionForItem(at: try XCTUnwrap(newMenu.items.firstIndex { $0.title == "Diagnostics" }))
        await fulfillment(of: [openedNew], timeout: 2)

        withExtendedLifetime(builder) {}
    }

    func testReportExactShape() {
        let now = Date(timeIntervalSince1970: 0.0)
        let p = BrainBarMenuPresentation(form: .needs, headline: "Needs attention", count: 1, icon: .marked,
            rows: [.init(kind: .headline, strings: ["Needs attention"]), .init(kind: .separator),
                   .init(kind: .needsYou, strings: ["Drive needs you"], action: .init(kind: .reconnectDrive, label: "Reconnect Google Drive"))])
        let v = BrainBarHealthVerdict(dot: .red, checks: [.init(name: "drive_auth", state: .red, note: "expired")], notes: [], evaluatedAt: now)
        XCTAssertEqual(BrainBarMenuReport.text(presentation: p, verdict: v, version: "1.2.3", now: now), """
        BrainBar report
        Version: 1.2.3
        Time: 1970-01-01T00:00:00Z

        Menu:
        Needs attention
        Drive needs you
        Reconnect Google Drive

        Checks:
        drive_auth: red — expired

        """)
    }

    func testModelSamplesAwakeClocksCachedInputsAndDamagedHistory() throws {
        let now = BrainBarStatusSnapshotTests.now
        let evaluated = now.addingTimeInterval(-3620.0)
        let history = BrainBarSleepHistory(uptime: { 0.0 })
        history.sleeping(at: evaluated.addingTimeInterval(10.0))
        history.woke(at: evaluated.addingTimeInterval(3610.0))
        let verdict = BrainBarHealthVerdict(dot: .green,
            checks: [.init(name: "stores", state: .green, note: "ok")], notes: [], evaluatedAt: evaluated)
        for damaged in [false, true] {
            var night = BrainBarStatusSnapshotTests.step()
            night.calendarHistoryDamaged = damaged
            night.calendarNightOutcome = .init(scheduledDate: "fixture", at: now, failed: true,
                abandoned: false, consecutiveFailures: 3)
            let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(),
                store: .init(loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
                launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
                embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: .unmeasurable),
                initialLaunchdObservations: [.backupDaily: BrainBarStatusSnapshotTests.step(consecutiveFailures: 2),
                    .jsonlBackup: BrainBarStatusSnapshotTests.step(), .maintenanceNightly: night],
                refreshStatusOnLoad: false, now: { now },
                initialWatcherHealth: .readable(.init(updatedAt: evaluated, pollCount: 1, alertReasons: [], dbProbeFailed: false)),
                awakeClock: { 10.0 }, sleepHistory: history, statusDeadlineScheduler: { _, _ in })
            let drive = BrainBarDriveAuthModel(runner: nil)
            drive.setForPreview(status: .init(state: .missing, reason: nil, expiresAt: nil))
            model.attachDriveAuth(drive)
            model.applyHealthBadge(.init(badgeOn: false, reason: "", activeCodes: [], healthVerdict: verdict), alerts: nil)
            let stats = DashboardStats(chunkCount: 100, enrichedChunkCount: 0, pendingEnrichmentCount: 0,
                enrichmentPercent: 0.0, enrichmentRatePerMinute: 0.0, databaseSizeBytes: 0,
                recentActivityBuckets: [], recentEnrichmentBuckets: [], ftsEligibleChunkCount: 100,
                ftsIndexedChunkCount: 75, signalCoverageIsAvailable: true)
            let c = model.menuCapture(stats: stats, timeZone: TimeZone(secondsFromGMT: 7200)!)
            XCTAssertEqual(c.verdict, verdict)
            XCTAssertEqual(c.verdictAwakeAgeS, 20.0)
            XCTAssertEqual(c.watcher?.fresh, true)
            XCTAssertEqual(c.driveAuthModel, .missing)
            XCTAssertEqual(c.keywordCoverage, 0.75)
            XCTAssertFalse(c.backupsMeasured)
            XCTAssertEqual(c.backupDailyFailures, 2)
            XCTAssertEqual(c.jsonlBackupFailures, 0)
            XCTAssertEqual(c.maintenanceNightFailures, damaged ? nil : 3)
            XCTAssertEqual(c.now, now)
            XCTAssertEqual(c.timeZone.secondsFromGMT(), 7200)
            model.systemDidWake()
            XCTAssertTrue(model.menuCapture(stats: nil).wakeCatchUpPending)
        }
    }

    func testReportRedactsPersonalDiagnosticIdentifiers() {
        let p = BrainBarMenuPresentation(form: .calm, headline: "", count: 0, icon: .resting, rows: [])
        let v = BrainBarHealthVerdict(dot: .red,
            checks: [.init(name: "disk", state: .red, note: "/home/fixture/db a@example.com aB3dE6gH9jK2mN5pQ8sT1vW4xY7zA0bC@example.com")], notes: [], evaluatedAt: .distantPast)
        let report = BrainBarMenuReport.text(presentation: p, verdict: v, version: "test", now: Date(timeIntervalSince1970: 0.0))
        XCTAssertEqual(report.components(separatedBy: "\n").first { $0.hasPrefix("disk:") },
                       "disk: red — /home/[user]/db [email] [email]")
    }

    func testReportRemovesUnlabelledMixedClassTokens() {
        let tokens: [String] = ["aB3dE6gH9jK2mN5pQ8sT1vW4xY7zA0bC", "0123456789abcdef0123456789abcdef",
                                "aB3d_6gH-9jK2mN5pQ8sT1vW4xY7zA0bC"]
        let p = BrainBarMenuPresentation(form: .calm, headline: "", count: 0, icon: .resting, rows: [])
        let v = BrainBarHealthVerdict(dot: .red, checks: [.init(name: "stores", state: .red,
            note: "command=worker " + tokens.joined(separator: " "))], notes: [], evaluatedAt: .distantPast)
        let report = BrainBarMenuReport.text(presentation: p, verdict: v, version: "test", now: .distantPast)
        for token in tokens { XCTAssertFalse(report.contains(token)) }
        XCTAssertTrue(report.contains("command=worker [redacted] [redacted] [redacted]"))
    }

}
