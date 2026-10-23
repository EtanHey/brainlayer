import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

final class BrainBarBackupStatusConsistencyTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_790_769_600)
    private let locale = Locale(identifier: "en_US")

    private func backups(age: Double = 2, error: String? = nil, verified: Bool? = true) -> ObservabilityDocument.Backups {
        .init(
            state: "measured", reason: "", inputs: [], freshness: "fresh", thresholdHours: 36,
            retentionInvariant: "PASS", survivingArchives30D: 6, errorType: error,
            lastVerifiedUpload: .init(at: now.addingTimeInterval(-3600), ageHours: 1, archiveId: "fixture-archive", verified: true),
            dbSnapshot: verified.map { .init(lastAt: now.addingTimeInterval(-age * 3600), destination: "fixture.db.zst", verified: $0) },
            launchd: .init(label: "fixture-backup", bootstrapped: true, disabledDirPresent: false)
        )
    }

    private func drive(_ state: DriveAuthStatus.State = .valid, expiresIn: Double = 48 * 3600) -> DriveAuthPresentation {
        .derive(status: .init(state: state, reason: nil, expiresAt: now.addingTimeInterval(expiresIn)),
                isReconnecting: false, lastOutcome: nil, now: now, formatDate: { ISO8601DateFormatter().string(from: $0) })
    }

    private func health(_ b: ObservabilityDocument.Backups, drive: DriveAuthPresentation) -> BrainBarBackupsHealth {
        .derive(job: .init(health: .healthy, attentionReason: nil, lastRunText: "", nextRunText: ""),
                drive: drive, status: ObservabilityPresentation.backupStatus(for: b, now: now), statusUnavailableReason: "Unmeasurable")
    }

    func testValidConsentExpiringInTwoDaysIsAdvisoryWithSixPassingChecks() {
        let b = backups()
        let checks = BrainBarBackupChecks.derive(b, now: now, locale: locale)
        XCTAssertEqual(checks.summary, "All 6 checks pass")
        XCTAssertTrue(checks.checks.allSatisfy { $0.tone == .green })
        XCTAssertEqual(drive().tone, .expiring)
        XCTAssertEqual(health(b, drive: drive()).badge, .expiring)
        for auth in [drive(expiresIn: -1), drive(.invalid), drive(.missing)] {
            XCTAssertEqual(auth.tone, .attention)
            XCTAssertEqual(health(b, drive: auth).badge, .attention)
        }
    }

    func testFreshFlagCannotHideOldMissingOrFailedDatabaseReceipt() {
        for b in [backups(age: 48), backups(verified: nil), backups(verified: false)] {
            let status = ObservabilityPresentation.backupStatus(for: b, now: now)
            let checks = BrainBarBackupChecks.derive(b, now: now, locale: locale)
            XCTAssertEqual(status.snapshot.tone, .red)
            XCTAssertNotEqual(checks.summary, "All 6 checks pass")
            XCTAssertEqual(health(b, drive: drive()).badge, .attention)
        }
        let stale = ObservabilityPresentation.backupStatus(for: backups(age: 48), now: now)
        XCTAssertEqual(stale.snapshot.text, "Last offsite DB copy is 2 days old")
        XCTAssertEqual(stale.freshness.tone, .red)
    }

    func testPassingRecoveryRowsCannotConcealFailedOrUnmeasurableEvidence() {
        for error in ["drive_credentials_missing", "job_alert:Fixture backup failed"] {
            let checks = BrainBarBackupChecks.derive(backups(error: error), now: now, locale: locale)
            XCTAssertTrue(checks.checks.allSatisfy { $0.tone == .green })
            XCTAssertNotEqual(checks.summary, "All 6 checks pass")
            XCTAssertNotNil(checks.attentionSentence)
            XCTAssertEqual(checks.verdictTone, .red)
            XCTAssertEqual(health(backups(error: error), drive: drive()).badge, .attention)
        }
        var unknown = backups()
        unknown = .init(state: "measured", reason: "", inputs: [], freshness: nil, thresholdHours: nil,
                        retentionInvariant: nil, survivingArchives30D: nil, errorType: nil,
                        lastVerifiedUpload: unknown.lastVerifiedUpload, dbSnapshot: unknown.dbSnapshot, launchd: unknown.launchd)
        let checks = BrainBarBackupChecks.derive(unknown, now: now)
        XCTAssertNotEqual(checks.summary, "All 6 checks pass")
    }

    func testFreshFlagCannotKeepAnOldTranscriptUploadGreenWithoutANewNoOpReceipt() {
        let base = backups()
        let b = ObservabilityDocument.Backups(
            state: base.state, reason: base.reason, inputs: [], freshness: "fresh", thresholdHours: 36,
            retentionInvariant: "PASS", survivingArchives30D: 6, errorType: nil,
            lastVerifiedUpload: .init(at: now.addingTimeInterval(-48 * 3600), ageHours: 1, archiveId: "fixture-archive", verified: true),
            dbSnapshot: base.dbSnapshot, launchd: base.launchd)
        let status = ObservabilityPresentation.backupStatus(for: b, now: now)
        XCTAssertEqual(status.upload.tone, .red)
        XCTAssertTrue(status.upload.text.contains("48 h ago"), "receipt age follows the shared clock, not a cached age_hours")
        XCTAssertEqual(status.freshness.tone, .red)
        XCTAssertEqual(health(b, drive: drive()).badge, .attention)
    }

    func testScheduleTimestampNamesTheAttemptAndItsLogRatherThanAVerifiedCopy() {
        let sources = BrainBarBackupSources(
            paths: .init(launchAgents: URL(fileURLWithPath: "/fixture/LA"),
                         databaseLog: URL(fileURLWithPath: "/fixture/db.log"),
                         archiveLog: URL(fileURLWithPath: "/fixture/transcripts.log"),
                         maintenanceLog: URL(fileURLWithPath: "/fixture/maintenance.log"),
                         snapshotDirectory: URL(fileURLWithPath: "/fixture/snapshots"),
                         archiveDirectory: URL(fileURLWithPath: "/fixture/archives")),
            readFile: { url in url.lastPathComponent == "db.log" ? Data(#"{"attempted_at":"2026-09-30T07:27:00Z","verified":true,"backup_log_provenance":"real"}"#.utf8) : nil },
            listDirectory: { _ in [] }, isRegularFile: { _ in false })
        let row = sources.rows(now: now, calendar: Calendar(identifier: .gregorian),
                               formatDate: { ISO8601DateFormatter().string(from: $0) })[0]
        XCTAssertTrue(row.lastRun.hasPrefix("Last attempt 2026-09-30T07:27:00Z"))
        XCTAssertTrue(row.lastRun.contains("db.log"))
    }

    func testCopyDatesUseReceiptFieldsWhileTheOlderLocalFileRemainsSeparate() {
        let iso = ISO8601DateFormatter()
        let referenceNow = iso.date(from: "2026-10-23T18:50:00Z")!
        let b = ObservabilityDocument.Backups(
            state: "measured", reason: "", inputs: [], freshness: "fresh", thresholdHours: 36,
            retentionInvariant: "PASS", survivingArchives30D: 6, errorType: nil,
            lastVerifiedUpload: .init(at: iso.date(from: "2026-10-23T01:00:14Z")!, ageHours: 18,
                                      archiveId: "fixture-archive", verified: true),
            dbSnapshot: .init(lastAt: iso.date(from: "2026-10-23T01:35:15Z")!, destination: "2026-10-23.db.zst", verified: true),
            launchd: .init(label: "fixture-backup", bootstrapped: true, disabledDirPresent: false))
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let lines = ObservabilityPresentation.backupReceiptLines(for: b, locale: locale, now: referenceNow, calendar: calendar)
        XCTAssertEqual(lines.map(\.text), ["Database · Drive · last good today 01:35", "Transcripts · Drive · last good today 01:00"])
        XCTAssertEqual(lines.map(\.tone), [.green, .green])
        let details = BrainBarBackupChecks.derive(b, now: referenceNow, locale: locale, calendar: calendar).details
        XCTAssertTrue(details.contains { $0.label == "Offsite database snapshot — created" })
        XCTAssertTrue(details.contains { $0.label == "Transcript upload receipt — attempt time" })
        let sources = BrainBarBackupSources(
            paths: .init(launchAgents: URL(fileURLWithPath: "/fixture/LA"),
                         databaseLog: URL(fileURLWithPath: "/fixture/db.log"), archiveLog: URL(fileURLWithPath: "/fixture/archive.log"),
                         maintenanceLog: URL(fileURLWithPath: "/fixture/maintenance.log"),
                         snapshotDirectory: URL(fileURLWithPath: "/fixture/snapshots"), archiveDirectory: URL(fileURLWithPath: "/fixture/archives")),
            readFile: { _ in nil }, listDirectory: { $0.lastPathComponent == "snapshots" ? ["2026-10-21.db.zst"] : [] },
            isRegularFile: { _ in true })
        let local = sources.rows(now: referenceNow, calendar: calendar, formatDate: { iso.string(from: $0) })[0].localCopy
        XCTAssertEqual(local?.lastPathComponent, "2026-10-21.db.zst")
        XCTAssertFalse(lines[0].text.contains("Oct 21"))
    }

    @MainActor
    func testMenuUsesPublishedReceiptSnapshotAndClockAndPreservesFailureSeverity() throws {
        guard case let .readable(base) = try BrainBarOnePageTestFixture.healthyResult() else { return XCTFail("fixture unreadable") }
        let cases: [(ObservabilityDocument.Backups, DriveAuthStatus.State, Bool)] = [
            (backups(), .valid, false), (backups(age: 48), .valid, true),
            (backups(error: "job_alert:Fixture backup failed"), .valid, true), (backups(), .invalid, true),
        ]
        for (b, auth, needsAttention) in cases {
            let document = try replacingBackups(base, with: b)
            let model = BrainBarSettingsViewModel(
                workspace: BrainBarRecordingWorkspace(),
                store: BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/fixture/config"),
                                             loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
                initialLaunchdStates: Dictionary(uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map { ($0, .running) }),
                refreshStatusOnLoad: false, now: { Date(timeIntervalSince1970: 1_790_769_600) },
                initialObservabilityResult: .readable(document), observabilityCadence: .known(300))
            let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
            runtime.jobsModel = model
            runtime.driveAuth.setForPreview(status: .init(state: auth, reason: nil, expiresAt: now.addingTimeInterval(48 * 3600)))
            let workspace = BrainBarRecordingWorkspace()
            let panel = BrainBarDashboardPanelController(workspace: workspace, runtime: runtime)
            let controller = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime,
                dashboardPanelController: panel, statusBar: TestStatusItemHost())
            defer { controller.stop() }
            let menu = controller.contextMenuForTesting
            controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []),
                                          alertsURL: URL(fileURLWithPath: "/fixture/absent-alerts"))
            controller.menuNeedsUpdate(menu)
            let receipts = menu.items.filter { $0.identifier?.rawValue.hasPrefix("brainbar.menu.backup.") == true }
            XCTAssertEqual(receipts.map(\.title), model.backupReceiptLines?.map(\.text))
            XCTAssertTrue(receipts.allSatisfy { !$0.isEnabled && $0.action == nil })
            XCTAssertTrue(receipts.allSatisfy { $0.toolTip?.contains("observability.json") == true })
            XCTAssertEqual(menu.items.first?.title.hasPrefix("Needs attention:"), needsAttention)
            if !needsAttention { XCTAssertTrue(menu.items.first?.title.hasPrefix("Info:") == true) }
            for (item, line) in zip(receipts, model.backupReceiptLines ?? []) where line.tone == .red {
                XCTAssertEqual(item.attributedTitle?.attribute(.foregroundColor, at: 0, effectiveRange: nil) as? NSColor,
                               BrainBarDesignTokens.Colors.statusError)
            }
            runtime.jobsModel = nil
            controller.menuNeedsUpdate(menu)
            XCTAssertFalse(menu.items.contains { $0.identifier?.rawValue.hasPrefix("brainbar.menu.backup.") == true })
        }
    }

    @MainActor
    func testDashboardAndBackupsShareReceiptsAcrossRefreshTicks() async throws {
        let base = try BrainBarOnePageTestFixture.healthyResult()
        guard case let .readable(document) = base else { return XCTFail("fixture unreadable") }
        let first = try replacingBackups(document, with: backups(age: 48))
        let second = try replacingBackups(document, with: backups())
        let referenceNow = now
        let reads = ReceiptReads(first: first, second: second)
        let model = BrainBarSettingsViewModel(
            workspace: BrainBarRecordingWorkspace(),
            store: BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/fixture/config"),
                                         loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false, now: { Date(timeIntervalSince1970: 1_790_769_600) },
            observabilityURL: URL(fileURLWithPath: "/fixture/observability.json"),
            initialObservabilityResult: .readable(first), observabilityRead: { _ in await reads.read() })
        await model.monitorObservabilityStatus {
            let tone = await MainActor.run { () -> ObservabilityStatusTone? in
                guard case let .readable(current) = model.dashboardObservabilityResult else {
                    XCTFail("shared dashboard snapshot unreadable"); return nil
                }
                let collector = BrainBarDashboardFixture.makeCollector()
                XCTAssertEqual(model.observabilityNow, referenceNow)
                let hero = BrainBarHeroPresentation.derive(
                    flow: DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: model.observabilityNow),
                    stats: collector.stats, backupTruth: BrainBarHeroBackupTruth.derive(from: model.dashboardObservabilityResult, now: model.observabilityNow, cadence: model.observabilityCadence))
                let dashboard = BrainBarOnePagePresentation.derive(
                    snapshotFreshness: .live(ageSeconds: 0), hero: hero, observability: model.dashboardObservabilityResult,
                    stats: collector.stats, agentActivity: BrainBarDashboardFixture.agentActivity, now: model.observabilityNow)
                XCTAssertEqual(model.backupReceiptLines, dashboard.backupLines)
                return dashboard.backupLines.first?.tone
            }
            if await reads.record(tone) == 2 { throw CancellationError() }
        }
        let tones = await reads.tones
        XCTAssertEqual(tones, [.red, .green])
    }

    @MainActor
    func testR1MountedDashboardReceiptAgesAdvanceForExistingAndArrivingReceipts() {
        let receipts = BrainBarOperationReceipts()
        receipts.record(.init(kind: .search, durationMillis: 10, count: 1, recordedAt: now.addingTimeInterval(-10)))
        let collector = BrainBarDashboardFixture.makeCollector()
        let state = BrainBarDashboardPanelState()
        func dashboard(_ clock: Date) -> AnyView {
            BrainBarDashboardPreview.make(collector: collector, receiptStore: receipts,
                observabilityResult: BrainBarDashboardFixture.healthyObservabilityResult,
                now: clock, panelState: state, observabilityCadence: .known(300))
        }
        let host = NSHostingView(rootView: dashboard(now))
        host.frame = NSRect(x: 0, y: 0, width: 1000, height: 1600)
        func text() -> String {
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.25))
            return state.renderedReceiptValues.values.joined(separator: "\n")
        }
        let initial = text()
        XCTAssertTrue(initial.contains("just now"), "Mounted receipt text unavailable: \(initial)")
        host.rootView = dashboard(now.addingTimeInterval(600))
        receipts.record(.init(kind: .ingest, durationMillis: 20, count: 2, recordedAt: now.addingTimeInterval(300)))
        let advanced = text()
        XCTAssertTrue(advanced.contains("10 min ago"), advanced)
        XCTAssertTrue(advanced.contains("5 min ago"), advanced)
    }

    @MainActor
    func testR1LiveReceiptAgesIgnoreAdvancingBackupClock() {
        let live = Date()
        let receipts = BrainBarOperationReceipts()
        receipts.record(.init(kind: .search, durationMillis: 10, count: 1, recordedAt: live.addingTimeInterval(-120)))
        let collector = BrainBarDashboardFixture.makeCollector()
        let state = BrainBarDashboardPanelState()
        func dashboard(_ backup: Date) -> AnyView {
            BrainBarDashboardPreview.make(collector: collector, receiptStore: receipts,
                observabilityResult: BrainBarDashboardFixture.healthyObservabilityResult,
                backupNow: backup, panelState: state, observabilityCadence: .known(300))
        }
        let host = NSHostingView(rootView: dashboard(live.addingTimeInterval(-600)))
        host.frame = NSRect(x: 0, y: 0, width: 1000, height: 1600)
        func settle() { host.layoutSubtreeIfNeeded(); RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.25)) }
        settle()
        XCTAssertTrue(state.renderedReceiptValues["Last search"]?.contains("2 min ago") == true)
        host.rootView = dashboard(live.addingTimeInterval(600))
        receipts.record(.init(kind: .ingest, durationMillis: 20, count: 2, recordedAt: live.addingTimeInterval(-180)))
        settle()
        XCTAssertTrue(state.renderedReceiptValues["Last search"]?.contains("2 min ago") == true)
        XCTAssertTrue(state.renderedReceiptValues["Last store"]?.contains("3 min ago") == true)
    }

    @MainActor
    func testR1StaleAndFutureEnvelopesCannotBeHealthyWithFreshCopyReceipts() throws {
        guard case let .readable(base) = try BrainBarOnePageTestFixture.healthyResult() else { return XCTFail("fixture") }
        for offset in [-1200.0, 600.0] {
            var root = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(try replacingBackups(base, with: backups()))) as? [String: Any])
            root["generatedAt"] = now.addingTimeInterval(offset).timeIntervalSinceReferenceDate
            let document = try JSONDecoder().decode(ObservabilityDocument.self, from: JSONSerialization.data(withJSONObject: root))
            let truth = BrainBarHeroBackupTruth.derive(from: .readable(document), now: now, cadence: .known(300))
            if case .measured = truth { XCTFail("untrusted envelope must not be measured") }
            let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(),
                store: BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/fixture/config"),
                    loadDocumentOverride: { BrainLayerEnvDocument(config: .defaultConfig) }),
                initialLaunchdStates: Dictionary(uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map { ($0, .running) }),
                refreshStatusOnLoad: false, now: { Date(timeIntervalSince1970: 1_790_769_600) },
                initialObservabilityResult: .readable(document), observabilityCadence: .known(300))
            XCTAssertEqual(model.backupTruth.statusNotice, truth.statusNotice)
            let collector = BrainBarDashboardFixture.makeCollector(watcherHeartbeatAt: now)
            let hero = BrainBarHeroPresentation.derive(
                flow: DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: now),
                stats: collector.stats, backupTruth: truth)
            XCTAssertEqual(hero.healthTone, .amber)
            XCTAssertEqual(hero.healthReason, truth.statusNotice)
            XCTAssertEqual(model.backupsHealth(drive: drive(expiresIn: 5 * 86400)).badge, .unknown)
            XCTAssertNotNil(model.backupsHealth(drive: drive(expiresIn: 5 * 86400)).reason)
            XCTAssertNotEqual(model.backupChecks?.summary, "All 6 checks pass")
            XCTAssertEqual(model.backupChecks?.verdictTone, .neutral)
            XCTAssertEqual(model.backupReceiptLines?.map(\.tone), [.green, .green], "historical copies are still verified")
            let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
            runtime.jobsModel = model
            runtime.driveAuth.setForPreview(status: .init(state: .valid, reason: nil, expiresAt: now.addingTimeInterval(5 * 86400)))
            let workspace = BrainBarRecordingWorkspace()
            let panel = BrainBarDashboardPanelController(workspace: workspace, runtime: runtime)
            let menu = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime,
                dashboardPanelController: panel, statusBar: TestStatusItemHost())
            defer { menu.stop() }
            menu.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []), alertsURL: URL(fileURLWithPath: "/fixture/absent-alerts"))
            menu.menuNeedsUpdate(menu.contextMenuForTesting)
            XCTAssertTrue(menu.contextMenuForTesting.items.first?.title.hasPrefix("Info:") == true)
            XCTAssertEqual(menu.contextMenuForTesting.items.first?.toolTip, model.backupsHealth(drive: drive(expiresIn: 5 * 86400)).reason)
            XCTAssertEqual(model.backupsHealth(drive: drive(.invalid)).badge, .attention)
            model.setObservabilityResultForTesting(.readable(document.replacingBackupsErrorType("job_alert:Fixture failure")))
            XCTAssertEqual(model.backupsHealth(drive: drive()).badge, .attention)
            XCTAssertEqual(model.backupChecks?.verdictTone, .red)
            menu.menuNeedsUpdate(menu.contextMenuForTesting)
            XCTAssertTrue(menu.contextMenuForTesting.items.first?.title.hasPrefix("Needs attention:") == true)
        }
    }

    private func replacingBackups(_ document: ObservabilityDocument, with backups: ObservabilityDocument.Backups) throws -> ObservabilityDocument {
        var root = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(document)) as? [String: Any])
        root["generatedAt"] = now.timeIntervalSinceReferenceDate
        root["backups"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(backups))
        return try JSONDecoder().decode(ObservabilityDocument.self, from: JSONSerialization.data(withJSONObject: root))
    }
}

private actor ReceiptReads {
    let first, second: ObservabilityDocument
    var tones: [ObservabilityStatusTone?] = []
    init(first: ObservabilityDocument, second: ObservabilityDocument) { self.first = first; self.second = second }
    func read() -> ObservabilityReadResult { .readable(tones.isEmpty ? first : second) }
    func record(_ tone: ObservabilityStatusTone?) -> Int { tones.append(tone); return tones.count }
}
