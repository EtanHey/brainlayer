import AppKit
import Darwin
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarMenuCoherenceTests: XCTestCase {
    typealias T = BrainBarStatusSnapshotTests
    private let backup = "Database backup failed; check the backup log"
    private let maintenance = "BrainLayer full maintenance failed: tonight's local verified snapshot unavailable; VACUUM skipped. Retry after resolving the failure; inspect /Users/fixture/.local/share/brainlayer/logs/maintenance.log"
    private let failure = "com.brainlayer.maintenance: failed 3 nights in a row; last exit 1"
    private let diagnostic = "Database snapshot failed (step backup-snapshot) at 2026-11-02 05:04 GMT+2; nightly runs have had problems 3 nights in a row — next retry 2026-11-03 03:15 GMT+2"

    private func date(_ value: String) -> Date { ISO8601DateFormatter().date(from: value)! }

    private func inputs() -> BrainBarStatusInputs {
        var input = T.healthy()
        input.now = date("2026-11-02T22:45:00Z")
        input.watcher = .running(heartbeatAt: input.now)
        input.watcherHeartbeatAt = input.now
        input.backups = .measured(T.backups(db: date("2026-11-01T17:00:00Z"), transcripts: date("2026-11-02T07:30:00Z")))
        input.jobAlerts = .init(active: ["backup-snapshot": backup, "maintenance-full": maintenance])
        let at = date("2026-11-02T03:04:00Z")
        let next = date("2026-11-03T01:15:00Z")
        var observation = T.step(code: 1, at: at, next: next, consecutiveFailures: 3)
        observation.calendarNightOutcome = .init(scheduledDate: "2026-11-02", at: at, failed: true,
            abandoned: false, consecutiveFailures: 3, failedSteps: ["backup-snapshot"],
            stepProblems: ["backup-snapshot": .init(at: at)])
        input.observations?[.maintenanceNightly] = observation
        input.observations?[.maintenanceWeekly] = observation
        let checks = BrainBarHealthVerdict.Check.labels.keys.sorted().map {
            BrainBarHealthVerdict.Check(name: $0, state: .green, note: "Within safety window")
        }
        input.badge?.healthVerdict = .init(dot: .green, checks: checks, notes: [
            .init(code: "job_alert_backup-snapshot", message: backup),
            .init(code: "job_alert_maintenance-full", message: maintenance),
            .init(code: "job_failure", message: failure),
        ], evaluatedAt: input.now)
        return input
    }

    func testIndexReportsFromSameNightCollapseWithDiskProblem() {
        var input = inputs()
        let at = date("2026-11-02T03:04:00Z")
        input.jobAlerts = .init(active: ["index": "BrainLayer index exited 1"])
        input.observations?[.index] = T.failed(.index, code: 1, at: at)
        for job in [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly] {
            input.observations?[job]?.calendarNightOutcome = .init(scheduledDate: "2026-11-02", at: at,
                failed: true, abandoned: false, consecutiveFailures: 1, failedSteps: ["index"],
                stepProblems: ["index": .init(at: at)])
        }
        input.badge?.healthVerdict = .init(dot: .amber, checks: [
            .init(name: "disk", state: .amber, note: "free space versus the next backup's need not measured (3 checks in a row)")
        ], notes: [.init(code: "job_alert_index", message: "BrainLayer index exited 1")], evaluatedAt: input.now)
        let snapshot = BrainBarStatusSnapshot.derive(input)
        XCTAssertEqual(snapshot.issues.count, 2, "\(snapshot.issues)")
        XCTAssertTrue(snapshot.issues.allSatisfy { $0.text.contains("Nothing to do.") })
        XCTAssertTrue(snapshot.issues.contains { $0.text.contains("Retries") })
        XCTAssertEqual(snapshot.verdict.dot, .amber)
    }

    func testDifferentFailedJobsRemainSeparate() {
        var input = T.healthy()
        input.jobAlerts = .init(active: ["index": "BrainLayer index exited 1", "wal-checkpoint": "Database checkpoint failed"])
        let snapshot = BrainBarStatusSnapshot.derive(input)
        XCTAssertTrue(snapshot.issues.contains { $0.alertKey == "index" })
        XCTAssertTrue(snapshot.issues.contains { $0.alertKey == "wal-checkpoint" })
        XCTAssertEqual(snapshot.issues.filter { $0.alertKey != nil }.count, 2)
    }

    func testExactReportedFourLinesBecomeOneProblem() throws {
        let input = inputs()
        let snapshot = BrainBarStatusSnapshot.derive(input)
        let rows = BrainBarStatusPopoverController.problemRows(snapshot.verdict, health: snapshot.healthVerdict)
        XCTAssertEqual(rows.count, 1, "\(rows)")
        let text = try XCTUnwrap(rows.first?.text)
        XCTAssertTrue(text.contains("3 nights in a row"), text)
        XCTAssertTrue(text.contains("Nov 2 05:04"), text)
        XCTAssertTrue(text.contains("next try 03:15"), text)
        XCTAssertEqual(snapshot.verdict.dot, .green, "Notes never change the outcome dot")
        XCTAssertEqual(snapshot.verdict.safeHeadline, "Memory safe · 1 thing to look at")
        // The fixture preserves the original strings, including the independently derived diagnostic.
        var legacy = input
        legacy.badge?.healthVerdict = nil
        XCTAssertTrue(BrainBarStatusSnapshot.derive(legacy).issues.contains { $0.text == diagnostic })
        XCTAssertEqual(input.badge?.healthVerdict?.notes.map(\.message), [backup, maintenance, failure])
    }

    func testHeaderMatrixAndMenuCap() {
        for dot in [BrainBarRenderedDot.green, .amber, .red] {
            for notes in [[], ["A problem"]] {
                let verdict = BrainBarStatusVerdict(attention: [], notices: notes, isChecking: false, dotOverride: dot)
                let headline = BrainBarStatusPopoverController.statusLineTitle(for: verdict)
                XCTAssertEqual(headline == "All good", dot == .green && notes.isEmpty)
                XCTAssertEqual(headline, verdict.dashboardStatus.headline)
                // Compact headline (2026-11-08): a non-green dot counts its rows; with none, the dot's words.
                if dot == .amber { XCTAssertEqual(headline, notes.isEmpty ? "Check soon" : "1 thing needs you") }
                if dot == .red { XCTAssertEqual(headline, notes.isEmpty ? "Needs attention" : "1 thing needs you") }
            }
        }
        let rows = (1...5).map { BrainBarStatusPopoverController.VerdictRow(text: "Problem \($0)") }
        let compact = BrainBarCompactNotices(rows: rows)
        XCTAssertEqual(compact.visible.map(\.title), ["Problem 1", "Problem 2", "Problem 3"])
        XCTAssertEqual(compact.overflowTitle, "+2 more")
    }

    func testOldJobAlertStaysVisibleButDoesNotCountInHeader() {
        var input = T.healthy()
        input.now = date("2026-11-03T00:00:00Z")
        input.watcher = .running(heartbeatAt: input.now)
        input.watcherHeartbeatAt = input.now
        input.backups = .measured(T.backups(db: input.now, transcripts: input.now))
        let reportedAt = input.now.addingTimeInterval(-3 * 60 * 60)
        let priorRun = input.now.addingTimeInterval(-4 * 60 * 60)
        input.jobAlerts = .init(active: ["wal-checkpoint": "BrainLayer WAL checkpoint was busy"],
                                reportedAt: ["wal-checkpoint": reportedAt])
        input.observations?[.walCheckpoint] = T.step(at: priorRun,
            next: input.now.addingTimeInterval(-4 * 60 * 60))
        let checks = BrainBarHealthVerdict.Check.labels.keys.sorted().map {
            BrainBarHealthVerdict.Check(name: $0, state: .green, note: "Within safety window")
        }
        input.badge?.healthVerdict = .init(dot: .green, checks: checks, notes: [], evaluatedAt: input.now)
        let snapshot = BrainBarStatusSnapshot.derive(input)
        let row = try! XCTUnwrap(BrainBarStatusPopoverController.problemRows(snapshot.verdict, health: snapshot.healthVerdict)
            .first(where: { $0.text.contains("WAL checkpoint") }))
        XCTAssertTrue(row.isOld)
        XCTAssertTrue(row.text.hasPrefix("Earlier: "), row.text)
        XCTAssertEqual(snapshot.verdict.safeHeadline, "All good")

        input.jobAlerts = .init(active: ["wal-checkpoint": "BrainLayer WAL checkpoint was busy"],
                                reportedAt: ["wal-checkpoint": input.now])
        let fresh = BrainBarStatusSnapshot.derive(input)
        XCTAssertFalse(BrainBarStatusPopoverController.problemRows(fresh.verdict, health: fresh.healthVerdict).first?.isOld ?? true)
        XCTAssertNotEqual(fresh.verdict.safeHeadline, "All good")

        input.jobAlerts = .init(active: ["wal-checkpoint": "BrainLayer WAL checkpoint was busy"])
        let unknownAge = BrainBarStatusSnapshot.derive(input)
        XCTAssertTrue(unknownAge.verdict.notices.contains { $0.contains("WAL checkpoint") },
                      "Missing optional timestamps keep the alert active and counted")
        XCTAssertFalse(unknownAge.verdict.oldNotices.contains { $0.contains("WAL checkpoint") })
        XCTAssertFalse(BrainBarStatusPopoverController.problemRows(unknownAge.verdict, health: unknownAge.healthVerdict)
            .first(where: { $0.text.contains("WAL checkpoint") })?.isOld ?? true)
        XCTAssertNotEqual(unknownAge.verdict.safeHeadline, "All good")
    }

    func testRenderOldAlertEvidence() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_S32_RENDER_DIR"] else {
            throw XCTSkip("Set BRAINBAR_S32_RENDER_DIR for isolated render evidence")
        }
        let directory = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var input = inputs()
        input.now = date("2026-11-03T12:00:00Z")
        input.watcher = .running(heartbeatAt: input.now)
        input.watcherHeartbeatAt = input.now
        input.backups = .measured(T.backups(db: date("2026-11-03T10:00:00Z"), transcripts: date("2026-11-03T11:00:00Z")))
        let reportedAt = date("2026-11-02T03:11:00Z")
        input.observations?[.index] = T.step(at: input.now.addingTimeInterval(-3600), next: input.now.addingTimeInterval(900))
        for job in [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly] {
            input.observations?[job] = T.step(at: input.now.addingTimeInterval(-3600), next: input.now.addingTimeInterval(900))
        }
        input.jobAlerts = .init(active: ["wal-checkpoint": "BrainLayer WAL checkpoint was busy"],
                                reportedAt: ["wal-checkpoint": reportedAt])
        input.observations?[.walCheckpoint] = T.step(at: reportedAt, next: date("2026-11-02T03:15:00Z"))
        let checks = BrainBarHealthVerdict.Check.labels.keys.sorted().map {
            BrainBarHealthVerdict.Check(name: $0, state: .green, note: "Within safety window")
        }
        input.badge?.healthVerdict = .init(dot: .green, checks: checks, notes: [], evaluatedAt: input.now)
        let snapshot = BrainBarStatusSnapshot.derive(input)
        XCTAssertEqual(snapshot.verdict.oldNotices.count, 1, "\(snapshot.issues)")
        guard case let .measured(backups) = input.backups else { return XCTFail() }
        try BrainBarRenderHarness.renderHealthVerdictPopover(snapshot,
            backupLines: ObservabilityPresentation.backupReceiptLines(for: backups, now: input.now),
            directory: directory, name: "menu-390")
        try BrainBarRenderHarness.renderMenuCoherenceDashboard(snapshot,
            result: BrainBarDashboardFixture.healthyObservabilityResult, now: BrainBarDashboardFixture.fetchedAt,
            directory: directory, name: "dashboard-960", width: 960, calendar: input.calendar)
    }

    func testReportedMenuHeaderMustAcknowledgeFailures() {
        let snapshot = BrainBarStatusSnapshot.derive(inputs())
        XCTAssertNotEqual(BrainBarStatusPopoverController.statusLineTitle(for: snapshot.verdict), "All good")
        XCTAssertEqual(snapshot.issues.count, 1, "The reported four lines must describe one problem")
    }

    func testRealMenuOverflowOpensDashboard() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil,
            databasePath: directory.appendingPathComponent("fixture.db").path)
        let panel = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: panel, statusBar: TestStatusItemHost())
        defer { controller.stop() }
        var badge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        badge.healthVerdict = .init(dot: .green, checks: [], notes: (1...5).map {
            .init(code: "fixture_\($0)", message: "Problem \($0)")
        }, evaluatedAt: Date())
        controller.setBadgeForTesting(badge, alertsURL: directory.appendingPathComponent("job-alerts.json"))
        let menu = controller.contextMenuForTesting
        controller.menuNeedsUpdate(menu)
        // "+2 more" expands as a submenu holding the hidden rows, then Open Dashboard.
        let overflow = try XCTUnwrap(menu.items.first { $0.title == "+2 more" })
        XCTAssertTrue(overflow.isEnabled)
        XCTAssertFalse(menu.items.contains { $0.title == "Problem 4" || $0.title == "Problem 5" })
        let submenu = try XCTUnwrap(overflow.submenu)
        XCTAssertEqual(submenu.items.filter { !$0.isSeparatorItem }.map(\.title), ["Problem 4", "Problem 5", "Open Dashboard"])
        let open = try XCTUnwrap(submenu.items.last)
        XCTAssertNotNil(open.action)
        XCTAssertTrue(open.target === controller)
    }

    func testBackupCardUsesActiveFailureAndKeepsOtherCopyGreen() throws {
        let input = inputs()
        let verdict = BrainBarStatusSnapshot.derive(input).verdict
        guard case let .measured(backups) = input.backups else { return XCTFail() }
        let lines = ObservabilityPresentation.backupReceiptLines(for: backups, now: input.now)
        XCTAssertEqual(lines.first?.tone, .green)
        let shown = BrainBarOnePagePresentation.backupRunLines(lines, verdict: verdict)
        XCTAssertEqual(shown.first?.tone, .amber)
        XCTAssertTrue(shown.first?.text.contains("last run failed") == true)
        XCTAssertEqual(shown.last?.tone, .green, "A snapshot failure must not invent a transcript failure")
        var recovered = verdict
        recovered.backupRunFailures = []
        XCTAssertEqual(BrainBarOnePagePresentation.backupRunLines(lines, verdict: recovered), lines)
        let missing = ObservabilityPresentation.backupReceiptLines(for: T.backups(db: nil), now: input.now)
        XCTAssertEqual(BrainBarOnePagePresentation.backupRunLines(missing, verdict: verdict).first?.tone, .red,
            "An active run failure must not soften a missing-copy warning")
    }

    func testIndependentProblemSurvivesGrouping() {
        var input = inputs()
        let health = input.badge!.healthVerdict!
        input.badge?.healthVerdict = .init(dot: health.dot, checks: health.checks,
            notes: health.notes + [.init(code: "job_failure", message: "com.brainlayer.index: failed 3 times")],
            evaluatedAt: input.now)
        let snapshot = BrainBarStatusSnapshot.derive(input)
        XCTAssertEqual(snapshot.issues.count, 2)
        XCTAssertTrue(snapshot.issues.contains { $0.text.contains("com.brainlayer.index: failed 3 times") })
    }

    func testMaintenanceJobFailureCountUsesUnresolvedCalendarNights() throws {
        var input = inputs()
        input.jobAlerts = .init(active: [:])
        let health = try XCTUnwrap(input.badge?.healthVerdict)
        input.badge?.healthVerdict = .init(dot: health.dot, checks: health.checks,
            notes: [.init(code: "job_failure", message: failure)], evaluatedAt: input.now)
        var night = try XCTUnwrap(input.observations?[.maintenanceNightly]?.calendarNightOutcome)
        night.failedSteps = ["wal-checkpoint"]
        night.stepProblems = ["wal-checkpoint": .init(at: date("2026-11-02T03:11:00Z"), reason: "checkpoint busy")]
        night.consecutiveFailures = 1
        for job in [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly] {
            input.observations?[job]?.calendarNightOutcome = night
        }
        let snapshot = BrainBarStatusSnapshot.derive(input)
        XCTAssertTrue(snapshot.issues.contains { $0.text.contains("failed 1 night in a row") }, "\(snapshot.issues)")
        XCTAssertFalse(snapshot.issues.contains { $0.text.contains("failed 3 nights in a row") }, "\(snapshot.issues)")
    }

    func testOldJobFailureNoteWithUnresolvedNightNeverBecomesOld() {
        var input = inputs()
        input.now = date("2026-11-03T22:45:00Z")
        input.watcher = .running(heartbeatAt: input.now)
        input.watcherHeartbeatAt = input.now
        input.backups = .measured(T.backups(db: input.now, transcripts: input.now))
        let verdict = input.badge!.healthVerdict!
        input.badge?.healthVerdict = .init(dot: verdict.dot, checks: verdict.checks,
            notes: verdict.notes, evaluatedAt: input.now)
        input.jobAlerts = .init(active: [:])
        let snapshot = BrainBarStatusSnapshot.derive(input)

        XCTAssertTrue(snapshot.issues.contains { $0.text.contains("failed 3 nights in a row") }, "\(snapshot.issues)")
        XCTAssertTrue(snapshot.verdict.oldNotices.isEmpty, "Calendar failures cannot age out: \(snapshot.verdict.oldNotices)")
        XCTAssertEqual(snapshot.verdict.dot, .green, "The note does not take dot authority from the health verdict")
        XCTAssertNotEqual(snapshot.verdict.safeHeadline, "All good", "The unresolved failure remains counted")
    }

    func testNightlyBackupFailuresKeepEachTargetsReasonAndTime() throws {
        var input = inputs()
        var night = try XCTUnwrap(input.observations?[.maintenanceNightly]?.calendarNightOutcome)
        night.failedSteps = ["backup-snapshot", "jsonl-backup"]
        night.stepProblems["backup-snapshot"] = .init(at: date("2026-11-02T03:04:00Z"), reason: "snapshot timed out")
        night.stepProblems["jsonl-backup"] = .init(at: date("2026-11-02T03:47:00Z"), reason: "upload rejected")
        for job in [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly] {
            input.observations?[job]?.calendarNightOutcome = night
        }
        let verdict = BrainBarStatusSnapshot.derive(input).verdict
        guard case let .measured(backups) = input.backups else { return XCTFail() }
        let lines = BrainBarOnePagePresentation.backupRunLines(
            ObservabilityPresentation.backupReceiptLines(for: backups, now: input.now, calendar: input.calendar), verdict: verdict)
        XCTAssertEqual(lines.map(\.tone), [.amber, .amber])
        XCTAssertTrue(lines.first?.text.contains("snapshot timed out at Nov 2 05:04") == true)
        XCTAssertTrue(lines.last?.text.contains("upload rejected at Nov 2 05:47") == true)
        XCTAssertFalse(lines.last?.text.contains("snapshot timed out") == true)
        XCTAssertFalse(lines.last?.text.contains("05:04") == true)
        try renderMacroScenarioIfRequested(input, name: "macro-both-failed-960")
    }

    func testGroupedLineKeepsRunnerAndUnknownFailures() throws {
        for (field, expected) in [("runner", "maintenance runner failed at Nov 2 05:20"), ("unknown", "unknown failure at Nov 2 05:20")] {
            var input = inputs()
            var night = try XCTUnwrap(input.observations?[.maintenanceNightly]?.calendarNightOutcome)
            if field == "runner" { night.runnerErrorAt = date("2026-11-02T03:20:00Z") } else { night.unknownFailureAt = date("2026-11-02T03:20:00Z") }
            for job in [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly] {
                input.observations?[job]?.calendarNightOutcome = night
            }
            let snapshot = BrainBarStatusSnapshot.derive(input)
            let rows = BrainBarStatusPopoverController.problemRows(snapshot.verdict, health: snapshot.healthVerdict)
            XCTAssertEqual(rows.count, 1, "\(rows)")
            XCTAssertTrue(rows.first?.text.contains(expected) == true, rows.first?.text ?? "")
            XCTAssertTrue(rows.first?.text.contains("snapshot failed at Nov 2 05:04") == true)
        }
    }

    func testEveryGroupedAlertKeepsItsOwnNoticeSeverity() {
        let snapshot = BrainBarStatusSnapshot.derive(inputs())
        XCTAssertEqual(snapshot.issues.filter { $0.id == "nightly.backup" }.count, 1)
        // A fresh verdict rates every job alert a notice before grouping; folding two alerts into
        // one line must not turn the second one back into a red Backups-page alert.
        for reason in [backup, maintenance] {
            XCTAssertTrue(snapshot.isNoticeAlert(reason: reason), reason)
        }
    }

    func testConfigurationNoticeDoesNotInventFailedBackupRuns() throws {
        var input = T.healthy()
        input.badge?.healthVerdict = .init(dot: .green, checks: [], notes: [], evaluatedAt: input.now)
        input.observations?[.maintenanceNightly]?.calendarScheduleProblem = "Fixture schedule invalid"
        let snapshot = BrainBarStatusSnapshot.derive(input)
        let notice = try XCTUnwrap(snapshot.issues.first { $0.id == "config.maintenance-plist" })
        XCTAssertTrue(notice.jobs.contains(.backupDaily) && notice.jobs.contains(.jsonlBackup))
        XCTAssertTrue(snapshot.verdict.backupRunFailures.isEmpty)
        guard case let .measured(backups) = input.backups else { return XCTFail() }
        let lines = BrainBarOnePagePresentation.backupRunLines(
            ObservabilityPresentation.backupReceiptLines(for: backups, now: input.now, calendar: input.calendar), verdict: snapshot.verdict)
        XCTAssertEqual(lines.map(\.tone), [.green, .green])
        XCTAssertFalse(lines.contains { $0.text.contains("last run failed") })
        try renderMacroScenarioIfRequested(input, name: "macro-config-notice-960")
    }

    func testActiveBackupFailureAlertsStillIdentifyTheirOwnTargets() {
        for (key, target) in [("backup-snapshot", BrainBarBackupRunFailure.Target.database), ("backup-daily", .database), ("jsonl-backup", .transcripts)] {
            var input = T.healthy()
            input.badge?.healthVerdict = .init(dot: .green, checks: [], notes: [], evaluatedAt: input.now)
            input.jobAlerts = .init(active: [key: "Fixture failure for \(key)"])
            let failures = BrainBarStatusSnapshot.derive(input).verdict.backupRunFailures
            XCTAssertEqual(failures.map(\.target), [target], key)
            XCTAssertEqual(failures.first?.text, "Fixture failure for \(key)")
        }
    }

    private func renderMacroScenarioIfRequested(_ input: BrainBarStatusInputs, name: String) throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_S25_RENDER_DIR"],
              case let .measured(backups) = input.backups,
              case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else { return }
        let directory = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let result = ObservabilityReadResult.readable(ObservabilityDocument(schemaVersion: 1,
            generatedAt: input.now, dbPath: "/tmp/fixture.db", windowHours: 24,
            stores: .init(state: "unmeasurable", reason: "No measurement yet today", inputs: [], totalChunks: nil, inWindow: nil),
            emitters: base.emitters, authorUnknown: base.authorUnknown, backups: backups))
        try BrainBarRenderHarness.renderMenuCoherenceDashboard(BrainBarStatusSnapshot.derive(input),
            result: result, now: input.now, directory: directory, name: name, width: 960, calendar: input.calendar)
    }

    func testTodayPlaceholderRendersAtBodyHeight() {
        let host = NSHostingView(rootView: BrainBarTodayEmptyState(text: "not measured yet today", isUnmeasured: true))
        host.layoutSubtreeIfNeeded()
        XCTAssertGreaterThan(host.fittingSize.height, 10)
        XCTAssertLessThan(host.fittingSize.height, 22, "Empty Today has one body-size line, without a hero dash")
    }

    func testGroupedBackupProblemBulletRendersNeutralBesideGreenVerdict() throws {
        let input = inputs()
        let snapshot = BrainBarStatusSnapshot.derive(input)
        XCTAssertEqual(snapshot.verdict.dot, .green)
        XCTAssertEqual(snapshot.issues.count, 1)
        let directory = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try BrainBarRenderHarness.renderMenuCoherenceDashboard(snapshot,
            result: BrainBarDashboardFixture.healthyObservabilityResult, now: input.now,
            directory: directory, name: "problem-bullet", width: 960, calendar: input.calendar)
        let bitmap = try XCTUnwrap(NSBitmapImageRep(data: Data(contentsOf: directory.appendingPathComponent("problem-bullet.png"))))
        let scale = CGFloat(bitmap.pixelsWide) / 960
        let neutral = try XCTUnwrap(BrainBarDesignTokens.Colors.statusUnknown.usingColorSpace(.sRGB))
        var neutralPixels = 0
        var greenPixels = 0
        // Fixed 960-point fixture: only the first problem's 5-point bullet, excluding its text
        // and the separate 9-point outcome dot. Scale to the bitmap's actual backing density.
        for y in Int(65 * scale)..<Int(83 * scale) {
            for x in Int(55 * scale)..<Int(65 * scale) {
                let color = try XCTUnwrap(bitmap.colorAt(x: x, y: y)?.usingColorSpace(.sRGB))
                if abs(color.redComponent - neutral.redComponent) < 0.05,
                   abs(color.greenComponent - neutral.greenComponent) < 0.05,
                   abs(color.blueComponent - neutral.blueComponent) < 0.05 { neutralPixels += 1 }
                if color.greenComponent > color.redComponent * 1.5,
                   color.greenComponent > color.blueComponent * 1.2 { greenPixels += 1 }
            }
        }
        XCTAssertGreaterThan(neutralPixels, 0, "The grouped failure must have a neutral problem marker")
        XCTAssertEqual(greenPixels, 0, "A failure bullet must never borrow the green outcome colour")
    }

    func testStructuredFailureReasonSurvivesCalendarReadAndGrouping() throws {
        let canonical = try XCTUnwrap(realpath(brainBarTestTemporaryDirectory().path, nil))
        defer { free(canonical) }
        let directory = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let at = date("2026-11-02T03:04:00Z").timeIntervalSince1970
        let rows: [[String: Any]] = [
            ["event": "runner", "kind": "nightly", "scheduled_date": "2026-11-02", "steps": ["backup-snapshot"], "ts": at - 60],
            ["event": "finished", "step": "backup-snapshot", "status": "failed", "returncode": 124, "reason": "timed out", "ts": at],
            ["event": "summary", "kind": "nightly", "scheduled_date": "2026-11-02", "failed": 1, "ts": at + 1],
        ]
        var data = Data()
        for row in rows { data.append(try JSONSerialization.data(withJSONObject: row)); data.append(10) }
        try data.write(to: directory.appendingPathComponent("runs.jsonl"))
        let history = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertNil(history.error)
        let night = try XCTUnwrap(history.maintenanceNight)
        XCTAssertEqual(night.stepProblems["backup-snapshot"]?.reason, "timed out")
        var input = inputs()
        input.observations?[.maintenanceNightly]?.calendarNightOutcome = night
        input.observations?[.maintenanceWeekly]?.calendarNightOutcome = night
        XCTAssertTrue(BrainBarStatusSnapshot.derive(input).issues.first?.text.contains("timed out at Nov 2 05:04") == true)
    }

    func testRenderEvidence() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_S25_RENDER_DIR"] else {
            throw XCTSkip("Set BRAINBAR_S25_RENDER_DIR for isolated render evidence")
        }
        let directory = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let input = inputs()
        let snapshot = BrainBarStatusSnapshot.derive(input)
        guard case let .measured(backups) = input.backups,
              case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else { return XCTFail() }
        let result = ObservabilityReadResult.readable(ObservabilityDocument(schemaVersion: 1,
            generatedAt: input.now, dbPath: "/tmp/fixture.db", windowHours: 24, stores: .init(state: "unmeasurable", reason: "No measurement yet today", inputs: [], totalChunks: nil, inWindow: nil),
            emitters: base.emitters, authorUnknown: base.authorUnknown, backups: backups))
        try BrainBarRenderHarness.renderHealthVerdictPopover(snapshot,
            backupLines: ObservabilityPresentation.backupReceiptLines(for: backups, now: input.now),
            directory: directory, name: "menu-390")
        for width in [390.0, 960.0] {
            try BrainBarRenderHarness.renderMenuCoherenceDashboard(snapshot, result: result, now: input.now,
                directory: directory, name: "dashboard-\(Int(width))", width: width, calendar: input.calendar)
        }
    }
}
