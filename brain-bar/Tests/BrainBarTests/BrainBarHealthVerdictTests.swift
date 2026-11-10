import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarHealthVerdictTests: XCTestCase {
    private let now = BrainBarStatusSnapshotTests.now

    private func badge(_ dot: String?, age: Double = 0, generatedAge: Double = 0,
                       mutate: (inout [String: Any]) -> Void = { _ in }) throws -> BadgeStatePresentation {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let iso = ISO8601DateFormatter()
        var document: [String: Any] = ["schema_version": 1, "generated_at": iso.string(from: now.addingTimeInterval(-generatedAge)),
            "alerts": ["state": "measured", "reason": "", "inputs": [], "badge_on": false, "active": []]]
        if let dot {
            document["verdict"] = ["dot": dot, "evaluated_at": iso.string(from: now.addingTimeInterval(-age)),
                "checks": [["name": "backup_local", "state": dot, "age_s": 108000,
                            "threshold_s": 108000, "since": NSNull(), "note": "Fixture backup age"]],
                "notes": [["code": "job", "message": "Weekly maintenance deferred"]]]
        }
        mutate(&document)
        let url = root.appendingPathComponent("badge-state.json")
        try JSONSerialization.data(withJSONObject: document).write(to: url)
        return BadgeStateReader.read(url: url, now: now, cadence: .known(300))
    }

    func testExportedRowsIncludeNoticesAndDeduplicateInWindowOrder() {
        let verdict = BrainBarStatusVerdict(attention: ["Backup failed"],
            notices: ["Maintenance failed", "Backup failed"], oldNotices: ["Yesterday"],
            isChecking: false, dotOverride: .amber)
        XCTAssertEqual(verdict.currentRows, ["Backup failed", "Maintenance failed"])
        XCTAssertEqual(verdict.currentRowCount, verdict.currentRows.count)
        XCTAssertEqual(verdict.dashboardStatus.headline, "1 thing needs you")
        let green = BrainBarStatusVerdict(attention: [], notices: ["Maintenance deferred"],
            isChecking: false, dotOverride: .green)
        XCTAssertFalse(green.dashboardStatus.headline.contains("need you"))
        XCTAssertFalse(green.dashboardStatus.headline.contains("needs you"))
    }

    func testStaleVerdictCannotReviveYesterdayCalendarFailure() throws {
        let iso = ISO8601DateFormatter()
        let evaluated = iso.date(from: "2026-11-04T05:08:17Z")!
        let rendered = iso.date(from: "2026-11-04T06:30:30Z")!
        let failure = iso.date(from: "2026-11-03T03:11:00Z")!
        for state in ["green", "amber", "red", "unknown"] {
            var inputs = BrainBarStatusSnapshotTests.healthy()
            inputs.now = rendered
            inputs.watcher = .running(heartbeatAt: rendered)
            inputs.watcherHeartbeatAt = rendered
            inputs.backups = .measured(BrainBarStatusSnapshotTests.backups(db: rendered, transcripts: rendered))
            var observation = BrainBarStatusSnapshotTests.step(at: rendered)
            observation.calendarNightOutcome = .init(scheduledDate: "2026-11-03", at: failure,
                failed: true, abandoned: false, consecutiveFailures: 4, failedSteps: ["wal-checkpoint"],
                stepProblems: ["wal-checkpoint": .init(at: failure)])
            inputs.observations?[.maintenanceNightly] = observation
            var raw = try badge(state)
            let health = try XCTUnwrap(raw.healthVerdict)
            raw.healthVerdict = .init(dot: health.dot, checks: health.checks, notes: health.notes, evaluatedAt: evaluated)
            inputs.badge = raw
            let snapshot = BrainBarStatusSnapshot.derive(inputs)
            XCTAssertEqual(snapshot.verdict.dot, state == "red" ? .red : .amber, state)
            XCTAssertNil(snapshot.healthVerdict, "Stale data is not a fresh verdict")
            XCTAssertTrue(snapshot.notices.contains { $0.text == "Health check stale since 07:08" })
            XCTAssertTrue(snapshot.notices.contains { $0.id == "maintenance.night" && $0.text.contains("4 nights") })
            XCTAssertFalse(snapshot.attention.contains { $0.id == "maintenance.night" })
            if state == "red" {
                XCTAssertTrue(snapshot.attention.contains { $0.id == "verdict.backup_local" })
            } else {
                XCTAssertTrue(snapshot.attention.isEmpty)
            }
            inputs.observations?[.maintenanceNightly] = BrainBarStatusSnapshotTests.step(at: rendered)
            XCTAssertEqual(BrainBarStatusSnapshot.derive(inputs).verdict.dot, state == "red" ? .red : .amber,
                           "Last-known red must hold even without any legacy failure")
        }
    }

    func testLegacyCalendarHistoryNeverTurnsDotRed() {
        var inputs = BrainBarStatusSnapshotTests.healthy()
        inputs.observations?[.maintenanceNightly] = BrainBarStatusSnapshotTests.failed(
            .maintenanceNightly, code: 1, consecutive: 4)
        let snapshot = BrainBarStatusSnapshot.derive(inputs)
        XCTAssertEqual(snapshot.verdict.dot, .green)
        XCTAssertTrue(snapshot.notices.contains { $0.id == "maintenance.night" })
    }

    func testVerdictFixtureMatrixAndLegacyFallback() throws {
        for (state, age, expected, source) in [
            ("green", 0.0, BrainBarRenderedDot.green, "verdict"), ("amber", 0, .amber, "verdict"),
            ("red", 0, .red, "verdict"), ("unknown", 0, .amber, "verdict"),
            ("green", 900, .green, "verdict"), ("green", 901, .amber, "fallback"),
            ("green", -1, .amber, "fallback"),
        ] {
            var inputs = BrainBarStatusSnapshotTests.healthy()
            inputs.backups = .measured(BrainBarStatusSnapshotTests.backups(db: nil))
            inputs.badge = try badge(state, age: age)
            let snapshot = BrainBarStatusSnapshot.derive(inputs)
            XCTAssertEqual(snapshot.verdict.dot, expected, "\(state), age \(age)")
            XCTAssertEqual(snapshot.healthVerdict == nil ? "fallback" : "verdict", source)
            if source == "fallback" {
                XCTAssertFalse(snapshot.verdict.needsAttention, "Staleness is uncertainty, not a new red outcome")
                XCTAssertGreaterThan(snapshot.verdict.currentRowCount, 0)
                XCTAssertEqual(BrainBarStatusPopoverController.statusLineTitle(for: snapshot.verdict), "Check soon")
            }
            if age == 901 { XCTAssertTrue(snapshot.notices.contains { $0.text.hasPrefix("Health check stale since ") }) }
            if state == "green", source == "verdict" {
                XCTAssertTrue(snapshot.notices.contains { $0.id == "database.copy" })
                XCTAssertTrue(snapshot.attention.isEmpty)
                XCTAssertTrue(BrainBarStatusPopoverController.verdictRows(try XCTUnwrap(snapshot.healthVerdict),
                    details: snapshot.notices.map(\.text)).map(\.text).contains("Database: no verified copy on record"))
            }
        }
        var legacy = BrainBarStatusSnapshotTests.healthy()
        legacy.badge = try badge(nil)
        XCTAssertEqual(BrainBarStatusSnapshot.derive(legacy).verdict.dot, .green)
        legacy.backups = .measured(BrainBarStatusSnapshotTests.backups(db: nil))
        XCTAssertEqual(BrainBarStatusSnapshot.derive(legacy).verdict.dot, .red)
    }

    func testVerdictSurvivesLegacyEnvelopeStalenessAndReconciliation() throws {
        let raw = try badge("green", generatedAge: 700)
        XCTAssertTrue(raw.badgeOn, "Legacy envelope is stale but verdict remains usable")
        let withJob = BadgeStatePresentation(badgeOn: true, reason: "Old job", activeCodes: ["job_alert_maintenance-full"],
            activeMessages: ["Old job"], healthVerdict: raw.healthVerdict, healthVerdictUnreadable: true)
        let reconciled = withJob.reconciled(with: BrainBarJobAlerts(active: [:]))
        XCTAssertFalse(reconciled.badgeOn)
        XCTAssertEqual(reconciled.healthVerdict, raw.healthVerdict)
        XCTAssertTrue(reconciled.healthVerdictUnreadable)
        var inputs = BrainBarStatusSnapshotTests.healthy()
        inputs.badge = raw
        inputs.now = now.addingTimeInterval(700)
        XCTAssertEqual(BrainBarStatusSnapshot.derive(inputs).verdict.dot, .green)
        inputs.now = now.addingTimeInterval(901)
        XCTAssertNil(BrainBarStatusSnapshot.derive(inputs).healthVerdict)
    }

    func testMalformedVerdictsPreserveLegacyAlertsAndHealthyFallback() throws {
        for shape in ["null-note", "unknown-state", "wrong-type", "missing-notes", "broken"] {
            for driveLost in [false, true] {
                let raw = try badge(shape == "broken" ? "broken" : "green") { document in
                    var verdict = document["verdict"] as! [String: Any]
                    var checks = verdict["checks"] as! [[String: Any]]
                    switch shape {
                    case "null-note": checks[0]["note"] = NSNull()
                    case "unknown-state": checks[0]["state"] = "graced"
                    case "missing-notes": verdict.removeValue(forKey: "notes")
                    default: break
                    }
                    verdict["checks"] = checks
                    document["verdict"] = shape == "wrong-type" ? "invalid" : verdict
                    if driveLost {
                        document["alerts"] = ["state": "measured", "reason": "", "inputs": [],
                            "badge_on": true, "active": [["code": "drive_auth_lost", "severity": "critical",
                            "message": "Drive authorization: Drive access lost"]]]
                    }
                }
                XCTAssertNil(raw.healthVerdict, shape)
                XCTAssertTrue(raw.healthVerdictUnreadable, shape)
                XCTAssertEqual(raw.badgeOn, driveLost, shape)
                XCTAssertFalse(raw.reason.contains("Badge state unreadable"), shape)
                var inputs = BrainBarStatusSnapshotTests.healthy()
                inputs.badge = raw.reconciled(with: BrainBarJobAlerts(active: [:]))
                let snapshot = BrainBarStatusSnapshot.derive(inputs)
                XCTAssertNil(snapshot.healthVerdict, "Receipt source must be fallback")
                XCTAssertEqual(snapshot.verdict.dot, .green, "Healthy Swift rules still apply: \(shape)")
                XCTAssertTrue(snapshot.notices.contains { $0.text == "Health check result could not be read" }, shape)
                if driveLost {
                    XCTAssertEqual(raw.activeMessages, ["Drive authorization: Drive access lost"])
                    XCTAssertTrue(snapshot.issues.contains { $0.text == "Drive authorization: Drive access lost" })
                }
            }
        }
    }

    func testReceiptWritesOnlyOnContentChangeOrHeartbeatOffMainThread() throws {
        let writes = HealthReceiptWrites()
        let publisher = BrainBarUIStatusPublisher { receipt, _ in
            XCTAssertFalse(Thread.isMainThread)
            writes.record(receipt)
        }
        let url = URL(fileURLWithPath: "/fixture/brainbar-ui-status.json")
        func publish(_ seconds: Double, dot: BrainBarRenderedDot = .green, source: String = "verdict",
                     evaluatedAt: Date? = nil, reasons: [String] = [], version: String = "1", commit: String = "abc") {
            publisher.publish(.init(renderedAt: now.addingTimeInterval(seconds), dot: dot, source: source,
                verdictEvaluatedAt: evaluatedAt, reasons: reasons, appVersion: version, gitCommit: commit), to: url)
            publisher.flush()
        }
        publish(0)
        XCTAssertEqual(writes.count, 1)
        publish(1)
        publish(299)
        XCTAssertEqual(writes.count, 1, "Identical renders produce zero extra writes")
        publish(300)
        XCTAssertEqual(writes.count, 2, "One heartbeat at 300 seconds")
        publish(301, dot: .amber)
        publish(302, dot: .amber, source: "fallback")
        publish(303, dot: .amber, source: "fallback", evaluatedAt: now)
        publish(304, dot: .amber, source: "fallback", evaluatedAt: now, reasons: ["Changed reason"])
        publish(305, dot: .amber, source: "fallback", evaluatedAt: now, reasons: ["Changed reason"], version: "2")
        publish(306, dot: .amber, source: "fallback", evaluatedAt: now, reasons: ["Changed reason"], version: "2", commit: "def")
        XCTAssertEqual(writes.count, 8)
    }

    func testVerdictRowHierarchyAndQuietBackupReceipts() throws {
        for state in ["red", "amber", "green"] {
            let health = try XCTUnwrap(badge(state).healthVerdict)
            let rows = BrainBarStatusPopoverController.verdictRows(health, details: ["Database: no verified copy on record"])
            XCTAssertEqual(rows.last?.state, nil, "Swift diagnostics are secondary")
            XCTAssertEqual(rows.last?.text, "Database: no verified copy on record")
            XCTAssertEqual(rows[rows.count - 2].state, nil, "Notes are secondary")
            if state == "green" {
                XCTAssertTrue(rows.allSatisfy { $0.state == nil })
            } else {
                XCTAssertEqual(rows.first?.state, health.dot)
            }
            var inputs = BrainBarStatusSnapshotTests.healthy()
            inputs.badge = try badge(state)
            XCTAssertEqual(BrainBarStatusPopoverController.statusLineTitle(for: BrainBarStatusSnapshot.derive(inputs).verdict),
                state == "green" ? "Memory safe · 1 thing to look at" : state == "red" ? "1 thing needs you" : "Check soon")
        }
        XCTAssertEqual(BrainBarStatusPopoverController.backupReceiptColor(.red, verdictMode: true), .secondaryLabelColor)
        XCTAssertEqual(BrainBarStatusPopoverController.backupReceiptColor(.red, verdictMode: false),
                       BrainBarDesignTokens.Colors.statusError)
    }

    func testRenderHarnessPopoverEvidence() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_S2_RENDER_DIR"] else {
            throw XCTSkip("Set BRAINBAR_S2_RENDER_DIR to produce visual evidence")
        }
        let directory = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for state in ["missing", "green", "red", "amber"] {
            var inputs = BrainBarStatusSnapshotTests.healthy()
            inputs.backups = .measured(BrainBarStatusSnapshotTests.backups(db: nil))
            inputs.badge = try badge(state == "missing" ? nil : state)
            try BrainBarRenderHarness.renderHealthVerdictPopover(BrainBarStatusSnapshot.derive(inputs),
                backupLines: ObservabilityPresentation.backupReceiptLines(
                    for: BrainBarStatusSnapshotTests.backups(db: nil), now: now),
                directory: directory, name: "s2-\(state)-390")
        }
    }

    func testReceiptFromActualIconRenderAndMenuRows() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil,
                        databasePath: root.appendingPathComponent("fixture.db").path)
        let panel = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: panel, statusBar: TestStatusItemHost())
        defer { controller.stop() }
        var previousIcon: Data?
        for state in ["red", "amber", "green"] {
            var raw = try badge(state)
            let health = try XCTUnwrap(raw.healthVerdict)
            raw.healthVerdict = .init(dot: health.dot, checks: health.checks, notes: health.notes, evaluatedAt: Date())
            controller.setBadgeForTesting(raw, alertsURL: root.appendingPathComponent("job-alerts.json"))
            controller.renderStatusIconForTesting(stats: BrainBarDashboardFixture.makeCollector().stats, state: .idle)
            let icon = try XCTUnwrap(controller.statusItemForTesting.button?.image?.tiffRepresentation)
            XCTAssertNotEqual(icon, previousIcon)
            previousIcon = icon
            controller.menuNeedsUpdate(controller.contextMenuForTesting)
            if state != "green" {
                XCTAssertTrue(controller.contextMenuForTesting.items.contains { $0.title == "Database snapshot (verified): Fixture backup age" })
            }
            controller.flushStatusReceiptForTesting()
            let url = BrainBarUIStatusReceipt.url(dbPath: root.appendingPathComponent("fixture.db").path, environment: [:])
            let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            XCTAssertEqual(json["dot"] as? String, state)
            XCTAssertEqual(json["source"] as? String, "verdict")
            XCTAssertNotNil(json["rendered_at"])
            XCTAssertNotNil(json["verdict_evaluated_at"])
            XCTAssertNotNil(json["app_version"])
            XCTAssertNotNil(json["git_commit"])
            XCTAssertEqual(json["reasons"] as? [String],
                BrainBarStatusPopoverController.verdictRows(raw.healthVerdict!).map(\.text))
            let collector = BrainBarDashboardFixture.makeCollector()
            let observed = try BrainBarOnePageTestFixture.healthyResult()
            let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: now)
            let hero = BrainBarHeroPresentation.derive(flow: flow, stats: collector.stats,
                backupTruth: BrainBarHeroBackupTruth.derive(from: observed, now: now, cadence: .known(300)),
                locale: Locale(identifier: "en_US"))
            let dashboard = BrainBarOnePagePresentation.derive(
                snapshotFreshness: collector.snapshotFreshnessState, hero: hero, observability: observed,
                stats: collector.stats, agentActivity: BrainBarDashboardFixture.agentActivity, now: now,
                calendar: BrainBarOnePageTestFixture.calendar, locale: Locale(identifier: "en_US"),
                observabilityCadence: .known(300), statusVerdict: controller.statusVerdictForTesting)
            XCTAssertEqual(json["reasons"] as? [String], dashboard.attentionItems)
            let items = try XCTUnwrap(json["items"] as? [[String: String]])
            XCTAssertEqual(items.compactMap { $0["text"] }, dashboard.attentionItems)
            XCTAssertEqual(items.map { $0["kind"] }, state == "red" ? ["attention", "notice"] :
                Array(repeating: "notice", count: dashboard.attentionItems.count))
            XCTAssertEqual(json["attention_count"] as? Int, controller.statusVerdictForTesting.attentionRowCount)
            if state != "green" {
                XCTAssertEqual(BrainBarStatusPopoverController.verdictRows(health).map(\.text),
                    ["Database snapshot (verified): Fixture backup age", "Weekly maintenance deferred"])
            }
        }
        let emptyAmber = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [],
            healthVerdict: .init(dot: .amber, checks: [], notes: [], evaluatedAt: Date()))
        controller.setBadgeForTesting(emptyAmber, alertsURL: root.appendingPathComponent("job-alerts.json"))
        controller.renderStatusIconForTesting(stats: BrainBarDashboardFixture.makeCollector().stats, state: .idle)
        controller.flushStatusReceiptForTesting()
        let emptyURL = BrainBarUIStatusReceipt.url(dbPath: root.appendingPathComponent("fixture.db").path, environment: [:])
        let emptyReceipt = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: emptyURL)) as? [String: Any])
        XCTAssertEqual(emptyReceipt["dot"] as? String, "amber")
        XCTAssertEqual(emptyReceipt["reasons"] as? [String], ["Health check: amber"])
        XCTAssertEqual(emptyReceipt["items"] as? [[String: String]], [["text": "Health check: amber", "kind": "notice"]])
        XCTAssertEqual(emptyReceipt["attention_count"] as? Int, 0)
        for state in ["green", "red"] {
            var raw = try badge(state)
            let health = try XCTUnwrap(raw.healthVerdict)
            raw.healthVerdict = .init(dot: health.dot, checks: health.checks, notes: health.notes,
                                     evaluatedAt: Date().addingTimeInterval(-4933))
            raw = .init(badgeOn: true, reason: "Yesterday's database checkpoint failed", activeCodes: [],
                        healthVerdict: raw.healthVerdict)
            controller.setBadgeForTesting(raw, alertsURL: root.appendingPathComponent("job-alerts.json"))
            controller.renderStatusIconForTesting(stats: BrainBarDashboardFixture.makeCollector().stats, state: .idle)
            controller.menuNeedsUpdate(controller.contextMenuForTesting)
            controller.flushStatusReceiptForTesting()
            let url = BrainBarUIStatusReceipt.url(dbPath: root.appendingPathComponent("fixture.db").path, environment: [:])
            let receipt = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            XCTAssertEqual(receipt["dot"] as? String, state == "red" ? "red" : "amber")
            XCTAssertEqual(receipt["source"] as? String, "fallback")
            XCTAssertNotNil(receipt["verdict_evaluated_at"] as? String)
            let reasons = try XCTUnwrap(receipt["reasons"] as? [String])
            XCTAssertTrue(reasons.contains { $0.hasPrefix("Health check stale since ") })
            XCTAssertFalse(reasons.contains(raw.reason))
            XCTAssertTrue(controller.contextMenuForTesting.items.contains { $0.title.hasPrefix("Health check stale since ") })
        }
        controller.setBadgeForTesting(.failVisible("Legacy failure"), alertsURL: root.appendingPathComponent("job-alerts.json"))
        controller.renderStatusIconForTesting(stats: BrainBarDashboardFixture.makeCollector().stats, state: .idle)
        controller.flushStatusReceiptForTesting()
        let fallbackURL = BrainBarUIStatusReceipt.url(dbPath: root.appendingPathComponent("fixture.db").path, environment: [:])
        let fallback = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fallbackURL)) as? [String: Any])
        XCTAssertEqual(fallback["source"] as? String, "fallback")
        XCTAssertEqual(fallback["dot"] as? String, "red")
        XCTAssertTrue(fallback["verdict_evaluated_at"] is NSNull)
        XCTAssertEqual(fallback["reasons"] as? [String], ["Legacy failure"])
        controller.setBadgeForTesting(try badge("broken"), alertsURL: root.appendingPathComponent("job-alerts.json"))
        controller.renderStatusIconForTesting(stats: BrainBarDashboardFixture.makeCollector().stats, state: .idle)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        controller.flushStatusReceiptForTesting()
        let unreadable = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fallbackURL)) as? [String: Any])
        XCTAssertEqual(unreadable["source"] as? String, "fallback")
        XCTAssertEqual(unreadable["dot"] as? String, "green")
        XCTAssertTrue(controller.contextMenuForTesting.items.contains { $0.title == "Health check result could not be read" })
        XCTAssertEqual(BrainBarUIStatusReceipt.url(dbPath: "/custom/fixture.db", environment: [:]).path,
                       "/custom/brainbar-ui-status.json")
        XCTAssertEqual(BrainBarUIStatusReceipt.url(dbPath: "/custom/fixture.db",
            environment: ["BRAINLAYER_BADGE_STATE_PATH": "/override/badge.json"]).path,
                       "/override/brainbar-ui-status.json")
    }
}

private final class HealthReceiptWrites: @unchecked Sendable {
    private let lock = NSLock()
    private var receipts: [BrainBarUIStatusReceipt] = []
    var count: Int { lock.withLock { receipts.count } }
    func record(_ receipt: BrainBarUIStatusReceipt) { lock.withLock { receipts.append(receipt) } }
}
