import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarHealthVerdictTests: XCTestCase {
    private let now = BrainBarStatusSnapshotTests.now

    private func badge(_ dot: String?, age: Double = 0, generatedAge: Double = 0) throws -> BadgeStatePresentation {
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
        let url = root.appendingPathComponent("badge-state.json")
        try JSONSerialization.data(withJSONObject: document).write(to: url)
        return BadgeStateReader.read(url: url, now: now, cadence: .known(300))
    }

    func testVerdictFixtureMatrixAndLegacyFallback() throws {
        for (state, age, expected, source) in [
            ("green", 0.0, BrainBarRenderedDot.green, "verdict"), ("amber", 0, .amber, "verdict"),
            ("red", 0, .red, "verdict"), ("unknown", 0, .amber, "verdict"),
            ("green", 900, .green, "verdict"), ("green", 901, .red, "fallback"),
            ("green", -1, .red, "fallback"), ("broken", 0, .red, "fallback"),
        ] {
            var inputs = BrainBarStatusSnapshotTests.healthy()
            inputs.backups = .measured(BrainBarStatusSnapshotTests.backups(db: nil))
            inputs.badge = try badge(state, age: age)
            let snapshot = BrainBarStatusSnapshot.derive(inputs)
            XCTAssertEqual(snapshot.verdict.dot, expected, "\(state), age \(age)")
            XCTAssertEqual(snapshot.healthVerdict == nil ? "fallback" : "verdict", source)
            if age == 901 { XCTAssertTrue(snapshot.notices.contains { $0.text == "Health check result is 15 min old" }) }
            if state == "green", source == "verdict" {
                XCTAssertTrue(snapshot.notices.contains { $0.id == "database.copy" })
                XCTAssertTrue(snapshot.attention.isEmpty)
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
            activeMessages: ["Old job"], healthVerdict: raw.healthVerdict)
        let reconciled = withJob.reconciled(with: BrainBarJobAlerts(active: [:]))
        XCTAssertFalse(reconciled.badgeOn)
        XCTAssertEqual(reconciled.healthVerdict, raw.healthVerdict)
        var inputs = BrainBarStatusSnapshotTests.healthy()
        inputs.badge = raw
        inputs.now = now.addingTimeInterval(700)
        XCTAssertEqual(BrainBarStatusSnapshot.derive(inputs).verdict.dot, .green)
        inputs.now = now.addingTimeInterval(901)
        XCTAssertNil(BrainBarStatusSnapshot.derive(inputs).healthVerdict)
    }

    func testRenderHarnessPopoverEvidence() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_S2_RENDER_DIR"] else { return }
        let directory = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for state in ["missing", "green", "red", "amber"] {
            var inputs = BrainBarStatusSnapshotTests.healthy()
            inputs.backups = .measured(BrainBarStatusSnapshotTests.backups(db: nil))
            inputs.badge = try badge(state == "missing" ? nil : state)
            for width in [560.0, 390.0] {
                try BrainBarRenderHarness.renderHealthVerdictPopover(BrainBarStatusSnapshot.derive(inputs),
                    directory: directory, name: "s2-\(state)-\(Int(width))", width: width)
            }
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
                XCTAssertTrue(controller.contextMenuForTesting.items.contains { $0.title == "Local database backup: Fixture backup age" })
            }
            let url = BrainBarUIStatusReceipt.url(dbPath: root.appendingPathComponent("fixture.db").path, environment: [:])
            let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            XCTAssertEqual(json["dot"] as? String, state)
            XCTAssertEqual(json["source"] as? String, "verdict")
            XCTAssertNotNil(json["rendered_at"])
            XCTAssertNotNil(json["verdict_evaluated_at"])
            XCTAssertNotNil(json["app_version"])
            XCTAssertNotNil(json["git_commit"])
            XCTAssertEqual((json["reasons"] as? [String])?.isEmpty, state == "green")
            if state != "green" {
                XCTAssertEqual(BrainBarStatusPopoverController.verdictRows(health),
                    ["Local database backup: Fixture backup age", "Weekly maintenance deferred"])
            }
        }
        controller.setBadgeForTesting(.failVisible("Legacy failure"), alertsURL: root.appendingPathComponent("job-alerts.json"))
        controller.renderStatusIconForTesting(stats: BrainBarDashboardFixture.makeCollector().stats, state: .idle)
        let fallbackURL = BrainBarUIStatusReceipt.url(dbPath: root.appendingPathComponent("fixture.db").path, environment: [:])
        let fallback = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fallbackURL)) as? [String: Any])
        XCTAssertEqual(fallback["source"] as? String, "fallback")
        XCTAssertEqual(fallback["dot"] as? String, "red")
        XCTAssertTrue(fallback["verdict_evaluated_at"] is NSNull)
        XCTAssertEqual(fallback["reasons"] as? [String], ["Legacy failure"])
        XCTAssertEqual(BrainBarUIStatusReceipt.url(dbPath: "/custom/fixture.db", environment: [:]).path,
                       "/custom/brainbar-ui-status.json")
        XCTAssertEqual(BrainBarUIStatusReceipt.url(dbPath: "/custom/fixture.db",
            environment: ["BRAINLAYER_BADGE_STATE_PATH": "/override/badge.json"]).path,
                       "/override/brainbar-ui-status.json")
    }
}
