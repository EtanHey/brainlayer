import Foundation
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarAmberNoticeTests: XCTestCase {
    private final class Center: BrainBarFaultNotificationCenter {
        var posts = 0
        func authorize() async throws -> Bool { true }
        func post(id: String, body: String) async throws { posts += 1 }
    }
    private func capture(_ name: String, _ state: BrainBarHealthVerdict.State, amber: String? = nil) -> BrainBarMenuCapture {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        var c = BrainBarMenuCapture(now: now, timeZone: TimeZone(secondsFromGMT: 0)!)
        c.verdict = .init(dot: state == .red ? .red : .amber, checks: BrainBarMenuPresenter.known.map {
            .init(name: $0, state: $0 == name ? state : $0 == amber ? .amber : .green, note: "Measured check", blind: false)
        }, notes: [], evaluatedAt: now)
        c.maintenanceNightFailures = 0
        c.backupDailyFailures = 0
        c.jsonlBackupFailures = 0
        return c
    }
    private final class Clock { var now = Date(timeIntervalSince1970: 1_800_000_000) }
    private func subject(_ center: Center, clock: @escaping () -> Date = Date.init) -> BrainBarFaultNotifications {
        let key = "BrainBarAmberNoticeTests.\(UUID())"
        let defaults = UserDefaults(suiteName: key)!
        addTeardownBlock { defaults.removePersistentDomain(forName: key) }
        return BrainBarFaultNotifications(center: center, defaults: defaults, clock: clock)
    }
    private func assertAmber(_ name: String) async {
        let c = capture(name, .amber)
        let p = BrainBarMenuPresenter.present(c)
        XCTAssertEqual(p.headline, "Check soon", name)
        XCTAssertEqual(p.count, 0, name)
        XCTAssertEqual(p.icon, .resting, name)
        XCTAssertFalse(p.rows.contains { $0.kind == .needsYou }, name)
        XCTAssertTrue(p.rows.contains { $0.kind == .statusLine && $0.state == .notice }, name)
        XCTAssertFalse(p.rows.contains { $0.action != nil }, name)
        let report = BrainBarMenuReport.text(presentation: p, verdict: c.verdict, version: "test", now: c.now)
        XCTAssertFalse(report.contains(BrainBarMenuPresenter.Say.cantFix.text), name)
        let center = Center()
        await subject(center).update(c)
        XCTAssertEqual(center.posts, 0, name)
    }
    func testAmberBackupLocal() async { await assertAmber("backup_local") }
    func testAmberBackupOffsite() async { await assertAmber("backup_offsite") }
    func testAmberTranscripts() async { await assertAmber("transcripts") }
    func testAmberDriveAuth() async { await assertAmber("drive_auth") }
    func testAmberIntegrity() async { await assertAmber("integrity") }
    func testAmberDisk() async { await assertAmber("disk") }
    func testAmberIngest() async { await assertAmber("ingest") }
    func testAmberStores() async { await assertAmber("stores") }
    func testAmberSearch() async { await assertAmber("search") }
    func testAmberEmbedding() async { await assertAmber("embedding") }
    func testAmberMonitor() async { await assertAmber("monitor") }

    func testRedAndMixedCountOrderAndStatusFileParity() throws {
        for name in BrainBarMenuPresenter.known {
            for state in [BrainBarHealthVerdict.State.amber, .red] {
                let c = capture(name, state)
                let p = BrainBarMenuPresenter.present(c)
                var input = BrainBarStatusSnapshotTests.healthy()
                input.now = c.now
                input.badge?.healthVerdict = c.verdict
                let verdict = BrainBarStatusSnapshot.derive(input).verdict
                let receipt = BrainBarUIStatusReceipt(renderedAt: c.now, dot: verdict.dot, source: "test",
                    verdictEvaluatedAt: c.now, reasons: [], appVersion: "test", gitCommit: "test", items: verdict.currentItems)
                let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: dir) }
                let file = dir.appendingPathComponent("brainbar-ui-status.json")
                try receipt.write(to: file)
                let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
                XCTAssertEqual(p.count, doc["attention_count"] as? Int, "\(name) \(state)")
                XCTAssertEqual(p.count, state == .red ? 1 : 0)
                if state == .red {
                    let titles = ["backup_local": BrainBarMenuPresenter.Say.noRecord.text,
                        "drive_auth": BrainBarMenuPresenter.Say.s13.text,
                        "embedding": BrainBarMenuPresenter.Say.f07.text,
                        "integrity": BrainBarMenuPresenter.Say.s19.text]
                    XCTAssertEqual(p.rows.first { $0.kind == .needsYou }?.strings.first,
                        titles[name] ?? BrainBarMenuPresenter.Say.generic.text, name)
                }
            }
            let other = ["backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk"].contains(name) ? "stores" : "disk"
            let p = BrainBarMenuPresenter.present(capture(name, .red, amber: other))
            XCTAssertEqual(p.count, 1, name)
            let rows = p.rows.filter { $0.kind == .needsYou || $0.kind == .statusLine }
            XCTAssertEqual(rows.first?.kind, .needsYou, name)
            XCTAssertEqual(rows.dropFirst().first?.state, .notice, name)
            XCTAssertEqual(rows.dropFirst().first?.strings.first, BrainBarMenuPresenter.Say.generic.text, other)
        }
    }
    private func assertOneGenericSentence(_ c: BrainBarMenuCapture) {
        let p = BrainBarMenuPresenter.present(c)
        XCTAssertEqual(p.rows.filter { $0.strings.first == BrainBarMenuPresenter.Say.generic.text }.count, 1)
    }
    func testNoDuplicateAtAmberLagTimeEscalation() {
        var c = capture("ingest", .amber)
        c.watcher?.alertReasons = ["offset_lag"]
        c.lagEpisodeStart = c.now.addingTimeInterval(-1801)
        assertOneGenericSentence(c)
    }
    func testNoDuplicateAtRedIngestStall() {
        var c = capture("ingest", .red)
        c.watcher?.alertReasons = ["file_ingestion_failure"]
        assertOneGenericSentence(c)
    }
    func testNoDuplicateAtRedBackupAmberTranscripts() {
        assertOneGenericSentence(capture("backup_offsite", .red, amber: "transcripts"))
    }

    func testDifferentAmberSentenceInSameThingStays() {
        var c = capture("integrity", .red, amber: "backup_local")
        c.backupDailyFailures = 2
        c.verdict = .init(dot: .red, checks: c.verdict!.checks.map { check in
            var check = check
            if check.name == "backup_local" { check.ageS = 10; check.thresholdS = 20 }
            return check
        }, notes: [], evaluatedAt: c.now)
        let rows = BrainBarMenuPresenter.present(c).rows
        XCTAssertEqual(rows.first { $0.kind == .needsYou }?.strings.first, BrainBarMenuPresenter.Say.s19.text)
        XCTAssertEqual(rows.first { $0.state == .notice }?.strings.first, BrainBarMenuPresenter.Say.s06("2"))
    }

    func testCountParityBoundaryForMergedRowsAndAppOnlyClocks() {
        var c = capture("backup_local", .red)
        c.verdict = .init(dot: .red, checks: c.verdict!.checks.map {
            $0.name == "integrity" ? .init(name: $0.name, state: .red, note: "Measured corruption") : $0
        }, notes: [], evaluatedAt: c.now)
        var input = BrainBarStatusSnapshotTests.healthy()
        input.now = c.now
        input.badge?.healthVerdict = c.verdict
        XCTAssertEqual(BrainBarMenuPresenter.present(c).count, 1, "M2 merges two red backup checks into one row")
        XCTAssertEqual(BrainBarStatusSnapshot.derive(input).verdict.attentionRowCount, 2, "Classic counts check items")
        c = capture("disk", .amber)
        c.maintenanceNightFailures = 3
        input.badge?.healthVerdict = c.verdict
        XCTAssertEqual(BrainBarMenuPresenter.present(c).count, 1, "App-only time escalation")
        XCTAssertEqual(BrainBarStatusSnapshot.derive(input).verdict.attentionRowCount, 0, "Receipt inputs lack the menu capture's streak")
    }

    func testAmberWatcherNoticeDoesNotWorsenRedSiblingInSameThing() async {
        let center = Center()
        let observed = subject(center)
        var c = capture("stores", .red, amber: "ingest")
        await observed.update(c)
        c.watcher?.alertReasons = ["file_ingestion_failure"]
        await observed.update(c)
        XCTAssertEqual(center.posts, 1)
        assertOneGenericSentence(c)
        XCTAssertFalse(BrainBarMenuPresenter.notificationFaults(c).flatMap { $0.causes }.contains("app:file_ingestion_failure"))
    }

    func testPairedJobCannotPromoteAnAmberCheck() async {
        for (job, check) in [("drive-consent", "drive_auth"), ("backup-daily", "backup_local"), ("jsonl-backup", "transcripts")] {
            let center = Center()
            await subject(center).update(capture(check, .amber), jobAlerts: [job: "Measured job alert"])
            XCTAssertEqual(center.posts, 0, check)
        }
    }

    func testAmberRedAmberTransitionsClearAndRearmEachThing() async {
        for name in BrainBarMenuPresenter.known {
            let center = Center(), clock = Clock()
            let notifier = subject(center, clock: { clock.now })
            await notifier.update(capture(name, .amber))
            XCTAssertEqual(center.posts, 0, name)
            await notifier.update(capture(name, .red))
            await notifier.update(capture(name, .red))
            XCTAssertEqual(center.posts, 1, name)
            await notifier.update(capture(name, .amber))
            clock.now = clock.now.addingTimeInterval(601)
            await notifier.update(capture(name, .red))
            XCTAssertEqual(center.posts, 2, name)
            XCTAssertTrue(BrainBarMenuPresenter.notificationFaults(capture(name, .amber)).isEmpty, name)
        }
    }
}
