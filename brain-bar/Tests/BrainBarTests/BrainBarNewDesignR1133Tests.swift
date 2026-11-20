import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarNewDesignR1133Tests: XCTestCase {
    typealias R113 = BrainBarNewDesignR113Tests

    struct KeyboardControl: View {
        var body: some View { Text("Reconnect").ndKeyboardFocus(action: {}) }
    }

    /// Inspect the declarative control subtree without creating a hosting view or window.
    func testWaitingRemovesTheKeyboardControlSubtree() {
        func containsControl(_ value: Any) -> Bool {
            if value is KeyboardControl { return true }
            return Mirror(reflecting: value).children.contains { containsControl($0.value) }
        }
        for waiting in [false, true, false] {
            let swap = NDWaitingSwap(waiting: waiting, button: { KeyboardControl() }, line: { Text("Waiting") })
            XCTAssertEqual(containsControl(swap.body), !waiting)
        }
    }

    func testCatchUpDoesNotExcuseAnotherIngestFailure() {
        for reason in ["file_ingestion_failure", "coverage_drop", "growth_not_indexed:codex", "unknown_failure"] {
            let c = R113.capture(ingest: .red, reasons: [reason], catchUp: .init(state: "catching_up", bytesBehind: 2048, etaS: 60))
            let p = BrainBarMenuPresenter.present(c)
            XCTAssertGreaterThan(p.count, 0, reason)
            XCTAssertFalse(BrainBarMenuPresenter.notificationFaults(c).isEmpty, reason)
            XCTAssertFalse(p.rows.flatMap(\.strings).contains { $0.contains("catching up by itself") }, reason)
        }
    }

    func testCatchUpDoesNotExcuseIngestContextOrBlindFailure() {
        for blind in [false, true] {
            var c = R113.capture(ingest: .red, reasons: [], catchUp: .init(state: "catching_up"))
            let checks = c.verdict!.checks.map { check -> BrainBarHealthVerdict.Check in
                var k = check
                if k.name == "ingest" {
                    k.contextCodes = blind ? [] : ["file_ingestion_failure"]
                    k.blind = blind
                    k.blindCodes = blind ? ["db_probe_failed"] : []
                }
                return k
            }
            c.verdict = .init(dot: .red, checks: checks, notes: [], evaluatedAt: c.now)
            XCTAssertGreaterThan(BrainBarMenuPresenter.present(c).count, 0)
            XCTAssertFalse(BrainBarMenuPresenter.notificationFaults(c).isEmpty)
        }
    }

    func testEmbeddingNoticeKeepsDistinctRedSearchAcrossSurfaces() throws {
        var i = BrainBarDesignFixture.inputs(.calm)
        let checks = i.capture.verdict!.checks.map { check -> BrainBarHealthVerdict.Check in
            var k = check
            if k.name == "search" { return .init(name: "search", state: .red, note: "", fails: 3) }
            if k.name == "embedding" { k = .init(name: "embedding", state: .amber, ageS: 120, thresholdS: 300, note: "") }
            return k
        }
        i.capture.verdict = .init(dot: .red, checks: checks, notes: [], evaluatedAt: i.capture.now)
        let m = NDModel.make(i), r = NDR1.make(i, NDModel.make(i))
        let search = try XCTUnwrap(m.faults.first { $0.concerns.contains("search") })
        XCTAssertTrue(r.items.contains { $0.concerns.contains("search") && $0.title == search.title && $0.severity == .red })
        XCTAssertGreaterThan(r.redCount, 0)
        XCTAssertEqual(NDIcon.state(m, r), .red(r.redCount))
        XCTAssertTrue(NDMenuHeadline.text(r).contains("needs you"))
        XCTAssertTrue(r.headline.contains("needs you"))
    }

    func testPollingWithoutReadsNeverClaimsLastRead() throws {
        for poll in [1, 40] {
            let c = R113.capture(ingest: .red, reasons: [], catchUp: nil, fresh: false)
            let file = WatcherHealthFile(updatedAt: c.now.addingTimeInterval(-600), pollCount: poll)
            let adapted = BrainBarMenuLiveCapture.capture(
                status: .init(issues: [], isChecking: false, healthVerdict: c.verdict),
                verdictAwakeAgeS: 0, wakeCatchUpPending: false, driveAuth: nil,
                backups: .notRead, backupsNotice: nil, watcher: .readable(file), watcherAwakeAgeS: 600,
                observations: nil, now: c.now, timeZone: c.timeZone)
            XCTAssertNil(adapted.watcher?.lastReadAt)
            XCTAssertFalse(BrainBarMenuPresenter.present(adapted).rows.flatMap(\.strings).contains { $0.contains("last read") })
            var i = NDInputs(capture: adapted)
            i.watcherHeartbeatAgeS = 600
            XCTAssertNil(NDPipelineState.statusLines(i)[.read])
            XCTAssertFalse(NDPipelineState.lastKnownLines(i, [.read: .init(text: "—")], at: c.now)[.read]?.text.contains("last read") == true)
        }
    }

    func testUnknownEmbeddingAgeKeepsFaultWithoutInventingZero() throws {
        for age: Double? in [nil, -.infinity, .infinity, .nan, -1] {
            var i = BrainBarDesignFixture.inputs(.calm)
            let checks = i.capture.verdict!.checks.map { k in
                k.name == "embedding" ? BrainBarHealthVerdict.Check(name: "embedding", state: .red, ageS: age, note: "") : k
            }
            i.capture.verdict = .init(dot: .red, checks: checks, notes: [], evaluatedAt: i.capture.now)
            let r = NDR1.make(i, NDModel.make(i))
            let item = try XCTUnwrap(r.items.first { $0.id == "meaning" })
            XCTAssertEqual(item.severity, .red)
            XCTAssertEqual(item.title, "Search by meaning is behind")
            XCTAssertEqual(r.redCount, 1)
        }
    }
}
