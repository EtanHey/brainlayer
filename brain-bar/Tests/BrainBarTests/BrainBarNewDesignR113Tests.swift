import Foundation
import XCTest
@testable import BrainBar

/// Revision 1.13: BrainLayer #1827's `catch_up` (release 1.6.6) gives the lag its minutes left ("about N min to go") and a
/// real "stops shrinking"; the decision stays inside `appEscalationCauses` (the R31 guard). L55: a lag that drains is a
/// calm notice; it needs him only when it stops shrinking past R20/R31's 30-minute clock or the reader stops.
final class BrainBarNewDesignR113Tests: XCTestCase {
    typealias Say = BrainBarMenuPresenter.Say
    typealias Check = BrainBarHealthVerdict.Check
    static let now = ISO8601DateFormatter().date(from: "2026-11-19T07:20:00+02:00")!
    static let mb117 = 117 * 1_048_576

    /// Every known check green but `ingest`, the streaks read, on the default lag switch (L55's notice side).
    static func capture(ingest: BrainBarHealthVerdict.State, reasons: [String], catchUp: WatcherCatchUp?, clock: Double? = nil,
                        fresh: Bool = true) -> BrainBarMenuCapture {
        var c = BrainBarMenuCapture(now: now, timeZone: TimeZone(secondsFromGMT: 7200)!)
        let checks = BrainBarMenuPresenter.known.map { Check(name: $0, state: $0 == "ingest" ? ingest : .green, note: "", unknownTicks: 0, blind: false) }
        c.verdict = BrainBarHealthVerdict(dot: ingest == .red ? .red : .amber, checks: checks, notes: [], evaluatedAt: now)
        c.maintenanceNightFailures = 0
        c.backupDailyFailures = 0
        c.jsonlBackupFailures = 0
        c.watcher = .init(fresh: fresh, alertReasons: reasons, lagBytes: mb117, lastReadAt: now.addingTimeInterval(-3000), catchUp: catchUp)
        c.lagEpisodeStart = clock.map { now.addingTimeInterval(-$0) }
        return c
    }

    func notifies(_ c: BrainBarMenuCapture) -> Bool { !BrainBarMenuPresenter.notificationFaults(c).isEmpty }

    // MARK: each state

    /// catch_up present and shrinking: a notice, "about N min to go", no count, no notification, past the clock too.
    func testCatchingUpIsANoticeWithItsMinutesToGo() {
        for clock in [nil, 2400.0] {
            let c = Self.capture(ingest: .amber, reasons: [], catchUp: .init(state: "catching_up", bytesBehind: Self.mb117, etaS: 1500), clock: clock)
            let p = BrainBarMenuPresenter.present(c)
            XCTAssertEqual(p.form, .calm, "\(String(describing: clock))")
            XCTAssertEqual(p.count, 0)
            XCTAssertEqual(p.headline, Say.lagBehind.text)
            let line = p.rows.first { $0.kind == .headlineDetail }?.strings.first ?? ""
            XCTAssertTrue(line.hasSuffix("It\u{2019}s catching up by itself, about 25\u{00A0}min to go."), line)
            XCTAssertTrue(line.hasPrefix("117\u{00A0}MB left to read"), line)
            XCTAssertFalse(notifies(c), "no notification while it shrinks")
            XCTAssertEqual(BrainBarMenuInvariantTests.violations(c, p), [])
        }
        // A red reader that BrainLayer measures catching up is still the notice (the catch-up is the red's whole story).
        let red = Self.capture(ingest: .red, reasons: [], catchUp: .init(state: "catching_up", bytesBehind: Self.mb117, etaS: 1500))
        XCTAssertEqual(BrainBarMenuPresenter.present(red).form, .calm)
        XCTAssertFalse(notifies(red))
    }

    /// catch_up absent (an older producer): L55's behaviour, the lag as the watcher's only alert.
    func testWithoutCatchUpL55Stands() {
        let draining = Self.capture(ingest: .red, reasons: ["offset_lag"], catchUp: nil, clock: 2400)
        let p = BrainBarMenuPresenter.present(draining)
        XCTAssertEqual(p.form, .calm)
        XCTAssertTrue(p.rows.first { $0.kind == .headlineDetail }?.strings.first?.hasSuffix("It\u{2019}s catching up by itself.") == true)
        XCTAssertFalse(notifies(draining))
        let notDraining = Self.capture(ingest: .amber, reasons: ["offset_lag", "file_ingestion_failure"], catchUp: nil, clock: 2400)
        XCTAssertEqual(BrainBarMenuPresenter.present(notDraining).form, .needs)
        XCTAssertTrue(notifies(notDraining))
    }

    /// offset_lag back (BrainLayer: it stopped shrinking): past the clock, A3 needs him, with the report, and notifies;
    /// under the clock it is A3's notice, never R02's "catching up".
    func testStoppedShrinkingPastTheClockNeedsYou() {
        let past = Self.capture(ingest: .amber, reasons: ["offset_lag"], catchUp: .init(state: "stalled", bytesBehind: Self.mb117), clock: 2400)
        let p = BrainBarMenuPresenter.present(past)
        XCTAssertEqual(p.form, .needs)
        XCTAssertEqual(p.count, 1)
        let row = p.rows.first { $0.kind == .needsYou }
        XCTAssertEqual(row?.strings.first, "Reading new conversations isn\u{2019}t catching up")
        XCTAssertEqual(row?.strings.last, Say.agent.text)
        XCTAssertEqual(row?.action?.kind, .copyReport)
        XCTAssertFalse(row?.strings.contains { $0.contains("catching up by itself") } ?? true, "never 'by itself' once it stopped")
        XCTAssertTrue(notifies(past))
        XCTAssertEqual(BrainBarMenuInvariantTests.violations(past, p), [])
        let under = Self.capture(ingest: .amber, reasons: ["offset_lag"], catchUp: .init(state: "stalled", bytesBehind: Self.mb117), clock: 600)
        let q = BrainBarMenuPresenter.present(under)
        XCTAssertEqual(q.form, .calm)
        XCTAssertEqual(q.headline, "Reading new conversations isn\u{2019}t catching up")
        XCTAssertFalse(Say.catchingUp(under), "R02's 'Catching up on new conversations' is not said")
        XCTAssertFalse(notifies(under))
        XCTAssertEqual(BrainBarMenuInvariantTests.violations(under, q), [])
    }

    /// The reader stopped (red, the file no longer fresh): needs him, whatever the catch-up said.
    func testReaderStoppedNeedsYou() {
        let stopped = Self.capture(ingest: .red, reasons: [], catchUp: .init(state: "catching_up", bytesBehind: Self.mb117, etaS: 1500), fresh: false)
        let p = BrainBarMenuPresenter.present(stopped)
        XCTAssertEqual(p.form, .needs)
        XCTAssertEqual(p.rows.first { $0.kind == .needsYou }?.strings.first, Say.stopped.text)
        XCTAssertFalse(Say.lagDraining(stopped), "a stale file says nothing about catching up")
        XCTAssertEqual(BrainBarMenuInvariantTests.violations(stopped, p), [])
    }

    // MARK: decoding

    func decode(_ catchUp: String?) -> WatcherHealthFile? {
        let block = catchUp.map { ", \"catch_up\": \($0)" } ?? ""
        let json = "{\"updated_at\": \"2026-11-19T05:20:00+00:00\", \"poll_count\": 9, \"alert_reasons\": [], \"max_offset_lag_bytes\": 2048\(block)}"
        guard case let .readable(file) = WatcherHealthReader.parse(Data(json.utf8), path: "fixture") else { return nil }
        return file
    }

    /// The field missing, `eta_s` missing (or null), `bytes_behind` missing: each leaves that part out; a bad part never
    /// makes the whole file unreadable.
    func testDecodingLeavesOutWhatIsMissing() throws {
        XCTAssertNil(try XCTUnwrap(decode(nil)).catchUp, "an older producer: no catch_up")
        XCTAssertEqual(try XCTUnwrap(decode(#"{"state": "catching_up", "bytes_behind": 122683392, "eta_s": 1500.4}"#)).catchUp,
                       WatcherCatchUp(state: "catching_up", bytesBehind: 122_683_392, etaS: 1500.4))
        XCTAssertEqual(try XCTUnwrap(decode(#"{"state": "catching_up", "bytes_behind": 122683392}"#)).catchUp?.etaS, nil)
        XCTAssertEqual(try XCTUnwrap(decode(#"{"state": "stalled", "bytes_behind": 122683392, "eta_s": null}"#)).catchUp?.etaS, nil)
        XCTAssertEqual(try XCTUnwrap(decode(#"{"state": "catching_up", "eta_s": 1500}"#)).catchUp?.bytesBehind, nil)
        XCTAssertEqual(try XCTUnwrap(decode(#"{"state": 7, "bytes_behind": "a lot", "eta_s": true}"#)).catchUp, WatcherCatchUp())
        XCTAssertEqual(try XCTUnwrap(decode("null")).catchUp, WatcherCatchUp(), "present but not an object: every part unknown")
        // What each missing part does to the words: no estimate, no "about … to go"; no bytes, the older field's size.
        let noEta = Self.capture(ingest: .amber, reasons: [], catchUp: .init(state: "catching_up", bytesBehind: Self.mb117))
        XCTAssertEqual(BrainBarMenuPresenter.lagDetail(noEta), "117\u{00A0}MB left to read. It\u{2019}s catching up by itself.")
        var noBytes = Self.capture(ingest: .amber, reasons: [], catchUp: .init(state: "catching_up", etaS: 1500))
        noBytes.watcher?.lagBytes = 3 * 1_048_576
        XCTAssertEqual(BrainBarMenuPresenter.lagDetail(noBytes), "3\u{00A0}MB left to read. It\u{2019}s catching up by itself, about 25\u{00A0}min to go.")
        noBytes.watcher?.lagBytes = nil
        XCTAssertEqual(BrainBarMenuPresenter.lagDetail(noBytes), "It\u{2019}s catching up by itself, about 25\u{00A0}min to go.")
        // The live adapter carries it.
        let file = try XCTUnwrap(decode(#"{"state": "catching_up", "bytes_behind": 122683392, "eta_s": 1500}"#))
        let adapted = BrainBarMenuLiveCapture.capture(status: .checking, verdictAwakeAgeS: nil, wakeCatchUpPending: false, driveAuth: nil,
            backups: .notRead, backupsNotice: nil, watcher: .readable(file), watcherAwakeAgeS: 40, observations: nil,
            now: Self.now, timeZone: TimeZone(secondsFromGMT: 7200)!)
        XCTAssertEqual(adapted.watcher?.catchUp?.etaS, 1500)
    }

    // MARK: the window

    /// The window's Read station and its panel use the same facts and words as the menu.
    func testTheWindowsReadStationSaysTheSame() throws {
        let i = BrainBarDesignFixture.inputs(.lagBehind)
        let r1 = NDR1.make(i, NDModel.make(i))
        let reader = try XCTUnwrap(r1.items.first { $0.id == "reader" })
        XCTAssertEqual(reader.severity, .orange)
        XCTAssertEqual(reader.title, BrainBarMenuPresenter.present(i.capture).headline)
        XCTAssertEqual(reader.detail, "117\u{00A0}MB left to read. It\u{2019}s catching up by itself, about 25\u{00A0}min to go.")
        XCTAssertNil(reader.action)
        XCTAssertEqual(NDStationPopover.faultLines(.read, item: reader, i), [reader.detail ?? ""], "its panel")
        XCTAssertEqual(r1.headline, reader.title)
        let stalled = BrainBarDesignFixture.inputs(.lagStalled)
        let s = NDR1.make(stalled, NDModel.make(stalled))
        let stalledReader = try XCTUnwrap(s.items.first { $0.id == "reader" })
        XCTAssertEqual(stalledReader.severity, .red)
        XCTAssertEqual(stalledReader.title, "Reading new conversations isn\u{2019}t catching up")
        XCTAssertEqual(stalledReader.action?.kind, .copyReport)
        XCTAssertEqual(stalledReader.detail, "117\u{00A0}MB left to read. " + BrainBarMenuPresenter.Say.agent.text, "who acts, as the menu says")
        XCTAssertTrue(s.pipeline.faults.contains(.read))
        // The Read station's own line: the menu's words, never "not being read" for a reader that reads.
        XCTAssertEqual(r1.pipeline.status[.read], .init(text: "catching up", severity: .orange))
        XCTAssertEqual(s.pipeline.status[.read], .init(text: "117\u{00A0}MB behind", severity: .red))
    }
}
