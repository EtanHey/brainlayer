import Foundation
import XCTest
@testable import BrainBar

/// Revision 1.11 (rulings L52, L53; CHANGES-R1.11): the menu names the thing, says who acts, and marks a problem once; the
/// lag's severity is one named switch. The new design's dropdown and the classic menu read this one presenter.
final class BrainBarNewDesignR111Tests: XCTestCase {
    typealias P = BrainBarMenuPresentation
    typealias Say = BrainBarMenuPresenter.Say
    typealias Check = BrainBarHealthVerdict.Check
    typealias State = BrainBarHealthVerdict.State

    static let now = ISO8601DateFormatter().date(from: "2026-11-18T12:20:00+02:00")!
    /// The words 1.11 removes from every menu path.
    static let removed = ["Something BrainBar checks needs attention", "can’t fix this itself", "can’t fix it itself", "can't fix",
                          "Give the report", "Copy report for an agent", "Nightly upkeep", "is disconnected", "go down"]
    static let mb117 = 117 * 1_048_576

    /// Every known check green except `states`, the streaks read, then `edit`.
    static func capture(_ states: [String: State] = [:], _ edit: (inout BrainBarMenuCapture) -> Void = { _ in }) -> BrainBarMenuCapture {
        var c = BrainBarMenuCapture(now: now, timeZone: TimeZone(secondsFromGMT: 7200)!)
        let checks = BrainBarMenuPresenter.known.map { Check(name: $0, state: states[$0] ?? .green, note: "", unknownTicks: 0, blind: false) }
        let dot: State = states.values.contains(.red) ? .red : states.values.contains(.amber) ? .amber : .green
        c.verdict = BrainBarHealthVerdict(dot: dot, checks: checks, notes: [], evaluatedAt: now)
        c.maintenanceNightFailures = 0
        c.backupDailyFailures = 0
        c.jsonlBackupFailures = 0
        edit(&c)
        return c
    }

    /// One check of the verdict replaced (or added).
    static func setting(_ c: inout BrainBarMenuCapture, _ check: Check) {
        let v = c.verdict!
        c.verdict = BrainBarHealthVerdict(dot: v.dot, checks: v.checks.filter { $0.name != check.name } + [check], notes: v.notes, evaluatedAt: v.evaluatedAt)
    }

    static func lag(_ ingest: State = .red, bytes: Int? = mb117, clock: Double? = nil, reasons: [String] = ["offset_lag"],
                    severity: BrainBarMenuPresenter.LagSeverity = .needsYouPastLimit) -> BrainBarMenuCapture {
        capture(["ingest": ingest]) {
            $0.watcher = .init(alertReasons: reasons, lagBytes: bytes, lastReadAt: now.addingTimeInterval(-30))
            $0.lagEpisodeStart = clock.map { now.addingTimeInterval(-$0) }
            $0.lagSeverity = severity
        }
    }

    /// A designer state: its capture, and the row it must draw (title, line, button, who acts, needs or notice).
    struct Row { var id: String, c: BrainBarMenuCapture, title: String, line: String?, action: P.ActionKind?, agent: Bool, needs: Bool }

    /// The states of CHANGES-R1.11 § 1 this capture type can hold (A2 and A3 need BrainLayer's estimate, not yet in the
    /// capture, so A1's words stand for them; C5, C6 and C9 are the cases file's S32, S06/S07/S19 and S13).
    static func rows() -> [Row] {
        let nb = "\u{00A0}"
        func with(_ states: [String: State], _ check: Check? = nil, _ edit: (inout BrainBarMenuCapture) -> Void = { _ in }) -> BrainBarMenuCapture {
            capture(states) { c in
                if let check { setting(&c, check) }
                edit(&c)
            }
        }
        return [
            Row(id: "A1 installed: bytes, no clock", c: lag(), title: Say.lagBehind.text,
                line: "117\(nb)MB left to read. It’s catching up by itself.", action: nil, agent: false, needs: true),
            Row(id: "A1 past its clock", c: lag(.amber, clock: 2400), title: Say.lagBehind.text,
                line: "117\(nb)MB left to read, behind for 40\(nb)min. It’s catching up by itself.", action: nil, agent: false, needs: true),
            Row(id: "A1 no bytes", c: lag(.amber, bytes: nil, clock: 2400), title: Say.lagBehind.text,
                line: "Behind for 40\(nb)min. It’s catching up by itself.", action: nil, agent: false, needs: true),
            Row(id: "A1 beside another alert", c: lag(reasons: ["offset_lag", "file_ingestion_failure"]), title: Say.lagBehind.text,
                line: "117\(nb)MB left to read.", action: nil, agent: false, needs: true),
            Row(id: "A4", c: with(["ingest": .red]) { $0.watcher = .init(fresh: false, lastReadAt: now.addingTimeInterval(-3000)) },
                title: Say.stopped.text, line: "It last read today at 11:30.", action: .copyReport, agent: true, needs: true),
            Row(id: "A5 red", c: with(["ingest": .red]) { $0.watcher = .init(alertReasons: ["coverage_drop"]) },
                title: Say.coverage.text, line: Say.coverageLine.text, action: .copyReport, agent: true, needs: true),
            Row(id: "A5 amber", c: with(["ingest": .amber]) { $0.watcher = .init(alertReasons: ["coverage_drop"]) },
                title: Say.coverage.text, line: Say.coverageLine.text, action: nil, agent: false, needs: false),
            Row(id: "A6", c: with(["ingest": .red]) { $0.watcher = .init(alertReasons: ["file_ingestion_failure"]) },
                title: Say.unreadFiles.text, line: Say.unreadFilesLine.text, action: .copyReport, agent: true, needs: true),
            Row(id: "A7", c: with([:], Check(name: "ingest", state: .red, note: "", contextCodes: ["growth_not_indexed:cursor"])),
                title: "Cursor’s sessions aren’t being read", line: Say.providerUnreadLine.text, action: .copyReport, agent: true, needs: true),
            Row(id: "B1", c: with([:], Check(name: "stores", state: .red, ageS: 7200, note: "")),
                title: Say.savesStuck.text, line: "The oldest has waited 2\(nb)h and isn’t in memory yet.", action: .copyReport, agent: true, needs: true),
            Row(id: "B2", c: with([:], Check(name: "stores", state: .amber, ageS: 720, note: "")),
                title: Say.savesSlow.text, line: "The oldest has waited 12\(nb)min.", action: nil, agent: false, needs: false),
            Row(id: "B3", c: with([:], Check(name: "search", state: .red, note: "", fails: 3)),
                title: Say.searchDown.text, line: "Search has failed its last 3 checks.", action: .copyReport, agent: true, needs: true),
            Row(id: "B4", c: with(["search": .amber]), title: Say.searchCheck.text, line: Say.searchOneOff.text, action: nil, agent: false, needs: false),
            Row(id: "B5", c: with([:], Check(name: "embedding", state: .red, ageS: 7300, thresholdS: 21_600, note: "")),
                title: Say.f07.text, line: "Its queue hasn’t shrunk in over 2 hours.", action: .copyReport, agent: true, needs: true),
            Row(id: "C1", c: with([:], Check(name: "backup_local", state: .red, ageS: 31 * 3600, note: "")),
                title: Say.localLate.text, line: "The newest is from Nov 17 at 05:20.", action: .copyReport, agent: true, needs: true),
            Row(id: "C2", c: with([:], Check(name: "backup_offsite", state: .red, ageS: 20 * 3600, note: "")),
                title: Say.driveLate.text, line: "The newest is 20 hours old.", action: .copyReport, agent: true, needs: true),
            Row(id: "C3", c: with([:], Check(name: "transcripts", state: .red, ageS: 20 * 3600, note: "")),
                title: Say.conversationsLate.text, line: "The newest is 20 hours old.", action: .copyReport, agent: true, needs: true),
            Row(id: "C4", c: with([:], Check(name: "backup_local", state: .amber, ageS: 22 * 3600, note: "")),
                title: Say.localRunningLate.text, line: "The newest is 22 hours old.", action: nil, agent: false, needs: false),
            Row(id: "C7", c: with(["disk": .red]), title: Say.noRoom.text, line: nil, action: .openStorageSettings, agent: false, needs: true),
            Row(id: "C8", c: with(["disk": .amber]), title: Say.roomTight.text, line: nil, action: nil, agent: false, needs: false),
            Row(id: "D1", c: with([:]) { $0.maintenanceNightFailures = 3 }, title: "Nightly check-up failed 3 nights",
                line: Say.stillSaved.text, action: .copyReport, agent: true, needs: true),
            Row(id: "D2", c: with(["monitor": .red]), title: Say.checksStopped.text,
                line: "The last one ran today at 12:20, so nothing since then is confirmed.", action: .copyReport, agent: true, needs: true),
            Row(id: "D3", c: with([:], Check(name: "new_check", state: .red, note: "")), title: Say.unknownCheck.text,
                line: "BrainLayer is newer than BrainBar. Updating BrainBar will name it.", action: .copyReport, agent: false, needs: true),
            Row(id: "D4", c: with([:], Check(name: "stores", state: .red, ageS: 7200, note: "")) { $0.verdictAwakeAgeS = 901 },
                title: Say.savesStuck.text, line: "The oldest has waited 2\(nb)h and isn’t in memory yet.", action: .copyReport, agent: true, needs: true),
            Row(id: "D5", c: with([:]) { c in
                    let v = c.verdict!
                    c.verdict = BrainBarHealthVerdict(dot: .red, checks: v.checks, notes: [], evaluatedAt: v.evaluatedAt)
                }, title: Say.d5.text, line: Say.d5Line.text, action: .copyReport, agent: false, needs: true),
        ]
    }

    static func words(_ p: P) -> [String] { [p.headline] + p.rows.flatMap(\.strings) + p.rows.compactMap(\.action?.label) }

    // MARK: no generic, "can't fix" or "give the report" on any menu path

    func testNoMenuPathShowsTheRemovedWords() throws {
        var captures = try BrainBarMenuPresentationTests.loadCases().map { try BrainBarMenuPresentationTests.capture($0["inputs"] as! [String: Any]) }
        captures += Self.rows().map(\.c)
        captures += BrainBarDesignFixture.Name.allCases.map { BrainBarDesignFixture.inputs($0).capture }
        // Every single check, red and amber, on a fresh and a stale verdict, with and without each watcher alert.
        for name in BrainBarMenuPresenter.known + ["new_check"] {
            for state in [State.red, .amber] {
                for reasons in [[], ["offset_lag"], ["coverage_drop"], ["file_ingestion_failure"], ["growth_not_indexed:codex"], ["unknown_alert"]] {
                    for awake in [0.0, 901] {
                        captures.append(Self.capture([:]) {
                            Self.setting(&$0, Check(name: name, state: state, note: ""))
                            $0.watcher = .init(alertReasons: reasons)
                            $0.verdictAwakeAgeS = awake
                        })
                    }
                }
            }
        }
        XCTAssertGreaterThan(captures.count, 300)
        for c in captures {
            let p = BrainBarMenuPresenter.present(c)
            for word in Self.removed { XCTAssertFalse(Self.words(p).contains { $0.contains(word) }, "\(word) in \(Self.words(p))") }
        }
        // And in the new design's surfaces built from them.
        for name in BrainBarDesignFixture.Name.allCases {
            let i = BrainBarDesignFixture.inputs(name)
            let m = NDModel.make(i), r1 = NDR1.make(i, m)
            let said = m.faults.flatMap { [$0.title] + $0.facts + [$0.action?.label ?? ""] } + r1.items.flatMap { [$0.title, $0.detail ?? ""] }
            for word in Self.removed { XCTAssertFalse(said.contains { $0.contains(word) }, "\(name): \(word)") }
        }
        // None of the removed sentences is registered: the closed world (BrainBarMenuInvariantTests) refuses them on every path.
        for word in Self.removed { XCTAssertFalse(Say.all.contains { $0.template.contains(word) }, word) }
    }

    // MARK: each state's row, under its condition, and the oracle agrees

    func testEachStateNamesTheThingAndWhoActs() {
        for row in Self.rows() {
            let p = BrainBarMenuPresenter.present(row.c)
            XCTAssertEqual(BrainBarMenuInvariantTests.violations(row.c, p), [], row.id)
            let drawn = p.rows.first { ($0.kind == .needsYou || $0.state == .notice) && $0.strings.first == row.title }
            guard let drawn else { XCTFail("\(row.id): no row titled \(row.title) in \(p.rows.map(\.strings))"); continue }
            var expected = [row.title] + (row.line.map { [$0] } ?? [])
            if row.id == "D4" { expected.append("(as of today at 12:20)") }
            if row.agent { expected.append(Say.agent.text) }
            XCTAssertEqual(drawn.strings, expected, row.id)
            XCTAssertEqual(drawn.action?.kind, row.action, row.id)
            XCTAssertEqual(drawn.kind == .needsYou, row.needs, row.id)
            XCTAssertEqual(p.form == .needs, row.needs, row.id)
            if row.action == .copyReport { XCTAssertEqual(drawn.action?.label, "Copy a report for your agent", row.id) }
        }
    }

    /// Each new sentence only under its condition, both ways: the bytes-behind part of the lag line with and without the
    /// fact, the clock part with and without the episode start, "catching up by itself" only while the lag is the only alert.
    func testTheLagLineSaysOnlyWhatIsKnown() {
        func line(_ c: BrainBarMenuCapture) -> [String]? { BrainBarMenuPresenter.present(c).rows.first { $0.strings.first == Say.lagBehind.text }?.strings }
        XCTAssertEqual(line(Self.lag())?.last, "117\u{00A0}MB left to read. It’s catching up by itself.")
        XCTAssertEqual(line(Self.lag(bytes: nil)), [Say.lagBehind.text, Say.byItself.text], "no bytes, no size; no clock, no span")
        XCTAssertEqual(line(Self.lag(bytes: 500_000))?.last, Say.byItself.text, "under 1 MB the size is left out")
        XCTAssertEqual(line(Self.lag(bytes: 3 * 1_073_741_824 / 2))?.last, "1.5\u{00A0}GB left to read. It’s catching up by itself.")
        XCTAssertEqual(line(Self.lag(reasons: ["offset_lag", "coverage_drop"]))?.last, "117\u{00A0}MB left to read.")
        XCTAssertEqual(line(Self.lag(reasons: ["offset_lag", "quarantined_record"]))?.last, "117\u{00A0}MB left to read. It’s catching up by itself.")
        // No row at all without its condition: a green reader under its limit is R02's, and a stale file is not a lag.
        XCTAssertNil(line(Self.lag(.green, clock: 600)))
        var stale = Self.lag()
        stale.watcher?.fresh = false
        XCTAssertNil(line(stale))
        XCTAssertEqual(BrainBarMenuPresenter.present(stale).rows.first { $0.kind == .needsYou }?.strings.first, Say.stopped.text)
        // The registry's condition holds only for the capture's own fact (the falsification and differential tests do this
        // for every capture they enumerate; here on the lag's own captures). A malformed size is no sentence at all.
        func holds(_ sentence: String, _ c: BrainBarMenuCapture) -> Bool? {
            let p = BrainBarMenuPresenter.present(c)
            return BrainBarMenuPresenter.entry(for: sentence, c).map { $0.0.holds?(c, p, $0.1) ?? true }
        }
        let c = Self.lag(clock: 2400)
        XCTAssertEqual(holds("117\u{00A0}MB left to read, behind for 40\u{00A0}min.", c), true)
        XCTAssertEqual(holds("118\u{00A0}MB left to read, behind for 40\u{00A0}min.", c), false)
        XCTAssertEqual(holds("117\u{00A0}MB left to read, behind for 41\u{00A0}min.", c), false)
        XCTAssertEqual(holds("117\u{00A0}MB left to read.", Self.lag(bytes: nil)), false)
        XCTAssertNil(holds("117.0\u{00A0}MB left to read.", c), "only the formatter's own words match")
    }

    /// The live adapter carries the watcher's bytes behind and last read into the capture; a zero is no fact.
    func testBytesBehindReachTheMenuFromTheWatcherFile() {
        func adapt(_ bytes: Int) -> BrainBarMenuCapture {
            BrainBarMenuLiveCapture.capture(status: .checking, verdictAwakeAgeS: nil, wakeCatchUpPending: false, driveAuth: nil,
                backups: .notRead, backupsNotice: nil,
                watcher: .readable(.init(updatedAt: Self.now.addingTimeInterval(-40), pollCount: 9, alertReasons: ["offset_lag"], maxOffsetLagBytes: bytes)),
                watcherAwakeAgeS: 40, observations: nil, now: Self.now, timeZone: TimeZone(secondsFromGMT: 7200)!)
        }
        XCTAssertEqual(adapt(Self.mb117).watcher?.lagBytes, Self.mb117)
        XCTAssertEqual(adapt(Self.mb117).watcher?.lastReadAt, Self.now.addingTimeInterval(-40))
        XCTAssertNil(adapt(0).watcher?.lagBytes)
    }

    // MARK: one mark per problem (L53)

    func testOneMarkPerProblem() {
        func problemMarks(_ p: P) -> [P.Mark] { p.rows.compactMap(\.mark).filter { [.marked, .alertCircle, .notice].contains($0) } }
        // A needs headline carries no mark when its rows do: one problem, and two.
        let one = BrainBarMenuPresenter.present(Self.lag())
        XCTAssertEqual(one.headline, "1 thing needs you")
        XCTAssertNil(one.rows.first { $0.kind == .headline }?.mark)
        XCTAssertEqual(problemMarks(one), [.alertCircle])
        let two = BrainBarMenuPresenter.present(Self.capture(["disk": .red]) { $0.maintenanceNightFailures = 3 })
        XCTAssertEqual(two.headline, "2 things need you")
        XCTAssertNil(two.rows.first { $0.kind == .headline }?.mark)
        XCTAssertEqual(problemMarks(two), [.alertCircle, .alertCircle])
        // A lone notice has exactly one: the amber mark on its headline.
        for c in [Self.capture(["disk": .amber]), Self.lag(severity: .noticeWhileDraining)] {
            let p = BrainBarMenuPresenter.present(c)
            XCTAssertEqual(problemMarks(p), [.notice], "\(p.rows.map(\.strings))")
            XCTAssertEqual(p.rows.first { $0.kind == .headline }?.mark, .notice)
        }
        // The window follows the same rule: the Overview and dropdown headlines carry no mark over their problem rows.
        for name in [BrainBarDesignFixture.Name.twoFaults, .driveSignedOut, .lagBehind, .searchFailing] {
            let i = BrainBarDesignFixture.inputs(name)
            let m = NDModel.make(i), r1 = NDR1.make(i, m)
            XCTAssertGreaterThan(r1.redCount, 0, "\(name)")
            XCTAssertNil(r1.headlineSymbol, "\(name)")
            XCTAssertEqual(m.headlineMark, .none, "\(name)")
            XCTAssertNil(m.shownPresentation.rows.first { $0.kind == .headline }?.mark, "\(name)")
        }
    }

    // MARK: the lag's severity, one named switch

    func testTheSeveritySwitchDefaultsToTodaysRule() {
        XCTAssertEqual(BrainBarMenuPresenter.lagSeverity, .needsYouPastLimit, "R20/R31 until he answers")
        XCTAssertEqual(BrainBarMenuCapture(now: Self.now, timeZone: .current).lagSeverity, BrainBarMenuPresenter.lagSeverity)
    }

    /// Today's side (R20, R31): past its limit the lag needs him, red, true words, no button, and it notifies.
    func testSeverityNeedsYouSide() {
        for c in [Self.lag(), Self.lag(.amber, clock: 2400)] {
            let p = BrainBarMenuPresenter.present(c)
            XCTAssertEqual(p.form, .needs)
            XCTAssertEqual(p.count, 1)
            XCTAssertEqual(p.icon, .marked)
            let row = p.rows.first { $0.kind == .needsYou }
            XCTAssertEqual(row?.strings.first, Say.lagBehind.text)
            XCTAssertNil(row?.action)
            XCTAssertFalse(BrainBarMenuPresenter.notificationFaults(c).isEmpty)
            XCTAssertEqual(BrainBarMenuInvariantTests.violations(c, p), [])
        }
    }

    /// The designer's side: while it drains, a notice whose headline names it, no count, no button, no notification; it
    /// needs him again once it is not the only alert, or the reader stops.
    func testSeverityNoticeSide() {
        for c in [Self.lag(severity: .noticeWhileDraining), Self.lag(.amber, clock: 2400, severity: .noticeWhileDraining)] {
            let p = BrainBarMenuPresenter.present(c)
            XCTAssertEqual(p.form, .calm)
            XCTAssertEqual(p.count, 0)
            XCTAssertEqual(p.icon, .resting)
            XCTAssertEqual(p.headline, Say.lagBehind.text)
            XCTAssertEqual(p.rows.first { $0.kind == .headlineDetail }?.strings, [BrainBarMenuPresenter.lagDetail(c)!])
            XCTAssertFalse(p.rows.contains { $0.kind == .needsYou || $0.action != nil })
            XCTAssertTrue(BrainBarMenuPresenter.notificationFaults(c).isEmpty)
            XCTAssertEqual(BrainBarMenuInvariantTests.violations(c, p), [])
        }
        var stopped = Self.lag(severity: .noticeWhileDraining)
        stopped.watcher?.fresh = false
        for c in [Self.lag(reasons: ["offset_lag", "file_ingestion_failure"], severity: .noticeWhileDraining), stopped] {
            let p = BrainBarMenuPresenter.present(c)
            XCTAssertEqual(p.form, .needs, "\(p.rows.map(\.strings))")
            XCTAssertEqual(BrainBarMenuInvariantTests.violations(c, p), [])
        }
        // The window takes the same decision: the reader's item is amber on this side and red on today's.
        var i = BrainBarDesignFixture.inputs(.lagBehind)
        XCTAssertEqual(NDR1.make(i, NDModel.make(i)).items.first { $0.id == "reader" }?.severity, .red)
        i.capture.lagSeverity = .noticeWhileDraining
        XCTAssertEqual(NDR1.make(i, NDModel.make(i)).items.first { $0.id == "reader" }?.severity, .orange)
    }

    // MARK: the classic menu reads the same rows

    func testTheClassicMenuDrawsOneWhoActsLineAndNoHeadlineMark() {
        let p = BrainBarMenuPresenter.present(Self.capture([:], { Self.setting(&$0, Check(name: "search", state: .red, note: "", fails: 3)) }))
        let row = p.rows.first { $0.kind == .needsYou }
        XCTAssertEqual(row?.strings, [Say.searchDown.text, "Search has failed its last 3 checks.", Say.agent.text])
        XCTAssertEqual(row?.action?.label, "Copy a report for your agent")
        XCTAssertNil(p.rows.first { $0.kind == .headline }?.mark)
    }
}
