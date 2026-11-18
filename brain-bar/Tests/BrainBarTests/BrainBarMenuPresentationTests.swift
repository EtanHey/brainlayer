import CryptoKit
import Foundation
import XCTest
@testable import BrainBar

/// The table test against the lead's `presentation-cases.json` (phase 8). Each case's capture is decoded from its
/// own `inputs`, literally, and every case's whole presentation is compared: rows, marks, states, actions, dividers,
/// spinners and progress included (P1 owed the six calm cases; P2 adds the eleven needs cases).
final class BrainBarMenuPresentationTests: XCTestCase {
    typealias P = BrainBarMenuPresentation

    /// The plan's `phase-8-native-build/presentation-cases.json`, byte for byte.
    static let casesSHA256 = BrainBarMenuCases.sha256
    static let calm = ["S1", "S2-today", "S2-after-seam", "S4", "S7", "S8"]

    func testCasesFileIsTheLeadsUnchangedCopy() throws {
        let digest = SHA256.hash(data: try Data(contentsOf: Self.casesURL()))
        XCTAssertEqual(digest.map { String(format: "%02x", $0) }.joined(), Self.casesSHA256)
    }

    func testEveryCaseMatchesWhole() throws {
        let cases = try Self.loadCases()
        XCTAssertEqual(cases.count, 17)
        XCTAssertEqual(cases.filter { $0["form"] as? String == "calm" }.compactMap { $0["id"] as? String }, Self.calm)
        for c in cases {
            let id = c["id"] as! String
            XCTAssertEqual(BrainBarMenuPresenter.present(try Self.capture(c["inputs"] as! [String: Any])), Self.expected(c), id)
        }
    }

    /// S4's inputs say its checks may hold any last-known values, because X02 masks them: a green verdict
    /// captured before the sleep gives the same menu as none.
    func testWakeMasksLastKnownGreenChecks() throws {
        let s4 = try Self.loadCases().first { $0["id"] as? String == "S4" }!
        var capture = try Self.capture(s4["inputs"] as! [String: Any])
        capture.verdict = BrainBarHealthVerdict(dot: .green, checks: BrainBarMenuPresenter.known.map {
            .init(name: $0, state: .green, note: "", unknownTicks: 0, blind: false)
        }, notes: [], evaluatedAt: capture.now.addingTimeInterval(-900))
        XCTAssertEqual(BrainBarMenuPresenter.present(capture), Self.expected(s4))
    }

    /// Round 1 of #1787, each from a case's own capture: S07 only with no copy on record (R21), a dead upload not worded
    /// (N2), D13's line directly under the Drive row (R22), and C14 proper (S19 over S06 with an aged local copy).
    func testRoundOneVariantsOfTheCases() throws {
        let cases = try Self.loadCases()
        func capture(_ id: String) throws -> BrainBarMenuCapture { try Self.capture(cases.first { $0["id"] as? String == id }!["inputs"] as! [String: Any]) }
        func with(_ v: BrainBarHealthVerdict, _ name: String, _ state: BrainBarHealthVerdict.State, age: Double) -> BrainBarHealthVerdict {
            .init(dot: v.dot, checks: v.checks.map { $0.name == name ? .init(name: name, state: state, ageS: age, note: "") : $0 }, notes: v.notes,
                  evaluatedAt: v.evaluatedAt)
        }
        func titles(_ c: BrainBarMenuCapture) -> [String] {
            BrainBarMenuPresenter.present(c).rows.filter { [.needsYou, .factLine].contains($0.kind) || $0.state == .notice }.compactMap(\.strings.first)
        }
        // R1.11: the rows these variants used to word generically now name their check.
        var s11b = try capture("S11b")
        s11b.verdict = with(s11b.verdict!, "backup_offsite", .green, age: 3600)
        XCTAssertEqual(titles(s11b).count, 1)
        XCTAssertFalse(titles(s11b).contains("A check failed, and BrainBar can’t tell which"), "\(titles(s11b))")
        var s5 = try capture("S5")
        s5.upload = .init(startedAt: s5.now.addingTimeInterval(-3420), live: false)
        XCTAssertEqual(titles(s5).count, 1)
        XCTAssertFalse(titles(s5).contains("A check failed, and BrainBar can’t tell which"), "\(titles(s5))")
        var c32 = try capture("C32")
        c32.maintenanceNightFailures = 3
        XCTAssertEqual(titles(c32), ["Google Drive is signed out", "Conversation backups can’t upload either until Drive is reconnected.",
                                     "Nightly check-up failed 3 nights"])
        var c14 = try capture("S11")
        c14.verdict = with(c14.verdict!, "backup_local", .red, age: 200_000)
        c14.backupDailyFailures = 2
        XCTAssertEqual(BrainBarMenuPresenter.present(c14), Self.expected(cases.first { $0["id"] as? String == "S11" }!))
        // CodeRabbit round 2: a copy's age is counted to `now`, not to `evaluatedAt` (the verdict stays awake-fresh in sleep).
        func driveDetail(_ c: BrainBarMenuCapture) -> String? {
            BrainBarMenuPresenter.present(c).rows.first { $0.strings.first == "Google Drive is signed out" }?.strings.last
        }
        var s3 = try capture("S3")
        s3.now = s3.now.addingTimeInterval(86_400)
        s3.backupDailyNextRunAt = s3.backupDailyNextRunAt?.addingTimeInterval(86_400)
        XCTAssertEqual(driveDetail(s3), "Backups can’t upload until you reconnect it. Your newest backup in Google Drive is from Nov 9 at 03:44.")
        var hours = try capture("S3")
        hours.verdict = with(hours.verdict!, "backup_offsite", .green, age: 3600)
        hours.now = hours.now.addingTimeInterval(7200)
        XCTAssertEqual(driveDetail(hours), "Backups can’t upload until you reconnect it. Your newest backup in Google Drive is 3 hours old.")
    }

    /// P2b, ruling R23: a bare `{HH:mm}` reads as a time today, so an upload that started on another day says its day with the
    /// words every other past time uses, in S03's line (calm) and inside S32's sentence. Same-day words do not change.
    func testAnUploadStartOnAnotherDaySaysTheDay() throws {
        let cases = try Self.loadCases()
        func capture(_ id: String) throws -> BrainBarMenuCapture { try Self.capture(cases.first { $0["id"] as? String == id }!["inputs"] as! [String: Any]) }
        func detail(_ c: BrainBarMenuCapture) -> [String]? { BrainBarMenuPresenter.present(c).rows.first { $0.kind == .headlineDetail }?.strings }
        func title(_ c: BrainBarMenuCapture) -> String? { BrainBarMenuPresenter.present(c).rows.first { $0.kind == .needsYou }?.strings.first }
        var s2 = try capture("S2-today")
        XCTAssertEqual(detail(s2), ["To Google Drive. Started at 11:28."])
        s2.upload?.startedAt = s2.now.addingTimeInterval(-44_400)  // 23:40 the night before (now is 12:00)
        XCTAssertEqual(detail(s2), ["To Google Drive. Started yesterday at 23:40."])
        let drive = "Your newest backup in Google Drive is from Nov 9 at 03:44. "
        var s32 = try capture("S32")
        s32.upload?.startedAt = s32.now.addingTimeInterval(-44_400)
        XCTAssertEqual(title(s32), drive + "A backup to Google Drive started yesterday at 23:40 and is still running.")
        s32.upload?.startedAt = s32.now.addingTimeInterval(-174_720)  // two days before, at 11:28
        XCTAssertEqual(title(s32), drive + "A backup to Google Drive started Nov 10 at 11:28 and is still running.")
    }

    // MARK: expected output, read from the cases file

    /// The cases file's expectation with each later ruling applied: R31 (below), then R1.11.
    static func expected(_ c: [String: Any]) -> P { r111(expectedR31(c)) }

    /// R1.11 (L52, L53) supersedes the frozen file's words and marks, every other byte still checked: the one who-acts
    /// sentence replaces the two hand-off lines, C5's replaces D10's, the button says "Copy a report for your agent", three
    /// titles take the window's names, and a needs headline carries no mark (one mark per problem).
    static func r111(_ p: P) -> P {
        var p = p
        // R1.12 (I 6): "in a row" goes from the nightly fault's sentence.
        let swaps = [("Google Drive is disconnected", "Google Drive is signed out"), ("Nightly upkeep failed", "Nightly check-up failed"),
                     (" nights in a row", " nights"),
                     ("BrainBar hasn’t seen the memories waiting for it go down", "Its queue hasn’t shrunk")]
        let pairs = [(["BrainBar can’t fix this itself.", "Give the report to an agent."], "An agent needs to look at this."),
                     (["If this backup doesn’t finish, BrainBar can’t fix it itself.", "Give the report to an agent."],
                      "If it doesn’t finish, an agent needs to look at it.")]
        p.rows = p.rows.map { row in
            var row = row
            row.strings = row.strings.map { s in swaps.reduce(s) { $0.replacingOccurrences(of: $1.0, with: $1.1) } }
            for (old, new) in pairs {
                if let i = row.strings.firstIndex(of: old[0]), row.strings.count > i + 1, row.strings[i + 1] == old[1] {
                    row.strings.replaceSubrange(i...(i + 1), with: [new])
                }
            }
            if row.action?.kind == .copyReport { row.action?.label = "Copy a report for your agent" }
            if row.kind == .headline, p.form == .needs { row.mark = nil }
            return row
        }
        return p
    }

    static func expectedR31(_ c: [String: Any]) -> P {
        let rows = (c["rows"] as! [[String: Any]]).map { r -> P.Row in
            var row = P.Row(kind: P.RowKind(rawValue: r["kind"] as! String)!)
            row.strings = (r["strings"] as? [[String: Any]] ?? []).map { $0["text"] as! String }
            if let mark = r["mark"] as? [String: String] {
                row.mark = mark["A1_disc"] != nil ? .marked : P.Mark(rawValue: String(mark["A2_sign"]!.prefix { $0 != " " }))!
            } else if let mark = r["mark"] as? String {
                row.mark = P.Mark(rawValue: String(mark.prefix { $0 != " " }))!
            }
            row.state = (r["state"] as? String).map { P.RowState(rawValue: $0)! }
            row.progressPercent = (r["progress_bar"] as? [String: Any])?["value"] as? Int
            row.spinner = r["spinner"] as? Bool ?? false
            row.dividerBefore = r["divider_before"] as? Bool ?? false
            if let action = r["action"] as? [String: Any] {
                row.action = P.Action(kind: P.ActionKind(rawValue: action["kind"] as! String)!,
                                      label: (action["strings"] as! [[String: Any]])[0]["text"] as! String)
            }
            row.keyEquivalent = r["key_equivalent"] as? String
            row.optionAlternate = ((r["option_alternate"] as? [String: Any])?["strings"] as? [[String: Any]])?.first?["text"] as? String
            return row
        }
        // R31 supersedes the frozen fixture's amber Needs-you rows; every other byte stays checked.
        let id = c["id"] as! String
        if ["S5", "S9", "S12"].contains(id) {
            var revised = rows
            let amberTitle = id == "S5" ? "The last 2 backups didn’t finish" : "Search by meaning isn’t catching up"
            let notices = revised.filter { $0.strings.first == amberTitle }.map { row -> P.Row in
                P.Row(kind: .statusLine, strings: Array(row.strings.dropLast(2)), state: .notice)
            }
            revised.removeAll { $0.strings.first == amberTitle }
            if id == "S12" {
                let index = revised.firstIndex { $0.kind == .everythingElse }!
                revised.insert(contentsOf: notices, at: index)
                revised[0].strings = ["2 things need you"]
                if let first = revised.firstIndex(where: { $0.kind == .needsYou }) { revised[first].dividerBefore = false }
                return .init(form: .needs, headline: "2 things need you", count: 2, icon: .marked, rows: revised)
            }
            let fine = id == "S5" ? "Saving, search and upkeep are fine." : "Saving, backups and upkeep are fine."
            return .init(form: .calm, headline: "Check soon", count: 0, icon: .resting, rows:
                [P.Row(kind: .headline, strings: ["Check soon"], mark: .notice), P.Row(kind: .separator)] + notices
                + [P.Row(kind: .everythingElse, strings: [fine], mark: .ok)] + Array(rows.suffix(3)))
        }
        return .init(form: P.Form(rawValue: c["form"] as! String)!, headline: c["headline"] as! String,
                     count: c["count"] as! Int, icon: P.Icon(rawValue: c["icon"] as! String)!, rows: rows)
    }

    static func casesURL() throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: "presentation-cases", withExtension: "json", subdirectory: "Fixtures"))
    }

    static func loadCases() throws -> [[String: Any]] {
        let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: casesURL())) as! [String: Any]
        return doc["cases"] as! [[String: Any]]
    }

    // MARK: the capture, decoded from a case's `inputs`: one helper, shared with the render harness

    static func date(_ value: Any?) -> Date? { BrainBarMenuCases.date(value) }

    static func capture(_ inputs: [String: Any]) throws -> BrainBarMenuCapture { try BrainBarMenuCases.capture(inputs) }
}
