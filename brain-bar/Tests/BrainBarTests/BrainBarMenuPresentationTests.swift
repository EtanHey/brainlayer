import CryptoKit
import Foundation
import XCTest
@testable import BrainBar

/// The table test: every case of the lead's `presentation-cases.json` (phase 8), built from its
/// `inputs`, must come out of `BrainBarMenuPresenter.present` exactly as the case says.
final class BrainBarMenuPresentationTests: XCTestCase {
    typealias P = BrainBarMenuPresentation
    typealias Check = BrainBarHealthVerdict.Check

    /// The plan's `phase-8-native-build/presentation-cases.json`, byte for byte, after the lead's S5 correction
    /// (`ageS` 165600 to 166380, review round 1). A drift means the copy is no longer the lead's file.
    static let casesSHA256 = "65e4e76375265706bd23e73e6869d5c387d2d0f221ccfb8218bab7d1286882a6"

    func testCasesFileIsTheLeadsUnchangedCopy() throws {
        let digest = SHA256.hash(data: try Data(contentsOf: Self.casesURL()))
        XCTAssertEqual(digest.map { String(format: "%02x", $0) }.joined(), Self.casesSHA256)
    }

    func testEveryCaseMatchesThePresentationTable() throws {
        let (cases, actions) = try Self.loadCases()
        XCTAssertEqual(cases.count, 17)
        for c in cases {
            let id = c["id"] as! String
            guard let capture = Self.captures[id] else { XCTFail("no capture for \(id)"); continue }
            XCTAssertEqual(BrainBarMenuPresenter.present(capture), Self.expected(c, actions: actions), id)
        }
    }

    /// The ruled variants each case names beside its main strings.
    func testVariantsTheCasesName() throws {
        let (cases, _) = try Self.loadCases()
        func row(_ id: String, _ kind: String) -> [String: Any] {
            let c = cases.first { $0["id"] as? String == id }!
            return (c["rows"] as! [[String: Any]]).first { $0["kind"] as? String == kind }!
        }
        var etaNull = Self.captures["S2-after-seam"]!
        etaNull.offsiteRecovering?.eta = nil
        func strings(_ c: BrainBarMenuCapture, _ kind: P.RowKind) -> [String] {
            BrainBarMenuPresenter.present(c).rows.first { $0.kind == kind }?.strings ?? []
        }
        XCTAssertEqual(strings(etaNull, .headlineDetail),
                       [(row("S2-after-seam", "headline_detail")["variant_eta_null"] as! [String: Any])["text"] as! String])
        var progressing = Self.captures["S32"]!
        progressing.offsiteRecovering = .init(progress: 0.4, eta: nil, startedAt: Self.date("2026-11-12T11:28:00+02:00"),
                                              updatedAt: Self.date("2026-11-12T12:00:00+02:00"), cadenceS: 30)
        XCTAssertEqual(strings(progressing, .needsYou).first,
                       (row("S32", "needs_you")["variant_after_PR2"] as! [String: Any])["text"] as? String)
        var noDBRow = Self.captures["S13-missing"]!
        noDBRow.verdict = Self.verdict("2026-11-12T12:48:00+02:00", [Self.check("drive_auth", .red)])
        let detail = ((row("S13-missing", "needs_you")["variant_without_db_row"] as! [String: Any])["detail"] as! [String: Any])
        XCTAssertEqual(strings(noDBRow, .needsYou).last, detail["text"] as? String)
    }

    /// Not in the table: escalations on a clock, and a check no row derives yet, must still count.
    func testEscalationsAndUnderivedChecksNeverUndercount() {
        var search = Self.captures["S7"]!
        search.now = Self.date("2026-11-12T13:11:00+02:00")
        search.verdictAwakeAgeS = 0
        XCTAssertEqual(BrainBarMenuPresenter.present(search).count, 2, "F02 after 15 min, R02 lag after 30 min")
        let stores = Self.capture("2026-11-12T12:48:00+02:00", [Self.check("stores", .red)])
        XCTAssertEqual(BrainBarMenuPresenter.present(stores).form, .needs)
        XCTAssertEqual(BrainBarMenuPresenter.present(stores).count, 1)
    }

    // MARK: review round 1 (each fails on cc673c90's logic)

    static func present(_ c: BrainBarMenuCapture) -> P { BrainBarMenuPresenter.present(c) }
    static func texts(_ p: P) -> [String] { p.rows.flatMap(\.strings) }
    static let s1Now = "2026-11-11T20:17:00+02:00"

    /// Fix 1 (Codex F1, Claude B1, N4): S08, never "Backed up to Google Drive".
    func testBackupChecksUnknownAbsentOrUnmeasuredAreCantTell() {
        var unknown = Self.captures["S1"]!
        unknown.verdict = Self.verdict(Self.s1Now, [Self.check("backup_offsite", .unknown), Self.check("transcripts", age: 58620)])
        var absent = Self.captures["S1"]!
        absent.verdict = BrainBarHealthVerdict(dot: .green, checks: Self.names.filter { $0 != "transcripts" }.map { Self.check($0, age: 58620) },
                                               notes: [], evaluatedAt: Self.date(Self.s1Now))
        var unmeasured = Self.captures["S1"]!
        unmeasured.backupsMeasured = false
        for (name, capture) in [("unknown", unknown), ("absent", absent), ("unmeasured", unmeasured)] {
            let p = Self.present(capture)
            XCTAssertEqual(p.headline, "Can’t confirm backups yet", name)
            XCTAssertTrue(Self.texts(p).contains("Can’t read your backup records right now."), name)
            XCTAssertFalse(Self.texts(p).contains { $0.hasPrefix("Backed up to Google Drive") }, name)
        }
    }

    /// Fix 2 (Claude B2): disk amber is S21 today, S31 with `blind` false, Can't tell only with `blind` true.
    func testDiskAmberCountsUnlessBlind() {
        var measured = Self.check("disk", .amber)
        let today = Self.present(Self.capture(Self.s1Now, [measured]))
        XCTAssertEqual([today.form.rawValue, "\(today.count)"], ["needs", "1"])
        XCTAssertEqual(today.rows[2].strings, ["Your Mac may not have enough spare room for backups."])
        XCTAssertEqual(today.rows[2].action, .init(kind: .openStorageSettings, label: "Open Storage Settings"))
        measured.blind = false
        XCTAssertEqual(Self.present(Self.capture(Self.s1Now, [measured])).rows[2].strings,
                       ["Your Mac is running low on room for backups. The next one still fits."])
        measured.blind = true
        XCTAssertEqual(Self.present(Self.capture(Self.s1Now, [measured])).count, 0)
    }

    /// Fix 3 (Codex F2, Claude B3): an unknown check name and a red dot never read "All good".
    func testUnknownCheckNameAndRedDotCount() {
        let foreign = Self.present(Self.capture(Self.s1Now, [Self.check("new_check", .red)]))
        XCTAssertEqual(foreign.count, 1)
        XCTAssertEqual(foreign.rows[2].strings.first, "A BrainLayer job reported a problem BrainBar doesn’t recognise.")
        var dot = Self.captures["S1"]!
        dot.verdict = BrainBarHealthVerdict(dot: .red, checks: dot.verdict!.checks, notes: [], evaluatedAt: dot.verdict!.evaluatedAt)
        XCTAssertEqual(Self.present(dot).form, .needs)
    }

    /// Fix 4 (Codex N2): X02 lets a red through (C9); X03 keeps a stale red's rows.
    func testRedIsNeverHiddenByChecking() {
        var wake = Self.captures["S3"]!
        wake.wakeCatchUpPending = true
        let c9 = Self.present(wake)
        XCTAssertEqual([c9.headline, c9.rows[2].strings.first], ["1 thing needs you", "Google Drive is disconnected"])
        XCTAssertEqual(c9.rows.filter { $0.kind == .statusLine }.map(\.strings), Array(repeating: ["Checking…"], count: 3))
        var stale = Self.captures["S11"]!
        stale.now = Self.date("2026-11-12T13:05:00+02:00")
        stale.verdictAwakeAgeS = 1020
        let kept = Self.present(stale)
        XCTAssertEqual(kept.count, 1)
        XCTAssertTrue(kept.rows[2].strings.contains("Last checked 17 min ago."))
        stale.verdictAwakeAgeS = 2701
        XCTAssertEqual(Self.present(stale).count, 2)
    }

    /// Fix 5 (Claude N1): an unreadable verdict escalates after 30 awake minutes (X04).
    func testUnreadableVerdictEscalates() {
        var unreadable = Self.capture("2026-11-12T12:48:00+02:00", [])
        unreadable.verdict = nil
        unreadable.unreadableVerdictEpisode = .init(start: Self.date("2026-11-12T12:00:00+02:00"), awakeS: 1801)
        XCTAssertEqual(Self.present(unreadable).rows[2].strings.first, "BrainBar hasn’t been able to check on your memory since 12:00.")
        unreadable.unreadableVerdictEpisode?.awakeS = 1700
        XCTAssertEqual(Self.present(unreadable).form, .calm)
    }

    /// Fix 6 (Claude N2): X08.
    func testStoppedChecksNoteCounts() {
        var notes = Self.captures["S1"]!
        notes.verdict = BrainBarHealthVerdict(dot: .green, checks: notes.verdict!.checks,
                                              notes: [.init(code: "observability_unloaded", message: "")], evaluatedAt: notes.verdict!.evaluatedAt)
        XCTAssertEqual(Self.present(notes).rows[2].strings.first, "BrainBar’s regular checks have stopped.")
    }

    /// Fix 7 (Claude N3): S12.
    func testDriveExpiringNeedsReconnect() {
        var expiring = Self.captures["S1"]!
        expiring.driveAuth = .expiring
        expiring.driveExpiresAt = Self.date("2026-11-12T03:15:00+02:00")
        let p = Self.present(expiring)
        XCTAssertEqual(p.rows[2].strings, ["Google Drive access ends tonight at 03:15. Reconnect to keep backups going."])
        XCTAssertEqual(p.rows[2].action?.kind, .reconnectDrive)
    }

    /// Fix 8 (Claude N9, N10, N8b): no percentage from a stalled upload, R14 only when nothing verified, no future age.
    func testNoUntrueSentences() {
        var stalled = Self.captures["S32"]!
        stalled.offsiteRecovering = .init(progress: 0.4, eta: nil, startedAt: Self.date("2026-11-12T11:28:00+02:00"),
                                          updatedAt: Self.date("2026-11-12T11:30:00+02:00"), cadenceS: 30)
        XCTAssertEqual(Self.present(stalled).rows[2].strings.first,
                       "Your newest backup in Google Drive is from Nov 9 at 03:44. A backup that started at 11:28 has stopped making progress.")
        var verified = Self.captures["S13-missing"]!
        verified.verdict = Self.verdict("2026-11-12T12:48:00+02:00", [Self.check("backup_offsite", age: 3600), Self.check("drive_auth", .red)])
        verified.backupDailyFailures = 1
        XCTAssertFalse(Self.texts(Self.present(verified)).contains { $0.contains("Nothing is backed up") })
        var future = Self.captures["S1"]!
        future.verdict = Self.verdict(Self.s1Now, [Self.check("backup_offsite", age: -600), Self.check("transcripts", age: 58620)])
        XCTAssertTrue(Self.texts(Self.present(future)).contains("Can’t make sense of your backup dates right now."))
    }

    /// Fix 9 (Claude N5): every check in every non-green state yields rows with words, never a bare hand-off.
    func testNoRowWithoutWords() {
        let handoff: Set = ["BrainBar can’t fix this itself.", "Give the report to an agent."]
        for name in Self.names + ["new_check"] {
            for state in [BrainBarHealthVerdict.State.amber, .red, .unknown] {
                for row in Self.present(Self.capture(Self.s1Now, [Self.check(name, state)])).rows
                where row.kind == .needsYou || row.kind == .statusLine {
                    XCTAssertFalse(row.strings.filter { !handoff.contains($0) }.isEmpty, "\(name) \(state)")
                }
            }
        }
    }

    /// Fix 11 (Claude N7, N8a; Codex 6): Gregorian words in any calendar, singular minutes, day edges, future times.
    func testFormatterEdges() {
        var hebrew = Self.captures["S5"]!
        hebrew.calendar = Calendar(identifier: .hebrew)
        hebrew.calendar.timeZone = Self.utcPlus2
        XCTAssertTrue(Self.texts(Self.present(hebrew)).contains("Your newest backup is from Nov 9 at 03:44."))
        XCTAssertEqual(BrainBarMenuPresenter.overDuration(90), "in over 1 minute")
        var c = Self.capture("2026-11-12T00:10:00+02:00", [])
        XCTAssertEqual(BrainBarMenuPresenter.at(Self.date("2026-11-12T00:00:00+02:00"), c), "today at 00:00")
        XCTAssertEqual(BrainBarMenuPresenter.at(Self.date("2026-11-11T23:59:00+02:00"), c), "yesterday at 23:59")
        XCTAssertEqual(BrainBarMenuPresenter.at(Self.date("2026-11-11T00:10:00+02:00"), c), "Nov 11 at 00:10")
        c.verdict = nil
        c.isChecking = true
        c.lastKnownBackupAt = Self.date("2026-11-12T01:00:00+02:00")
        XCTAssertFalse(Self.texts(Self.present(c)).contains { $0.contains("Last backup") })
    }

    // MARK: expected output, read from the cases file

    static func expected(_ c: [String: Any], actions: [String: Any]) -> P {
        let rows = (c["rows"] as! [[String: Any]]).map { r -> P.Row in
            var row = P.Row(kind: P.RowKind(rawValue: r["kind"] as! String)!)
            row.strings = (r["strings"] as? [[String: Any]] ?? []).map { $0["text"] as! String }
            if let a = r["action"] as? [String: Any] {
                let kind = a["kind"] as! String
                let label = ((a["strings"] as? [[String: Any]])?.first?["text"] as? String)
                    ?? (actions[kind] as! [String: Any])["label"] as! String
                row.action = .init(kind: P.ActionKind(rawValue: kind)!, label: label)
            }
            if let mark = r["mark"] as? [String: String] {
                row.mark = mark["A1_disc"] != nil ? .marked : P.Mark(rawValue: String(mark["A2_sign"]!.prefix { $0 != " " }))!
            } else if let mark = r["mark"] as? String {
                row.mark = P.Mark(rawValue: String(mark.prefix { $0 != " " }))!
            }
            row.state = (r["state"] as? String).map { P.RowState(rawValue: $0)! }
            row.dividerBefore = r["divider_before"] as? Bool ?? false
            row.spinner = r["spinner"] as? Bool ?? false
            row.progressPercent = (r["progress_bar"] as? [String: Any])?["value"] as? Int
            row.keyEquivalent = r["key_equivalent"] as? String
            row.optionAlternate = ((r["option_alternate"] as? [String: Any])?["strings"] as? [[String: Any]])?.first?["text"] as? String
            return row
        }
        return .init(form: P.Form(rawValue: c["form"] as! String)!, headline: c["headline"] as! String,
                     count: c["count"] as! Int, icon: P.Icon(rawValue: c["icon"] as! String)!, rows: rows)
    }

    static func casesURL() throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: "presentation-cases", withExtension: "json", subdirectory: "Fixtures"))
    }

    static func loadCases() throws -> ([[String: Any]], [String: Any]) {
        let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: casesURL())) as! [String: Any]
        return (doc["cases"] as! [[String: Any]], doc["actions"] as! [String: Any])
    }

    // MARK: captures, one per case, built from its `inputs` (values quoted from the file)

    static let names = ["backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk",
                        "ingest", "embedding", "stores", "search", "monitor"]
    static let utcPlus2 = TimeZone(secondsFromGMT: 7200)!

    static func date(_ iso: String) -> Date { ISO8601DateFormatter().date(from: iso)! }

    /// Every check carries blind false and unknown_ticks 0 (the file's conventions, seam PR-1 fields).
    static func check(_ name: String, _ state: BrainBarHealthVerdict.State = .green, age: Double? = nil,
                      threshold: Double? = nil, since: String? = nil) -> Check {
        Check(name: name, state: state, ageS: age, thresholdS: threshold, since: since, note: "", unknownTicks: 0, blind: false)
    }

    /// Any of the 11 checks not given is green (the file's conventions). A check under another name is kept.
    static func verdict(_ at: String, _ given: [Check]) -> BrainBarHealthVerdict {
        let checks = names.map { name in given.first { $0.name == name } ?? check(name) } + given.filter { !names.contains($0.name) }
        return .init(dot: .green, checks: checks, notes: [], evaluatedAt: date(at))
    }

    static func capture(_ now: String, _ given: [Check], _ edit: (inout BrainBarMenuCapture) -> Void = { _ in }) -> BrainBarMenuCapture {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = utcPlus2
        var c = BrainBarMenuCapture(now: date(now), calendar: calendar, verdict: verdict(now, given))
        edit(&c)
        return c
    }

    static let lag = BrainBarMenuCapture.Watcher(alertReasons: ["offset_lag"])
    static let upload = BrainBarMenuCapture.Upload(startedAt: date("2026-11-12T11:28:00+02:00"), live: true)

    static let captures: [String: BrainBarMenuCapture] = [
        "S1": capture("2026-11-11T20:17:00+02:00", [check("backup_offsite", age: 58380), check("transcripts", age: 58620)]),
        "S2-today": capture("2026-11-12T12:00:00+02:00", [check("backup_offsite", .amber, age: 114480, threshold: 280800)]) {
            $0.upload = upload
        },
        "S2-after-seam": capture("2026-11-12T12:00:00+02:00", [check("backup_offsite", .amber, age: 114480, threshold: 280800)]) {
            $0.upload = upload
            $0.offsiteRecovering = .init(progress: 0.4, eta: date("2026-11-12T12:48:00+02:00"), startedAt: date("2026-11-12T11:28:00+02:00"),
                                         updatedAt: date("2026-11-12T12:00:00+02:00"), cadenceS: 30)
        },
        "S4": capture("2026-11-11T10:39:00+02:00", []) {
            $0.verdict = nil
            $0.wakeCatchUpPending = true
            $0.lastKnownBackupAt = date("2026-11-11T04:00:00+02:00")
        },
        "S7": capture("2026-11-12T12:48:00+02:00", [check("backup_offsite", age: 27660), check("transcripts", age: 31620),
                                                    check("search", .amber, since: "2026-11-12T12:45:00+02:00")]) {
            $0.watcher = lag
            $0.lagEpisodeStart = date("2026-11-12T12:40:00+02:00")
        },
        // `now` is null in the file (first launch); any time does.
        "S8": capture("2026-11-12T12:48:00+02:00", []) {
            $0.verdict = nil
            $0.isChecking = true
        },
        "S3": capture("2026-11-10T00:35:00+02:00", [check("backup_offsite", age: 75060), check("drive_auth", .red)]) {
            $0.driveAuth = .invalid
            $0.backupDailyFailures = 1
        },
        "S5": capture("2026-11-11T01:57:00+02:00", [check("backup_local", .amber, age: 166380, threshold: 194400),
                                                    check("backup_offsite", .amber, age: 166380, threshold: 280800)]) {
            $0.backupDailyFailures = 2
        },
        "S6": capture("2026-11-12T12:48:00+02:00", [check("backup_offsite", age: 31680), check("transcripts", age: 31680)]) {
            $0.maintenanceNightFailures = 3
        },
        "S9": capture("2026-11-12T12:49:00+02:00", [check("backup_offsite", age: 31680), check("transcripts", age: 31740),
                                                    check("embedding", .amber, age: 3960, threshold: 21600)]) {
            $0.keywordCoverage = 1.0
        },
        "S10": capture("2026-11-12T12:48:00+02:00", [check("drive_auth", .red), check("search", .amber, since: "2026-11-12T12:45:00+02:00")]) {
            $0.driveAuth = .invalid
            $0.watcher = lag
            $0.lagEpisodeStart = date("2026-11-12T12:40:00+02:00")
        },
        // backup_local: "any ageS past red (not drawn)".
        "S11": capture("2026-11-12T12:48:00+02:00", [check("backup_local", .red, age: 200000, threshold: 194400), check("integrity", .red)]),
        "S11b": capture("2026-11-12T12:48:00+02:00", [check("backup_local", .red)]),
        "S12": capture("2026-11-12T12:48:00+02:00", [check("drive_auth", .red), check("embedding", .amber, age: 3960, threshold: 21600)]) {
            $0.driveAuth = .invalid
            $0.maintenanceNightFailures = 3
            $0.keywordCoverage = 1.0
        },
        "S32": capture("2026-11-12T12:00:00+02:00", [check("backup_offsite", .red, age: 288960, threshold: 280800)]) { $0.upload = upload },
        "C32": capture("2026-11-10T00:35:00+02:00", [check("backup_offsite", age: 75060), check("transcripts", .amber, age: 144000),
                                                     check("drive_auth", .red)]) {
            $0.driveAuth = .invalid
            $0.backupDailyFailures = 1
            $0.jsonlBackupFailures = 2
        },
        "S13-missing": capture("2026-11-12T12:48:00+02:00", [check("backup_offsite", .red), check("drive_auth", .red)]) {
            $0.driveAuth = .missing
        },
    ]
}
