import CryptoKit
import Foundation
import XCTest
@testable import BrainBar

/// The table test: every case of the lead's `presentation-cases.json` (phase 8), built from its
/// `inputs`, must come out of `BrainBarMenuPresenter.present` exactly as the case says.
final class BrainBarMenuPresentationTests: XCTestCase {
    typealias P = BrainBarMenuPresentation
    typealias Check = BrainBarHealthVerdict.Check

    /// FIXTURES-CHANGES-v3.md, "file 2". A drift here means the copy is no longer the lead's file.
    static let casesSHA256 = "c1f29c7460bb94a485ee7f42bbbaed83eceec606b2db1252b1e97d8716dccfe5"

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
            if let mark = r["mark"] as? [String: String], row.kind == .headline {
                row.sign = mark["A1_disc"] != nil ? .marked : P.Sign(rawValue: String(mark["A2_sign"]!.prefix { $0 != " " }))
            }
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

    /// Any of the 11 checks not given is green (the file's conventions).
    static func verdict(_ at: String, _ given: [Check]) -> BrainBarHealthVerdict {
        let checks = names.map { name in given.first { $0.name == name } ?? check(name) }
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
        // The file gives ageS 165600 (46.0 h, rounded), which is Nov 9 03:57; 166380 is its 03:44.
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
