import CryptoKit
import Foundation
import XCTest
@testable import BrainBar

/// The table test against the lead's `presentation-cases.json` (phase 8). Each case's capture is decoded from its
/// own `inputs`, literally. P1 owes the six calm cases whole; for the eleven needs cases it owes form, count and icon
/// (P2 turns those into whole-value equality).
final class BrainBarMenuPresentationTests: XCTestCase {
    typealias P = BrainBarMenuPresentation

    /// The plan's `phase-8-native-build/presentation-cases.json`, byte for byte.
    static let casesSHA256 = "65e4e76375265706bd23e73e6869d5c387d2d0f221ccfb8218bab7d1286882a6"
    static let calm = ["S1", "S2-today", "S2-after-seam", "S4", "S7", "S8"]

    func testCasesFileIsTheLeadsUnchangedCopy() throws {
        let digest = SHA256.hash(data: try Data(contentsOf: Self.casesURL()))
        XCTAssertEqual(digest.map { String(format: "%02x", $0) }.joined(), Self.casesSHA256)
    }

    func testCalmCasesMatchWholeAndNeedsCasesMatchFormCountAndIcon() throws {
        let cases = try Self.loadCases()
        XCTAssertEqual(cases.count, 17)
        XCTAssertEqual(cases.filter { $0["form"] as? String == "calm" }.compactMap { $0["id"] as? String }, Self.calm)
        for c in cases {
            let id = c["id"] as! String
            let got = BrainBarMenuPresenter.present(try Self.capture(c["inputs"] as! [String: Any]))
            if Self.calm.contains(id) {
                XCTAssertEqual(got, Self.expected(c), id)
            } else {
                XCTAssertEqual([got.form.rawValue, "\(got.count)", got.icon.rawValue],
                               [c["form"] as! String, "\(c["count"] as! Int)", c["icon"] as! String], id)
            }
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

    // MARK: expected output, read from the cases file

    static func expected(_ c: [String: Any]) -> P {
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

    static func loadCases() throws -> [[String: Any]] {
        let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: casesURL())) as! [String: Any]
        return doc["cases"] as! [[String: Any]]
    }

    // MARK: the capture, decoded from a case's `inputs` (named as the truth table names them)

    /// A time as the file writes it; some carry a remark after the timestamp, as in "…+02:00 (synthetic; …)".
    static func date(_ value: Any?) -> Date? {
        (value as? String).flatMap { ISO8601DateFormatter().date(from: String($0.prefix { $0 != " " })) }
    }

    static func capture(_ inputs: [String: Any]) throws -> BrainBarMenuCapture {
        // S8's `now` is null (first launch): any time does.
        let now = date(inputs["now"]) ?? date("2026-11-12T12:48:00+02:00")!
        var c = BrainBarMenuCapture(now: now, timeZone: TimeZone(secondsFromGMT: 7200)!)
        let verdict = inputs["verdict"] as? [String: Any]
        let checks = inputs["checks"] as? [String: [String: Any]] ?? [:]
        if verdict?["decodable"] as? Bool == true {
            // The file's convention: any of the 11 checks not given is green, with blind false and unknown_ticks 0.
            let built = BrainBarMenuPresenter.known.map { name -> BrainBarHealthVerdict.Check in
                let k = checks[name] ?? [:]
                return .init(name: name, state: .init(rawValue: k["state"] as? String ?? "green")!, ageS: k["ageS"] as? Double,
                             thresholdS: k["thresholdS"] as? Double, since: k["since"] as? String, note: "", unknownTicks: 0, blind: false)
            }
            XCTAssertEqual(verdict?["fresh"] as? Bool, true)
            c.verdict = BrainBarHealthVerdict(dot: .green, checks: built, notes: [], evaluatedAt: try XCTUnwrap(date(verdict?["evaluatedAt"])))
        }
        c.isChecking = inputs["app.isChecking"] as? Bool ?? false
        c.wakeCatchUpPending = inputs["app.wakeCatchUpPending"] as? Bool ?? false
        c.backupsMeasured = (inputs["backupsObservation.state"] as? String ?? "measured") == "measured"
        c.lastKnownBackupAt = date(inputs["backupsObservation.lastMeasured.newestVerifiedCopyAt"])
        if let w = inputs["watcherFile"] as? [String: Any] {
            c.watcher = .init(readable: w["readable"] as! Bool, fresh: w["fresh"] as! Bool,
                              alertReasons: w["alertReasons"] as! [String], dbProbeFailed: w["dbProbeFailed"] as! Bool)
        }
        if inputs["localSnapshot.uploadState"] as? String == "in_progress" {
            c.upload = .init(startedAt: try XCTUnwrap(date(inputs["uploadProcessStartedAt"])), live: inputs["uploadProcessLive"] as! Bool)
        }
        if let r = checks["backup_offsite"]?["recovering"] as? [String: Any] {
            c.offsiteRecovering = .init(progress: r["progress"] as? Double, eta: date(r["eta"]), startedAt: date(r["started_at"])!,
                                        updatedAt: date(r["updated_at"])!, cadenceS: r["cadence_s"] as! Double)
        }
        c.lagEpisodeStart = date(inputs["uiClock.offsetLagEpisodeStart"])
        c.maintenanceNightFailures = inputs["maintenance.night.consecutiveFailures"] as? Int ?? 0
        return c
    }
}
