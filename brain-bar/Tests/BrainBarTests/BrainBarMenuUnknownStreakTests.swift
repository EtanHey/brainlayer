import Foundation
import XCTest
@testable import BrainBar

extension BrainBarMenuInvariantTests {
    static var streaks: [WritableKeyPath<Capture, Int?>] { [\Capture.maintenanceNightFailures, \Capture.backupDailyFailures, \Capture.jsonlBackupFailures] }

    /// The second key: the oracle's coverage table, independent of the registry's thing names and helpers.
    static func oracleCoverage(_ id: String, _ values: [String], _ f: Facts) -> Set<String> {
        switch id {
        case "S1": return Set(things.map(\.word))
        case "A2.fine.is", "A2.fine.are": return Set(f.fineWords(values[0]))
        case "M5": return Set(f.everythingElse)
        case "U03.stillSaved": return ["saving", "backups"]
        case "F07.exactWords": return ["search"]
        default: return []
        }
    }

    static func unknownViolations(_ c: Capture, _ s: Int) -> [String] {
        var zero = c, unknown = c
        zero[keyPath: streaks[s]] = 0
        unknown[keyPath: streaks[s]] = nil
        let p0 = BrainBarMenuPresenter.present(zero), pn = BrainBarMenuPresenter.present(unknown)
        let thing: BrainBarMenuPresenter.Thing = s == 0 ? .upkeep : .safe
        let word = s == 0 ? "upkeep" : "backups"
        var out: [String] = []
        // R30: only unsupported reassurance may leave a Needs-you row; all other fields and ordering stay.
        for oracle in [false, true] {
            func covers(_ sentence: String, _ p: P, _ capture: Capture) -> Bool {
                guard let (entry, values) = BrainBarMenuPresenter.entry(for: sentence, capture), entry.kind == .reassurance else { return false }
                let coverage = oracleCoverage(entry.id, values, Facts(c: capture, p: p))
                if oracle { return coverage.contains(word) }
                return BrainBarMenuPresenter.reassuranceCoverage(entry.id, values, p, capture).contains(thing)
            }
            let expected = p0.rows.filter { $0.kind == .needsYou }.map { row in
                var row = row
                row.strings = row.strings.compactMap { text in
                    let kept = sentences(text).filter { !covers($0, p0, zero) }
                    return kept.isEmpty ? nil : kept.joined(separator: " ")
                }
                return row
            }
            if expected != pn.rows.filter({ $0.kind == .needsYou }) { out.append("I10 rows changed (oracle=\(oracle))") }
            if ([pn.headline] + pn.rows.flatMap(\.strings)).flatMap(sentences).contains(where: { covers($0, pn, unknown) }) {
                out.append("I10 reassurance covers \(word) (oracle=\(oracle))")
            }
        }
        if p0.form == .needs && (pn.form != .needs || pn.count != p0.count) { out.append("I10 needs form/count changed") }
        if pn.headline == "All good" { out.append("I10 All good with unknown \(word)") }
        let countID = s == 0 ? "U03" : s == 1 ? "S06" : "no transcript count sentence yet"
        if pn.rows.flatMap(\.strings).flatMap(sentences).contains(where: { BrainBarMenuPresenter.entry(for: $0, unknown)?.0.id == countID }) {
            out.append("I10 unmeasured count")
        }
        // Fresh unflagged captures use the existing family checking line. Stale/flagged captures may use X01's global line.
        if c.verdict != nil, (0...900).contains(c.verdictAwakeAgeS), !c.isChecking, !c.wakeCatchUpPending,
           !pn.rows.contains(where: { $0.kind == .statusLine && $0.state == .cantTell && $0.strings == ["Checking…"] }) {
            out.append("I10 missing checking line")
        }
        return out
    }

    func testI10UnknownComparedWithZero() throws {
        XCTAssertEqual(Set(BrainBarMenuPresenter.Say.all.filter { $0.kind == .reassurance }.map(\.id)),
                       ["S1", "A2.fine.is", "A2.fine.are", "M5", "U03.stillSaved", "F07.exactWords"])
        var count = 0, found: [String] = []
        // Every known/nil input, on the differential suite's reduced cross (pairs: two contexts, singles: eight).
        let cross: [(State, Freshness, Bool, Bool, Bool)] = [
            (.green, .fresh0, true, false, false), (.red, .fresh0, false, false, false),
            (.green, .fresh0, true, true, false), (.green, .fresh0, true, false, true),
            (.amber, .fresh900, true, true, true), (.green, .stale901, true, true, false),
            (.green, .staleNegative, true, false, true), (.green, .missing, false, false, false),
        ]
        for input in Self.inputs() {
            for (dot, age, measured, checking, wake) in input.settings.count == 2 ? Array(cross.prefix(2)) : cross {
                let c = Self.capture(input, dot: dot, checking: checking, wake: wake, freshness: age, measured: measured)
                for s in 0...2 { count += 1; found += Self.unknownViolations(c, s).map { "\($0) streak \(s) input \(input)" } }
            }
        }
        let edges = Self.edgeCaptures() + (try BrainBarMenuPresentationTests.loadCases().map {
            try BrainBarMenuPresentationTests.capture($0["inputs"] as! [String: Any])
        })
        for c in edges { for s in 0...2 { count += 1; found += Self.unknownViolations(c, s) } }
        print("I10: \(count) zero/nil comparisons")
        XCTAssertEqual(Array(found.prefix(10)), [], "\(found.count) violations")
    }

    private func assertUnknownDoesNotReassure(_ s: Int) {
        let c = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
        XCTAssertEqual(Self.unknownViolations(c, s), [], "streak \(s)")
    }

    func testStreaksStartUnknown() {
        let c = Capture(now: Self.now, timeZone: TimeZone(secondsFromGMT: 0)!)
        for streak in Self.streaks { XCTAssertNil(c[keyPath: streak]) }
    }

    func testI10UnknownUpkeepDoesNotReassure() { assertUnknownDoesNotReassure(0) }
    func testUnknownDailyBackupDoesNotReassure() { assertUnknownDoesNotReassure(1) }
    func testUnknownTranscriptBackupDoesNotReassure() { assertUnknownDoesNotReassure(2) }

    func testUnknownDropsCountOnlyFaultAndUnsupportedReassurance() {
        var c = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
        c.maintenanceNightFailures = 3
        let known = BrainBarMenuPresenter.present(c)
        XCTAssertEqual(known.count, 1)
        XCTAssertTrue(known.rows.flatMap(\.strings).contains { $0.contains("Your memory is still being saved and backed up") })
        // R30 refined clause 1: the upkeep title/action/count stay; only the now-unproven backup reassurance leaves.
        for s in [1, 2] { XCTAssertEqual(Self.unknownViolations(c, s), []) }
        c.maintenanceNightFailures = nil
        let unknown = BrainBarMenuPresenter.present(c)
        XCTAssertEqual(unknown.form, .calm)
        XCTAssertEqual(unknown.count, 0)
        XCTAssertTrue(unknown.rows.contains { $0.state == .cantTell && $0.strings == ["Checking…"] })
        XCTAssertFalse(unknown.rows.flatMap(\.strings).contains { $0.contains("Nightly upkeep failed") })
    }
}
