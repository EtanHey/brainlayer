import Foundation
import XCTest
@testable import BrainBar

extension BrainBarMenuInvariantTests {
    static var streaks: [WritableKeyPath<Capture, Int?>] { [\Capture.maintenanceNightFailures, \Capture.backupDailyFailures, \Capture.jsonlBackupFailures] }

    /// The second key: the oracle's coverage table, independent of the registry's thing names and helpers.
    static func oracleCoverage(_ id: String, _ values: [String], _ f: Facts) -> Set<String>? {
        switch id {
        case "S1": return Set(things.map(\.word))
        case "A2.fine.is", "A2.fine.are": return Set(f.fineWords(values[0]))
        case "M5": return Set(f.everythingElse)
        case "U03.stillSaved": return ["saving", "backups"]
        case "F07.exactWords": return ["search"]
        default: return nil
        }
    }

    static func unknownViolations(_ c: Capture, _ s: Int, zeroPresentation: P? = nil) -> [String] {
        var zero = c, unknown = c
        zero[keyPath: streaks[s]] = 0
        unknown[keyPath: streaks[s]] = nil
        let p0 = zeroPresentation ?? BrainBarMenuPresenter.present(zero), pn = BrainBarMenuPresenter.present(unknown)
        let thing: BrainBarMenuPresenter.Thing = s == 0 ? .upkeep : .safe
        let word = s == 0 ? "upkeep" : "backups"
        var out: [String] = []
        // Match each unique sentence once per capture, then reuse it for both coverage keys and the count check.
        func matches(_ texts: [String], _ capture: Capture) -> [String: (BrainBarMenuPresenter.Sentence, [String])] {
            var result: [String: (BrainBarMenuPresenter.Sentence, [String])] = [:]
            for text in Set(texts.flatMap(sentences)) { result[text] = BrainBarMenuPresenter.entry(for: text, capture) }
            return result
        }
        let before = matches(p0.rows.filter { $0.kind == .needsYou }.flatMap(\.strings), zero)
        let after = matches([pn.headline] + pn.rows.flatMap(\.strings), unknown)
        let f0 = Facts(c: zero, p: p0), fn = Facts(c: unknown, p: pn)
        // R30: only unsupported reassurance may leave a Needs-you row; all other fields and ordering stay.
        for oracle in [false, true] {
            func covers(_ sentence: String, _ unmeasured: Bool) -> Bool {
                guard let (entry, values) = (unmeasured ? after : before)[sentence], entry.kind == .reassurance else { return false }
                if oracle { return oracleCoverage(entry.id, values, unmeasured ? fn : f0)?.contains(word) == true }
                return BrainBarMenuPresenter.reassuranceCoverage(entry.id, values, unmeasured ? pn : p0, unmeasured ? unknown : zero)?.contains(thing) == true
            }
            let expected = p0.rows.filter { $0.kind == .needsYou }.map { row in
                var row = row
                row.strings = row.strings.compactMap { text in
                    let kept = sentences(text).filter { !covers($0, false) }
                    return kept.isEmpty ? nil : kept.joined(separator: " ")
                }
                return row
            }
            if expected != pn.rows.filter({ $0.kind == .needsYou }) { out.append("I10 rows changed (oracle=\(oracle))") }
            if after.keys.contains(where: { covers($0, true) }) { out.append("I10 reassurance covers \(word) (oracle=\(oracle))") }
        }
        if p0.form == .needs && (pn.form != .needs || pn.count != p0.count) { out.append("I10 needs form/count changed") }
        if pn.headline == "All good" { out.append("I10 All good with unknown \(word)") }
        let countID = s == 0 ? "U03" : s == 1 ? "S06" : "no transcript count sentence yet"
        if after.values.contains(where: { $0.0.id == countID }) {
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
        // Two core contexts per input, skipping its own nil streak (the known twin covers it); reuse zero presentations.
        func check(_ c: Capture, skipping: Int? = nil) {
            var cachedZero: P?
            for s in 0...2 where skipping != s {
                if c[keyPath: Self.streaks[s]] == 0, cachedZero == nil { cachedZero = BrainBarMenuPresenter.present(c) }
                count += 1
                found += Self.unknownViolations(c, s, zeroPresentation: c[keyPath: Self.streaks[s]] == 0 ? cachedZero : nil)
                    .map { "\($0) streak \(s)" }
            }
        }
        // Every input takes two core contexts. Flag/staleness interactions have explicit examples below.
        let core: [(State, Freshness, Bool, Bool, Bool)] = [
            (.green, .fresh0, true, false, false), (.red, .fresh0, false, false, false),
        ]
        for input in Self.inputs() {
            for (dot, age, measured, checking, wake) in core {
                check(Self.capture(input, dot: dot, checking: checking, wake: wake, freshness: age, measured: measured), skipping: input.unknownStreak)
            }
        }
        let examples: [Input] = [
            Input(settings: [:], base: 0, upkeep: 0),
            Input(settings: ["drive_auth": .red, "transcripts": .red], base: 0, upkeep: 0, profile: 2),
            Input(settings: ["embedding": .red], base: 0, upkeep: 3, profile: 1),
            Input(settings: [:], base: 3, upkeep: 0),
            Input(settings: ["backup_local": .red, "backup_offsite": .amber], base: 1, upkeep: 3, profile: 2),
        ]
        let interactions: [(State, Freshness, Bool, Bool, Bool)] = [
            (.green, .fresh0, true, true, false), (.green, .fresh0, true, false, true),
            (.amber, .fresh900, true, true, true), (.green, .stale901, true, false, false),
            (.green, .staleNegative, true, false, false), (.green, .missing, false, false, false),
        ]
        for input in examples {
            for (dot, age, measured, checking, wake) in interactions {
                check(Self.capture(input, dot: dot, checking: checking, wake: wake, freshness: age, measured: measured))
            }
        }
        let edges = Self.edgeCaptures() + (try BrainBarMenuPresentationTests.loadCases().map {
            try BrainBarMenuPresentationTests.capture($0["inputs"] as! [String: Any])
        })
        for c in edges { check(c) }
        print("I10: \(count) zero/nil comparisons")
        XCTAssertEqual(Array(found.prefix(10)), [], "\(found.count) violations")
    }

    private func assertUnknownDoesNotReassure(_ s: Int) {
        let c = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
        XCTAssertEqual(Self.unknownViolations(c, s), [], "streak \(s)")
        var unknown = c
        unknown[keyPath: Self.streaks[s]] = nil
        XCTAssertEqual(BrainBarMenuPresenter.present(unknown), Self.expectedUnknown(
            s == 0 ? "Still checking upkeep" : "Can’t confirm backups yet",
            s == 0 ? "Saving, search and backups are fine." : "Saving, search and upkeep are fine."))
    }

    static func expectedUnknown(_ headline: String, _ fine: String, lines: Int = 1) -> P {
        P(form: .calm, headline: headline, count: 0, icon: .resting, rows:
            [P.Row(kind: .headline, strings: [headline], mark: .wait), P.Row(kind: .separator)]
            + Array(repeating: P.Row(kind: .statusLine, strings: ["Checking…"], mark: .wait, state: .cantTell), count: lines)
            + [P.Row(kind: .everythingElse, strings: [fine], mark: .ok), P.Row(kind: .separator),
               P.Row(kind: .standardItem, strings: ["Settings…"], keyEquivalent: "⌘,"),
               P.Row(kind: .standardItem, strings: ["Quit BrainBar"], keyEquivalent: "⌘Q", optionAlternate: "Restart BrainBar")])
    }

    func testEveryReassuranceHasBothCoverageEntries() {
        let c = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
        let p = BrainBarMenuPresenter.present(c), f = Facts(c: c, p: p)
        for entry in BrainBarMenuPresenter.Say.all where entry.kind == .reassurance {
            XCTAssertNotNil(BrainBarMenuPresenter.reassuranceCoverage(entry.id, ["Saving"], p, c), entry.id)
            XCTAssertNotNil(Self.oracleCoverage(entry.id, ["Saving"], f), entry.id)
            XCTAssertFalse(BrainBarMenuPresenter.reassuranceCoverage(entry.id, ["Saving"], p, c)?.isEmpty ?? true, entry.id)
            XCTAssertFalse(Self.oracleCoverage(entry.id, ["Saving"], f)?.isEmpty ?? true, entry.id)
        }
    }

    func testUnknownUpkeepIsNeverProvenFine() {
        var c = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
        c.maintenanceNightFailures = nil
        XCTAssertFalse(BrainBarMenuPresenter.provenFine(c).contains(.upkeep))
    }

    func testAllStreaksUnknownAtLaunch() {
        var c = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
        for streak in Self.streaks { c[keyPath: streak] = nil }
        XCTAssertEqual(BrainBarMenuPresenter.present(c), Self.expectedUnknown("Can’t confirm backups yet", "Saving and search are fine.", lines: 2))
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
