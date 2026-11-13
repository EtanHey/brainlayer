import Foundation
import XCTest
@testable import BrainBar

/// The safety property of the M menu, enumerated rather than hunted for (phase 8 revision, 13 Nov). Every later
/// presentation PR runs under this test; one that adds a truth-table exception changes it on purpose and says so.
///
/// Enumerated: each of the 11 known checks and one unknown name in each state (green, amber, red, unknown, absent),
/// alone on three bases (plain, a live upload, a lagging watcher), and every pair of them on the plain base; each
/// crossed with the dot (green, amber, red), `isChecking` and the wake flag (off, on), the verdict (fresh at awake
/// age 0 and 900, stale at -1 and 901, missing) and backups measured or not. Not enumerated: the watcher's other
/// fields, `maintenanceNightFailures`, ages, `since` values, recovering, notes, and triples of checks.
final class BrainBarMenuInvariantTests: XCTestCase {
    typealias P = BrainBarMenuPresentation
    typealias Capture = BrainBarMenuCapture
    typealias Check = BrainBarHealthVerdict.Check
    typealias State = BrainBarHealthVerdict.State

    static let known = ["backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk",
                        "ingest", "embedding", "stores", "search", "monitor"]
    static let names = known + ["new_check"]
    enum Setting: CaseIterable { case green, amber, red, unknown, absent }
    enum Freshness: CaseIterable { case fresh0, fresh900, staleNegative, stale901, missing }

    static let now = ISO8601DateFormatter().date(from: "2026-11-12T12:48:00+02:00")!

    static func check(_ name: String, _ setting: Setting) -> Check? {
        let state: State
        switch setting {
        case .green: state = .green
        case .amber: state = .amber
        case .red: state = .red
        case .unknown: state = .unknown
        case .absent: return nil
        }
        let age: Double? = name == "backup_offsite" ? 27660 : name == "transcripts" ? 31620 : nil
        let since = name == "search" && setting == .amber ? "2026-11-12T12:45:00+02:00" : nil
        return Check(name: name, state: state, ageS: age, since: since, note: "", unknownTicks: 0, blind: false)
    }

    static func capture(_ settings: [String: Setting], base: Int, dot: State, checking: Bool, wake: Bool,
                        freshness: Freshness, measured: Bool) -> Capture {
        let checks = names.compactMap { check($0, settings[$0] ?? ($0 == "new_check" ? .absent : .green)) }
        var c = Capture(now: now, timeZone: TimeZone(secondsFromGMT: 7200)!)
        c.verdict = freshness == .missing ? nil : BrainBarHealthVerdict(dot: dot, checks: checks, notes: [], evaluatedAt: now)
        c.verdictAwakeAgeS = [.fresh0: 0, .fresh900: 900, .staleNegative: -1, .stale901: 901, .missing: 0][freshness]!
        c.isChecking = checking
        c.wakeCatchUpPending = wake
        c.backupsMeasured = measured
        if base == 1 { c.upload = .init(startedAt: now.addingTimeInterval(-1920), live: true) }
        if base == 2 {
            c.watcher = .init(alertReasons: ["offset_lag"])
            c.lagEpisodeStart = now.addingTimeInterval(-480)
        }
        return c
    }

    /// Every check configuration enumerated, with the bases it runs on.
    static func configurations() -> [([String: Setting], [Int])] {
        let nonGreen: [Setting] = [.amber, .red, .unknown, .absent]
        var all: [([String: Setting], [Int])] = [([:], [0, 1, 2])]
        for name in names { for s in nonGreen { all.append(([name: s], [0, 1, 2])) } }
        for (i, a) in names.enumerated() {
            for b in names[(i + 1)...] { for sa in nonGreen { for sb in nonGreen { all.append(([a: sa, b: sb], [0])) } } }
        }
        return all
    }

    func testInvariantsHoldOnEveryEnumeratedInput() {
        var outputs = 0
        var violations: [String] = []
        for (settings, bases) in Self.configurations() {
            for base in bases {
                for dot in [State.green, .amber, .red] {
                    for checking in [false, true] {
                        for wake in [false, true] {
                            for freshness in Freshness.allCases {
                                for measured in [true, false] {
                                    let c = Self.capture(settings, base: base, dot: dot, checking: checking, wake: wake,
                                                         freshness: freshness, measured: measured)
                                    outputs += 1
                                    for v in Self.violations(c, BrainBarMenuPresenter.present(c)) where violations.count < 25 {
                                        violations.append("\(v) | checks \(settings) base \(base) dot \(dot) checking \(checking) wake \(wake) \(freshness) measured \(measured)")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        print("BrainBarMenuInvariantTests: \(outputs) outputs checked")
        XCTAssertGreaterThan(outputs, 100_000)
        XCTAssertEqual(violations, [], violations.joined(separator: "\n"))
    }

    /// The guard on its own: handed a presentation that breaks each invariant, it returns one that does not.
    func testFinalGuardCorrectsAPresentationThatBreaksEachInvariant() {
        func base(_ edit: (inout Capture) -> Void = { _ in }) -> Capture {
            var c = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
            edit(&c)
            return c
        }
        let allGood = P(form: .calm, headline: "All good", count: 0, icon: .resting,
                        rows: [P.Row(kind: .headline, strings: ["All good"], mark: .ok), P.Row(kind: .headlineDetail, strings: ["Backed up."])])
        let oneNeeds = P(form: .needs, headline: "1 thing needs you", count: 1, icon: .marked,
                         rows: [P.Row(kind: .headline, strings: ["1 thing needs you"], mark: .marked), P.Row(kind: .separator),
                                P.Row(kind: .needsYou, strings: ["Something BrainBar checks is failing."], mark: .alertCircle)])
        let red = base { $0.verdict = BrainBarHealthVerdict(dot: .green, checks: $0.verdict!.checks + [Self.check("disk", .red)!], notes: [], evaluatedAt: Self.now) }
        let unmeasured = base { $0.backupsMeasured = false }
        let searchAmber = base { $0.verdict = BrainBarHealthVerdict(dot: .green, checks: $0.verdict!.checks.filter { $0.name != "search" }
                                                                    + [Self.check("search", .amber)!], notes: [], evaluatedAt: Self.now) }
        var falseFine = allGood
        falseFine.headline = "Backing up now"
        falseFine.rows[0].strings = ["Backing up now"]
        falseFine.rows[1] = P.Row(kind: .everythingElse, strings: ["Saving, search and upkeep are fine."])
        var miscounted = oneNeeds
        miscounted.count = 2
        miscounted.icon = .resting
        var badWords = oneNeeds
        badWords.rows[2].strings = ["", "Can't"]
        badWords.rows.append(P.Row(kind: .statusLine, strings: [""]))
        var strayDriveLine = falseFine
        strayDriveLine.rows[1] = P.Row(kind: .headlineDetail, strings: ["Backed up to Google Drive today at 04:00."])
        let cases: [(String, Capture, P)] = [
            ("I1: calm over a red check", red, allGood),
            ("I2: All good while backups are unmeasured", unmeasured, allGood),
            ("I2: the all-good Drive line under another headline", unmeasured, strayDriveLine),
            ("I3: search called fine while amber", searchAmber, falseFine),
            ("I4: count and icon disagree with the rows", red, miscounted),
            ("I5: empty strings and a straight apostrophe", red, badWords),
        ]
        for (name, c, bad) in cases {
            XCTAssertFalse(Self.violations(c, bad).isEmpty, "\(name): the crafted presentation should break it")
            XCTAssertEqual(Self.violations(c, BrainBarMenuPresenter.finalGuard(c, bad)), [], name)
        }
        // A duplicate check name hides a red from a lookup by first match; the guard still sees it.
        let duplicate = base { $0.verdict = BrainBarHealthVerdict(dot: .green, checks: [Self.check("stores", .green)!] + $0.verdict!.checks
                                                                  + [Self.check("stores", .red)!], notes: [], evaluatedAt: Self.now) }
        XCTAssertEqual(BrainBarMenuPresenter.finalGuard(duplicate, allGood).form, .needs)
    }

    // MARK: the invariants, written from the brief, not from the presenter

    static func violations(_ c: Capture, _ p: P) -> [String] {
        var out: [String] = []
        let v = c.verdict
        let fresh = v != nil && (0...900).contains(c.verdictAwakeAgeS)
        func all(_ name: String, _ state: State) -> Bool {
            guard let v, v.checks.contains(where: { $0.name == name }) else { return false }
            return v.checks.allSatisfy { $0.name != name || $0.state == state }
        }
        let needsRows = p.rows.filter { $0.kind == .needsYou }.count

        // I1: no hidden red.
        if let v, v.dot == .red || v.checks.contains(where: { $0.state == .red }), p.form != .needs {
            out.append("I1 a red with a verdict present is not the needs form")
        }
        // I2: "All good" only when proven (and its Drive line only beside it).
        let texts = [p.headline] + p.rows.flatMap(\.strings)
        if texts.contains("All good") {
            let proven = fresh && v!.dot == .green && known.allSatisfy { all($0, .green) }
                && v!.checks.allSatisfy { known.contains($0.name) || $0.state == .green }
                && c.backupsMeasured && !c.isChecking && !c.wakeCatchUpPending
            if !proven { out.append("I2 All good without proof") }
        } else if texts.contains(where: { $0.hasPrefix("Backed up to Google Drive") }) {
            out.append("I2 the all-good Drive line without All good")
        }
        // I3: reassurance only when proven.
        func provenFine(_ names: [String]) -> Bool {
            fresh && !c.isChecking && !c.wakeCatchUpPending && all("monitor", .green) && names.allSatisfy { all($0, .green) }
        }
        let things: [(word: String, checks: [String])] = [
            ("saving", ["ingest", "stores"]), ("search", ["search", "embedding"]),
            ("backups", ["backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk"]), ("upkeep", []),
        ]
        for row in p.rows where row.kind == .everythingElse {
            let text = row.strings.joined(separator: " ").lowercased()
            let covered = text.hasPrefix("everything else") ? things : things.filter { text.contains($0.word) }
            if covered.isEmpty { out.append("I3 a reassurance that names nothing: \(text)") }
            for thing in covered where !provenFine(thing.checks) { out.append("I3 \(thing.word) called fine without proof") }
            if covered.contains(where: { $0.word == "backups" }), !c.backupsMeasured { out.append("I3 backups called fine while unmeasured") }
            if covered.contains(where: { $0.word == "saving" }), c.watcher?.alertReasons.isEmpty != true {
                out.append("I3 saving called fine beside a watcher alert")
            }
        }
        // I4: one rule.
        if p.count != needsRows { out.append("I4 count \(p.count) but \(needsRows) Needs-you rows") }
        if (p.icon == .marked) != (p.form == .needs) { out.append("I4 icon \(p.icon) with form \(p.form)") }
        if (p.form == .needs) != (p.count >= 1) { out.append("I4 form \(p.form) with count \(p.count)") }
        // I5: words.
        for row in p.rows where row.kind != .separator {
            if row.strings.isEmpty || row.strings.contains("") { out.append("I5 a \(row.kind) row with an empty string list or string") }
        }
        let every = texts + p.rows.compactMap(\.action?.label) + p.rows.compactMap(\.optionAlternate)
        if every.contains(where: { $0.contains("'") }) { out.append("I5 a straight apostrophe") }
        return out
    }
}
