import Foundation
import XCTest
@testable import BrainBar

/// The safety property of the M menu, enumerated rather than hunted for (phase 8 revision, 13 Nov). Every later
/// presentation PR runs under this test; one that adds a truth-table exception changes it on purpose and says so.
///
/// Enumerated: the 11 known checks in green, amber, red, unknown and absent, and one unknown name in amber, red,
/// unknown and absent (never green: a green unknown name says nothing). Each alone on five bases (plain, a live
/// upload, a lagging watcher under its 30-minute limit, a watcher alert `file_ingestion_failure`, a lagging watcher
/// past its limit) with nightly upkeep failures 0 and 3, and every pair of them on the plain base with upkeep 0; each
/// crossed with the dot (green, amber, red; never unknown), `isChecking` and the wake flag (off, on), the verdict
/// (fresh at awake age 0 and 900, stale at -1 and 901, missing) and backups measured or not. P2 crosses every input with
/// five profiles of its capture fields (job streaks, the Drive model, keyword coverage, a Drive copy with no age).
/// Not enumerated: watcher fields other than these alerts, other ages and `since` values, recovering, notes, and triples.
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
        // Copies under their red line and an embedding stall under its threshold, so P2's S06, S32 and F07 can fire.
        let (age, threshold): (Double?, Double?) = ["backup_offsite": (27660, 280800), "transcripts": (31620, 280800),
                                                     "embedding": (3960, 21600)][name] ?? (nil, nil)
        let since = name == "search" && setting == .amber ? "2026-11-12T12:45:00+02:00" : nil
        return Check(name: name, state: state, ageS: age, thresholdS: threshold, since: since, note: "", unknownTicks: 0, blind: false)
    }

    struct Input { var settings: [String: Setting], base: Int, upkeep: Int
        var profile = 0
    }

    static func capture(_ input: Input, dot: State, checking: Bool, wake: Bool, freshness: Freshness, measured: Bool) -> Capture {
        // Profiles 1 and 4: no Drive copy has ever verified (`backup_offsite` carries no age), so R14 and S07 are reachable.
        let checks = names.compactMap { check($0, input.settings[$0] ?? ($0 == "new_check" ? .absent : .green)) }.map {
            [1, 4].contains(input.profile) && $0.name == "backup_offsite"
                ? Check(name: $0.name, state: $0.state, thresholdS: $0.thresholdS, note: "", unknownTicks: 0, blind: false) : $0
        }
        var c = Capture(now: now, timeZone: TimeZone(secondsFromGMT: 7200)!)
        // Profiles 2 and 3: the verdict was evaluated a day before `now` and is still awake-fresh (it stayed fresh through a
        // sleep), so a copy age counted from `evaluatedAt` instead of `now` would understate (CodeRabbit, round 2).
        c.verdict = freshness == .missing ? nil : BrainBarHealthVerdict(dot: dot, checks: checks, notes: [],
                                                                         evaluatedAt: [2, 3].contains(input.profile) ? now.addingTimeInterval(-86_400) : now)
        c.verdictAwakeAgeS = [.fresh0: 0, .fresh900: 900, .staleNegative: -1, .stale901: 901, .missing: 0][freshness]!
        c.isChecking = checking
        c.wakeCatchUpPending = wake
        c.backupsMeasured = measured
        c.maintenanceNightFailures = input.upkeep
        // P2's fields: 0 none; 1 one database failure with a next run (S02), transcripts failing (D13), Drive never
        // connected, exact-word coverage; 2 two database failures (S06), transcripts failing, Drive invalid, coverage;
        // 3 one database failure with a next run, Drive invalid, no coverage figure; 4 two database failures, Drive invalid.
        c.backupDailyFailures = [0, 1, 2, 1, 2][input.profile]
        c.backupDailyNextRunAt = [1, 3].contains(input.profile) ? now.addingTimeInterval(52_200) : nil
        c.jsonlBackupFailures = [0, 2, 2, 0, 0][input.profile]
        c.driveAuthModel = [nil, .missing, .invalid, .invalid, .invalid][input.profile]
        c.keywordCoverage = [nil, 1, 1, nil, nil][input.profile]
        switch input.base {
        case 1: c.upload = .init(startedAt: now.addingTimeInterval(-1920), live: true)
        case 2, 4:
            c.watcher = .init(alertReasons: ["offset_lag"])
            c.lagEpisodeStart = now.addingTimeInterval(input.base == 2 ? -480 : -1860)
        case 3: c.watcher = .init(alertReasons: ["file_ingestion_failure"])
        default: break
        }
        return c
    }

    static func capture(_ settings: [String: Setting], base: Int, dot: State, checking: Bool, wake: Bool,
                        freshness: Freshness, measured: Bool) -> Capture {
        capture(Input(settings: settings, base: base, upkeep: 0), dot: dot, checking: checking, wake: wake,
                freshness: freshness, measured: measured)
    }

    /// Every check configuration enumerated: singles on every base and upkeep value, pairs on the plain base.
    static func inputs() -> [Input] {
        let nonGreen: [Setting] = [.amber, .red, .unknown, .absent]
        var singles: [[String: Setting]] = [[:]]
        for name in names { for s in nonGreen { singles.append([name: s]) } }
        var all = singles.flatMap { s in (0...4).flatMap { b in [0, 3].map { Input(settings: s, base: b, upkeep: $0) } } }
        for (i, a) in names.enumerated() {
            for b in names[(i + 1)...] {
                for sa in nonGreen { for sb in nonGreen { all.append(Input(settings: [a: sa, b: sb], base: 0, upkeep: 0)) } }
            }
        }
        return all.flatMap { input in (0...4).map { var i = input; i.profile = $0; return i } }
    }

    /// Each input's 120 captures, checked; I6 compares every flagged result with the same capture's flags-off result.
    static func check(_ input: Input) -> (outputs: Int, violations: [String], seen: Set<String>) {
        var outputs = 0
        var violations: [String] = []
        var seen = Set<String>()
        for dot in [State.green, .amber, .red] {
            for freshness in Freshness.allCases {
                for measured in [true, false] {
                    var calm: P?
                    for (checking, wake) in [(false, false), (true, false), (false, true), (true, true)] {
                        let c = capture(input, dot: dot, checking: checking, wake: wake, freshness: freshness, measured: measured)
                        let p = BrainBarMenuPresenter.present(c)
                        outputs += 1
                        for row in p.rows {
                            if let first = row.strings.first, let title = speaksFor.first(where: { first.hasPrefix($0.prefix) }) { seen.insert(title.prefix) }
                            seen.formUnion(row.strings.flatMap(sentences).filter { ["Everything else is fine.", stillSaved, exactWords, nothingInDrive].contains($0) })
                            if row.strings.contains(where: { $0.hasPrefix("Backups can’t upload") && $0.contains(" Your newest backup in Google Drive is ") }) {
                                seen.insert("R15 in the Drive row")
                            }
                        }
                        var found = Self.violations(c, p)
                        if let calm, [.fresh0, .fresh900].contains(freshness), calm.form == .needs,
                           p.form != .needs || Self.facts(p) != Self.facts(calm) {
                            found.append("I6 a checking flag made a fresh verdict calmer")
                        }
                        if calm == nil { calm = p }
                        violations += found.map { "\($0) | \(input) dot \(dot) checking \(checking) wake \(wake) \(freshness) measured \(measured)" }
                    }
                }
            }
        }
        return (outputs, violations, seen)
    }

    final class Tally: @unchecked Sendable {
        let lock = NSLock()
        var outputs = 0, violations: [String] = [], seen = Set<String>()
        func add(_ r: (outputs: Int, violations: [String], seen: Set<String>)) {
            lock.withLock { outputs += r.outputs; violations += r.violations; seen.formUnion(r.seen) }
        }
    }

    /// The presenter is pure, so the inputs are checked on every core; the result does not depend on the order.
    func testInvariantsHoldOnEveryEnumeratedInput() {
        let inputs = Self.inputs(), tally = Tally(), start = Date()
        DispatchQueue.concurrentPerform(iterations: inputs.count) { tally.add(Self.check(inputs[$0])) }
        print("BrainBarMenuInvariantTests: \(tally.outputs) outputs checked in \(String(format: "%.1f", Date().timeIntervalSince(start))) s")
        XCTAssertEqual(tally.outputs, inputs.count * 120)
        XCTAssertGreaterThan(tally.outputs, 900_000)
        // Not vacuous: every worded row and every reassuring sentence is reached somewhere in the enumeration.
        XCTAssertEqual(Set(Self.speaksFor.map(\.prefix) + ["Everything else is fine.", Self.stillSaved, Self.exactWords, Self.nothingInDrive,
                                                         "R15 in the Drive row"]).subtracting(tally.seen), [])
        let first = Array(tally.violations.sorted().prefix(25))
        let byInvariant = Dictionary(grouping: tally.violations) { String($0.prefix(2)) }.mapValues(\.count).sorted { $0.key < $1.key }
        XCTAssertEqual(first, [], "\(tally.violations.count) violations \(byInvariant):\n" + first.joined(separator: "\n"))
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
        // I3 by sentence (P2): reassurances inside Needs-you rows, beside them and in status lines.
        func needsForm(_ rows: [P.Row]) -> P {
            P(form: .needs, headline: "1 thing needs you", count: 1, icon: .marked,
              rows: [P.Row(kind: .headline, strings: ["1 thing needs you"], mark: .marked), P.Row(kind: .separator)] + rows)
        }
        let handoff = ["BrainBar can’t fix this itself.", "Give the report to an agent."]
        let report = P.Action(kind: .copyReport, label: "Copy report for an agent")
        let drive = P.Row(kind: .needsYou, strings: ["Google Drive is disconnected", "Backups can’t upload until you reconnect it."],
                          action: P.Action(kind: .reconnectDrive, label: "Reconnect Google Drive"), mark: .alertCircle)
        let fine = P.Row(kind: .everythingElse, strings: ["Everything else is fine."], mark: .checkmark)
        let driveAndSearchAmber = base { $0.verdict = BrainBarHealthVerdict(dot: .green, checks: $0.verdict!.checks.filter { !["search", "drive_auth"].contains($0.name) }
                                                                           + [Self.check("search", .amber)!, Self.check("drive_auth", .red)!], notes: [], evaluatedAt: Self.now) }
        let upkeepUnmeasured = base { $0.maintenanceNightFailures = 3; $0.backupsMeasured = false }
        let embeddingRed = base { $0.verdict = BrainBarHealthVerdict(dot: .green, checks: $0.verdict!.checks.filter { $0.name != "embedding" }
                                                                    + [Self.check("embedding", .red)!], notes: [], evaluatedAt: Self.now) }
        let upkeepRow = P.Row(kind: .needsYou, strings: ["Nightly upkeep failed 3 nights in a row", Self.stillSaved] + handoff, action: report, mark: .alertCircle)
        let meaningRow = P.Row(kind: .needsYou, strings: ["Search by meaning isn’t catching up",
                                                          "BrainBar hasn’t seen the memories waiting for it go down. " + Self.exactWords] + handoff,
                               action: report, mark: .alertCircle)
        let generic = P.Row(kind: .needsYou, strings: ["Something BrainBar checks is failing."] + handoff, action: report, mark: .alertCircle)
        // Round 1 of #1787: the still-saved sentence needs a watcher file that proves saving (Codex P2-F2), and the I8 claims.
        func upkeepWith(_ w: Capture.Watcher) -> Capture { base { $0.maintenanceNightFailures = 3; $0.watcher = w } }
        let localNoRecord = base { $0.verdict = BrainBarHealthVerdict(dot: .green, checks: $0.verdict!.checks.filter { $0.name != "backup_local" }
                                                                     + [Self.check("backup_local", .red)!], notes: [], evaluatedAt: Self.now) }
        let s07 = P.Row(kind: .needsYou, strings: ["No verified backup on record"] + handoff, action: report, mark: .alertCircle)
        let driveMissing = base {
            $0.verdict = BrainBarHealthVerdict(dot: .green, checks: $0.verdict!.checks.filter { $0.name != "drive_auth" } + [Self.check("drive_auth", .red)!],
                                               notes: [], evaluatedAt: Self.now)
            $0.driveAuthModel = .missing
        }
        let staleDrive = base {
            $0.verdict = BrainBarHealthVerdict(dot: .green, checks: $0.verdict!.checks.filter { $0.name != "drive_auth" } + [Self.check("drive_auth", .red)!],
                                               notes: [], evaluatedAt: Self.now.addingTimeInterval(-86_400))
        }
        let driveAged = P.Row(kind: .needsYou, strings: ["Google Drive is disconnected",
                                                         "Backups can’t upload until you reconnect it. Your newest backup in Google Drive is 8 hours old."],
                              action: P.Action(kind: .reconnectDrive, label: "Reconnect Google Drive"), mark: .alertCircle)
        let connect = P.Row(kind: .needsYou, strings: ["Google Drive isn’t connected", "Backups can’t upload until you connect it. " + Self.nothingInDrive],
                            action: P.Action(kind: .connectDrive, label: "Connect Google Drive"), mark: .alertCircle)
        let cases: [(String, Capture, P)] = [
            ("I1: calm over a red check", red, allGood),
            ("I2: All good while backups are unmeasured", unmeasured, allGood),
            ("I2: the all-good Drive line under another headline", unmeasured, strayDriveLine),
            ("I3: search called fine while amber", searchAmber, falseFine),
            ("I4: count and icon disagree with the rows", red, miscounted),
            ("I5: empty strings and a straight apostrophe", red, badWords),
            ("I3: Everything else is fine. beside the Drive row while search is amber", driveAndSearchAmber, needsForm([drive, fine])),
            ("I3: U03’s still-saved sentence while backups are unmeasured", upkeepUnmeasured, needsForm([upkeepRow])),
            ("I3: the exact-words sentence inside F07’s line with no coverage figure", embeddingRed, needsForm([meaningRow])),
            ("I3: a reassurance sitting in a status line", driveAndSearchAmber,
             needsForm([drive, P.Row(kind: .statusLine, strings: ["Search is fine."], state: .cantTell)])),
            ("I3: Everything else is fine. beside a generic row stands for all four", red, needsForm([generic, fine])),
            ("I3: still saved while the watcher file is unreadable", upkeepWith(.init(readable: false)), needsForm([upkeepRow])),
            ("I3: still saved while the watcher file is stale", upkeepWith(.init(fresh: false)), needsForm([upkeepRow])),
            ("I3: still saved while the database probe fails", upkeepWith(.init(dbProbeFailed: true)), needsForm([upkeepRow])),
            ("I8: S07’s title while the Drive copy has an age (R21)", localNoRecord, needsForm([s07])),
            ("I8: R14’s clause while the Drive copy has an age", driveMissing, needsForm([connect])),
            // CodeRabbit round 2 (Codex R2-F2): the reassurance is proven against the rows after the S07 correction.
            ("I3 after I8: Everything else is fine. beside a false S07 title", localNoRecord, needsForm([s07, fine])),
            ("I8: an hours-old Drive copy counted from evaluatedAt, a day before now", staleDrive, needsForm([driveAged])),
            ("I7: calm over upkeep failing, verdict stale", base { $0.maintenanceNightFailures = 3; $0.verdictAwakeAgeS = 901 }, allGood),
            ("I7: calm over a watcher alert, no verdict", base { $0.verdict = nil; $0.watcher = .init(alertReasons: ["file_ingestion_failure"]) }, allGood),
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

    /// R24: a failed database probe makes only `coverage_drop` unprovable; R09 and X10 still count (review round 1, N4).
    func testFailedDatabaseProbeHidesOnlyCoverageDrop() {
        func present(_ reasons: [String]) -> P {
            var c = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
            c.watcher = .init(alertReasons: reasons, dbProbeFailed: true)
            return BrainBarMenuPresenter.present(c)
        }
        XCTAssertEqual(present(["coverage_drop"]).count, 0)
        XCTAssertEqual(present(["file_ingestion_failure"]).count, 1)
        XCTAssertEqual(present(["a_reason_nobody_knows"]).count, 1)
        XCTAssertEqual(present(["coverage_drop", "file_ingestion_failure"]).count, 1)
        // R10: `quarantined_record` is Fine, and the oracle agrees (review round 1, Codex N3, Claude N3).
        var quarantined = Self.capture(["search": .amber], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
        quarantined.watcher = .init(alertReasons: ["quarantined_record"])
        let p = BrainBarMenuPresenter.present(quarantined)
        XCTAssertTrue(p.rows.contains { $0.kind == .everythingElse && $0.strings == ["Saving, backups and upkeep are fine."] })
        XCTAssertEqual(Self.violations(quarantined, p), [])
    }

    /// `present` is total over its capture type: numbers that are negative, zero, NaN or infinite, and dates at the
    /// edges of time, never trap, and the result still holds the invariants (review round 1, Codex N1, Claude N2).
    func testPresentIsTotalOverOddNumbersAndDates() {
        let odd: [Double] = [-1, 0, .nan, .infinity, -.infinity, .greatestFiniteMagnitude]
        let dates = [Date.distantPast, .distantFuture, Date(timeIntervalSinceReferenceDate: .nan),
                     Date(timeIntervalSinceReferenceDate: .infinity), Date(timeIntervalSinceReferenceDate: -.infinity)]
        var captures: [Capture] = []
        for x in odd {
            var c = Self.capture(["backup_offsite": .amber], base: 1, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
            c.offsiteRecovering = .init(progress: 0.4, eta: nil, startedAt: Self.now.addingTimeInterval(-60), updatedAt: Self.now, cadenceS: x)
            captures.append(c)
            c.offsiteRecovering?.cadenceS = 30
            c.offsiteRecovering?.progress = x
            captures.append(c)
            var aged = Self.capture([:], base: 0, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
            aged.verdict = BrainBarHealthVerdict(dot: .green, checks: aged.verdict!.checks.map {
                ["backup_offsite", "transcripts"].contains($0.name) ? Check(name: $0.name, state: .green, ageS: x, note: "") : $0
            }, notes: [], evaluatedAt: Self.now)
            captures.append(aged)
            aged.verdictAwakeAgeS = x
            aged.maintenanceNightFailures = x.isFinite ? Int(max(min(x, 9), -9)) : 0
            captures.append(aged)
        }
        for x in odd {  // P2: copy, stall and coverage numbers, and streaks, through every worded row's formatter
            for (drive, base, failures) in [(Setting.red, 0, 1), (.green, 0, 2), (.green, 1, 0)] {
                var c = Self.capture(["drive_auth": drive], base: base, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
                c.verdict = BrainBarHealthVerdict(dot: .green, checks: c.verdict!.checks.map {
                    ["backup_local", "backup_offsite", "embedding"].contains($0.name)
                        ? Check(name: $0.name, state: $0.name == "backup_offsite" ? .red : .amber, ageS: x, thresholdS: x, note: "") : $0
                }, notes: [], evaluatedAt: Self.now)
                c.keywordCoverage = x
                c.backupDailyFailures = failures
                c.backupDailyNextRunAt = Self.now.addingTimeInterval(x.isFinite ? max(min(x, 1e9), -1e9) : 0)
                captures.append(c)
                c.backupDailyFailures = x.isFinite ? Int(max(min(x, 9), -9)) : .max
                captures.append(c)
            }
        }
        for d in dates {
            var c = Self.capture(["drive_auth": .red, "backup_offsite": .red], base: 0, dot: .green, checking: false, wake: false,
                                 freshness: .fresh0, measured: true)
            c.verdict = BrainBarHealthVerdict(dot: .green, checks: c.verdict!.checks, notes: [], evaluatedAt: d)
            c.backupDailyNextRunAt = d
            captures.append(c)
            // A copy dated after now is never given an age (local review, CodeRabbit, before the PR opened).
            if !(d.addingTimeInterval(-27660) <= Self.now) {
                XCTAssertFalse(BrainBarMenuPresenter.present(c).rows.flatMap(\.strings).contains { $0.contains(" old.") }, "\(d)")
            }
        }
        for d in dates {
            var c = Self.capture(["backup_offsite": .amber, "search": .amber], base: 1, dot: .green, checking: false, wake: false,
                                 freshness: .fresh0, measured: true)
            c.upload = .init(startedAt: d, live: true)
            captures.append(c)
            c.upload = nil
            c.offsiteRecovering = .init(progress: 0.4, eta: d, startedAt: d, updatedAt: d, cadenceS: 30)
            captures.append(c)
            var lag = Self.capture([:], base: 2, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
            lag.lagEpisodeStart = d
            lag.lastKnownBackupAt = d
            captures.append(lag)
            lag.verdict = BrainBarHealthVerdict(dot: .green, checks: lag.verdict!.checks, notes: [], evaluatedAt: d)
            captures.append(lag)
            lag.now = d
            captures.append(lag)
            lag.verdict = nil
            lag.wakeCatchUpPending = true
            captures.append(lag)
        }
        for c in captures { XCTAssertEqual(Self.violations(c, BrainBarMenuPresenter.present(c)), []) }
    }

    // MARK: the invariants, written from the brief, not from the presenter

    /// The reassuring sentences the presenter may emit beyond "All good" and "… fine." (each with its own proof below).
    static let stillSaved = "Your memory is still being saved and backed up, as of the last checks."
    static let exactWords = "Agents can still find every memory by exact words."
    /// I8: sentences that state a fact about the copies (A1's S07 title, the truth table's S07 sentence, R14's clause).
    static let noRecord = ["No verified backup on record", "BrainBar has no record of a verified backup of your memory."]
    static let nothingInDrive = "Nothing is backed up to Google Drive yet."

    static func sentences(_ string: String) -> [String] {
        let parts = string.components(separatedBy: ". ")
        return parts.indices.map { $0 < parts.count - 1 ? parts[$0] + "." : parts[$0] }
    }

    /// I6 compares the Needs-you rows' facts: a flag may take a reassurance away (it proves nothing), never a row or a fact.
    static func facts(_ p: P) -> [P.Row] {
        p.rows.filter { $0.kind == .needsYou }.map { row in
            var row = row
            row.strings = row.strings.flatMap(sentences).filter { ![stillSaved, exactWords].contains($0) }
            return row
        }
    }

    /// The thing each worded row speaks for, read from the cases file's titles and status lines. A row not listed here
    /// (R17's generic rows) speaks for no one thing, so "Everything else is fine." beside it covers all four.
    static let speaksFor: [(prefix: String, word: String)] = [
        ("Google Drive is disconnected", "backups"), ("Google Drive isn’t connected", "backups"), ("The last ", "backups"),
        ("No verified backup on record", "backups"), ("The newest checked backup is damaged", "backups"),
        ("Your newest backup in Google Drive", "backups"), ("A backup to Google Drive started", "backups"),
        ("Conversation backups can’t upload", "backups"), ("To Google Drive. Started", "backups"),
        ("Search by meaning isn’t catching up", "search"), ("Can’t confirm agents can search", "search"),
        ("Catching up on new conversations", "saving"), ("Nightly upkeep failed", "upkeep"),
    ]

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
        // I3, by sentence: every reassuring sentence, wherever it sits, only when everything it covers is proven.
        func provenFine(_ names: [String]) -> Bool {
            fresh && !c.isChecking && !c.wakeCatchUpPending && all("monitor", .green) && names.allSatisfy { all($0, .green) }
        }
        let things: [(word: String, checks: [String])] = [
            ("saving", ["ingest", "stores"]), ("search", ["search", "embedding"]),
            ("backups", ["backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk"]), ("upkeep", []),
        ]
        /// Every check of this name satisfies `ok`, and there is at least one.
        func every(_ name: String, _ ok: (Check) -> Bool) -> Bool {
            guard let v, v.checks.contains(where: { $0.name == name }) else { return false }
            return v.checks.filter { $0.name == name }.allSatisfy(ok)
        }
        func requireFine(_ words: [String], _ said: String) {
            for thing in things where words.contains(thing.word) && !provenFine(thing.checks) { out.append("I3 \(thing.word) called fine without proof: \(said)") }
            if words.contains("backups"), !c.backupsMeasured { out.append("I3 backups called fine while unmeasured: \(said)") }
            // Saving needs a watcher file that can prove it: readable, fresh, its database probe working, and no alert
            // except `quarantined_record` (R10: Fine).
            if words.contains("saving"), c.watcher.map({ $0.readable && $0.fresh && !$0.dbProbeFailed
                && $0.alertReasons.allSatisfy { $0 == "quarantined_record" } }) != true {
                out.append("I3 saving called fine without a clean watcher file: \(said)")
            }
            if words.contains("backups"), v?.checks.contains(where: { ["backup_local", "backup_offsite", "transcripts"].contains($0.name) && ($0.ageS ?? 0) < 0 }) == true {
                out.append("I3 backups called fine beside a future-dated copy: \(said)")
            }
            if words.contains("upkeep"), c.maintenanceNightFailures >= 3 { out.append("I3 upkeep called fine while failing: \(said)") }
        }
        /// "Everything else": the four things minus those a drawn row speaks for; all four beside a row no title places.
        func everythingElse() -> [String] {
            var left = things.map(\.word)
            for row in p.rows where [.needsYou, .statusLine, .factLine, .headlineDetail].contains(row.kind) {
                guard let first = row.strings.first, let word = speaksFor.first(where: { first.hasPrefix($0.prefix) })?.word else { return things.map(\.word) }
                left.removeAll { $0 == word }
            }
            return left
        }
        for row in p.rows {
            for sentence in row.strings.flatMap(sentences) {
                let text = sentence.lowercased()
                if text.hasSuffix(" fine.") {
                    let covered = text.hasPrefix("everything else") ? everythingElse() : things.map(\.word).filter { text.contains($0) }
                    if covered.isEmpty, !text.hasPrefix("everything else") { out.append("I3 a reassurance that names nothing: \(text)") }
                    requireFine(covered, sentence)
                } else if sentence == stillSaved {
                    requireFine(["saving", "backups"], sentence)
                } else if sentence == exactWords {
                    if c.keywordCoverage != 1 || !provenFine(["search"]) { out.append("I3 exact-word search called working without proof") }
                } else if noRecord.contains(sentence) {  // I8
                    if !(every("backup_local") { $0.state == .red && $0.ageS == nil } && every("backup_offsite") { $0.ageS == nil }) {
                        out.append("I8 no verified backup claimed while a copy has a record: \(sentence)")
                    }
                } else if sentence.hasSuffix(" hour old.") || sentence.hasSuffix(" hours old.") {  // I8: an age counted to now
                    let named = sentence.contains("in Google Drive") ? ["backup_offsite"] : ["backup_local", "backup_offsite"]
                    let ages = (v?.checks ?? []).filter { named.contains($0.name) }.compactMap(\.ageS).filter { $0.isFinite && $0 >= 0 }
                    let atNow = ages.min().map { $0 + c.now.timeIntervalSince(v!.evaluatedAt) }
                    if !(atNow.map { $0 >= 0 && $0 < 86_400 } ?? false) { out.append("I8 a copy age in hours that is not under a day at now: \(sentence)") }
                } else if sentence == nothingInDrive {  // I8
                    if !(every("backup_offsite") { $0.ageS == nil } && c.backupsMeasured) { out.append("I8 nothing in Drive claimed while unproven") }
                } else if text.contains("still"), !text.hasPrefix("still checking"), !text.contains("is still running") {
                    out.append("I3 an unlisted reassurance: \(sentence)")
                }
            }
        }
        // I4: one rule.
        if p.count != needsRows { out.append("I4 count \(p.count) but \(needsRows) Needs-you rows") }
        if (p.icon == .marked) != (p.form == .needs) { out.append("I4 icon \(p.icon) with form \(p.form)") }
        if (p.form == .needs) != (p.count >= 1) { out.append("I4 form \(p.form) with count \(p.count)") }
        // I7 (R20): a Needs-you fact the app reads without the producer counts, whatever the verdict or the flags.
        if let w = c.watcher, w.readable, w.fresh {
            let alert = w.alertReasons.contains { !["offset_lag", "quarantined_record"].contains($0) && !($0 == "coverage_drop" && w.dbProbeFailed) }
            let lagPast = w.alertReasons.contains("offset_lag") && c.lagEpisodeStart.map { c.now.timeIntervalSince($0) > 1800 } == true
            if (alert || lagPast), p.form != .needs { out.append("I7 a readable watcher fact did not count") }
        }
        if c.maintenanceNightFailures >= 3, p.form != .needs { out.append("I7 nightly upkeep failing did not count") }
        // I5: words.
        for row in p.rows where row.kind != .separator {
            if row.strings.isEmpty || row.strings.contains("") { out.append("I5 a \(row.kind) row with an empty string list or string") }
        }
        let every = texts + p.rows.compactMap(\.action?.label) + p.rows.compactMap(\.optionAlternate)
        if every.contains(where: { $0.contains("'") }) { out.append("I5 a straight apostrophe") }
        return out
    }
}
