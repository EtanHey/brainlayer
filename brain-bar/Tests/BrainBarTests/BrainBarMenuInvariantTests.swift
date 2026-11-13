import Foundation
import XCTest
@testable import BrainBar

/// The safety property of the M menu, enumerated rather than hunted for (phase 8 revision, 13 Nov). Every later
/// presentation PR runs under this test; one that adds a truth-table exception changes it on purpose and says so.
///
/// Enumerated: the 11 known checks in green, amber, red, unknown and absent, and one unknown name in amber, red,
/// unknown and absent (never green: a green unknown name says nothing). Each alone on six bases (plain, a live
/// upload, a lagging watcher under its 30-minute limit, a watcher alert `file_ingestion_failure`, a lagging watcher
/// past its limit, and (P2b) a lagging watcher beside a live upload that started the day before) with nightly upkeep failures 0 and 3, and every pair of them on the plain base with upkeep 0; each
/// crossed with the dot (green, amber, red; never unknown), `isChecking` and the wake flag (off, on), the verdict
/// (fresh at awake age 0 and 900, stale at -1 and 901, missing) and backups measured or not. P2 crosses every input with
/// five profiles of its capture fields (job streaks, the Drive model, keyword coverage, a Drive copy with no age).
/// Not enumerated: watcher fields other than these alerts, other ages and `since` values, recovering, notes, and triples.
/// P2b: every sentence shown must be a registered one (`BrainBarMenuPresenter.Say.all`), and every claim and reassurance
/// is checked against this file's own condition for its id (`oracle`), never against the registry's.
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
        // Profiles 1 and 4: no Drive database copy has ever verified (`backup_offsite` carries no age); in 4 no conversation
        // backup has either (round 1 of #1788, R25), so R14 and S07 are reachable there.
        let checks = names.compactMap { check($0, input.settings[$0] ?? ($0 == "new_check" ? .absent : .green)) }.map {
            ([1, 4].contains(input.profile) && $0.name == "backup_offsite") || (input.profile == 4 && $0.name == "transcripts")
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
        // 3 one database failure with a next run, Drive invalid, no coverage figure; 4 two database failures, Drive
        // never connected (round 1 of #1788: R14 needs the `missing` model and no Drive record of either kind).
        c.backupDailyFailures = [0, 1, 2, 1, 2][input.profile]
        c.backupDailyNextRunAt = [1, 3].contains(input.profile) ? now.addingTimeInterval(52_200) : nil
        c.jsonlBackupFailures = [0, 2, 2, 0, 0][input.profile]
        c.driveAuthModel = [nil, .missing, .invalid, .invalid, .missing][input.profile]
        c.keywordCoverage = [nil, 1, 1, nil, nil][input.profile]
        if [1, 5].contains(input.base) { c.upload = .init(startedAt: now.addingTimeInterval(input.base == 1 ? -1920 : -93_600), live: true) }
        if [2, 4, 5].contains(input.base) {
            c.watcher = .init(alertReasons: ["offset_lag"])
            c.lagEpisodeStart = now.addingTimeInterval(input.base == 4 ? -1860 : -480)
        }
        if input.base == 3 { c.watcher = .init(alertReasons: ["file_ingestion_failure"]) }
        // P2b: on the two live-upload bases, profiles 2 to 4 also carry an on-schedule `recovering` with progress, its ETA today
        // (2), none (3) or tomorrow (4): S03's and S32's progress sentences, and R27.
        if [1, 5].contains(input.base), [2, 3, 4].contains(input.profile) {
            c.offsiteRecovering = .init(progress: 0.4, eta: [2: now.addingTimeInterval(1800), 4: now.addingTimeInterval(86_400)][input.profile],
                                        startedAt: now.addingTimeInterval(input.base == 1 ? -1920 : -93_600), updatedAt: now, cadenceS: 30)
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
        var all = singles.flatMap { s in (0...5).flatMap { b in [0, 3].map { Input(settings: s, base: b, upkeep: $0) } } }
        for (i, a) in names.enumerated() {
            for b in names[(i + 1)...] {
                for sa in nonGreen { for sb in nonGreen { all.append(Input(settings: [a: sa, b: sb], base: 0, upkeep: 0)) } }
            }
        }
        return all.flatMap { input in (0...4).map { var i = input; i.profile = $0; return i } }
    }

    /// Each input's 120 captures, checked; I6 compares every flagged result with the same capture's flags-off result.
    static func check(_ input: Input) -> (outputs: Int, violations: [String], seen: [String: [String]]) {
        var outputs = 0
        var violations: [String] = []
        var seen: [String: [String]] = [:]
        for dot in [State.green, .amber, .red] {
            for freshness in Freshness.allCases {
                for measured in [true, false] {
                    var calm: P?
                    for (checking, wake) in [(false, false), (true, false), (false, true), (true, true)] {
                        let c = capture(input, dot: dot, checking: checking, wake: wake, freshness: freshness, measured: measured)
                        let p = BrainBarMenuPresenter.present(c)
                        outputs += 1
                        if p.rows.contains(where: { $0.strings.contains { $0.hasPrefix("Backups can’t upload") && $0.contains(" Your newest backup in Google Drive is ") } }) {
                            seen["R15 in the Drive row"] = []
                        }
                        var (found, shown) = Self.audit(c, p)
                        seen.merge(shown) { old, _ in old }
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
        var outputs = 0, violations: [String] = [], seen: [String: [String]] = [:]
        func add(_ r: (outputs: Int, violations: [String], seen: [String: [String]])) {
            lock.withLock { outputs += r.outputs; violations += r.violations; seen.merge(r.seen) { old, _ in old } }
        }
    }

    /// The presenter is pure, so the inputs are checked on every core; the result does not depend on the order.
    func testInvariantsHoldOnEveryEnumeratedInput() throws {
        let inputs = Self.inputs(), tally = Tally(), start = Date()
        DispatchQueue.concurrentPerform(iterations: inputs.count) { tally.add(Self.check(inputs[$0])) }
        print("BrainBarMenuInvariantTests: \(tally.outputs) outputs checked in \(String(format: "%.1f", Date().timeIntervalSince(start))) s")
        XCTAssertEqual(tally.outputs, inputs.count * 120)
        XCTAssertGreaterThan(tally.outputs, 900_000)
        // The seventeen contract cases count for reach, and hold the invariants too.
        for c in try BrainBarMenuPresentationTests.loadCases().map({ try BrainBarMenuPresentationTests.capture($0["inputs"] as! [String: Any]) }) {
            let (found, shown) = Self.audit(c, BrainBarMenuPresenter.present(c))
            tally.add((0, found, shown))
        }
        // Not vacuous: every registry entry is shown somewhere, except those listed with their reason.
        XCTAssertEqual(Set(BrainBarMenuPresenter.Say.all.map(\.id)).subtracting(tally.seen.keys), Self.unreached)
        XCTAssertNotNil(tally.seen["R15 in the Drive row"])
        // Nor the other way: every claim and reassurance, as it was shown, is untrue on some enumerated capture, and shown
        // there it is caught by the oracle and removed by the guard (each condition exercised as a mutation).
        var unfalsified = Set(Self.oracle.keys).intersection(tally.seen.keys)
        print("BrainBarMenuInvariantTests: \(unfalsified.count) of \(Self.oracle.count) conditions shown, each to be falsified")
        func caught(_ id: String, _ c: Capture, _ p: P) -> Bool { Self.violations(c, p).contains { $0.contains(" \(id) shown while untrue") } }
        search: for input in inputs {
            for dot in [State.green, .amber, .red] { for freshness in Freshness.allCases { for measured in [true, false] { for flag in [false, true] {
                let c = Self.capture(input, dot: dot, checking: flag, wake: flag, freshness: freshness, measured: measured)
                let f = Facts(c: c, p: BrainBarMenuPresenter.present(c))
                unfalsified = unfalsified.filter { id in
                    guard !Self.oracle[id]!(f, tally.seen[id]!) else { return true }
                    let sentence = BrainBarMenuPresenter.Say.all.first { $0.id == id }!.fill(tally.seen[id]!)
                    var shown = f.p, alternate = f.p
                    shown.rows.append(P.Row(kind: .standardItem, strings: [sentence]))
                    alternate.rows[alternate.rows.count - 1].optionAlternate = sentence  // the Quit row's Option alternate (B5)
                    for bad in [shown, alternate] {
                        XCTAssertTrue(caught(id, c, bad), "the oracle misses \(id) shown falsely")
                        XCTAssertFalse(caught(id, c, BrainBarMenuPresenter.finalGuard(c, bad)), "the guard keeps \(id) shown falsely")
                    }
                    return false
                }
                if unfalsified.isEmpty { break search }
            } } } }
        }
        XCTAssertEqual(unfalsified, [])
        let first = Array(tally.violations.sorted().prefix(25))
        let byInvariant = Dictionary(grouping: tally.violations) { String($0.prefix(2)) }.mapValues(\.count).sorted { $0.key < $1.key }
        XCTAssertEqual(first, [], "\(tally.violations.count) violations \(byInvariant):\n" + first.joined(separator: "\n"))
    }

    /// Registry entries no enumerated capture or case shows, each with its reason.
    static let unreached: Set<String> = [
        "action.storage",  // the Open Storage Settings action exists for P3's disk rows; no row takes it yet
        "C.few",  // "Catching up on a few things": the two Behind rows (R02, S03) each carry their own headline
    ]

    /// The two keys: a claim or reassurance in the registry has a condition here, and every condition here has an entry
    /// there. A neutral entry carries no condition, and none says "fine" or "still" (the words of a reassurance).
    func testEveryClaimAndReassuranceHasAnOracleCondition() {
        let all = BrainBarMenuPresenter.Say.all
        XCTAssertEqual(Set(all.filter { $0.kind != .neutral }.map(\.id)), Set(Self.oracle.keys))
        XCTAssertEqual(all.count, Set(all.map(\.id)).count, "ids are unique")
        XCTAssertEqual(all.filter { ($0.kind == .neutral) != ($0.holds == nil) }.map(\.id), [])
        XCTAssertEqual(all.filter { $0.kind == .neutral && ($0.template.hasSuffix(" fine.") || $0.template.contains(" still ")) }.map(\.id), [])
        XCTAssertEqual(all.filter { $0.template.contains("'") }.map(\.id), [], "I5: no registered words carry a straight apostrophe")
    }

    /// No user-visible words are written in the presenter outside the registry: every string literal left in
    /// BrainBarMenuPresenter.swift is a check or alert name (snake_case) or letter-free punctuation. Comments are skipped.
    func testThePresenterWritesNoWordsOutsideTheRegistry() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBar/BrainBarMenuPresenter.swift")
        let code = try String(contentsOf: url, encoding: .utf8).components(separatedBy: "\n").map { $0.components(separatedBy: "//")[0] }.joined(separator: "\n")
        let literal = try NSRegularExpression(pattern: #""((?:[^"\\\n]|\\.)*)""#)
        let free = literal.matches(in: code, range: NSRange(code.startIndex..., in: code)).map { String(code[Range($0.range(at: 1), in: code)!]) }
            .filter { $0.contains(where: \.isLetter) && $0.range(of: "^[a-z_]+$", options: .regularExpression) == nil }
        XCTAssertGreaterThan(code.count, 10_000)
        XCTAssertEqual(free, [])
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
                                P.Row(kind: .needsYou, strings: ["Something BrainBar checks needs attention."], mark: .alertCircle)])
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
                                                          "BrainBar hasn’t seen the memories waiting for it go down in over an hour. " + Self.exactWords] + handoff,
                               action: report, mark: .alertCircle)
        let generic = P.Row(kind: .needsYou, strings: ["Something BrainBar checks needs attention."] + handoff, action: report, mark: .alertCircle)
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
        /// The real presentation of the plain capture, its Quit row edited.
        func standing(_ edit: (inout P.Row) -> Void) -> P {
            var p = BrainBarMenuPresenter.present(base())
            edit(&p.rows[p.rows.count - 1])
            return p
        }
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
            // P2b (Codex R2-F1): a claim on no list before, and a sentence no entry makes.
            ("I8: U03’s title with no failed nights", base(), needsForm([P.Row(kind: .needsYou, strings: ["Nightly upkeep failed 3 nights in a row"] + handoff,
                                                                         action: report, mark: .alertCircle)])),
            ("I8: the Drive row with Drive green", base(), needsForm([drive])),
            ("I8: a Behind headline with nothing running", base(), P(form: .calm, headline: "Backing up now", count: 0, icon: .resting,
                                                                    rows: [P.Row(kind: .headline, strings: ["Backing up now"], mark: .busy)])),
            ("CW: a sentence no entry makes", red, needsForm([P.Row(kind: .needsYou, strings: ["Your backups are perfectly safe."] + handoff,
                                                                   action: report, mark: .alertCircle)])),
            // Round 1 of #1788 (Codex B1, Claude N8): an option alternate and a key equivalent hold only their one permitted value.
            ("B5: a false option alternate", base(), standing { $0.optionAlternate = "Backing up now" }),
            ("B5: a stray key equivalent", base(), standing { $0.keyEquivalent = "⌘Z" }),
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

    /// Round 1 of #1788: three sentences the function showed falsely (Claude B1 with N5 and ruling R25, B2 with R27, B3 with
    /// N11), and D13's word under the never-connected row (N6, R26).
    func testP2bRoundOneWrongOutputs() throws {
        func capture(_ settings: [String: Setting], profile: Int) -> Capture {
            var input = Input(settings: settings, base: 0, upkeep: 0)
            input.profile = profile
            return Self.capture(input, dot: .green, checking: false, wake: false, freshness: .fresh0, measured: true)
        }
        func sentences(_ c: Capture) -> [String] { BrainBarMenuPresenter.present(c).rows.flatMap(\.strings).flatMap(Self.sentences) }
        func detail(_ c: Capture) -> [String]? { BrainBarMenuPresenter.present(c).rows.first { $0.kind == .headlineDetail }?.strings }
        func adding(_ c: Capture, _ check: Check) -> Capture {
            var c = c
            c.verdict = BrainBarHealthVerdict(dot: .green, checks: c.verdict!.checks + [check], notes: [], evaluatedAt: c.verdict!.evaluatedAt)
            return c
        }
        // A1 (R25): the conversation backup in Drive is verified (transcripts carries an age), so neither clause is true.
        XCTAssertFalse(sentences(capture(["backup_offsite": .red, "drive_auth": .red], profile: 1)).contains("Nothing is backed up to Google Drive yet."))
        XCTAssertFalse(sentences(capture(["backup_local": .red], profile: 1)).contains("No verified backup on record"))
        // A2 (R27): an ETA tomorrow at 10:00 is not said as "about 10:00".
        let cases = try BrainBarMenuPresentationTests.loadCases()
        var seam = try BrainBarMenuPresentationTests.capture(cases.first { $0["id"] as? String == "S2-after-seam" }!["inputs"] as! [String: Any])
        seam.offsiteRecovering?.eta = seam.now.addingTimeInterval(79_200)
        XCTAssertEqual(detail(seam), ["To Google Drive. Started at 11:28. 40% done."])
        // A3: a second Drive copy, 25 hours old, is the older copy (U10); a second `search` check unconfirmed for two hours.
        let duplicateDrive = adding(capture([:], profile: 0), Check(name: "backup_offsite", state: .green, ageS: 90_000, thresholdS: 280_800, note: ""))
        XCTAssertEqual(detail(duplicateDrive), ["Backed up to Google Drive Nov 11 at 11:48."])
        let duplicateSearch = adding(capture(["search": .amber], profile: 0),
                                     Check(name: "search", state: .amber, since: "2026-11-12T10:48:00+02:00", note: "", unknownTicks: 0, blind: false))
        XCTAssertEqual(BrainBarMenuPresenter.present(duplicateSearch).count, 1)
        // A4 (R26): under the never-connected row (profile 1, model `missing`).
        XCTAssertTrue(sentences(capture(["drive_auth": .red, "transcripts": .amber], profile: 1))
            .contains("Conversation backups can’t upload either until Drive is connected."))
    }

    /// Round 1 of #1788 (C9): the two keys agree. For every conditioned entry, over every enumerated input on a reduced cross
    /// (singles, on every base, profile and upkeep value: six settings of dot, verdict age, measured and the flags that cover
    /// every value of each; pairs: two) plus the 17 cases, and over candidate values (every value shown anywhere and its
    /// perturbations), the registry's condition and the oracle's give the same answer wherever the filled sentence is one the
    /// matcher accepts (no other sentence reaches a condition). The test compares their answers; it shares neither's code.
    func testTheRegistryAndTheOracleAgree() throws {
        let start = Date()
        let cross: [(State, Freshness, Bool, Bool)] = [(.green, .fresh0, true, false), (.green, .fresh0, true, true), (.red, .fresh0, false, false),
                                                       (.amber, .fresh900, true, false), (.green, .stale901, true, true), (.green, .missing, false, false)]
        var captures: [Capture] = []
        for input in Self.inputs() {
            for (dot, freshness, measured, flag) in input.settings.count == 2 ? Array(cross.prefix(3).suffix(2)) : cross {
                captures.append(Self.capture(input, dot: dot, checking: flag, wake: flag, freshness: freshness, measured: measured))
            }
        }
        captures += try BrainBarMenuPresentationTests.loadCases().map { try BrainBarMenuPresentationTests.capture($0["inputs"] as! [String: Any]) }
        let presented = captures.map(BrainBarMenuPresenter.present)
        let entries = BrainBarMenuPresenter.Say.all.filter { $0.kind != .neutral }
        var pool: [String: Set<[String]>] = [:]
        for p in presented {
            for sentence in ([p.headline] + p.rows.flatMap(\.strings)).flatMap(Self.sentences) {
                if let (entry, values) = BrainBarMenuPresenter.entry(for: sentence), entry.kind != .neutral { pool[entry.id, default: []].insert(values) }
            }
        }
        // Each entry's candidates: every value shown, perturbed slot by slot, kept where the matcher accepts the sentence.
        let candidates: [(BrainBarMenuPresenter.Sentence, [[String]])] = entries.map { entry in
            let slots = entry.template.components(separatedBy: "{").dropFirst().map { String($0.prefix { $0 != "}" }) }
            var tuples: Set<[String]> = slots.isEmpty ? [[]] : []
            for shown in pool[entry.id] ?? [Array(repeating: "", count: slots.count)] {
                var grid: [[String]] = [[]]
                for (slot, value) in zip(slots, shown) { grid = grid.flatMap { prefix in Self.perturbed(slot, value).map { prefix + [$0] } } }
                tuples.formUnion(grid.filter { entry.match(entry.fill($0)) == $0 })
            }
            return (entry, Array(tuples))
        }
        let tally = Tally()
        DispatchQueue.concurrentPerform(iterations: captures.count) { i in
            let f = Facts(c: captures[i], p: presented[i])
            var compared = 0, disagree: [String] = []
            for (entry, tuples) in candidates {
                for values in tuples {
                    compared += 1
                    let registry = entry.holds!(captures[i], presented[i], values), oracle = Self.oracle[entry.id]!(f, values)
                    if registry != oracle { disagree.append("\(entry.id) \(values) registry \(registry) oracle \(oracle) | capture \(i)") }
                }
            }
            tally.add((compared, disagree, [:]))
        }
        print("BrainBarMenuInvariantTests: \(tally.outputs) comparisons over \(captures.count) captures and \(candidates.map(\.1.count).reduce(0, +)) candidate values in \(String(format: "%.1f", Date().timeIntervalSince(start))) s")
        XCTAssertGreaterThan(tally.outputs, 1_000_000)
        let byId = Dictionary(grouping: tally.violations) { String($0.prefix { $0 != " " }) }.mapValues(\.count)
        XCTAssertEqual(Array(tally.violations.sorted().prefix(20)), [], "\(tally.violations.count) disagreements \(byId)")
    }

    /// A slot's value and its neighbours: the count minus and plus one, zero and one; the other day words and the next day; a
    /// clock a minute off; a shorter and a longer duration; other and malformed lists.
    static func perturbed(_ slot: String, _ value: String) -> [String] {
        func shift(_ clock: String, _ minutes: Int) -> String {
            guard let h = Int(clock.prefix(2)), let m = Int(clock.suffix(2)) else { return clock }
            let t = ((h * 60 + m + minutes) % 1440 + 1440) % 1440
            return String(format: "%02d:%02d", t / 60, t % 60)
        }
        switch slot {
        case "N2", "N3", "p": return Int(value).map { n in [value, String(n - 1), String(n + 1), "0", "1", "2", "3"] } ?? [value]
        case "HH:mm": return [value, shift(value, 1), shift(value, -1)]
        case "at":
            let halves = value.components(separatedBy: " at ")
            guard halves.count == 2 else { return [value] }
            let date = halves[0].split(separator: " "), next = date.count == 2 ? Int(date[1]).map { ["\(date[0]) \($0 - 1)", "\(date[0]) \($0 + 1)"] } ?? [] : []
            return (["today", "yesterday", halves[0]] + next).flatMap { day in [halves[1], shift(halves[1], 1)].map { day + " at " + $0 } }
        case "age":
            let hours = Int(value.prefix { $0.isNumber }).map { ["\($0 - 1) hours old", "\($0 + 1) hours old"] } ?? []
            let days = value.hasPrefix("from ") ? perturbed("at", String(value.dropFirst(5))).map { "from " + $0 } : []
            return [value, "1 hour old", "2 hours old", "23 hours old"] + hours + days
        case "over": return [value, "", " in over a minute", " in over 2 minutes", " in over an hour", " in over 2 hours", " in over 3 hours"]
        case "thing": return ["new conversations", "search", "backups", "upkeep"]
        case "two": return ["new conversations and backups", "new conversations and search", "search and backups"]
        case "fine": return ["Saving", "Search", "Backups", "Upkeep", "Saving and saving"]
        case "fines": return ["Saving and search", "Saving, search and backups", "Saving, search, backups and upkeep", "Search, backups and upkeep",
                              "Saving and upkeep", value, "Saving, saving and search"]
        default: return [value]
        }
    }

    /// Round 1 of #1788 (Codex B2): a placeholder matches only the words its formatter makes: canonical lists in order, each
    /// noun once, with the verb for its count; counts at the floor the truth table names; real clocks, days and hours.
    func testTheMatcherAcceptsOnlyWhatTheFormattersMake() {
        let refused = [
            "Saving, saving and saving are fine.", "Saving and saving is fine.", "Search and saving are fine.", "Saving are fine.",
            "Saving and search is fine.", "saving is fine.", "Your newest backup in Google Drive is 1 hours old.",
            "Your newest backup in Google Drive is 24 hours old.", "Your newest backup is 0 hours old.", "Started at 25:00.", "Started at 12:60.",
            "0 things need you", "1 things need you", "02 things need you", "The last 0 backups didn’t finish", "The last 1 backups didn’t finish",
            "Nightly upkeep failed 2 nights in a row", "Nightly upkeep failed 03 nights in a row", "Last backup Foo 9 at 04:00.",
            "Last backup Nov 32 at 04:00.", "Last backup Nov 09 at 04:00.", "Catching up on backups and new conversations",
            "Catching up on backups and backups", "BrainBar hasn’t seen the memories waiting for it go down in over 1 hours.",
            "BrainBar hasn’t seen the memories waiting for it go down in over 60 minutes.", "101% done.", "040% done.",
        ]
        XCTAssertEqual(refused.filter { BrainBarMenuPresenter.entry(for: $0) != nil }, [])
        let accepted = [
            "Saving, search and upkeep are fine.", "Search is fine.", "Saving and backups are fine.", "Your newest backup in Google Drive is 1 hour old.",
            "Your newest backup is 23 hours old.", "The newest backup BrainBar can confirm is from Nov 9 at 03:44.", "Started at 23:59.",
            "Started yesterday at 23:40.", "2 things need you", "The last 2 backups didn’t finish", "Nightly upkeep failed 3 nights in a row",
            "Last backup today at 04:00.", "Catching up on new conversations and backups", "100% done.", "0% done, about 13:05.",
            "BrainBar hasn’t seen the memories waiting for it go down in over 2 minutes.", "BrainBar hasn’t seen the memories waiting for it go down.",
        ]
        XCTAssertEqual(accepted.filter { BrainBarMenuPresenter.entry(for: $0) == nil }, [])
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
        ("Conversation backups can’t upload", "backups"), ("To Google Drive.", "backups"),
        ("Search by meaning isn’t catching up", "search"), ("Can’t confirm agents can search", "search"),
        ("Catching up on new conversations", "saving"), ("Nightly upkeep failed", "upkeep"),
    ]

    static let things: [(word: String, checks: [String])] = [
        ("saving", ["ingest", "stores"]), ("search", ["search", "embedding"]),
        ("backups", ["backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk"]), ("upkeep", []),
    ]

    /// What an oracle condition may read: the capture and the value, never the presenter's own helpers or conditions.
    struct Facts: Sendable {
        let c: Capture, p: P
        var fresh: Bool { c.verdict != nil && (0...900).contains(c.verdictAwakeAgeS) }
        func named(_ name: String) -> [Check] { c.verdict?.checks.filter { $0.name == name } ?? [] }
        /// Every check of this name satisfies `ok`, and there is at least one.
        func every(_ name: String, _ ok: (Check) -> Bool) -> Bool { !named(name).isEmpty && named(name).allSatisfy(ok) }
        func red(_ name: String) -> Bool { named(name).contains { $0.state == .red } }
        /// F07: an `embedding` check red, or amber within its threshold (amber past it is F06, progress seen).
        func stalled(_ k: Check) -> Bool {
            k.name == "embedding" && (k.state == .red || (k.state == .amber && k.ageS.map { $0.isFinite && $0 >= 0 } == true && k.thresholdS.map { k.ageS! <= $0 } == true))
        }
        func provenFine(_ names: [String]) -> Bool {
            fresh && !c.isChecking && !c.wakeCatchUpPending && every("monitor") { $0.state == .green } && names.allSatisfy { every($0) { $0.state == .green } }
        }
        /// I3: each named thing proven fine. Saving needs a watcher file that can prove it (readable, fresh, its probe working,
        /// no alert but R10's `quarantined_record`); backups a measured observation with no future-dated copy; upkeep < 3 nights.
        func fine(_ words: [String]) -> Bool {
            BrainBarMenuInvariantTests.things.filter { words.contains($0.word) }.allSatisfy { provenFine($0.checks) }
                && (!words.contains("backups") || (c.backupsMeasured && !checks("backup_local", "backup_offsite", "transcripts").contains { ($0.ageS ?? 0) < 0 }))
                && (!words.contains("saving") || c.watcher.map { $0.readable && $0.fresh && !$0.dbProbeFailed && $0.alertReasons.allSatisfy { $0 == "quarantined_record" } } == true)
                && (!words.contains("upkeep") || c.maintenanceNightFailures < 3)
        }
        func checks(_ names: String...) -> [Check] { names.flatMap { named($0) } }
        var allGoodProven: Bool {
            fresh && c.verdict!.dot == .green && known.allSatisfy { every($0) { $0.state == .green } }
                && c.verdict!.checks.allSatisfy { known.contains($0.name) || $0.state == .green } && c.backupsMeasured && !c.isChecking && !c.wakeCatchUpPending
                && fine(["saving", "search", "backups", "upkeep"])
        }
        var needsRows: Int { p.rows.filter { $0.kind == .needsYou }.count }
        /// "Everything else": the four things minus those a drawn row speaks for; all four beside a row no title places.
        var everythingElse: [String] {
            var left = BrainBarMenuInvariantTests.things.map(\.word)
            for row in p.rows where [.needsYou, .statusLine, .factLine, .headlineDetail].contains(row.kind) {
                guard let first = row.strings.first, let word = speaksFor.first(where: { first.hasPrefix($0.prefix) })?.word else { return BrainBarMenuInvariantTests.things.map(\.word) }
                left.removeAll { $0 == word }
            }
            return left
        }
        /// R02: lag on a readable, fresh watcher file whose probe works, `ingest` green, the episode not past 30 minutes.
        var catchingUp: Bool {
            c.watcher.map { $0.readable && $0.fresh && !$0.dbProbeFailed && $0.alertReasons.contains("offset_lag") } == true
                && c.lagEpisodeStart.map { c.now.timeIntervalSince($0) > 1800 } != true && every("ingest") { $0.state == .green }
        }
        /// A time the words can name (Foundation's own range; NaN is in no range).
        func nameable(_ d: Date) -> Bool { d >= .distantPast && d <= .distantFuture }
        /// U13: a `recovering` on schedule (heard from within two cadences, not overdue) is the running upload; without
        /// one, a live upload process is. Its start, never after now.
        var running: Date? {
            if let r = c.offsiteRecovering {
                let quiet = c.now.timeIntervalSince(r.updatedAt)
                return r.cadenceS > 0 && r.cadenceS.isFinite && quiet >= 0 && quiet <= 2 * r.cadenceS && (r.eta.map { c.now <= $0 && nameable($0) } ?? true)
                    && r.progress.map { (0...1).contains($0) } != false && r.startedAt <= c.now && nameable(r.startedAt) ? r.startedAt : nil
            }
            return c.upload.flatMap { $0.live && $0.startedAt <= c.now && nameable($0.startedAt) ? $0.startedAt : nil }
        }
        var progress: String? { running == nil ? nil : c.offsiteRecovering?.progress.flatMap { (0...1).contains($0) ? String(Int(($0 * 100).rounded())) : nil } }
        func clock(_ date: Date, _ format: String = "HH:mm") -> String { BrainBarMenuInvariantTests.formatted(date, format, c.timeZone) }
        /// Part 0's `{at}`, written here from the truth table: "today at 04:00" or "yesterday at 23:10" under a day ago, else
        /// "Nov 9 at 03:44".
        func dayWords(_ d: Date) -> String {
            let recent = c.now.timeIntervalSince(d) < 86_400
            if recent, calendar.isDate(d, inSameDayAs: c.now) { return "today at " + clock(d) }
            if recent, let y = calendar.date(byAdding: .day, value: -1, to: c.now), calendar.isDate(d, inSameDayAs: y) { return "yesterday at " + clock(d) }
            return clock(d, "MMM d") + " at " + clock(d)
        }
        var calendar: Calendar {
            var calendar = Calendar(identifier: .gregorian)
            calendar.timeZone = c.timeZone
            return calendar
        }
        /// A bare clock time reads as a time today, so it is true only for a start on today's date (P2b, ruling R23).
        func startedToday(_ value: String) -> Bool { running.map { calendar.isDate($0, inSameDayAs: c.now) && clock($0) == value } == true }
        /// R23: a start on another day says its day.
        func startedOn(_ value: String) -> Bool { running.map { !calendar.isDate($0, inSameDayAs: c.now) && dayWords($0) == value } == true }
        /// `{copy age}` of the newest copy among `names`, whole: "1 hour old", "N hours old" (N the rounded hours, 2 to 23)
        /// under a day at now, else "from {day words}".
        func age(_ names: [String], _ text: String) -> Bool {
            guard let v = c.verdict, let newest = names.flatMap({ named($0) }).compactMap(\.ageS).filter({ $0.isFinite && $0 >= 0 }).min() else { return false }
            let copied = v.evaluatedAt.addingTimeInterval(-newest), elapsed = c.now.timeIntervalSince(copied)
            guard nameable(copied), elapsed >= 0 else { return false }
            let hours = min(23, max(1, Int((elapsed / 3600).rounded())))
            return text == (elapsed >= 86_400 ? "from " + dayWords(copied) : hours == 1 ? "1 hour old" : "\(hours) hours old")
        }
        /// `{over d}`, whole: the largest whole number of hours (2 or more), else "an hour", else minutes, strictly below the
        /// stall's age; nothing under a minute.
        func overWords(_ age: Double?) -> String {
            guard let age, age.isFinite, age > 60 else { return "" }
            func below(_ unit: Double) -> Int { let a = min(age, 1e12); var n = Int(a / unit); if Double(n) * unit >= a { n -= 1 }; return n }
            let hours = below(3600)
            if hours >= 2 { return " in over \(hours) hours" }
            if hours == 1 { return " in over an hour" }
            return below(60) == 1 ? " in over a minute" : " in over \(below(60)) minutes"
        }
        /// R25: no copy in Drive of either kind has a verified record.
        var noDriveRecord: Bool { every("backup_offsite") { $0.ageS == nil } && named("transcripts").allSatisfy { $0.ageS == nil } }
        func fineWords(_ text: String) -> [String] { BrainBarMenuInvariantTests.things.map(\.word).filter { text.lowercased().contains($0) } }
    }

    /// The oracle's own condition for every claim and reassurance, keyed by the registry's ids and written from the truth
    /// table and the brief, not from the presenter. `testEveryClaimAndReassuranceHasAnOracleCondition` keeps the two key
    /// sets equal, so no sentence can be added without stating twice, independently, when it is true.
    static let oracle: [String: @Sendable (Facts, [String]) -> Bool] = [
        "S1": { f, _ in f.allGoodProven },
        "S01": { f, v in
            let ages = f.checks("backup_offsite", "transcripts").compactMap(\.ageS)
            guard f.p.headline == "All good", f.allGoodProven, let older = ages.max(), ages.allSatisfy({ $0.isFinite && $0 >= 0 }) else { return false }
            let copied = f.c.verdict!.evaluatedAt.addingTimeInterval(-older)
            return copied <= f.c.now && f.nameable(copied) && f.dayWords(copied) == v[0]
        },
        "X02": { f, _ in f.c.wakeCatchUpPending },
        "S4": { f, v in f.c.lastKnownBackupAt.map { $0 <= f.c.now && f.nameable($0) && f.dayWords($0) == v[0] } == true },
        "R02.headline": { f, _ in f.catchingUp },
        "R02": { f, _ in f.catchingUp },
        "S03.headline": { f, _ in f.running != nil },
        "S03": { f, _ in f.running != nil },
        "S03.started": { f, v in f.startedToday(v[0]) },
        "S03.startedOn": { f, v in f.startedOn(v[0]) },
        "S03.progress": { f, v in f.progress == v[0] },
        "S03.progressEta": { f, v in f.progress == v[0] && f.c.offsiteRecovering?.eta.map { f.calendar.isDate($0, inSameDayAs: f.c.now) && f.clock($0) == v[1] } == true },
        "C3": { f, v in v[0].components(separatedBy: " and ").allSatisfy { ($0 == "new conversations" && f.catchingUp) || ($0 == "backups" && f.running != nil) } },
        "C.few": { f, _ in [f.catchingUp, f.running != nil].filter { $0 }.count >= 3 },
        "A2.fine.is": { f, v in f.fine(f.fineWords(v[0])) },
        "A2.fine.are": { f, v in f.fine(f.fineWords(v[0])) },
        "M5": { f, _ in f.fine(f.everythingElse) },
        "A1.count.one": { f, _ in f.needsRows == 1 },
        "A1.count.many": { f, v in Int(v[0]) == f.needsRows && f.needsRows >= 2 },
        "U03": { f, v in Int(v[0]) == f.c.maintenanceNightFailures && f.c.maintenanceNightFailures >= 3 },
        "U03.stillSaved": { f, _ in f.fine(["saving", "backups"]) },
        "S19": { f, _ in f.red("integrity") },
        "S19.memory": { f, _ in f.red("integrity") },
        "F07": { f, _ in f.named("embedding").contains(where: f.stalled) },
        "F07.since": { f, v in f.named("embedding").contains { f.stalled($0) && f.overWords($0.ageS) == v[0] } },
        "F07.exactWords": { f, _ in f.c.keywordCoverage == 1 && f.provenFine(["search"]) },
        "S07": { f, _ in f.every("backup_local") { $0.state == .red && $0.ageS == nil } && f.noDriveRecord },
        "R14": { f, _ in f.noDriveRecord && f.c.backupsMeasured },
        "S32": { f, v in f.startedToday(v[0]) },
        "S32.progress": { f, v in f.startedToday(v[0]) && f.progress == v[1] },
        "S32.on": { f, v in f.startedOn(v[0]) },
        "S32.progressOn": { f, v in f.startedOn(v[0]) && f.progress == v[1] },
        "S06": { f, v in Int(v[0]) == f.c.backupDailyFailures && f.c.backupDailyFailures >= 2 },
        "S06.newest": { f, v in f.c.backupsMeasured && f.age(["backup_local", "backup_offsite"], v[0]) },
        "S06.confirmed": { f, v in f.age(["backup_local", "backup_offsite"], v[0]) },
        "R15": { f, v in f.c.backupsMeasured && f.age(["backup_offsite"], v[0]) },
        "S13": { f, _ in f.red("drive_auth") && f.c.driveAuthModel != .missing },
        "S13.missing": { f, _ in f.red("drive_auth") && f.c.driveAuthModel == .missing },
        "S13.detail": { f, _ in f.red("drive_auth") && f.c.driveAuthModel != .missing },
        "S13.detailMissing": { f, _ in f.red("drive_auth") && f.c.driveAuthModel == .missing },
        "D13": { f, _ in f.red("drive_auth") && f.c.driveAuthModel != .missing },
        "D13.missing": { f, _ in f.red("drive_auth") && f.c.driveAuthModel == .missing },
    ]

    /// The oracle's own formatters, one per format and zone on each thread.
    static func formatted(_ date: Date, _ format: String, _ zone: TimeZone) -> String {
        let key = "oracle|" + format + "|" + zone.identifier
        let f = Thread.current.threadDictionary[key] as? DateFormatter ?? {
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.timeZone = zone
            f.dateFormat = format
            Thread.current.threadDictionary[key] = f
            return f
        }()
        return f.string(from: date)
    }

    static func violations(_ c: Capture, _ p: P) -> [String] { audit(c, p).violations }

    /// The invariants, and every sentence shown with its placeholder values (for the reach and falsify checks).
    static func audit(_ c: Capture, _ p: P) -> (violations: [String], shown: [String: [String]]) {
        var out: [String] = [], shown: [String: [String]] = [:]
        let f = Facts(c: c, p: p)
        // I1: no hidden red.
        if let v = c.verdict, v.dot == .red || v.checks.contains(where: { $0.state == .red }), p.form != .needs {
            out.append("I1 a red with a verdict present is not the needs form")
        }
        // The closed world, and I2, I3 and I8 by sentence: every sentence shown is a registered one, and a claim or a
        // reassurance only where the oracle's own condition holds.
        let texts = [p.headline] + p.rows.flatMap(\.strings)
        for sentence in (texts + p.rows.compactMap(\.action?.label) + p.rows.compactMap(\.optionAlternate)).flatMap(sentences) {
            guard let (entry, values) = BrainBarMenuPresenter.entry(for: sentence) else { out.append("CW an unregistered sentence: \(sentence)"); continue }
            shown[entry.id] = values
            guard entry.kind != .neutral else { continue }
            guard let holds = oracle[entry.id] else { out.append("KEYS \(entry.id) has no oracle condition"); continue }
            let law = ["S1", "S01"].contains(entry.id) ? "I2" : entry.kind == .claim ? "I8" : "I3"
            if !holds(f, values) { out.append("\(law) \(entry.id) shown while untrue: \(sentence)") }
        }
        // B5: the standing items keep their one key and Option alternate (Settings ⌘,; Quit ⌘Q, Restart BrainBar); no other row has either.
        for row in p.rows {
            let standing = row.kind == .standardItem ? row.strings : []
            if row.keyEquivalent != (standing == ["Settings…"] ? "⌘," : standing == ["Quit BrainBar"] ? "⌘Q" : nil) { out.append("B5 a key equivalent on \(standing)") }
            if row.optionAlternate != nil, !(standing == ["Quit BrainBar"] && row.optionAlternate == "Restart BrainBar") { out.append("B5 an option alternate on \(standing)") }
        }
        // I4: one rule.
        if p.count != f.needsRows { out.append("I4 count \(p.count) but \(f.needsRows) Needs-you rows") }
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
        return (out, shown)
    }
}
