import Foundation

/// The one presentation function for the M menu and icon (phase-8 PRESENTATION-SPEC.md, truth table v3 Part 2).
/// Pure and deterministic: the same capture always gives the same presentation. Nothing in production calls it.
///
/// P1 derives the six calm cases (S1, S2 today and after the seam, S4, S7, S8); P2 the Needs-you rows of the eleven
/// needs cases (S13 with M1 and D13, S06, S07, S19, S32, F07, U03) with merges M2 and M5. Everything else takes ruling
/// R17's generic rows: a thing that needs him reads "Something BrainBar checks is failing.", a thing that cannot be
/// told reads "Checking…". P3 to P5 replace them family by family. `finalGuard` runs last, on every path.
enum BrainBarMenuPresenter {
    typealias P = BrainBarMenuPresentation
    typealias Capture = BrainBarMenuCapture
    typealias State = BrainBarHealthVerdict.State

    private enum Thing: Int, CaseIterable { case remembering, finding, safe, upkeep, system }
    private enum Level { case cantTell, behind, needs }
    /// U11: custody-critical, then confirmed, then escalated.
    private enum Rank: Int { case custody, confirmed, escalated }

    private struct Line {
        var thing: Thing
        var level: Level
        var strings: [String]
        /// A Behind row's headline (Part 2 step 8).
        var headline: String? = nil
        var spinner = false
        var progress: Int? = nil
        /// False for R17's generic rows: they merge only with each other, and no "Everything else" stands beside them.
        var worded = true
        var rank = Rank.confirmed
        /// M2 inside a class, by severity (truth table erratum, 13 Nov): S19 0, S07 1, S32 2, S06 3, S13 4; generic 9.
        var priority = 9
        var action = P.ActionKind.copyReport
        var handoff = BrainBarMenuPresenter.handoff
        /// D13's line, drawn directly under this row (R22) and never counted.
        var fact: [String]? = nil
    }

    private static let checksOf: [Thing: [String]] = [
        .safe: ["backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk"],
        .remembering: ["ingest", "stores"], .finding: ["search", "embedding"], .upkeep: [], .system: ["monitor"],
    ]
    static let known = Thing.allCases.flatMap { checksOf[$0]! }
    private static let four: [Thing] = [.remembering, .finding, .safe, .upkeep]
    private static let copies = ["backup_local", "backup_offsite", "transcripts"]

    private static func thing(of name: String) -> Thing { Thing.allCases.first { checksOf[$0]!.contains(name) } ?? .system }

    /// R17's generic rows.
    private static func needs(_ thing: Thing) -> Line { Line(thing: thing, level: .needs, strings: ["Something BrainBar checks is failing."], worded: false) }
    private static func checking(_ thing: Thing) -> Line { Line(thing: thing, level: .cantTell, strings: ["Checking…"], worded: false) }
    private static func worded(_ thing: Thing, _ strings: [String], _ rank: Rank, priority: Int = 9) -> Line {
        Line(thing: thing, level: .needs, strings: strings, rank: rank, priority: priority)
    }
    private static let handoff = ["BrainBar can’t fix this itself.", "Give the report to an agent."]

    /// The only entry point. It is total over its capture type: no input traps, and the guard runs last on every path.
    static func present(_ c: Capture) -> P { finalGuard(c, derived(c)) }

    private static func derived(_ c: Capture) -> P {
        if let v = c.verdict, (0...900).contains(c.verdictAwakeAgeS) {
            let lines = derive(c, v)
            if !c.isChecking, !c.wakeCatchUpPending { return assemble(c, lines, allGood: allGoodDetail(c, v)) }
            // R19: with a fresh verdict, checking never calms; every Needs-you row stays (X01: a decodable verdict ends it).
            return lines.contains { $0.level == .needs } ? assemble(c, lines, allGood: nil) : checkingState(c)
        }
        // X03, X04 (P5 words them): the verdict's checks are Can't tell, except that its reds count (I1),
        // and so do the rows the app reads without the producer (R20).
        var reds: Set<Thing> = Set((c.verdict?.checks ?? []).filter { $0.state == .red }.map { thing(of: $0.name) })
        if c.verdict?.dot == .red { reds.insert(.system) }
        let lines: [Line] = Thing.allCases.filter(reds.contains).map(needs) + appFacts(c)
        guard !lines.isEmpty else { return checkingState(c) }
        return assemble(c, lines + four.filter { t in !lines.contains { $0.thing == t } }.map(checking), allGood: nil)
    }

    /// The worst state a check reports under this name, or nil when no check has the name.
    private static func worst(_ v: BrainBarHealthVerdict, _ name: String) -> State? {
        let states = v.checks.filter { $0.name == name }.map(\.state)
        return [State.red, .amber, .unknown, .green].first(where: states.contains)
    }

    /// R20: Needs-you rows whose facts the app reads without the producer, so they count whatever the verdict's state.
    /// A failed database probe makes only `coverage_drop` unprovable (R24); other alerts still count (R09, X10).
    private static func appFacts(_ c: Capture) -> [Line] {
        var lines: [Line] = []
        if c.maintenanceNightFailures >= 3 {  // U03; its second sentence only when saving and backups are proven fine
            lines.append(worded(.upkeep, ["Nightly upkeep failed \(c.maintenanceNightFailures) nights in a row"]
                + (provenFine(c).isSuperset(of: [.remembering, .safe]) ? [stillSaved] : []), .confirmed))
        }
        guard let w = c.watcher, w.readable, w.fresh else { return lines }
        if w.alertReasons.contains("offset_lag"), lagPastLimit(c) { lines.append(needs(.remembering)) }  // R02 escalated
        for reason in w.alertReasons where !["offset_lag", "quarantined_record"].contains(reason) {
            if reason == "coverage_drop", w.dbProbeFailed { continue }
            lines.append(needs(["coverage_drop", "file_ingestion_failure"].contains(reason) ? .remembering : .system))
        }
        return lines
    }

    private static func lagPastLimit(_ c: Capture) -> Bool { c.lagEpisodeStart.map { c.now.timeIntervalSince($0) > 1800 } == true }

    // MARK: Part 1 rows from one fresh verdict

    private static func derive(_ c: Capture, _ v: BrainBarHealthVerdict) -> [Line] {
        let (wordedLines, consumed) = needsRows(c, v)
        var lines: [Line] = appFacts(c) + wordedLines
        for name in known where name != "monitor" && !consumed.contains(name) {
            switch worst(v, name) {
            case .green?: break
            case .red?: lines.append(needs(thing(of: name)))
            case .amber?: lines.append(calmAmber(name, c, v) ?? needs(thing(of: name)))
            case .unknown?, nil: lines.append(checking(thing(of: name)))
            }
        }
        // X05, X06: a monitor that is not green blinds every thing.
        if worst(v, "monitor") != .green {
            if [.red, .amber].contains(worst(v, "monitor")) { lines.append(needs(.system)) }
            lines += four.map(checking)
        }
        for k in v.checks where !known.contains(k.name) && k.state != .green { lines.append(needs(.system)) }
        if !c.backupsMeasured || v.checks.contains(where: { copies.contains($0.name) && ($0.ageS ?? 0) < 0 }) {
            lines.append(checking(.safe))
        }
        if let w = c.watcher, w.readable, w.fresh, !w.dbProbeFailed {
            if w.alertReasons.contains("offset_lag"), worst(v, "ingest") == .green, !lagPastLimit(c) {  // R02, under 30 min
                lines.append(Line(thing: .remembering, level: .behind, strings: ["Catching up on new conversations."],
                                  headline: "Catching up on new conversations", spinner: true))
            }
        } else {
            lines.append(checking(.remembering))
        }
        if v.dot == .red, !lines.contains(where: { $0.level == .needs }) { lines.append(needs(.system)) }
        if v.dot != .green, v.checks.allSatisfy({ $0.state == .green }) { lines += four.map(checking) }
        return lines
    }

    /// The two amber rows that do not need him in P1: S03 (a backup to Drive is running) and F02 (search unconfirmed
    /// for under 15 minutes). Every other amber is a generic Needs-you row until its family is derived.
    private static func calmAmber(_ name: String, _ c: Capture, _ v: BrainBarHealthVerdict) -> Line? {
        switch name {
        case "backup_offsite" where worst(v, "backup_local") == .green:
            guard let (started, onSchedule) = runningUpload(c) else { return nil }
            var text = "To Google Drive. Started at \(hhmm(started, c))."
            if let p = onSchedule?.progress { text += " \(percent(p))% done" + (onSchedule?.eta.map { ", about \(hhmm($0, c))." } ?? ".") }
            return Line(thing: .safe, level: .behind, strings: [text], headline: "Backing up now", progress: onSchedule?.progress.map(percent))
        case "search":
            let since = v.checks.first { $0.name == "search" }?.since.flatMap(parse)
            guard let since, (0...900).contains(c.now.timeIntervalSince(since)) else { return nil }
            return Line(thing: .finding, level: .cantTell, strings: ["Can’t confirm agents can search right now."])
        default:
            return nil
        }
    }

    /// U13: once `recovering` exists it is the evidence of a running upload; before PR-2 the live upload is. A cadence that
    /// is not a finite positive number makes the recovering unusable (no range is built from it).
    private static func runningUpload(_ c: Capture) -> (started: Date, onSchedule: Capture.Recovering?)? {
        let onSchedule = c.offsiteRecovering.flatMap { r -> Capture.Recovering? in
            let quiet = c.now.timeIntervalSince(r.updatedAt)
            let sane = r.cadenceS.isFinite && r.cadenceS > 0 && r.startedAt <= c.now && r.progress.map { (0...1).contains($0) } != false
            return sane && quiet >= 0 && quiet <= 2 * r.cadenceS && r.eta.map { c.now <= $0 } != false ? r : nil
        }
        let upload = c.upload.flatMap { $0.live ? $0.startedAt : nil }
        guard let started = c.offsiteRecovering == nil ? upload : onSchedule?.startedAt, started <= c.now else { return nil }
        return (started, onSchedule)
    }

    // MARK: P2: the Needs-you rows of the eleven needs cases

    /// The worded rows, with M1 and D13 applied, and the checks they speak for (those derive no generic row). A check name
    /// carried more than once is never read for its fields: it keeps its generic row. Every other case keeps P1's rows.
    private static func needsRows(_ c: Capture, _ v: BrainBarHealthVerdict) -> (lines: [Line], consumed: Set<String>) {
        func one(_ name: String) -> BrainBarHealthVerdict.Check? {
            let named = v.checks.filter { $0.name == name }
            return named.count == 1 ? named[0] : nil
        }
        func within(_ k: BrainBarHealthVerdict.Check) -> Bool { valid(k.ageS) && k.thresholdS.map { k.ageS! <= $0 } == true }
        var lines: [Line] = [], consumed: Set<String> = []
        if one("integrity")?.state == .red {  // S19
            lines.append(worded(.safe, ["The newest checked backup is damaged", "Your memory itself may be damaged too."], .custody, priority: 0))
            consumed.insert("integrity")
        }
        if let e = one("embedding"), e.state == .red || (e.state == .amber && within(e)) {  // F07
            let exact = c.keywordCoverage == 1 && searchProven(c) ? " " + exactWords : ""
            lines.append(worded(.finding, ["Search by meaning isn’t catching up", "BrainBar hasn’t seen the memories waiting for it go down"
                + (e.ageS.flatMap(over).map { " " + $0 } ?? "") + "." + exact], .confirmed))
            consumed.insert("embedding")
        }
        // The database copies: S07, S32, S06 with failures, and S02 (drawn only through M1).
        let local = one("backup_local"), offsite = one("backup_offsite")
        let reds = [local, offsite].compactMap { $0 }.filter { $0.state == .red }
        var db: Line? = nil, dbNames: [String] = [], s02 = false
        if let local, let offsite {
            let faults = [local, offsite].filter { $0.state != .green }
            if local.state == .red, local.ageS == nil, offsite.ageS == nil {  // S07, only with no copy on record anywhere (R21)
                (db, dbNames) = (worded(.safe, [noRecord], .custody, priority: 1), ["backup_local"])
            } else if !reds.isEmpty, let upload = runningUpload(c) {  // S32, R8, R9
                let age = c.backupsMeasured ? copyAge(offsite.ageS, c, v).map { "Your newest backup in Google Drive is \($0). " } : nil
                let progress = upload.onSchedule?.progress.map { "is \(percent($0))% done" } ?? "is still running"
                db = worded(.safe, [(age ?? "") + "A backup to Google Drive started at \(hhmm(upload.started, c)) and \(progress)."],
                            local.state == .red ? .custody : .confirmed, priority: 2)
                db?.handoff = ["If this backup doesn’t finish, BrainBar can’t fix it itself.", "Give the report to an agent."]
                dbNames = reds.map(\.name)
            } else if c.backupDailyFailures >= 2, !faults.isEmpty, c.upload == nil, c.offsiteRecovering == nil,
                      faults.allSatisfy({ ($0.state == .red && valid($0.ageS)) || ($0.state == .amber && within($0)) }) {  // S06
                let newest = [local, offsite].compactMap(\.ageS).filter { valid($0) }.min().flatMap { copyAge($0, c, v) }.map {
                    c.backupsMeasured ? "Your newest backup is \($0)." : "The newest backup BrainBar can confirm is \($0)."
                }
                db = worded(.safe, ["The last \(c.backupDailyFailures) backups didn’t finish"] + (newest.map { [$0] } ?? []),
                            local.state == .red ? .custody : reds.isEmpty ? .escalated : .confirmed, priority: 3)
                dbNames = faults.map(\.name)
            } else if faults.isEmpty, c.backupDailyFailures == 1, c.backupDailyNextRunAt.map({ $0 > c.now }) == true {
                s02 = true
            }
        }
        guard one("drive_auth")?.state == .red else {
            if let db { lines.append(db) }
            return (lines, consumed.union(dbNames))
        }
        // S13, and M1: it absorbs every database-backup row, takes its rank, and adds the Drive copy's age (R15, R14).
        let missing = c.driveAuthModel == .missing, absorbs = db != nil || s02 || !reds.isEmpty
        var detail = "Backups can’t upload until you \(missing ? "connect" : "reconnect") it."
        if absorbs, c.backupsMeasured, let age = copyAge(offsite?.ageS, c, v) {
            detail += " Your newest backup in Google Drive is \(age)."
        } else if absorbs, c.backupsMeasured, missing, offsite?.state == .red, offsite?.ageS == nil {
            detail += " " + nothingInDrive
        }
        let transcriptsFailing = one("transcripts").map { [.amber, .red].contains($0.state) && $0.ageS != nil } == true && c.jsonlBackupFailures >= 2
        var s13 = worded(.safe, [missing ? "Google Drive isn’t connected" : "Google Drive is disconnected", detail],
                         db?.rank == .custody || local?.state == .red ? .custody : .confirmed, priority: 4)
        s13.action = missing ? .connectDrive : .reconnectDrive
        // D13, R22: a failing transcript backup cannot upload either while Drive is red; a fact line directly under the
        // Drive row, not counted.
        if transcriptsFailing { s13.fact = ["Conversation backups can’t upload either until Drive is reconnected."] }
        lines.append(s13)
        consumed.formUnion(["drive_auth"] + dbNames + reds.map(\.name) + (transcriptsFailing ? ["transcripts"] : []))
        return (lines, consumed)
    }

    // MARK: Part 2: merge, count, order, headline

    private static func assemble(_ c: Capture, _ input: [Line], allGood: String?) -> P {
        // R17 and M5: one row per thing and level; the first line of a thing at that level speaks.
        func at(_ level: Level) -> [Line] { Thing.allCases.compactMap { t in input.first { $0.thing == t && $0.level == level } } }
        let behind = at(.behind), cantTell = at(.cantTell)
        let fine = four.filter { t in !input.contains { $0.thing == t } }
        func status(_ line: Line, mark: P.Mark?) -> P.Row {
            P.Row(kind: .statusLine, strings: line.strings, mark: mark, state: line.level == .behind ? .behind : .cantTell, spinner: line.spinner)
        }
        // M2, M3: worded Needs-you rows of one thing and one action merge, the highest class speaking, then severity; R17's
        // generic rows merge only with each other. Step 6 orders by class, thing, severity. Swift's sort is not stable,
        // so the input index is the last key.
        func ordered(_ lines: [Line], _ key: (Line) -> (Int, Int, Int)) -> [Line] {
            lines.enumerated().sorted {
                let (a, b) = (key($0.element), key($1.element))
                return (a.0, a.1, a.2, $0.offset) < (b.0, b.1, b.2, $1.offset)
            }.map(\.element)
        }
        var needs: [Line] = []
        for line in ordered(input.filter { $0.level == .needs }, { ($0.rank.rawValue, $0.priority, 0) })
            where !needs.contains(where: { $0.thing == line.thing && $0.worded == line.worded && (!line.worded || $0.action == line.action) }) {
            needs.append(line)
        }
        if !needs.isEmpty {
            let rows = ordered(needs) { ($0.rank.rawValue, $0.thing.rawValue, $0.priority) }.flatMap { line in
                [P.Row(kind: .needsYou, strings: line.strings + (line.action == .copyReport ? line.handoff : []), action: action(line.action))]
                    + (line.fact.map { [P.Row(kind: .factLine, strings: $0)] } ?? [])
            }
            // "Everything else is fine." stands for the things no row speaks for, never beside a custody-critical or generic row.
            let everythingElse = !fine.isEmpty && !needs.contains { $0.rank == .custody } && input.allSatisfy(\.worded)
                && Set(fine).isSubset(of: provenFine(c))
            return needsForm(rows, statuses: (behind + cantTell).map { status($0, mark: nil) }, everythingElse: everythingElse)
        }
        let nouns: [Thing: String] = [.remembering: "new conversations", .finding: "search", .safe: "backups", .upkeep: "upkeep"]
        let headline: String
        let mark: P.Mark
        switch behind.count {
        case 1: (headline, mark) = (behind[0].headline ?? "Catching up on a few things", .busy)
        case 2: (headline, mark) = ("Catching up on \(nouns[behind[0].thing] ?? "") and \(nouns[behind[1].thing] ?? "")", .busy)
        case 3...: (headline, mark) = ("Catching up on a few things", .busy)
        default:
            let things = cantTell.map(\.thing)
            headline = things.contains(.safe) ? "Can’t confirm backups yet"
                : things.count == 1 ? "Still checking \(nouns[things[0]] ?? "a few things")"
                : things.isEmpty ? "All good" : "Still checking a few things"
            mark = things.isEmpty ? .ok : .wait
        }
        var rows = [P.Row(kind: .headline, strings: [headline], mark: mark)]
        let folded = behind.count == 1 ? behind[0] : nil
        if let folded, folded.strings.first?.dropLast() != headline[...] {  // R02 repeats its headline: no line (R16)
            rows.append(P.Row(kind: .headlineDetail, strings: folded.strings, state: .behind, progressPercent: folded.progress))
        }
        if headline == "All good", let allGood { rows.append(P.Row(kind: .headlineDetail, strings: [allGood])) }
        let listed = (folded == nil ? behind : []) + cantTell
        var fineNouns = fine.map { [.remembering: "saving", .finding: "search", .safe: "backups", .upkeep: "upkeep"][$0] ?? "" }
        let last = fineNouns.popLast() ?? ""
        let joined = fineNouns.isEmpty ? last : fineNouns.joined(separator: ", ") + " and " + last
        let fineRow = P.Row(kind: .everythingElse, strings: [joined.prefix(1).uppercased() + joined.dropFirst() + (fine.count == 1 ? " is fine." : " are fine.")],
                            mark: listed.isEmpty ? nil : .ok)
        if !listed.isEmpty { rows.append(P.Row(kind: .separator)) }
        rows += listed.map { status($0, mark: $0.level == .behind ? .busy : .wait) }
        if !fine.isEmpty, headline != "All good" { rows.append(fineRow) }
        return .init(form: .calm, headline: headline, count: 0, icon: .resting, rows: rows + bottom)
    }

    private static let copyReport = action(.copyReport)

    private static func action(_ kind: P.ActionKind) -> P.Action {
        switch kind {
        case .reconnectDrive: P.Action(kind: kind, label: "Reconnect Google Drive")
        case .connectDrive: P.Action(kind: kind, label: "Connect Google Drive")
        case .copyReport: P.Action(kind: kind, label: "Copy report for an agent")
        case .openStorageSettings: P.Action(kind: kind, label: "Open Storage Settings")
        }
    }

    private static func needsForm(_ needsRows: [P.Row], statuses: [P.Row], everythingElse: Bool = false) -> P {
        let count = needsRows.filter { $0.kind == .needsYou }.count
        let headline = count == 1 ? "1 thing needs you" : "\(count) things need you"
        var rows = [P.Row(kind: .headline, strings: [headline], mark: .marked), P.Row(kind: .separator)]
        for row in needsRows {  // a fact line rides directly under its Needs-you row (R22)
            let first = !rows.contains { $0.kind == .needsYou }
            rows.append(row.kind == .needsYou ? P.Row(kind: .needsYou, strings: row.strings, action: row.action, mark: .alertCircle, dividerBefore: !first) : row)
        }
        rows += statuses.map { var row = $0; row.mark = nil; return row }
        if everythingElse { rows.append(P.Row(kind: .everythingElse, strings: ["Everything else is fine."], mark: .checkmark)) }
        return .init(form: .needs, headline: headline, count: count, icon: .marked, rows: rows + bottom)
    }

    /// X01, X02, and in P1 every path without a readable, fresh verdict and no red: nothing is confirmed.
    private static func checkingState(_ c: Capture) -> P {
        let launching = c.isChecking && c.verdict == nil
        let headline = c.wakeCatchUpPending && !launching ? "Checking after sleep…" : "Checking on your memory…"
        let last = c.lastKnownBackupAt.flatMap { $0 <= c.now ? " Last backup \(at($0, c))." : nil } ?? ""
        return .init(form: .calm, headline: headline, count: 0, icon: .resting,
                     rows: [P.Row(kind: .headline, strings: [headline], mark: .wait),
                            P.Row(kind: .headlineDetail, strings: ["Saving, search, backups and upkeep aren’t confirmed yet." + last],
                                  state: .cantTell)] + bottom)
    }

    /// R4: one bottom block for both forms.
    private static let bottom = [
        P.Row(kind: .separator),
        P.Row(kind: .standardItem, strings: ["Settings…"], keyEquivalent: "⌘,"),
        P.Row(kind: .standardItem, strings: ["Quit BrainBar"], keyEquivalent: "⌘Q", optionAlternate: "Restart BrainBar"),
    ]

    /// S01, the all-good line: the older of the offsite and transcript copies (U10), never a future time.
    private static func allGoodDetail(_ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        let ages = ["backup_offsite", "transcripts"].compactMap { name in v.checks.first { $0.name == name }?.ageS }
        guard let age = ages.max(), ages.allSatisfy({ $0.isFinite && $0 >= 0 }) else { return nil }
        let date = v.evaluatedAt.addingTimeInterval(-age)
        return date <= c.now ? "Backed up to Google Drive \(at(date, c))." : nil
    }

    // MARK: the final guard (invariants I1 to I5), the last step of every path

    /// Whatever was derived, the returned value satisfies the invariants; where it would not, the value is corrected
    /// conservatively (a reassurance removed, or the needs form with R17's row) rather than trusted.
    /// Internal only as the seam for the guard's own test (BrainBarMenuInvariantTests); production calls `present`.
    static func finalGuard(_ c: Capture, _ derived: P) -> P {
        var p = derived
        // I5: no straight apostrophe, no empty string, no row without words (a Needs-you row takes R17's).
        func words(_ s: String) -> String { s.replacingOccurrences(of: "'", with: "’") }
        p.headline = words(p.headline)
        p.rows = p.rows.compactMap { row in
            var row = row
            row.strings = row.strings.map(words).filter { !$0.isEmpty }
            if let label = row.action?.label { row.action?.label = words(label) }
            row.optionAlternate = row.optionAlternate.map(words)
            if row.kind == .separator || !row.strings.isEmpty { return row }
            return row.kind == .needsYou ? P.Row(kind: .needsYou, strings: needs(.system).strings + handoff, action: copyReport) : nil
        }
        // I2: "All good" only when proven; otherwise nothing is confirmed.
        if p.headline == "All good" || p.rows.contains(where: { $0.strings.contains("All good") }), !allGoodProven(c) {
            p = checkingState(c)
        }
        if p.headline != "All good" { p.rows.removeAll { $0.strings.first?.hasPrefix("Backed up to Google Drive") == true } }
        // I1, I7 and I4: a red, or a fact the app reads without the producer, always needs him; form, count and icon
        // follow the Needs-you rows drawn.
        var needsRows = p.rows.filter { $0.kind == .needsYou }
        let red = c.verdict.map { $0.dot == .red || $0.checks.contains { $0.state == .red } } ?? false
        let w = c.watcher.flatMap { $0.readable && $0.fresh ? $0 : nil }
        let appFact = c.maintenanceNightFailures >= 3
            || w?.alertReasons.contains { !["offset_lag", "quarantined_record"].contains($0) && !($0 == "coverage_drop" && w?.dbProbeFailed == true) } == true
            || (w?.alertReasons.contains("offset_lag") == true && c.lagEpisodeStart.map { c.now.timeIntervalSince($0) > 1800 } == true)
        if needsRows.isEmpty, red || appFact || p.form == .needs || p.count != 0 || p.icon == .marked {
            needsRows = [P.Row(kind: .needsYou, strings: needs(.system).strings + handoff, action: copyReport)]
        }
        if !needsRows.isEmpty, p.form != .needs || p.count != needsRows.count || p.icon != .marked || p.rows.filter({ $0.kind == .needsYou }) != needsRows {
            let carried = p.rows.contains { $0.kind == .needsYou } ? p.rows.filter { [.needsYou, .factLine].contains($0.kind) }
                : needsRows + p.rows.filter { $0.kind == .factLine }
            p = needsForm(carried, statuses: p.rows.filter { $0.kind == .statusLine }, everythingElse: p.rows.contains { $0.kind == .everythingElse })
        }
        // I3 and I8, by sentence: a reassurance, or a claim about the copies, stays wherever it sits only when what it says is
        // proven; an unproven S07 title gives way to R17's.
        let drawn = p
        p.rows = p.rows.compactMap { row in
            var row = row
            row.strings = row.strings.compactMap { string in
                let kept = sentences(string).compactMap { sentence -> String? in
                    if let holds = claimHolds(sentence, c) { return holds ? sentence : sentence == nothingInDrive ? nil : needs(.system).strings[0] }
                    return proven(sentence, drawn, c) != false ? sentence : nil
                }
                return kept.isEmpty ? nil : kept.joined(separator: " ")
            }
            if row.kind == .needsYou, row.strings.isEmpty { row.strings = needs(.system).strings + handoff }
            return row.strings.isEmpty && row.kind != .separator ? nil : row
        }
        return p
    }

    // MARK: I3's reassurances, every one the function can emit, in one place

    private static let stillSaved = "Your memory is still being saved and backed up, as of the last checks."  // U03
    private static let exactWords = "Agents can still find every memory by exact words."  // F07

    // MARK: I8's claims about the copies, each with the condition that makes it true

    private static let noRecord = "No verified backup on record"  // S07, A1's words
    private static let noRecordTT = "BrainBar has no record of a verified backup of your memory."  // S07, the truth table's
    private static let nothingInDrive = "Nothing is backed up to Google Drive yet."  // R14

    /// Whether a claim about the copies holds for this capture; nil when the sentence is not such a claim. S07's needs the
    /// local copy red with no clock and no Drive clock either (R21); R14's needs no Drive clock and backups measured.
    private static func claimHolds(_ sentence: String, _ c: Capture) -> Bool? {
        guard [noRecord, noRecordTT, nothingInDrive].contains(sentence) else { return nil }
        let checks = c.verdict?.checks ?? []
        func every(_ name: String, _ ok: (BrainBarHealthVerdict.Check) -> Bool) -> Bool {
            checks.contains { $0.name == name } && checks.filter { $0.name == name }.allSatisfy(ok)
        }
        let noDriveClock = every("backup_offsite") { $0.ageS == nil }
        return sentence == nothingInDrive ? noDriveClock && c.backupsMeasured
            : noDriveClock && every("backup_local") { $0.state == .red && $0.ageS == nil }
    }

    /// Whether a reassuring sentence is proven for this capture; nil when the sentence is not a reassurance. "All good" and
    /// the Drive line under it are I2's.
    private static func proven(_ sentence: String, _ p: P, _ c: Capture) -> Bool? {
        switch sentence {
        case stillSaved: return provenFine(c).isSuperset(of: [.remembering, .safe])
        case exactWords: return c.keywordCoverage == 1 && searchProven(c)
        case _ where sentence.hasSuffix(" fine."): return covered(sentence, p).isSubset(of: provenFine(c))
        default: return nil
        }
    }

    /// The things a "… fine." sentence speaks for: the things it names (all four if none); for "Everything else is fine.",
    /// every thing no drawn row speaks for, and all four beside a row it cannot place (R17's generic rows).
    private static func covered(_ sentence: String, _ p: P) -> Set<Thing> {
        let text = sentence.lowercased()
        guard text.hasPrefix("everything else") else {
            let named = Set([(Thing.remembering, "saving"), (.finding, "search"), (.safe, "backups"), (.upkeep, "upkeep")]
                .filter { text.contains($0.1) }.map(\.0))
            return named.isEmpty ? Set(four) : named
        }
        var left = Set(four)
        for row in p.rows where [.needsYou, .statusLine, .factLine, .headlineDetail].contains(row.kind) {
            guard let first = row.strings.first, let thing = speaksFor.first(where: { first.hasPrefix($0.0) })?.1 else { return Set(four) }
            left.remove(thing)
        }
        return left
    }

    /// The worded rows' first words and the thing each speaks for.
    private static let speaksFor: [(String, Thing)] = [
        ("Google Drive ", .safe), ("The last ", .safe), ("No verified backup", .safe), ("The newest checked backup", .safe),
        ("Your newest backup in Google Drive", .safe), ("A backup to Google Drive", .safe), ("Conversation backups", .safe),
        ("To Google Drive.", .safe), ("Search by meaning", .finding), ("Can’t confirm agents can search", .finding),
        ("Catching up on new conversations", .remembering), ("Nightly upkeep", .upkeep),
    ]

    /// A string's sentences: "A. B." gives ["A.", "B."].
    private static func sentences(_ string: String) -> [String] {
        let parts = string.components(separatedBy: ". ")
        return parts.indices.map { $0 < parts.count - 1 ? parts[$0] + "." : parts[$0] }
    }

    /// F07's exact-words sentence needs search itself proven, beyond the coverage figure (stricter than the truth table).
    private static func searchProven(_ c: Capture) -> Bool {
        guard let v = c.verdict, (0...900).contains(c.verdictAwakeAgeS), !c.isChecking, !c.wakeCatchUpPending else { return false }
        return allGreen(v, ["monitor", "search"])
    }

    private static func allGreen(_ v: BrainBarHealthVerdict, _ names: [String]) -> Bool {
        names.allSatisfy { name in v.checks.contains { $0.name == name } && v.checks.allSatisfy { $0.name != name || $0.state == .green } }
    }

    /// A thing is proven fine only under a fresh verdict, with no checking flag, the monitor green, and all of its own
    /// checks present and green (plus, for saving, a clean watcher file; for backups, a measured, sane observation).
    private static func provenFine(_ c: Capture) -> Set<Thing> {
        guard let v = c.verdict, (0...900).contains(c.verdictAwakeAgeS), !c.isChecking, !c.wakeCatchUpPending,
              allGreen(v, ["monitor"]) else { return [] }
        var fine = Set(four.filter { allGreen(v, checksOf[$0]!) })
        let w = c.watcher
        if w == nil || w?.readable == false || w?.fresh == false || w?.dbProbeFailed == true
            || w?.alertReasons.contains(where: { $0 != "quarantined_record" }) == true { fine.remove(.remembering) }
        if !c.backupsMeasured || v.checks.contains(where: { copies.contains($0.name) && ($0.ageS ?? 0) < 0 }) { fine.remove(.safe) }
        if c.maintenanceNightFailures >= 3 { fine.remove(.upkeep) }
        return fine
    }

    private static func allGoodProven(_ c: Capture) -> Bool {
        guard let v = c.verdict, v.dot == .green, allGreen(v, known), v.checks.allSatisfy({ known.contains($0.name) || $0.state == .green })
        else { return false }
        return c.backupsMeasured && provenFine(c) == Set(four)
    }

    // MARK: words for times (truth table Part 0): Gregorian, in the capture's time zone

    private static func formatter(_ format: String, _ c: Capture) -> DateFormatter {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = Calendar(identifier: .gregorian)
        f.timeZone = c.timeZone
        f.dateFormat = format
        return f
    }

    private static func hhmm(_ date: Date, _ c: Capture) -> String { formatter("HH:mm", c).string(from: date) }

    /// `{at}`, a past event: "today at 04:00", "yesterday at 23:10" (both under 24 h), else "Nov 9 at 03:44".
    private static func at(_ date: Date, _ c: Capture) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = c.timeZone
        if c.now.timeIntervalSince(date) < 86_400 {
            if calendar.isDate(date, inSameDayAs: c.now) { return "today at \(hhmm(date, c))" }
            if let yesterday = calendar.date(byAdding: .day, value: -1, to: c.now), calendar.isDate(date, inSameDayAs: yesterday) {
                return "yesterday at \(hhmm(date, c))"
            }
        }
        return formatter("MMM d 'at' HH:mm", c).string(from: date)
    }

    private static func percent(_ progress: Double) -> Int { Int((progress * 100).rounded()) }

    private static func valid(_ age: Double?) -> Bool { age.map { $0.isFinite && $0 >= 0 } == true }

    /// `{copy age}` from a backup check's wall-clock `ageS`: "21 hours old" under a day, else "from Nov 9 at 03:44"
    /// (`evaluatedAt − ageS`). Nil for no, a negative or non-finite age, or a time after now.
    private static func copyAge(_ age: Double?, _ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        guard let age, valid(age), v.evaluatedAt.addingTimeInterval(-age) <= c.now else { return nil }
        guard age >= 86_400 else {
            let hours = min(23, max(1, Int((age / 3600).rounded())))
            return hours == 1 ? "1 hour old" : "\(hours) hours old"
        }
        return "from \(at(v.evaluatedAt.addingTimeInterval(-age), c))"
    }

    /// `{over d}`: a strict lower bound from an awake-adjusted age, rounded down ("in over an hour"); nil under a minute.
    private static func over(_ age: Double) -> String? {
        guard age.isFinite, age > 60 else { return nil }
        func below(_ unit: Double) -> Int { Int((min(age, 1e12) / unit).rounded(.up)) - 1 }
        if below(3600) >= 2 { return "in over \(below(3600)) hours" }
        if below(3600) == 1 { return "in over an hour" }
        return below(60) == 1 ? "in over a minute" : "in over \(below(60)) minutes"
    }

    private static func parse(_ iso: String) -> Date? { ISO8601DateFormatter().date(from: iso) }
}
