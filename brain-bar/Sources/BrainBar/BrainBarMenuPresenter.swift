import Foundation

/// The one presentation function for the M menu and icon (phase-8 PRESENTATION-SPEC.md, truth table v3 Part 2).
/// Pure and deterministic: the same capture always gives the same presentation. Nothing in production calls it.
///
/// P1 derives the six calm cases (S1, S2 today and after the seam, S4, S7, S8); P2 the Needs-you rows of the eleven
/// needs cases (S13 with M1 and D13, S06, S07, S19, S32, F07, U03) with merges M2 and M5. Everything else takes ruling
/// R17's generic rows: a thing that needs him reads "Something BrainBar checks needs attention." (R24), a thing that cannot be
/// told reads "Checking…". P3 to P5 replace them family by family. `finalGuard` runs last, on every path.
enum BrainBarMenuPresenter {
    typealias P = BrainBarMenuPresentation
    typealias Capture = BrainBarMenuCapture
    typealias State = BrainBarHealthVerdict.State

    enum Thing: Int, CaseIterable, Sendable { case remembering, finding, safe, upkeep, system }
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
    static let four: [Thing] = [.remembering, .finding, .safe, .upkeep]
    private static let copies = ["backup_local", "backup_offsite", "transcripts"]

    private static func thing(of name: String) -> Thing { Thing.allCases.first { checksOf[$0]!.contains(name) } ?? .system }

    /// R17's generic rows.
    private static func needs(_ thing: Thing) -> Line { Line(thing: thing, level: .needs, strings: [Say.generic.text], worded: false) }
    private static func checking(_ thing: Thing) -> Line { Line(thing: thing, level: .cantTell, strings: [Say.checking.text], worded: false) }
    private static func worded(_ thing: Thing, _ strings: [String], _ rank: Rank, priority: Int = 9) -> Line {
        Line(thing: thing, level: .needs, strings: strings, rank: rank, priority: priority)
    }
    private static let handoff = [Say.cantFix.text, Say.giveReport.text]

    /// The only entry point. It is total over its capture type: no input traps, and the guard runs last on every path.
    static func present(_ c: Capture) -> P { finalGuard(c, derived(c)) }

    /// What the presenter says before its guard. Internal only as the seam for the property test (round 1 of #1790): every
    /// sentence it says is registered for `c`, which `present`'s output alone cannot show, as the guard drops one that is not.
    static func derived(_ c: Capture) -> P {
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
        if let failures = c.maintenanceNightFailures, failures >= 3 {  // U03; its second sentence only when saving and backups are proven fine
            lines.append(worded(.upkeep, [Say.u03(String(failures))]
                + (provenFine(c).isSuperset(of: [.remembering, .safe]) ? [Say.stillSaved.text] : []), .confirmed))
        }
        guard let w = c.watcher, w.readable, w.fresh else { return lines }
        if w.alertReasons.contains("offset_lag"), lagPastLimit(c) { lines.append(needs(.remembering)) }  // R02 escalated
        for reason in w.alertReasons where !["offset_lag", "quarantined_record"].contains(reason) {
            if reason == "coverage_drop", w.dbProbeFailed { continue }
            lines.append(needs(["coverage_drop", "file_ingestion_failure"].contains(reason) ? .remembering : .system))
        }
        return lines
    }

    static func lagPastLimit(_ c: Capture) -> Bool { c.lagEpisodeStart.map { c.now.timeIntervalSince($0) > 1800 } == true }

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
        if c.maintenanceNightFailures == nil { lines.append(checking(.upkeep)) }
        if c.backupDailyFailures == nil || c.jsonlBackupFailures == nil { lines.append(checking(.safe)) }
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
                lines.append(Line(thing: .remembering, level: .behind, strings: [Say.r02.text], headline: Say.r02Headline.text, spinner: true))
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
            guard let (_, onSchedule) = runningUpload(c), let start = uploadStart(c) else { return nil }
            let started = !hasWords(start.date, c) ? [] : [start.today ? Say.s03Started(hhmm(start.date, c)) : Say.s03StartedOn(at(start.date, c))]
            // R27: the estimate is said as a clock time only when it falls today.
            let eta = onSchedule?.eta.flatMap { calendar(c).isDate(minute($0, c), inSameDayAs: c.now) && hasWords($0, c) ? $0 : nil }
            let done = onSchedule?.progress.map { p in eta.map { Say.s03ProgressEta(String(percent(p)), hhmm($0, c)) } ?? Say.s03Progress(String(percent(p))) }
            let text = ([Say.s03.text] + started + (done.map { [$0] } ?? [])).joined(separator: " ")
            return Line(thing: .safe, level: .behind, strings: [text], headline: Say.s03Headline.text, progress: onSchedule?.progress.map(percent))
        case "search":
            // A name carried more than once is never read for its fields (round 1 of #1788, N11): its row stays generic.
            let searches = v.checks.filter { $0.name == "search" }
            let since = searches.count == 1 ? searches[0].since.flatMap(parse) : nil
            guard let since, (0...900).contains(c.now.timeIntervalSince(since)) else { return nil }
            return Line(thing: .finding, level: .cantTell, strings: [Say.f02.text])
        default:
            return nil
        }
    }

    /// U13: once `recovering` exists it is the evidence of a running upload; before PR-2 the live upload is. A cadence that
    /// is not a finite positive number makes the recovering unusable (no range is built from it).
    static func runningUpload(_ c: Capture) -> (started: Date, onSchedule: Capture.Recovering?)? {
        let onSchedule = c.offsiteRecovering.flatMap { r -> Capture.Recovering? in
            let quiet = c.now.timeIntervalSince(r.updatedAt)
            let sane = r.cadenceS.isFinite && r.cadenceS > 0 && r.startedAt <= c.now && r.progress.map { (0...1).contains($0) } != false
            return sane && quiet >= 0 && quiet <= 2 * r.cadenceS && r.eta.map { c.now <= $0 && sayable($0) } != false ? r : nil
        }
        let upload = c.upload.flatMap { $0.live ? $0.startedAt : nil }
        guard let started = c.offsiteRecovering == nil ? upload : onSchedule?.startedAt, started <= c.now, sayable(started) else { return nil }
        return (started, onSchedule)
    }

    /// The running upload's start, and whether it is on today's date in the capture's zone: a bare `{HH:mm}` reads as today,
    /// so a start on another day takes the day words every other past time uses (P2b, ruling R23).
    static func uploadStart(_ c: Capture) -> (today: Bool, date: Date)? {
        runningUpload(c).map {
            let date = minute($0.started, c)
            return (calendar(c).isDate(date, inSameDayAs: c.now), $0.started)
        }
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
            lines.append(worded(.safe, [Say.s19.text, Say.s19Memory.text], .custody, priority: 0))
            consumed.insert("integrity")
        }
        if let e = one("embedding"), e.state == .red || (e.state == .amber && within(e)) {  // F07
            let exact = c.keywordCoverage == 1 && searchProven(c) ? " " + Say.exactWords.text : ""
            lines.append(worded(.finding, [Say.f07.text, Say.f07Since(e.ageS.flatMap(over).map { " " + $0 } ?? "") + exact], .confirmed))
            consumed.insert("embedding")
        }
        // The database copies: S07, S32, S06 with failures, and S02 (drawn only through M1).
        let local = one("backup_local"), offsite = one("backup_offsite")
        let reds = [local, offsite].compactMap { $0 }.filter { $0.state == .red }
        var db: Line? = nil, dbNames: [String] = [], s02 = false
        if let local, let offsite {
            let faults = [local, offsite].filter { $0.state != .green }
            if local.state == .red, local.ageS == nil, Say.noDriveClock(c) {  // S07, only with no copy of any kind on record (R21, R25)
                (db, dbNames) = (worded(.safe, [Say.noRecord.text], .custody, priority: 1), ["backup_local"])
            } else if !reds.isEmpty, let upload = runningUpload(c), let start = uploadStart(c), hasWords(start.date, c) {  // S32, R8, R9, R23
                let age = c.backupsMeasured ? copyAge(offsite.ageS, c, v).map { Say.r15($0) + " " } : nil
                let p = upload.onSchedule?.progress.map { String(percent($0)) }
                let progress = start.today ? p.map { Say.s32Progress(hhmm(start.date, c), $0) } ?? Say.s32(hhmm(start.date, c))
                    : p.map { Say.s32ProgressOn(at(start.date, c), $0) } ?? Say.s32On(at(start.date, c))
                db = worded(.safe, [(age ?? "") + progress], local.state == .red ? .custody : .confirmed, priority: 2)
                db?.handoff = [Say.ifNotFinished.text, Say.giveReport.text]
                dbNames = reds.map(\.name)
            } else if let failures = c.backupDailyFailures, failures >= 2, !faults.isEmpty, c.upload == nil, c.offsiteRecovering == nil,
                      faults.allSatisfy({ ($0.state == .red && valid($0.ageS)) || ($0.state == .amber && within($0)) }) {  // S06
                let newest = [local, offsite].compactMap(\.ageS).filter { valid($0) }.min().flatMap { copyAge($0, c, v) }.map {
                    c.backupsMeasured ? Say.s06Newest($0) : Say.s06Confirmed($0)
                }
                db = worded(.safe, [Say.s06(String(failures))] + (newest.map { [$0] } ?? []),
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
        var detail = missing ? Say.s13DetailMissing.text : Say.s13Detail.text
        if absorbs, c.backupsMeasured, let age = copyAge(offsite?.ageS, c, v) {
            detail += " " + Say.r15(age)
        } else if absorbs, c.backupsMeasured, missing, offsite?.state == .red, Say.noDriveClock(c) {  // R14, R25
            detail += " " + Say.nothingInDrive.text
        }
        let transcriptsFailing = one("transcripts").map { [.amber, .red].contains($0.state) && $0.ageS != nil } == true && c.jsonlBackupFailures.map { $0 >= 2 } == true
        var s13 = worded(.safe, [missing ? Say.s13Missing.text : Say.s13.text, detail],
                         db?.rank == .custody || local?.state == .red ? .custody : .confirmed, priority: 4)
        s13.action = missing ? .connectDrive : .reconnectDrive
        // D13, R22: a failing transcript backup cannot upload either while Drive is red; a fact line directly under the
        // Drive row, not counted.
        if transcriptsFailing { s13.fact = [missing ? Say.d13Missing.text : Say.d13.text] }  // R26
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
        let nouns = Say.behindNoun
        let headline: String
        let mark: P.Mark
        let folded = behind.count == 1 && behind[0].headline != nil ? behind[0] : nil
        switch behind.count {
        case 1 where folded != nil: (headline, mark) = (folded?.headline ?? "", .busy)
        case 2: (headline, mark) = (Say.catchingTwo(Say.joined(behind.compactMap { nouns[$0.thing] })), .busy)
        case 3...: (headline, mark) = (Say.catchingFew.text, .busy)
        default:
            // A lone Behind line with no headline of its own is listed under a Can't-tell headline, never "a few things" (N3).
            let things = (cantTell + behind).map(\.thing)
            headline = things.contains(.safe) ? Say.cantConfirmBackups.text
                : things.count == 1 ? nouns[things[0]].map { Say.stillChecking($0) } ?? Say.stillCheckingFew.text
                : things.isEmpty ? Say.allGood.text : Say.stillCheckingFew.text
            mark = things.isEmpty ? .ok : .wait
        }
        var rows = [P.Row(kind: .headline, strings: [headline], mark: mark)]
        if let folded, folded.strings.first?.dropLast() != headline[...] {  // R02 repeats its headline: no line (R16)
            rows.append(P.Row(kind: .headlineDetail, strings: folded.strings, state: .behind, progressPercent: folded.progress))
        }
        if headline == Say.allGood.text, let allGood { rows.append(P.Row(kind: .headlineDetail, strings: [allGood])) }
        let listed = (folded == nil ? behind : []) + cantTell
        let fineList = Say.list(fine.compactMap { Say.fineNoun[$0] })
        let fineRow = P.Row(kind: .everythingElse, strings: [fine.count == 1 ? Say.fineIs(fineList) : Say.fineAre(fineList)], mark: listed.isEmpty ? nil : .ok)
        if !listed.isEmpty { rows.append(P.Row(kind: .separator)) }
        rows += listed.map { status($0, mark: $0.level == .behind ? .busy : .wait) }
        if !fine.isEmpty, headline != Say.allGood.text { rows.append(fineRow) }
        return .init(form: .calm, headline: headline, count: 0, icon: .resting, rows: rows + bottom)
    }

    private static let copyReport = action(.copyReport)

    private static func action(_ kind: P.ActionKind) -> P.Action {
        switch kind {
        case .reconnectDrive: P.Action(kind: kind, label: Say.reconnect.text)
        case .connectDrive: P.Action(kind: kind, label: Say.connect.text)
        case .copyReport: P.Action(kind: kind, label: Say.copyReport.text)
        case .openStorageSettings: P.Action(kind: kind, label: Say.storage.text)
        }
    }

    private static func needsForm(_ needsRows: [P.Row], statuses: [P.Row], everythingElse: Bool = false) -> P {
        let count = needsRows.filter { $0.kind == .needsYou }.count
        let headline = count == 1 ? Say.countOne.text : Say.countMany(String(count))
        var rows = [P.Row(kind: .headline, strings: [headline], mark: .marked), P.Row(kind: .separator)]
        for row in needsRows {  // a fact line rides directly under its Needs-you row (R22)
            let first = !rows.contains { $0.kind == .needsYou }
            rows.append(row.kind == .needsYou ? P.Row(kind: .needsYou, strings: row.strings, action: row.action, mark: .alertCircle, dividerBefore: !first) : row)
        }
        rows += statuses.map { var row = $0; row.mark = nil; return row }
        if everythingElse { rows.append(P.Row(kind: .everythingElse, strings: [Say.everythingElse.text], mark: .checkmark)) }
        return .init(form: .needs, headline: headline, count: count, icon: .marked, rows: rows + bottom)
    }

    /// X01, X02, and in P1 every path without a readable, fresh verdict and no red: nothing is confirmed.
    private static func checkingState(_ c: Capture) -> P {
        let launching = c.isChecking && c.verdict == nil
        let headline = c.wakeCatchUpPending && !launching ? Say.checkingSleep.text : Say.checkingMemory.text
        let last = c.lastKnownBackupAt.flatMap { $0 <= c.now && sayable($0) && hasWords($0, c) ? " " + Say.lastBackup(at($0, c)) : nil } ?? ""
        return .init(form: .calm, headline: headline, count: 0, icon: .resting,
                     rows: [P.Row(kind: .headline, strings: [headline], mark: .wait),
                            P.Row(kind: .headlineDetail, strings: [Say.notConfirmed.text + last],
                                  state: .cantTell)] + bottom)
    }

    /// R4: one bottom block for both forms.
    private static let bottom = [
        P.Row(kind: .separator),
        P.Row(kind: .standardItem, strings: [Say.settings.text], keyEquivalent: Say.settingsKey),
        P.Row(kind: .standardItem, strings: [Say.quit.text], keyEquivalent: Say.quitKey, optionAlternate: Say.restart.text),
    ]

    /// S01, the all-good line: the older of the offsite and transcript copies (U10), never a future time.
    private static func allGoodDetail(_ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        allGoodDate(c).flatMap { hasWords($0, c) ? Say.backedUp(at($0, c)) : nil }  // no words, no sentence (CI on #1790)
    }

    static func allGoodDate(_ c: Capture) -> Date? {
        guard let v = c.verdict else { return nil }
        // U10: the older copy, across every check of the two names (a duplicate carries another copy; round 1, Claude B3).
        let ages = v.checks.filter { ["backup_offsite", "transcripts"].contains($0.name) }.compactMap(\.ageS)
        guard let age = ages.max(), ages.allSatisfy({ $0.isFinite && $0 >= 0 }) else { return nil }
        let date = v.evaluatedAt.addingTimeInterval(-age)
        return date <= c.now && sayable(date) ? date : nil
    }

    // MARK: the final guard (invariants I1 to I8), the last step of every path

    /// Whatever was derived, the returned value satisfies the invariants; where it would not, the value is corrected
    /// conservatively (a sentence removed, a title replaced by R17's, the checking state, or the needs form with R17's row)
    /// rather than trusted. Every sentence it returns is a registered one (`Say.all`) whose condition holds.
    /// Internal only as the seam for the guard's own test (BrainBarMenuInvariantTests); production calls `present`.
    static func finalGuard(_ c: Capture, _ derived: P) -> P {
        var p = derived
        // I5: no straight apostrophe, no empty string, no row without words (a Needs-you row takes R17's). Labels and the
        // option alternate are the registered ones.
        func words(_ s: String) -> String { s.replacingOccurrences(of: "'", with: "’") }
        p.headline = words(p.headline)
        p.rows = p.rows.compactMap { row in
            var row = row
            row.strings = row.strings.map(words).filter { !$0.isEmpty }
            row.action = row.action.map { action($0.kind) }
            if row.kind == .separator || !row.strings.isEmpty { return row }
            return row.kind == .needsYou ? P.Row(kind: .needsYou, strings: needs(.system).strings + handoff, action: copyReport) : nil
        }
        /// A sentence is registered and, when it is of one of `kinds`, its condition holds on `drawn`.
        func holds(_ sentence: String, on drawn: P, _ kinds: Set<Sentence.Kind> = [.claim, .reassurance]) -> Bool {
            guard let (entry, values) = entry(for: sentence, c) else { return false }
            return !kinds.contains(entry.kind) || entry.holds?(c, drawn, values) != false
        }
        // I2 and every calm headline ("All good", a Behind headline, "Checking after sleep…"): one that is unregistered or
        // does not hold gives way to the checking state, in which nothing is confirmed.
        if p.form == .calm, p.count == 0, p.icon == .resting, !p.rows.contains(where: { $0.kind == .needsYou }), !holds(p.headline, on: p) {
            p = checkingState(c)
        }
        // I1, I7 and I4: a red, or a fact the app reads without the producer, always needs him; form, count, icon and the
        // count headline follow the Needs-you rows drawn.
        var needsRows = p.rows.filter { $0.kind == .needsYou }
        let red = c.verdict.map { $0.dot == .red || $0.checks.contains { $0.state == .red } } ?? false
        let w = c.watcher.flatMap { $0.readable && $0.fresh ? $0 : nil }
        let appFact = c.maintenanceNightFailures.map { $0 >= 3 } == true
            || w?.alertReasons.contains { !["offset_lag", "quarantined_record"].contains($0) && !($0 == "coverage_drop" && w?.dbProbeFailed == true) } == true
            || (w?.alertReasons.contains("offset_lag") == true && c.lagEpisodeStart.map { c.now.timeIntervalSince($0) > 1800 } == true)
        if needsRows.isEmpty, red || appFact || p.form == .needs || p.count != 0 || p.icon == .marked {
            needsRows = [P.Row(kind: .needsYou, strings: needs(.system).strings + handoff, action: copyReport)]
        }
        // A needs form is headed, in its headline and its headline row, by the count sentence for the rows it returns, never by
        // another registered sentence (P2c, CodeRabbit on #1788).
        let count = needsRows.count == 1 ? Say.countOne.text : Say.countMany(String(needsRows.count))
        if !needsRows.isEmpty, p.form != .needs || p.count != needsRows.count || p.icon != .marked || p.rows.filter({ $0.kind == .needsYou }) != needsRows
            || p.headline != count || p.rows.filter({ $0.kind == .headline }).map(\.strings) != [[count]] {
            let carried = p.rows.contains { $0.kind == .needsYou } ? p.rows.filter { [.needsYou, .factLine].contains($0.kind) }
                : needsRows + p.rows.filter { $0.kind == .factLine }
            p = needsForm(carried, statuses: p.rows.filter { $0.kind == .statusLine }, everythingElse: p.rows.contains { $0.kind == .everythingElse })
        }
        // I8 by sentence, then I3: a sentence no entry makes, or a claim whose condition fails, is corrected first (a Needs-you
        // row's title gives way to R17's, any other sentence is dropped); then every reassurance is proven against the rows
        // as corrected (R2-F2). Neither step adds or removes a Needs-you row, so I1, I4 and I7 still hold.
        func bySentence(_ keep: (String, _ title: Bool) -> String?) {
            p.rows = p.rows.compactMap { row in
                var row = row, title = row.kind == .needsYou
                row.strings = row.strings.compactMap { string in
                    let kept = sentences(string).compactMap { sentence in defer { title = false }; return keep(sentence, title) }
                    return kept.isEmpty ? nil : kept.joined(separator: " ")
                }
                if row.kind == .needsYou, row.strings.isEmpty { row.strings = needs(.system).strings + handoff }
                return row.strings.isEmpty && row.kind != .separator ? nil : row
            }
        }
        let drawn = p
        bySentence { holds($0, on: drawn, [.claim]) ? $0 : $1 ? Say.generic.text : nil }
        let corrected = p
        bySentence { sentence, _ in holds(sentence, on: corrected, [.reassurance]) ? sentence : nil }
        // B5 (round 1), last since P2c (CodeRabbit on #1788): once both passes have dropped what they drop, the standing items
        // keep their one permitted key and Option alternate, and no other row has either.
        p.rows = p.rows.map { row in
            var row = row
            let standing = row.kind == .standardItem ? row.strings : []
            row.keyEquivalent = standing == [Say.settings.text] ? Say.settingsKey : standing == [Say.quit.text] ? Say.quitKey : nil
            row.optionAlternate = standing == [Say.quit.text] && row.optionAlternate != nil ? Say.restart.text : nil
            return row
        }
        return p
    }

    /// The things "Everything else is fine." speaks for: every thing no drawn row speaks for, and all four beside a row
    /// whose first sentence speaks for no one thing (R17's generic rows).
    static func uncovered(_ p: P, _ c: Capture) -> Set<Thing> {
        var left = Set(four)
        for row in p.rows where [.needsYou, .statusLine, .factLine, .headlineDetail].contains(row.kind) {
            guard let first = row.strings.first.flatMap({ sentences($0).first }), let thing = entry(for: first, c)?.0.thing else { return Set(four) }
            left.remove(thing)
        }
        return left
    }

    /// A string's sentences: "A. B." gives ["A.", "B."].
    static func sentences(_ string: String) -> [String] {
        let parts = string.components(separatedBy: ". ")
        return parts.indices.map { $0 < parts.count - 1 ? parts[$0] + "." : parts[$0] }
    }

    /// F07's exact-words sentence needs search itself proven, beyond the coverage figure (stricter than the truth table).
    static func searchProven(_ c: Capture) -> Bool {
        guard let v = c.verdict, (0...900).contains(c.verdictAwakeAgeS), !c.isChecking, !c.wakeCatchUpPending else { return false }
        return allGreen(v, ["monitor", "search"])
    }

    private static func allGreen(_ v: BrainBarHealthVerdict, _ names: [String]) -> Bool {
        names.allSatisfy { name in v.checks.contains { $0.name == name } && v.checks.allSatisfy { $0.name != name || $0.state == .green } }
    }

    /// A thing is proven fine only under a fresh verdict, with no checking flag, the monitor green, and all of its own
    /// checks present and green (plus, for saving, a clean watcher file; for backups, a measured, sane observation).
    static func provenFine(_ c: Capture) -> Set<Thing> {
        guard let v = c.verdict, (0...900).contains(c.verdictAwakeAgeS), !c.isChecking, !c.wakeCatchUpPending,
              allGreen(v, ["monitor"]) else { return [] }
        var fine = Set(four.filter { allGreen(v, checksOf[$0]!) })
        let w = c.watcher
        if w == nil || w?.readable == false || w?.fresh == false || w?.dbProbeFailed == true
            || w?.alertReasons.contains(where: { $0 != "quarantined_record" }) == true { fine.remove(.remembering) }
        if !c.backupsMeasured || v.checks.contains(where: { copies.contains($0.name) && ($0.ageS ?? 0) < 0 }) { fine.remove(.safe) }
        if c.maintenanceNightFailures.map({ $0 < 3 }) != true { fine.remove(.upkeep) }
        if c.backupDailyFailures == nil || c.jsonlBackupFailures == nil { fine.remove(.safe) }
        return fine
    }

    static func allGoodProven(_ c: Capture) -> Bool {
        guard let v = c.verdict, v.dot == .green, allGreen(v, known), v.checks.allSatisfy({ known.contains($0.name) || $0.state == .green })
        else { return false }
        return c.backupsMeasured && provenFine(c) == Set(four)
    }

    private static func parse(_ iso: String) -> Date? { ISO8601DateFormatter().date(from: iso) }
}
