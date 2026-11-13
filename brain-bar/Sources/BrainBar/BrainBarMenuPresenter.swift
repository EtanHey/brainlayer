import Foundation

/// The one presentation function for the M menu and icon (phase-8 PRESENTATION-SPEC.md, truth table v3 Part 2).
/// Pure and deterministic: the same capture always gives the same presentation. Nothing in production calls it.
///
/// P1 of five derives the six calm cases (S1, S2 today and after the seam, S4, S7, S8). Everything else takes
/// ruling R17's generic rows: a thing that needs him reads "Something BrainBar checks is failing.", a thing that
/// cannot be told reads "Checking…". P2 to P5 replace them family by family. `finalGuard` runs last, on every path.
enum BrainBarMenuPresenter {
    typealias P = BrainBarMenuPresentation
    typealias Capture = BrainBarMenuCapture
    typealias State = BrainBarHealthVerdict.State

    private enum Thing: Int, CaseIterable { case remembering, finding, safe, upkeep, system }
    private enum Level { case cantTell, behind, needs }

    private struct Line {
        var thing: Thing
        var level: Level
        var strings: [String]
        /// A Behind row's headline (Part 2 step 8).
        var headline: String? = nil
        var spinner = false
        var progress: Int? = nil
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
    private static func needs(_ thing: Thing) -> Line { Line(thing: thing, level: .needs, strings: ["Something BrainBar checks is failing."]) }
    private static func checking(_ thing: Thing) -> Line { Line(thing: thing, level: .cantTell, strings: ["Checking…"]) }
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
        if c.maintenanceNightFailures >= 3 { lines.append(needs(.upkeep)) }  // U03
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
        var lines: [Line] = appFacts(c)
        for name in known where name != "monitor" {
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
            // U13: once `recovering` exists it is the evidence; before PR-2 the live upload is. A cadence that is not a
            // finite positive number makes the recovering unusable (no range is built from it).
            let onSchedule = c.offsiteRecovering.flatMap { r -> BrainBarMenuCapture.Recovering? in
                let quiet = c.now.timeIntervalSince(r.updatedAt)
                let sane = r.cadenceS.isFinite && r.cadenceS > 0 && r.startedAt <= c.now && r.progress.map { (0...1).contains($0) } != false
                return sane && quiet >= 0 && quiet <= 2 * r.cadenceS && r.eta.map { c.now <= $0 } != false ? r : nil
            }
            let upload = c.upload.flatMap { $0.live ? $0.startedAt : nil }
            guard let started = c.offsiteRecovering == nil ? upload : onSchedule?.startedAt, started <= c.now else { return nil }
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

    // MARK: Part 2: merge, count, order, headline

    private static func assemble(_ c: Capture, _ input: [Line], allGood: String?) -> P {
        // R17 and M5: one row per thing and level; the first line of a thing at that level speaks.
        func at(_ level: Level) -> [Line] { Thing.allCases.compactMap { t in input.first { $0.thing == t && $0.level == level } } }
        let needs = at(.needs), behind = at(.behind), cantTell = at(.cantTell)
        let fine = four.filter { t in !input.contains { $0.thing == t } }
        func status(_ line: Line, mark: P.Mark?) -> P.Row {
            P.Row(kind: .statusLine, strings: line.strings, mark: mark, state: line.level == .behind ? .behind : .cantTell, spinner: line.spinner)
        }
        if !needs.isEmpty {
            let rows = needs.map { P.Row(kind: .needsYou, strings: $0.strings + handoff, action: copyReport) }
            return needsForm(rows, statuses: (behind + cantTell).map { status($0, mark: nil) })
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

    private static let copyReport = P.Action(kind: .copyReport, label: "Copy report for an agent")

    private static func needsForm(_ needsRows: [P.Row], statuses: [P.Row]) -> P {
        let headline = needsRows.count == 1 ? "1 thing needs you" : "\(needsRows.count) things need you"
        var rows = [P.Row(kind: .headline, strings: [headline], mark: .marked), P.Row(kind: .separator)]
        for (i, row) in needsRows.enumerated() {
            rows.append(P.Row(kind: .needsYou, strings: row.strings, action: row.action, mark: .alertCircle, dividerBefore: i > 0))
        }
        rows += statuses.map { var row = $0; row.mark = nil; return row }
        return .init(form: .needs, headline: headline, count: needsRows.count, icon: .marked, rows: rows + bottom)
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
        // I3: a reassurance stays only when every thing it covers is proven fine.
        let fine = provenFine(c)
        p.rows.removeAll { $0.kind == .everythingElse && !covered($0).isSubset(of: fine) }
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
            p = needsForm(needsRows, statuses: p.rows.filter { $0.kind == .statusLine })
        }
        return p
    }

    /// The things a reassurance speaks for. P1 draws no "Everything else is fine.", so it must stand for all four.
    private static func covered(_ row: P.Row) -> Set<Thing> {
        let text = row.strings.joined(separator: " ").lowercased()
        let named = Set([(Thing.remembering, "saving"), (.finding, "search"), (.safe, "backups"), (.upkeep, "upkeep")]
            .filter { text.contains($0.1) }.map(\.0))
        return text.hasPrefix("everything else") || named.isEmpty ? Set(four) : named
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

    private static func parse(_ iso: String) -> Date? { ISO8601DateFormatter().date(from: iso) }
}
