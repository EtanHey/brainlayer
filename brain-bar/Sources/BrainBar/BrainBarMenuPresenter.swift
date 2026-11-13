import Foundation

/// The one presentation function for the M menu and icon (phase-8 PRESENTATION-SPEC.md, truth table v3
/// Part 2). Pure and deterministic: the same capture always gives the same presentation.
///
/// Rows the spec's table does not cover are still derived from their check's state, without words:
/// a red check is a Needs-you row with the hand-off, anything else not green is a Can't-tell row. A check
/// this function cannot word yet never hides a fault and never reads "All good".
enum BrainBarMenuPresenter {
    typealias P = BrainBarMenuPresentation
    typealias Capture = BrainBarMenuCapture

    enum Thing: Int, CaseIterable { case remembering, finding, safe, upkeep, system }
    enum Level: Int { case fine, cantTell, behind, needs }
    /// U11: custody-critical, then confirmed, then escalated.
    enum Rank: Int { case custody, confirmed, escalated }

    struct Line {
        var thing: Thing
        var level: Level
        var rank = Rank.confirmed
        /// Within one thing, the lower speaks for a merged row (C14: S19 before S06 and S07).
        var priority = 9
        var strings: [String] = []
        var action: P.ActionKind? = nil
        var handoff = ["BrainBar can’t fix this itself.", "Give the report to an agent."]
        var spinner = false
        var progress: Int? = nil
        var factLine = false
    }

    static func present(_ c: Capture) -> P {
        var lines: [Line]
        if let v = c.verdict, !c.wakeCatchUpPending, (0...900).contains(c.verdictAwakeAgeS) {
            lines = derive(c, v)
        } else if let v = c.verdict, !c.wakeCatchUpPending, c.verdictAwakeAgeS > 2700 {
            lines = [Line(thing: .system, level: .needs, rank: .escalated,
                          strings: ["BrainBar hasn’t been able to check on your memory since \(hhmm(v.evaluatedAt, c))."],
                          action: .copyReport)]
                + [Thing.remembering, .finding, .safe, .upkeep].map { Line(thing: $0, level: .cantTell, strings: ["Checking…"]) }
        } else {
            // X01 launch, X02 wake, X03 stale, X04 unreadable: all four Can't tell.
            let afterSleep = c.wakeCatchUpPending && !(c.isChecking && c.verdict == nil)
            let detail = "Saving, search, backups and upkeep aren’t confirmed yet."
                + (c.lastKnownBackupAt.map { " Last backup \(at($0, c))." } ?? "")
            return .init(form: .calm, headline: afterSleep ? "Checking after sleep…" : "Checking on your memory…", count: 0, icon: .resting,
                         rows: [P.Row(kind: .headline, strings: [afterSleep ? "Checking after sleep…" : "Checking on your memory…"], mark: .wait),
                                P.Row(kind: .headlineDetail, strings: [detail])] + bottom)
        }
        return assemble(c, lines, safeDetail: c.verdict.flatMap { allGoodDetail(c, $0) })
    }

    // MARK: Part 1 rows from one capture

    static func derive(_ c: Capture, _ v: BrainBarHealthVerdict) -> [Line] {
        func check(_ name: String) -> BrainBarHealthVerdict.Check? { v.checks.first { $0.name == name } }
        var lines: [Line] = []
        func underived(_ thing: Thing, _ name: String) {
            let state = check(name)?.state ?? .unknown
            guard state != .green else { return }
            let needs = state == .red || (name == "monitor" && state == .amber)
            lines.append(Line(thing: thing, level: needs ? .needs : .cantTell, action: needs ? .copyReport : nil))
        }

        // Safe: the database copies (S02, S03, S06, S07, S32), then M1 into S13.
        let local = check("backup_local"), off = check("backup_offsite")
        let db = [local, off].compactMap { $0 }
        let dbRed = db.contains { $0.state == .red }, dbAmber = db.contains { $0.state == .amber }
        let onSchedule = c.offsiteRecovering.flatMap { r in
            c.now.timeIntervalSince(r.updatedAt) <= 2 * r.cadenceS && r.eta.map { c.now <= $0 } != false ? r : nil
        }
        let started = onSchedule?.startedAt ?? (c.upload?.live == true ? c.upload?.startedAt : nil)
        let custody = local?.state == .red
        var dbLine: Line?
        if local?.state == .red, local?.ageS == nil {
            dbLine = Line(thing: .safe, level: .needs, rank: .custody, priority: 1, strings: ["No verified backup on record"], action: .copyReport)
        } else if dbRed, let started {
            let progress = c.offsiteRecovering?.progress.map { "is \(percent($0))% done" } ?? "is still running"
            let copy = copyAge(off?.ageS, c, v).map { "Your newest backup in Google Drive is \($0). " } ?? ""
            dbLine = Line(thing: .safe, level: .needs, rank: custody ? .custody : .confirmed, priority: 2,
                          strings: [copy + "A backup to Google Drive started at \(hhmm(started, c)) and \(progress)."], action: .copyReport,
                          handoff: ["If this backup doesn’t finish, BrainBar can’t fix it itself.", "Give the report to an agent."])
        } else if dbAmber, let started {
            var text = "To Google Drive. Started at \(hhmm(started, c))."
            if let r = onSchedule, let p = r.progress { text += " \(percent(p))% done" + (r.eta.map { ", about \(hhmm($0, c))." } ?? ".") }
            dbLine = Line(thing: .safe, level: .behind, strings: [text], progress: onSchedule?.progress.map(percent))
        } else if dbRed || (dbAmber && c.backupDailyFailures >= 2) {
            let n = c.backupDailyFailures
            let newest = db.compactMap(\.ageS).min().flatMap { copyAge($0, c, v) }.map {
                c.backupsMeasured ? "Your newest backup is \($0)." : "The newest backup BrainBar can confirm is \($0)."
            }
            dbLine = Line(thing: .safe, level: .needs, rank: custody ? .custody : dbRed ? .confirmed : .escalated, priority: 3,
                          strings: [n >= 2 ? "The last \(n) backups didn’t finish" : "Scheduled backups aren’t running"] + (newest.map { [$0] } ?? []),
                          action: .copyReport)
        } else if dbAmber {
            dbLine = Line(thing: .safe, level: .cantTell)  // S04, S05: the deadline and next run are not in the capture
        } else if c.backupDailyFailures == 1 {
            dbLine = Line(thing: .safe, level: .fine)  // S02: a fact M draws only when M1 absorbs it
        }
        let driveRed = check("drive_auth")?.state == .red
        if driveRed {
            let missing = c.driveAuth == .missing
            let absorbs = dbLine.map { $0.level != .behind } ?? false
            var detail = "Backups can’t upload until you \(missing ? "connect" : "reconnect") it."
            if absorbs {
                detail += missing ? " Nothing is backed up to Google Drive yet."
                    : copyAge(off?.ageS, c, v).map { " Your newest backup in Google Drive is \($0)." } ?? ""
            }
            lines.append(Line(thing: .safe, level: .needs, rank: absorbs && dbLine?.rank == .custody ? .custody : .confirmed, priority: 4,
                              strings: [missing ? "Google Drive isn’t connected" : "Google Drive is disconnected", detail],
                              action: missing ? .connectDrive : .reconnectDrive))
            if absorbs { dbLine = nil }
        } else {
            underived(.safe, "drive_auth")
        }
        if let dbLine, dbLine.level != .fine { lines.append(dbLine) }
        if let t = check("transcripts"), t.state != .green {
            let failing = t.state == .red || (t.state == .amber && c.jsonlBackupFailures >= 2)
            if failing && driveRed {  // D13: a fact line under the Drive row, not counted
                lines.append(Line(thing: .safe, level: .fine, strings: ["Conversation backups can’t upload either until Drive is reconnected."], factLine: true))
            } else {
                lines.append(Line(thing: .safe, level: failing ? .needs : .cantTell, rank: t.state == .red ? .confirmed : .escalated,
                                  action: failing ? .copyReport : nil))
            }
        }
        if check("integrity")?.state == .red {
            lines.append(Line(thing: .safe, level: .needs, rank: .custody, priority: 0,
                              strings: ["The newest checked backup is damaged", "Your memory itself may be damaged too."], action: .copyReport))
        } else {
            underived(.safe, "integrity")
        }
        underived(.safe, "disk")

        // Remembering: R02 from the watcher file; ingest and stores by state.
        underived(.remembering, "ingest")
        underived(.remembering, "stores")
        if let w = c.watcher, w.readable, w.fresh {
            if w.dbProbeFailed {
                lines.append(Line(thing: .remembering, level: .cantTell))
            } else if w.alertReasons.contains(where: { !["offset_lag", "quarantined_record"].contains($0) }) {
                lines.append(Line(thing: .remembering, level: .needs, action: .copyReport))
            }
            if w.alertReasons.contains("offset_lag"), check("ingest")?.state == .green {
                if let start = c.lagEpisodeStart, c.now.timeIntervalSince(start) > 1800 {
                    lines.append(Line(thing: .remembering, level: .needs, rank: .escalated,
                                      strings: ["BrainLayer has been behind on saving new conversations for over 30 minutes."], action: .copyReport))
                } else {
                    lines.append(Line(thing: .remembering, level: .behind, strings: ["Catching up on new conversations."], spinner: true))
                }
            }
        } else {
            lines.append(Line(thing: .remembering, level: .cantTell))
        }

        // Finding: F02 (escalates at 15 min, U5) and F07.
        if let s = check("search"), s.state == .amber {
            if let since = s.since.flatMap(parse), c.now.timeIntervalSince(since) >= 900 {
                let when = c.calendar.isDate(since, inSameDayAs: c.now) ? hhmm(since, c) : at(since, c)
                lines.append(Line(thing: .finding, level: .needs, rank: .escalated,
                                  strings: ["BrainBar hasn’t been able to confirm agents can search since \(when)."], action: .copyReport))
            } else {
                lines.append(Line(thing: .finding, level: .cantTell, strings: ["Can’t confirm agents can search right now."]))
            }
        } else {
            underived(.finding, "search")
        }
        if let e = check("embedding"), e.state == .red || (e.state == .amber && (e.ageS ?? .infinity) <= (e.thresholdS ?? -1)) {
            var detail = "BrainBar hasn’t seen the memories waiting for it go down" + (e.ageS.map { " \(overDuration($0))" } ?? "") + "."
            if c.keywordCoverage == 1 { detail += " Agents can still find every memory by exact words." }
            lines.append(Line(thing: .finding, level: .needs, strings: ["Search by meaning isn’t catching up", detail], action: .copyReport))
        } else {
            underived(.finding, "embedding")
        }

        // Upkeep: U03. Its second sentence needs Remembering and Safe both Fine.
        if c.maintenanceNightFailures >= 3 {
            let calm = !lines.contains { ($0.thing == .remembering || $0.thing == .safe) && $0.level != .fine }
            lines.append(Line(thing: .upkeep, level: .needs,
                              strings: ["Nightly upkeep failed \(c.maintenanceNightFailures) nights in a row"]
                                  + (calm ? ["Your memory is still being saved and backed up, as of the last checks."] : []),
                              action: .copyReport))
        }
        underived(.system, "monitor")
        return lines
    }

    // MARK: Part 2: merge, count, order, headline

    static func assemble(_ c: Capture, _ lines: [Line], safeDetail: String?) -> P {
        // M2, M3: one Needs-you row per thing and action, spoken by the highest class, then priority.
        var needs: [Line] = []
        for line in lines.filter({ $0.level == .needs }).sorted(by: { ($0.rank.rawValue, $0.priority) < ($1.rank.rawValue, $1.priority) })
            where !needs.contains(where: { $0.thing == line.thing && $0.action == line.action }) { needs.append(line) }
        needs.sort { ($0.rank.rawValue, $0.thing.rawValue, $0.priority) < ($1.rank.rawValue, $1.thing.rawValue, $1.priority) }
        // M5: the first Behind row of a thing speaks.
        var behind: [Line] = []
        for line in lines where line.level == .behind && !behind.contains(where: { $0.thing == line.thing }) { behind.append(line) }
        behind.sort { $0.thing.rawValue < $1.thing.rawValue }
        let cantTell = lines.filter { $0.level == .cantTell }.sorted { $0.thing.rawValue < $1.thing.rawValue }
        let fine = [Thing.remembering, .finding, .safe, .upkeep].filter { thing in !lines.contains { $0.thing == thing && $0.level != .fine } }

        var rows: [P.Row]
        let headline: String
        if !needs.isEmpty {
            headline = needs.count == 1 ? "1 thing needs you" : "\(needs.count) things need you"
            rows = [P.Row(kind: .headline, strings: [headline], mark: .marked), P.Row(kind: .separator)]
            for (i, line) in needs.enumerated() {
                rows.append(P.Row(kind: .needsYou, strings: line.strings + (line.action == .copyReport ? line.handoff : []),
                                  action: line.action.map { P.Action(kind: $0, label: label($0)) }, dividerBefore: i > 0))
            }
            rows += lines.filter(\.factLine).map { P.Row(kind: .factLine, strings: $0.strings) }
            rows += (behind + cantTell).map { P.Row(kind: .statusLine, strings: $0.strings, spinner: $0.spinner) }
            if !fine.isEmpty, !needs.contains(where: { $0.rank == .custody }) {
                rows.append(P.Row(kind: .everythingElse, strings: ["Everything else is fine."]))
            }
        } else {
            let sign: P.Mark
            var folded: Line?
            let nouns: [Thing: String] = [.remembering: "new conversations", .finding: "search", .safe: "backups", .upkeep: "upkeep"]
            switch behind.count {
            case 1:
                folded = behind[0]
                headline = [.remembering: "Catching up on new conversations", .finding: "Making new memories searchable"][behind[0].thing] ?? "Backing up now"
                sign = .busy
            case 2:
                headline = "Catching up on \(nouns[behind[0].thing]!) and \(nouns[behind[1].thing]!)"
                sign = .busy
            case 3...:
                headline = "Catching up on a few things"
                sign = .busy
            default:
                let things = Set(cantTell.map(\.thing))
                headline = things.contains(.safe) ? "Can’t confirm backups yet"
                    : things.count == 1 ? "Still checking \(nouns[things.first!] ?? "a few things")"
                    : things.isEmpty ? "All good" : "Still checking a few things"
                sign = things.isEmpty ? .ok : .wait
            }
            rows = [P.Row(kind: .headline, strings: [headline], mark: sign)]
            if let folded, folded.strings.first?.dropLast() != headline[...] {
                rows.append(P.Row(kind: .headlineDetail, strings: folded.strings, progressPercent: folded.progress))
            }
            if behind.isEmpty, cantTell.isEmpty, let safeDetail { rows.append(P.Row(kind: .headlineDetail, strings: [safeDetail])) }
            let listed = (behind + cantTell).filter { $0.thing != folded?.thing || $0.level != .behind }
            var fineNouns = fine.map { [.remembering: "saving", .finding: "search", .safe: "backups", .upkeep: "upkeep"][$0]! }
            let last = fineNouns.popLast()
            let joined = (fineNouns.isEmpty ? last ?? "" : fineNouns.joined(separator: ", ") + " and " + (last ?? ""))
            let everythingElse = fine.isEmpty || (behind.isEmpty && cantTell.isEmpty) ? []
                : [P.Row(kind: .everythingElse, strings: [joined.prefix(1).uppercased() + joined.dropFirst() + (fine.count == 1 ? " is fine." : " are fine.")])]
            rows += listed.isEmpty ? everythingElse
                : [P.Row(kind: .separator)] + listed.map { P.Row(kind: .statusLine, strings: $0.strings, spinner: $0.spinner) } + everythingElse
        }
        return .init(form: needs.isEmpty ? .calm : .needs, headline: headline, count: needs.count,
                     icon: needs.isEmpty ? .resting : .marked, rows: rows + bottom)
    }

    /// R4: one bottom block for both forms.
    static let bottom = [
        P.Row(kind: .separator),
        P.Row(kind: .standardItem, strings: ["Settings…"], keyEquivalent: "⌘,"),
        P.Row(kind: .standardItem, strings: ["Quit BrainBar"], keyEquivalent: "⌘Q", optionAlternate: "Restart BrainBar"),
    ]

    static func label(_ kind: P.ActionKind) -> String {
        switch kind {
        case .reconnectDrive: "Reconnect Google Drive"
        case .connectDrive: "Connect Google Drive"
        case .copyReport: "Copy report for an agent"
        case .openStorageSettings: "Open Storage Settings"
        }
    }

    /// S01, the all-good line: the older of the offsite and transcript copies (wall-clock ages, U10).
    static func allGoodDetail(_ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        let ages = v.checks.filter { ["backup_offsite", "transcripts"].contains($0.name) }.compactMap(\.ageS)
        return ages.max().map { "Backed up to Google Drive \(at(v.evaluatedAt.addingTimeInterval(-$0), c))." }
    }

    // MARK: words for times (truth table Part 0). No existing app formatter produces these forms.

    static func formatter(_ format: String, _ c: Capture) -> DateFormatter {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = c.calendar
        f.timeZone = c.calendar.timeZone
        f.dateFormat = format
        return f
    }

    static func hhmm(_ date: Date, _ c: Capture) -> String { formatter("HH:mm", c).string(from: date) }

    /// `{at}`: "today at 04:00", "yesterday at 23:10", else "Nov 9 at 03:44".
    static func at(_ date: Date, _ c: Capture) -> String {
        if c.now.timeIntervalSince(date) < 86_400 {
            if c.calendar.isDate(date, inSameDayAs: c.now) { return "today at \(hhmm(date, c))" }
            if let yesterday = c.calendar.date(byAdding: .day, value: -1, to: c.now), c.calendar.isDate(date, inSameDayAs: yesterday) {
                return "yesterday at \(hhmm(date, c))"
            }
        }
        return formatter("MMM d 'at' HH:mm", c).string(from: date)
    }

    /// `{copy age}`: "21 hours old" under a day, else "from Nov 9 at 03:44" (`evaluatedAt − ageS`).
    static func copyAge(_ age: Double?, _ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        guard let age else { return nil }
        guard age >= 86_400 else {
            let hours = max(1, Int((age / 3600).rounded()))
            return hours == 1 ? "1 hour old" : "\(hours) hours old"
        }
        return "from \(at(v.evaluatedAt.addingTimeInterval(-age), c))"
    }

    /// `{over d}`: a lower bound from an awake-adjusted age, rounded down.
    static func overDuration(_ age: Double) -> String {
        let hours = Int(age / 3600)
        return hours >= 2 ? "in over \(hours) hours" : hours == 1 ? "in over an hour" : "in over \(max(1, Int(age / 60))) minutes"
    }

    static func percent(_ progress: Double) -> Int { Int((progress * 100).rounded()) }

    static func parse(_ iso: String) -> Date? { ISO8601DateFormatter().date(from: iso) }
}
