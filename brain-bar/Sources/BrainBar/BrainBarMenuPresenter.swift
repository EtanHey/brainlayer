import Foundation

/// The one presentation function for the M menu and icon (phase-8 PRESENTATION-SPEC.md, truth table v3
/// Part 2). Pure and deterministic: the same capture always gives the same presentation.
///
/// Each row speaks its truth-table row's words (Part 1; outside the case table they are ruled, untested).
/// A check, note or dot it cannot place still counts: a red is never hidden, and it never reads "All good"
/// over something it cannot confirm.
enum BrainBarMenuPresenter {
    typealias P = BrainBarMenuPresentation
    typealias Capture = BrainBarMenuCapture
    typealias Check = BrainBarHealthVerdict.Check

    enum Thing: Int, CaseIterable { case remembering, finding, safe, upkeep, system }
    enum Level: Int { case fine, cantTell, behind, needs }
    /// U11: custody-critical, then confirmed, then escalated.
    enum Rank: Int { case custody, confirmed, escalated }

    struct Line {
        var thing: Thing
        var level: Level
        var strings: [String]
        var rank = Rank.confirmed
        /// Inside a class, by severity (truth table, "Erratum, 13 Nov 2026": S19, then S07, then S06).
        var priority = 9
        var action: P.ActionKind? = nil
        var handoff = ["BrainBar can’t fix this itself.", "Give the report to an agent."]
        var spinner = false
        var progress: Int? = nil
        var factLine = false
        /// A Behind row's headline (Part 2 step 8).
        var headline: String? = nil
        /// The red check a Needs-you row comes from: X02 lets it through, X03 keeps it.
        var redCheck: String? = nil
        /// A database-backup row that M1 absorbs into S13.
        var absorbable = false
        var id = ""
    }

    static func needs(_ thing: Thing, _ strings: [String], _ rank: Rank = .confirmed, _ action: P.ActionKind = .copyReport,
                      red: String? = nil, priority: Int = 9) -> Line {
        Line(thing: thing, level: .needs, strings: strings, rank: rank, priority: priority, action: action, redCheck: red)
    }

    static func cantTell(_ thing: Thing, _ text: String) -> Line { Line(thing: thing, level: .cantTell, strings: [text]) }

    static func behind(_ thing: Thing, _ text: String, headline: String) -> Line {
        Line(thing: thing, level: .behind, strings: [text], headline: headline)
    }

    static let known = ["backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk",
                        "ingest", "embedding", "stores", "search", "monitor"]
    static let checking = [Thing.remembering, .finding, .safe, .upkeep].map { cantTell($0, "Checking…") }
    static let unrecognised = "A BrainLayer job reported a problem BrainBar doesn’t recognise."  // X10

    static func present(_ c: Capture) -> P {
        if let v = c.verdict, !c.wakeCatchUpPending, (0...900).contains(c.verdictAwakeAgeS) {
            return assemble(c, derive(c, v), allGood: allGoodDetail(c, v))
        }
        // X01 launch, X02 wake, X03 stale, X04 unreadable: Can't tell, except the reds that stay.
        let launching = c.isChecking && c.verdict == nil
        var kept: [Line] = []
        if let v = c.verdict, !launching {
            kept = derive(c, v).filter { $0.level == .needs && $0.redCheck != nil }
            if c.wakeCatchUpPending {  // X02: a red outside ingest, stores and embedding breaks through (C9)
                kept.removeAll { ["ingest", "stores", "embedding"].contains($0.redCheck ?? "") }
            } else {  // X03: a stale red keeps its rows, each with when it was last checked
                for i in kept.indices { kept[i].strings.append("Last checked \(agoText(v.evaluatedAt, c)).") }
                if c.verdictAwakeAgeS > 2700 {
                    kept.append(needs(.system, ["BrainBar hasn’t been able to check on your memory \(sinceText(v.evaluatedAt, c))."], .escalated))
                }
            }
        } else if c.verdict == nil, !launching, !c.wakeCatchUpPending, let episode = c.unreadableVerdictEpisode, episode.awakeS > 1800 {
            kept = [needs(.system, ["BrainBar hasn’t been able to check on your memory \(sinceText(episode.start, c))."], .escalated)]
        }
        if !kept.isEmpty {
            return assemble(c, kept + checking.filter { line in !kept.contains { $0.thing == line.thing } }, allGood: nil)
        }
        let headline = c.wakeCatchUpPending && !launching ? "Checking after sleep…" : "Checking on your memory…"
        let last = c.lastKnownBackupAt.flatMap { $0 <= c.now ? " Last backup \(at($0, c))." : nil } ?? ""
        return .init(form: .calm, headline: headline, count: 0, icon: .resting,
                     rows: [P.Row(kind: .headline, strings: [headline], mark: .wait),
                            P.Row(kind: .headlineDetail, strings: ["Saving, search, backups and upkeep aren’t confirmed yet." + last],
                                  state: .cantTell)] + bottom)
    }

    // MARK: Part 1 rows from one capture

    static func derive(_ c: Capture, _ v: BrainBarHealthVerdict) -> [Line] {
        func check(_ name: String) -> Check? { v.checks.first { $0.name == name } }
        func state(_ name: String) -> BrainBarHealthVerdict.State { check(name)?.state ?? .unknown }
        /// U5: a not-confirmed amber escalates once its typed `since` is older than its window.
        func past(_ name: String, _ window: TimeInterval) -> Bool {
            check(name)?.since.flatMap(parse).map { c.now.timeIntervalSince($0) > window } ?? false
        }
        func since(_ name: String) -> String { check(name)?.since.flatMap(parse).map { " " + sinceText($0, c) } ?? "" }
        var lines: [Line] = []

        // Safe. The copies first (S02 to S09, S25, S32), then Drive (S12, S13 with M1, S15, S16).
        let driveRed = state("drive_auth") == .red
        var db = backupLine(c, v, [check("backup_local"), check("backup_offsite")], failures: c.backupDailyFailures,
                            last: c.backupDailyLastRunAt, next: c.backupDailyNextRunAt, database: true)
        if driveRed {
            let missing = c.driveAuth == .missing
            let absorbs = db?.absorbable == true
            var detail = "Backups can’t upload until you \(missing ? "connect" : "reconnect") it."
            if absorbs, let age = copyAge(check("backup_offsite")?.ageS, c, v) {
                detail += " Your newest backup in Google Drive is \(age)."  // R15
            } else if absorbs, missing, check("backup_offsite")?.ageS == nil {
                detail += " Nothing is backed up to Google Drive yet."  // R14: the Drive copy never verified
            }
            lines.append(needs(.safe, [missing ? "Google Drive isn’t connected" : "Google Drive is disconnected", detail],
                               absorbs && db?.rank == .custody ? .custody : .confirmed, missing ? .connectDrive : .reconnectDrive,
                               red: "drive_auth", priority: 4))
            if absorbs { db = nil }
        } else if state("drive_auth") == .amber {
            let offered = [.expiring, .invalid, .missing].contains(c.driveAuth)
            lines.append(past("drive_auth", 3600)
                ? needs(.safe, ["BrainBar hasn’t been able to confirm Google Drive access\(since("drive_auth"))."], .escalated,
                        offered ? .reconnectDrive : .copyReport)
                : cantTell(.safe, "Can’t confirm Google Drive access right now."))
        } else if state("drive_auth") == .unknown {
            lines.append(cantTell(.safe, "Can’t check Google Drive access right now."))
        }
        if c.driveAuth == .expiring {
            let ends = c.driveExpiresAt.map { "Google Drive access ends \(nextText($0, c)). " } ?? ""
            lines.append(needs(.safe, [ends + "Reconnect to keep backups going."], .confirmed, .reconnectDrive, priority: 5))
        }
        if let db, db.level != .fine { lines.append(db) }
        if let t = backupLine(c, v, [check("transcripts")], failures: c.jsonlBackupFailures,
                              last: c.jsonlBackupLastRunAt, next: c.jsonlBackupNextRunAt, database: false) {
            if driveRed, t.id == "S06" {  // D13: a fact line under the Drive row, not counted
                lines.append(Line(thing: .safe, level: .fine, strings: ["Conversation backups can’t upload either until Drive is reconnected."],
                                  factLine: true))
            } else if t.level != .fine {
                lines.append(t)
            }
        }
        switch state("integrity") {
        case .red:
            lines.append(needs(.safe, ["The newest checked backup is damaged", "Your memory itself may be damaged too."], .custody,
                               red: "integrity", priority: 0))
        case .amber where past("integrity", 1800):
            lines.append(needs(.safe, ["BrainBar hasn’t been able to confirm your newest backup is sound\(since("integrity"))."], .escalated))
        case .amber, .unknown: lines.append(cantTell(.safe, "Can’t confirm your newest backup is sound yet."))
        case .green: break
        }
        switch (state("disk"), check("disk")?.blind) {
        case (.red, _):
            lines.append(needs(.safe, ["Your Mac didn’t have room for the next backup at the last check."], .confirmed, .openStorageSettings,
                               red: "disk"))
        case (.amber, true?):
            lines.append(past("disk", 1800)
                ? needs(.safe, ["BrainBar hasn’t been able to check free space for backups\(since("disk"))."], .escalated)
                : cantTell(.safe, "Can’t check free space for backups right now."))
        case (.amber, false?):
            lines.append(needs(.safe, ["Your Mac is running low on room for backups. The next one still fits."], .confirmed, .openStorageSettings))
        case (.amber, nil):  // S21: measured and blind collide until the producer sends `blind`
            lines.append(needs(.safe, ["Your Mac may not have enough spare room for backups."], .confirmed, .openStorageSettings))
        case (.unknown, _): lines.append(cantTell(.safe, "Can’t check free space for backups right now."))
        case (.green, _): break
        }

        // Remembering: ingest (R03 to R06, R23), stores (R14, R15, R26), the watcher file (R02, R08, R09, R24).
        let notSaving = "BrainBar hasn’t been able to confirm new conversations are being saved\(since("ingest"))."
        let cantConfirmSaving = "Can’t confirm new conversations are being saved right now."
        let ingest = check("ingest")
        switch state("ingest") {
        case .red: lines.append(needs(.remembering, [notSaving], red: "ingest"))
        case .amber:
            if let age = ingest?.ageS, let limit = ingest?.thresholdS, age > limit {  // R04: the maintenance-pause demotion
                lines.append(past("ingest", 9000) ? needs(.remembering, [notSaving], .escalated)
                    : behind(.remembering, "Saving is paused while nightly upkeep runs.", headline: "Saving is paused for nightly upkeep"))
            } else if ingest?.ageS == nil, past("ingest", 1800) {
                lines.append(needs(.remembering, [notSaving], .escalated))
            } else {
                lines.append(cantTell(.remembering, cantConfirmSaving))
            }
        case .unknown: lines.append(cantTell(.remembering, cantConfirmSaving))
        case .green: break
        }
        let unconfirmedSaves = "Your agents’ saves haven’t been confirmed\(since("stores"))."
        switch state("stores") {
        case .red: lines.append(needs(.remembering, [unconfirmedSaves], red: "stores"))
        case .amber where past("stores", 1800): lines.append(needs(.remembering, [unconfirmedSaves], .escalated))
        case .amber where (check("stores")?.ageS ?? 0) > 0:
            lines.append(cantTell(.remembering, "Some memories your agents sent are waiting to be saved."))
        case .amber, .unknown: lines.append(cantTell(.remembering, "Can’t confirm your agents’ saves are getting through right now."))
        case .green: break
        }
        if let w = c.watcher, w.readable, w.fresh {
            if w.dbProbeFailed {
                lines.append(c.dbProbeEpisodeStart.map { c.now.timeIntervalSince($0) > 300 } == true
                    ? needs(.remembering, ["BrainLayer hasn’t been able to check its own writes for over 5 minutes, so BrainBar can’t confirm new conversations are being saved."], .escalated)
                    : cantTell(.remembering, cantConfirmSaving))
            } else if w.alertReasons.contains("coverage_drop") {
                lines.append(needs(.remembering, ["Fewer new conversations are being saved than are being read."]))
            }
            if w.alertReasons.contains("file_ingestion_failure") {
                let n = max(w.fileIngestionFailureCount, 1)
                lines.append(needs(.remembering, ["\(n) conversation \(n == 1 ? "file" : "files") couldn’t be added to your memory."]))
            }
            if w.alertReasons.contains(where: { !["offset_lag", "coverage_drop", "file_ingestion_failure", "quarantined_record"].contains($0) }) {
                lines.append(needs(.system, [unrecognised]))
            }
            if w.alertReasons.contains("offset_lag"), state("ingest") == .green {
                if let start = c.lagEpisodeStart, c.now.timeIntervalSince(start) > 1800 {
                    lines.append(needs(.remembering, ["BrainLayer has been behind on saving new conversations for over 30 minutes."], .escalated))
                } else {
                    var lag = behind(.remembering, "Catching up on new conversations.", headline: "Catching up on new conversations")
                    lag.spinner = true
                    lines.append(lag)
                }
            }
        } else {
            lines.append(cantTell(.remembering, cantConfirmSaving))
        }

        // Finding: search (F02 to F04), embedding (F06 to F08).
        switch state("search") {
        case .amber where past("search", 900):
            lines.append(needs(.finding, ["BrainBar hasn’t been able to confirm agents can search\(since("search"))."], .escalated))
        case .amber: lines.append(cantTell(.finding, "Can’t confirm agents can search right now."))
        case .red: lines.append(needs(.finding, ["Agents couldn’t search your memory at the last checks."], red: "search"))
        case .unknown: lines.append(cantTell(.finding, "Can’t check search right now."))
        case .green: break
        }
        let embedding = check("embedding")
        let embeddingAge = embedding?.ageS, embeddingLimit = embedding?.thresholdS
        if let e = embedding, e.state == .red || (e.state == .amber && embeddingAge != nil && embeddingAge! <= (embeddingLimit ?? -1)) {
            var detail = "BrainBar hasn’t seen the memories waiting for it go down" + (embeddingAge.map { " \(overDuration($0))" } ?? "") + "."
            if c.keywordCoverage == 1 { detail += " Agents can still find every memory by exact words." }
            lines.append(needs(.finding, ["Search by meaning isn’t catching up", detail], red: e.state == .red ? "embedding" : nil))
        } else if embedding?.state == .amber, let age = embeddingAge, age > (embeddingLimit ?? .infinity) {
            lines.append(behind(.finding, "Making new memories searchable by meaning, as fast as your Mac allows.",
                                headline: "Making new memories searchable"))
        } else if state("embedding") == .amber, past("embedding", 1800) {
            lines.append(needs(.finding, ["BrainBar hasn’t been able to check whether new memories are searchable by meaning\(since("embedding"))."],
                               .escalated))
        } else if state("embedding") != .green {
            lines.append(cantTell(.finding, "Can’t check whether new memories are searchable by meaning."))
        }

        // Upkeep: U03. Its second sentence needs Remembering and Safe both Fine.
        if c.maintenanceNightFailures >= 3 {
            let calm = !lines.contains { ($0.thing == .remembering || $0.thing == .safe) && $0.level != .fine }
            lines.append(needs(.upkeep, ["Nightly upkeep failed \(c.maintenanceNightFailures) nights in a row"]
                + (calm ? ["Your memory is still being saved and backed up, as of the last checks."] : [])))
        }

        // System: monitor (X05, X06), notes (X08), unknown check names (X10), and the dot as a last line of defence.
        switch state("monitor") {
        case .amber, .red:
            lines.append(needs(.system, ["BrainBar can’t fully check on your memory right now."], .escalated,
                               red: state("monitor") == .red ? "monitor" : nil))
        case .unknown: lines += checking  // X05 has no sentence of its own: each thing reads "Checking…"
        case .green: break
        }
        if v.notes.contains(where: { ["health_check_unloaded", "observability_unloaded"].contains($0.code) }) {
            lines.append(needs(.system, ["BrainBar’s regular checks have stopped."]))
        }
        for k in v.checks where !known.contains(k.name) && k.state != .green {
            lines.append(needs(.system, [unrecognised], red: k.state == .red ? k.name : nil))
        }
        if v.dot == .red, !lines.contains(where: { $0.level == .needs }) {
            lines.append(needs(.system, [unrecognised], red: "dot"))
        }
        return lines
    }

    /// One copy family's row: the two database copies (`database`) or the transcripts.
    static func backupLine(_ c: Capture, _ v: BrainBarHealthVerdict, _ given: [Check?], failures n: Int,
                           last: Date?, next: Date?, database: Bool) -> Line? {
        let checks = given.compactMap { $0 }
        let local = database ? given[0] : nil, offsite = database ? given[1] : nil
        let red = checks.first { $0.state == .red }
        let aged = checks.filter { $0.state == .amber && $0.ageS != nil }
        let blind = checks.filter { $0.state == .amber && $0.ageS == nil }
        let newest = checks.compactMap(\.ageS).min().flatMap { copyAge($0, c, v) }.map {
            c.backupsMeasured ? "Your newest backup is \($0)." : "The newest backup BrainBar can confirm is \($0)."
        }
        let recovering = database ? c.offsiteRecovering : nil
        let onSchedule = recovering.flatMap { r in
            c.now.timeIntervalSince(r.updatedAt) <= 2 * r.cadenceS && r.eta.map { c.now <= $0 } != false ? r : nil
        }
        // U13: once `recovering` exists it is the evidence; before PR-2 the live upload is.
        let started = recovering == nil ? (database && c.upload?.live == true ? c.upload?.startedAt : nil) : onSchedule?.startedAt
        let stopped = onSchedule == nil ? recovering.map { "A backup that started at \(hhmm($0.startedAt, c)) has stopped making progress." } : nil
        var line: Line
        if checks.contains(where: { ($0.ageS ?? 0) < 0 }) {
            return cantTell(.safe, "Can’t make sense of your backup dates right now.")  // S25
        } else if let local, local.state == .red, local.ageS == nil {
            line = needs(.safe, ["No verified backup on record"], .custody, red: local.name, priority: 1)  // S07
        } else if let red, let started {  // S32
            let copy = copyAge(offsite?.ageS, c, v).map { "Your newest backup in Google Drive is \($0). " } ?? ""
            let progress = onSchedule?.progress.map { "is \(percent($0))% done" } ?? "is still running"
            line = needs(.safe, [copy + "A backup to Google Drive started at \(hhmm(started, c)) and \(progress)."],
                         local?.state == .red ? .custody : .confirmed, red: red.name, priority: 2)
            line.handoff = ["If this backup doesn’t finish, BrainBar can’t fix it itself.", "Give the report to an agent."]
        } else if !aged.isEmpty || !blind.isEmpty, red == nil, let started {  // S03
            var text = "To Google Drive. Started at \(hhmm(started, c))."
            if let r = onSchedule, let p = r.progress { text += " \(percent(p))% done" + (r.eta.map { ", about \(hhmm($0, c))." } ?? ".") }
            line = behind(.safe, text, headline: "Backing up now")
            line.progress = onSchedule?.progress.map(percent)
            return line
        } else if red == nil, aged.contains(where: { $0.ageS! > ($0.thresholdS ?? .infinity) }) {  // S04
            return behind(.safe, "A backup came due while your Mac was off." + (newest.map { " " + $0 } ?? ""),
                          headline: "Catching up on a missed backup")
        } else if red != nil || !aged.isEmpty {
            let crossing = aged.compactMap { k in k.thresholdS.map { v.evaluatedAt.addingTimeInterval($0 - k.ageS!) } }.min()
            if red == nil, n <= 1, let next, next > c.now.addingTimeInterval(-7200), crossing.map({ next < $0 }) ?? true {  // S05
                let failed = n == 1 ? last.map { "The scheduled backup \(at($0, c)) didn’t finish. " } ?? "" : ""
                let scheduled = n == 1 ? "It’s scheduled again \(nextText(next, c))." : "The next backup is scheduled \(nextText(next, c))."
                line = behind(.safe, (stopped.map { $0 + " " } ?? failed) + (newest.map { $0 + " " } ?? "") + scheduled,
                              headline: "Backup due \(nextText(next, c))")
            } else {  // S06
                let title = n >= 2 ? (database ? "The last \(n) backups didn’t finish" : "The last \(n) scheduled backups didn’t finish.")
                    : "Scheduled backups aren’t running."
                line = needs(.safe, [stopped ?? title] + (newest.map { [$0] } ?? []),
                             local?.state == .red ? .custody : red != nil ? .confirmed : .escalated, red: red?.name, priority: 3)
                line.id = "S06"
            }
        } else if checks.count < given.count || checks.contains(where: { $0.state == .unknown }) || !blind.isEmpty || !c.backupsMeasured {
            let blindSince = blind.compactMap { $0.since.flatMap(parse) }.min().flatMap { c.now.timeIntervalSince($0) > 1800 ? $0 : nil }
            let unreadableSince = c.backupsUnreadableFor.flatMap { $0 > 3600 ? c.now.addingTimeInterval(-$0) : nil }
            if let start = unreadableSince ?? blindSince {  // S09
                line = needs(.safe, ["BrainBar hasn’t been able to read your backup records \(sinceText(start, c))."], .escalated)
            } else {  // S08
                let last = c.lastKnownBackupAt.flatMap { $0 <= c.now ? " Last known backup \(at($0, c))." : nil } ?? ""
                line = cantTell(.safe, "Can’t read your backup records right now." + last)
            }
        } else if n == 1 {
            line = Line(thing: .safe, level: .fine, strings: [])  // S02: a fact M draws only when M1 absorbs it
        } else {
            return nil
        }
        line.absorbable = database
        return line
    }

    // MARK: Part 2: merge, count, order, headline

    static func sorted(_ lines: [Line], by key: (Line) -> [Int]) -> [Line] {
        lines.enumerated().sorted { (key($0.element) + [$0.offset]).lexicographicallyPrecedes(key($1.element) + [$1.offset]) }.map(\.element)
    }

    static func assemble(_ c: Capture, _ input: [Line], allGood: String?) -> P {
        var lines: [Line] = []
        for line in input where !lines.contains(where: { $0.thing == line.thing && $0.level == line.level && $0.strings == line.strings }) {
            lines.append(line)
        }
        // M2, M3: one Needs-you row per thing and action, spoken by the highest class, then severity.
        var needs: [Line] = []
        for line in sorted(lines.filter { $0.level == .needs }, by: { [$0.rank.rawValue, $0.priority] })
            where !needs.contains(where: { $0.thing == line.thing && $0.action == line.action }) { needs.append(line) }
        needs = sorted(needs) { [$0.rank.rawValue, $0.thing.rawValue, $0.priority] }
        // M5: the first Behind row of a thing speaks.
        var behind: [Line] = []
        for line in lines where line.level == .behind && !behind.contains(where: { $0.thing == line.thing }) { behind.append(line) }
        behind = sorted(behind) { [$0.thing.rawValue] }
        let cantTell = sorted(lines.filter { $0.level == .cantTell }) { [$0.thing.rawValue] }
        let fine = [Thing.remembering, .finding, .safe, .upkeep].filter { thing in !lines.contains { $0.thing == thing && $0.level != .fine } }
        func status(_ line: Line, mark: P.Mark?) -> P.Row {
            P.Row(kind: .statusLine, strings: line.strings, mark: mark, state: line.level == .behind ? .behind : .cantTell, spinner: line.spinner)
        }

        var rows: [P.Row]
        let headline: String
        if !needs.isEmpty {
            headline = needs.count == 1 ? "1 thing needs you" : "\(needs.count) things need you"
            rows = [P.Row(kind: .headline, strings: [headline], mark: .marked), P.Row(kind: .separator)]
            for (i, line) in needs.enumerated() {
                rows.append(P.Row(kind: .needsYou, strings: line.strings + (line.action == .copyReport ? line.handoff : []),
                                  action: line.action.map { P.Action(kind: $0, label: label($0)) }, mark: .alertCircle, dividerBefore: i > 0))
            }
            rows += lines.filter(\.factLine).map { P.Row(kind: .factLine, strings: $0.strings) }
            rows += (behind + cantTell).map { status($0, mark: nil) }
            if !fine.isEmpty, !needs.contains(where: { $0.rank == .custody }) {
                rows.append(P.Row(kind: .everythingElse, strings: ["Everything else is fine."], mark: .checkmark))
            }
        } else {
            let mark: P.Mark
            let nouns: [Thing: String] = [.remembering: "new conversations", .finding: "search", .safe: "backups", .upkeep: "upkeep"]
            switch behind.count {
            case 1:
                headline = behind[0].headline ?? "Backing up now"
                mark = .busy
            case 2:
                headline = "Catching up on \(nouns[behind[0].thing] ?? "") and \(nouns[behind[1].thing] ?? "")"
                mark = .busy
            case 3...:
                headline = "Catching up on a few things"
                mark = .busy
            default:
                let things = Set(cantTell.map(\.thing))
                headline = things.contains(.safe) ? "Can’t confirm backups yet"
                    : things.count == 1 ? "Still checking \(nouns[things.first!] ?? "a few things")"
                    : things.isEmpty ? "All good" : "Still checking a few things"
                mark = things.isEmpty ? .ok : .wait
            }
            rows = [P.Row(kind: .headline, strings: [headline], mark: mark)]
            let folded = behind.count == 1 ? behind[0] : nil
            if let folded, folded.strings.first?.dropLast() != headline[...] {
                rows.append(P.Row(kind: .headlineDetail, strings: folded.strings, state: .behind, progressPercent: folded.progress))
            }
            if behind.isEmpty, cantTell.isEmpty, let allGood { rows.append(P.Row(kind: .headlineDetail, strings: [allGood])) }
            let listed = (behind.count == 1 ? [] : behind) + cantTell
            var fineNouns = fine.map { [.remembering: "saving", .finding: "search", .safe: "backups", .upkeep: "upkeep"][$0] ?? "" }
            let lastNoun = fineNouns.popLast() ?? ""
            let joined = fineNouns.isEmpty ? lastNoun : fineNouns.joined(separator: ", ") + " and " + lastNoun
            let fineText = joined.prefix(1).uppercased() + joined.dropFirst() + (fine.count == 1 ? " is fine." : " are fine.")
            let showFine = !fine.isEmpty && !(behind.isEmpty && cantTell.isEmpty)
            if listed.isEmpty {
                if showFine { rows.append(P.Row(kind: .everythingElse, strings: [fineText])) }
            } else {
                rows.append(P.Row(kind: .separator))
                rows += listed.map { status($0, mark: $0.level == .behind ? .busy : .wait) }
                if showFine { rows.append(P.Row(kind: .everythingElse, strings: [fineText], mark: .ok)) }
            }
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

    /// S01, the all-good line: every copy green and measured; the older of the offsite and transcript copies (U10).
    static func allGoodDetail(_ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        let copies = ["backup_local", "backup_offsite", "transcripts"].map { name in v.checks.first { $0.name == name } }
        guard c.backupsMeasured, copies.allSatisfy({ $0?.state == .green }) else { return nil }
        let ages = copies.dropFirst().compactMap { $0?.ageS }.filter { $0 >= 0 }
        return ages.max().map { "Backed up to Google Drive \(at(v.evaluatedAt.addingTimeInterval(-$0), c))." }
    }

    // MARK: words for times (truth table Part 0). No existing app formatter produces these forms.

    /// Month names and day boundaries are Gregorian whatever calendar the Mac uses; the time zone is the capture's.
    static func gregorian(_ c: Capture) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = c.calendar.timeZone
        return calendar
    }

    static func formatter(_ format: String, _ c: Capture) -> DateFormatter {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = gregorian(c)
        f.timeZone = c.calendar.timeZone
        f.dateFormat = format
        return f
    }

    static func hhmm(_ date: Date, _ c: Capture) -> String { formatter("HH:mm", c).string(from: date) }

    /// `{at}`, a past event: "today at 04:00", "yesterday at 23:10" (both under 24 h), else "Nov 9 at 03:44".
    static func at(_ date: Date, _ c: Capture) -> String {
        let calendar = gregorian(c)
        if c.now.timeIntervalSince(date) < 86_400 {
            if calendar.isDate(date, inSameDayAs: c.now) { return "today at \(hhmm(date, c))" }
            if let yesterday = calendar.date(byAdding: .day, value: -1, to: c.now), calendar.isDate(date, inSameDayAs: yesterday) {
                return "yesterday at \(hhmm(date, c))"
            }
        }
        return formatter("MMM d 'at' HH:mm", c).string(from: date)
    }

    /// `{since}`: "since 11:53" today, else "since yesterday at 23:10" or "since Nov 9 at 03:44".
    static func sinceText(_ date: Date, _ c: Capture) -> String {
        gregorian(c).isDate(date, inSameDayAs: c.now) ? "since \(hhmm(date, c))" : "since \(at(date, c))"
    }

    /// `{next}`, a scheduled time: "at 14:00", "tonight at 03:15", "tomorrow at 09:00", else "Nov 9 at 03:44".
    static func nextText(_ date: Date, _ c: Capture) -> String {
        let calendar = gregorian(c)
        let hour = calendar.component(.hour, from: date)
        if calendar.isDate(date, inSameDayAs: c.now) { return (hour < 18 ? "at " : "tonight at ") + hhmm(date, c) }
        if let tomorrow = calendar.date(byAdding: .day, value: 1, to: c.now), calendar.isDate(date, inSameDayAs: tomorrow) {
            return (hour < 6 ? "tonight at " : "tomorrow at ") + hhmm(date, c)
        }
        return formatter("MMM d 'at' HH:mm", c).string(from: date)
    }

    /// `{ago}` from a wall-clock event: "just now", "12 min ago", "3 h ago"; a day or more uses `{at}`.
    static func agoText(_ date: Date, _ c: Capture) -> String {
        let seconds = c.now.timeIntervalSince(date)
        if seconds < 60 { return "just now" }
        if seconds < 3600 { return "\(Int(seconds / 60)) min ago" }
        return seconds < 86_400 ? "\(Int(seconds / 3600)) h ago" : at(date, c)
    }

    /// `{copy age}`: "21 hours old" under a day, else "from Nov 9 at 03:44" (`evaluatedAt − ageS`). Nil for no or a future age.
    static func copyAge(_ age: Double?, _ c: Capture, _ v: BrainBarHealthVerdict) -> String? {
        guard let age, age >= 0 else { return nil }
        guard age >= 86_400 else {
            let hours = min(23, max(1, Int((age / 3600).rounded())))
            return hours == 1 ? "1 hour old" : "\(hours) hours old"
        }
        return "from \(at(v.evaluatedAt.addingTimeInterval(-age), c))"
    }

    /// `{over d}`: a lower bound from an awake-adjusted age, rounded down.
    static func overDuration(_ age: Double) -> String {
        let hours = Int(age / 3600), minutes = max(1, Int(age / 60))
        if hours >= 2 { return "in over \(hours) hours" }
        if hours == 1 { return "in over an hour" }
        return minutes == 1 ? "in over 1 minute" : "in over \(minutes) minutes"
    }

    static func percent(_ progress: Double) -> Int { Int((progress * 100).rounded()) }

    static func parse(_ iso: String) -> Date? { ISO8601DateFormatter().date(from: iso) }
}
