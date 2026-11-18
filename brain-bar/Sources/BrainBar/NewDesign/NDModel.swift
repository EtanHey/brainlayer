import Foundation

/// What the new design draws, derived once from one `NDInputs` (R1: the icon, the dropdown and the window read this one
/// value). Every state comes from `BrainBarMenuPresenter.present` and the sources the classic UI reads (L5); this file
/// only decides which visual each state takes. It never makes a state calmer than the presentation: the truth test
/// (`BrainBarNewDesignTruthTests`) enumerates the presenter's inputs and holds it to that.
struct NDModel: Equatable, Sendable {
    typealias P = BrainBarMenuPresentation
    typealias Thing = BrainBarMenuPresenter.Thing
    typealias Say = BrainBarMenuPresenter.Say

    enum Mark: Equatable, Sendable { case none, fault, unknown, running, check }
    enum Tone: Equatable, Sendable { case fine, warn, fault }

    /// A number with its unit in the secondary ink ("48" "s"), or a word ("none", "—").
    struct Number: Equatable, Sendable {
        var value: String
        var unit: String = ""
        var tone: Tone = .fine
        var strong = false
        var secondary = false

        static let unknown = Number(value: "—", secondary: true)
        var text: String { unit.isEmpty ? value : value + "\u{00A0}" + unit }
    }

    /// An age against its limit (V2).
    struct Gauge: Equatable, Sendable {
        var fraction: Double
        var tone: Tone
        /// The active fault concerns this row: Fault to the notch, a Fault dashed line past it.
        var concerned = false
        var pinned: Bool { fraction >= 1 }
    }

    enum Visual: Equatable, Sendable {
        case gauge(Gauge)
        /// An empty queue: the track only (N7).
        case emptyTrack
        case fill(Double)
        case unknown
        /// No visual: a state word stands in the number cell and runs left.
        case none
        /// Failures in a row: the notch pinned in Fault, then the dots.
        case streak(Int)
        /// An agent: live sessions after the name, a saves bar scaled to the busiest agent.
        case agent(sessions: Int?, saves: Int?, scale: Int)
    }

    struct Row: Equatable, Sendable, Identifiable {
        var id: String
        var name: String
        var visual: Visual
        var number: Number
        var strong = false
        /// The number cell holds a state word, which may run left over the empty lane.
        var stateWord = false
        var accessibility: String
    }

    struct Area: Equatable, Sendable, Identifiable {
        var id: String { name }
        var thing: Thing?
        var name: String
        var mark: Mark
        var rows: [Row]
        /// One limit line joins the gauges' ends (no line for Agents: its bars are amounts).
        var limitLine = true
        var aside: String? = nil
        /// Why the area carries the dashed ring when none of its rows is dashed (its values are the app's own reads, but
        /// the last check did not confirm it): the help tag and VoiceOver say it.
        var unknownReason: String? = nil
        var fault: Bool { mark == .fault }
        /// Fix 14: the ring is drawn only when a row on this surface is dashed; otherwise the reason is in the help tag
        /// and VoiceOver, and the mark stays empty.
        var shownMark: Mark { mark == .unknown && unknownReason != nil ? .none : mark }
    }

    /// The dropdown's row for one area: its single visual and number.
    struct MenuArea: Equatable, Sendable, Identifiable {
        enum Visual: Equatable, Sendable {
            case wave
            case fill(Double)
            case gauge(Gauge)
            case streak(Int)
            case unknown
        }
        var id: String { name }
        var thing: Thing
        var name: String
        var mark: Mark
        var visual: Visual
        var number: Number
        var accessibility: String
        /// The ring's reason when the row's own value is known (design pre-pass 5b).
        var unknownReason: String? = nil
        /// Fix 14: as `Area.shownMark`: no ring beside a known value; the reason goes to the help tag and VoiceOver.
        var shownMark: Mark { mark == .unknown && unknownReason != nil ? .none : mark }
    }

    struct Fault: Equatable, Sendable, Identifiable {
        var id: String { title }
        var title: String
        /// The facts as the presenter says them, every sentence (notifications, the report and VoiceOver keep these).
        var facts: [String]
        /// The facts where a gauge already shows the age (L4): the dropdown and the window's needs band draw these.
        var shownFacts: [String]
        var action: P.Action?
        var thing: Thing?
        /// The rows this fault concerns: their gauges turn Fault (review finding 4).
        var concerns: Set<String>
        var jobs: Set<BrainLayerLaunchdJob> = []
    }

    var presentation: P
    var headline: String
    var headlineMark: Mark
    /// A line under the headline only when something is being handled (S03's progress); never a reassurance.
    var headlineDetail: String?
    var count: Int
    var icon: P.Icon
    var faults: [Fault]
    var everythingElse: Bool
    var menuAreas: [MenuArea]
    /// Overview's panel: Remembering, Finding, Agents (left); Backups, Upkeep (right).
    var areas: [Area]
    var checked: String?
    var memories: String?
    var backupsTabCount: Int
    var jobsTabCount: Int
    var unknownThings: Set<Thing>
    var faultThings: Set<Thing>
    var runningThings: Set<Thing>
    /// What the surfaces show, as a presentation: the report copies this, runtime warnings included (F03).
    var shownPresentation: P
    /// Google Drive for Settings, from the same accepted Backups row every other surface draws (F01).
    var drive: DriveSetting

    struct DriveSetting: Equatable, Sendable {
        /// "Connected", "Disconnected", "Not connected", or "—" when the last check cannot say.
        var state: String
        /// Reconnect or Connect, offered only while a counted fault says Drive needs it.
        var action: P.Action?
    }

    static let unconfirmed = BrainBarMenuPresenter.SurfaceSay.unconfirmed
}

// MARK: - Derivation

extension NDModel {
    /// The watcher job's name on the Jobs page (copy deck): the gauge is how long ago it last reported.
    static let readerRow = "Read new conversations"
    static let areaNames: [Thing: String] = [.remembering: "Remembering", .finding: "Finding", .safe: "Backups", .upkeep: "Upkeep"]
    /// The upkeep gauges' drawing scale only (L3): it places the notch and decides nothing.
    static let nightlyScaleS: Double = 36 * 3600
    static let weeklyScaleS: Double = 8 * 86_400

    static func make(_ i: NDInputs) -> NDModel {
        let c = i.capture
        let p = BrainBarMenuPresenter.present(c)
        let freshVerdict: BrainBarHealthVerdict? = c.verdict.flatMap { (0...900).contains(c.verdictAwakeAgeS) ? $0 : nil }

        // The presenter's Needs-you rows, with their fact lines (R22) and the thing each speaks for.
        var faults: [Fault] = []
        for row in p.rows {
            switch row.kind {
            case .needsYou:
                let title = row.strings.first ?? ""
                let thing = BrainBarMenuPresenter.entry(for: title, c)?.0.thing
                let facts = Array(row.strings.dropFirst())
                faults.append(Fault(title: title, facts: facts, shownFacts: withoutGaugeAges(facts, c), action: row.action,
                                    thing: thing, concerns: concerns(title: title, thing: thing, c)))
            case .factLine:
                guard !faults.isEmpty else { continue }
                faults[faults.count - 1].facts += row.strings
                faults[faults.count - 1].shownFacts += withoutGaugeAges(row.strings, c)
            default: continue
            }
        }
        // The runtime warnings that need him become rows in their own (the snapshot's) words; one already said is not
        // said twice (R29) but still turns its area red.
        let faultExtras = i.extras.filter { $0.level == .fault }
        for extra in faultExtras where !faults.contains(where: { $0.title == extra.text }) {
            faults.append(Fault(title: extra.text, facts: [Say.cantFix.text, Say.giveReport.text],
                                shownFacts: [Say.cantFix.text, Say.giveReport.text],
                                action: P.Action(kind: .copyReport, label: Say.copyReport.text), thing: extra.thing,
                                concerns: [], jobs: extra.jobs))
        }

        // Which areas are red: a worded row's thing, the things behind the generic rows (their non-green checks and the
        // facts the app reads by itself), and the runtime warnings.
        var faultThings = Set(faults.compactMap(\.thing))
        let generic = p.rows.contains { $0.kind == .needsYou && $0.strings.first == Say.generic.text }
        if generic {
            for name in BrainBarMenuPresenter.known where name != "monitor" {
                let states = (c.verdict?.checks ?? []).filter { $0.name == name }.map(\.state)
                if states.contains(.red) || (freshVerdict != nil && states.contains(.amber) && !calmAmber(name, p)),
                   let thing = thingOf(name) {
                    faultThings.insert(thing)
                }
            }
            if let w = c.watcher, w.readable, w.fresh {
                let alerts = w.alertReasons.filter { !["offset_lag", "quarantined_record"].contains($0) && !($0 == "coverage_drop" && w.dbProbeFailed) }
                if alerts.contains(where: { ["coverage_drop", "file_ingestion_failure"].contains($0) })
                    || (w.alertReasons.contains("offset_lag") && BrainBarMenuPresenter.lagPastLimit(c)) {
                    faultThings.insert(.remembering)
                }
            }
        }
        faultThings.formUnion(faultExtras.compactMap(\.thing))
        if p.form != .needs && faultExtras.isEmpty { faultThings = [] }

        // Running: what the presenter says is being handled. Unknown: anything the presenter cannot prove fine.
        var running = Set<Thing>()
        if Say.catchingUp(c) { running.insert(.remembering) }
        if BrainBarMenuPresenter.runningUpload(c) != nil { running.insert(.safe) }
        running.subtract(faultThings)
        let proven = BrainBarMenuPresenter.provenFine(c)
        var unknown = Set(BrainBarMenuPresenter.four).subtracting(proven).subtracting(faultThings).subtracting(running)
        // A notice the presentation does not carry (an unverified vector check, a notice-level job alert) keeps its area
        // from reading fine.
        for extra in i.extras where extra.level == .notice {
            if let thing = extra.thing, !faultThings.contains(thing) { unknown.insert(thing) }
        }
        func mark(_ t: Thing) -> Mark {
            faultThings.contains(t) ? .fault : running.contains(t) ? .running : unknown.contains(t) ? .unknown : .none
        }

        // Headline, count and icon: the presentation's, plus the runtime warnings it cannot see.
        let extraRows = faults.count - p.rows.filter { $0.kind == .needsYou }.count
        var headline = p.headline, count = p.count, icon = p.icon
        var headlineMark: Mark = switch p.rows.first(where: { $0.kind == .headline })?.mark {
        case .marked?: .fault
        case .busy?: .running
        case .wait?: .unknown
        default: .none
        }
        if extraRows > 0 {
            count = faults.count
            headline = count == 1 ? Say.countOne.text : Say.countMany(String(count))
            icon = .marked
            headlineMark = .fault
        } else if headline == Say.allGood.text, !unknown.isEmpty {
            // "All good" is never said beside something the presentation does not carry and cannot prove.
            let nouns = BrainBarMenuPresenter.four.filter(unknown.contains).compactMap { Say.behindNoun[$0] }
            headline = nouns.count == 1 ? Say.stillChecking(nouns[0]) : Say.stillCheckingFew.text
            headlineMark = .unknown
        }
        let everythingElse = extraRows == 0 && i.extras.isEmpty && p.rows.contains { $0.kind == .everythingElse && $0.strings == [Say.everythingElse.text] }
        let detail: String? = headlineMark == .running
            ? p.rows.first(where: { $0.kind == .headlineDetail && $0.state == .behind })?.strings.joined(separator: " ") : nil

        // L4 and the words budget: beside "Everything else is fine." a fault drops its own reassurance (U03's "still
        // being saved and backed up"); VoiceOver and the report keep every sentence.
        if everythingElse {
            for index in faults.indices {
                faults[index].shownFacts = faults[index].shownFacts.compactMap { string in
                    let kept = BrainBarMenuPresenter.sentences(string).filter { BrainBarMenuPresenter.entry(for: $0, c)?.0.kind != .reassurance }
                    return kept.isEmpty ? nil : kept.joined(separator: " ")
                }
            }
        }
        let ctx = Context(i: i, c: c, verdict: freshVerdict, faults: faults, faultThings: faultThings, unknown: unknown)
        let areas = [
            remembering(ctx, mark(.remembering)), finding(ctx, mark(.finding)), agentsArea(ctx),
            backups(ctx, mark(.safe)), upkeep(ctx, mark(.upkeep)),
        ].map { area -> Area in
            var area = area
            if area.mark == .unknown, !area.rows.contains(where: { $0.visual == .unknown }) { area.unknownReason = unconfirmed }
            return area
        }
        let menuAreas = menu(ctx, areas: areas, marks: Dictionary(uniqueKeysWithValues: BrainBarMenuPresenter.four.map { ($0, mark($0)) }))

        let jobsFaults = Set(faults.filter { $0.thing == .upkeep }.map(\.title)).count
            + faultExtras.filter { !$0.jobs.isEmpty && $0.thing != .upkeep && $0.thing != .safe }.count
        return NDModel(
            presentation: p, headline: headline, headlineMark: headlineMark, headlineDetail: detail, count: count, icon: icon,
            faults: faults, everythingElse: everythingElse, menuAreas: menuAreas, areas: areas,
            checked: checkedLine(c), memories: i.memories.map(NDFormat.integer),
            backupsTabCount: faultThings.contains(.safe) ? max(1, faults.filter { $0.thing == .safe }.count) : 0,
            jobsTabCount: jobsFaults, unknownThings: unknown, faultThings: faultThings, runningThings: running,
            shownPresentation: shown(p, headline: headline, count: count, icon: icon, faults: faults,
                                     extraRows: extraRows, everythingElse: everythingElse),
            drive: driveSetting(areas, faults))
    }

    struct Context {
        var i: NDInputs
        var c: BrainBarMenuCapture
        var verdict: BrainBarHealthVerdict?
        var faults: [Fault]
        var faultThings: Set<Thing>
        var unknown: Set<Thing>

        /// The one check of this name in the fresh verdict; a name carried twice is never read for its fields.
        func check(_ name: String) -> BrainBarHealthVerdict.Check? {
            let named = verdict?.checks.filter { $0.name == name } ?? []
            return named.count == 1 ? named[0] : nil
        }
        func concerned(_ name: String) -> Bool { faults.contains { $0.concerns.contains(name) } }
    }

    static func thingOf(_ name: String) -> Thing? {
        switch name {
        case "ingest", "stores": .remembering
        case "search", "embedding": .finding
        case "backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity", "disk": .safe
        default: nil
        }
    }

    /// The two ambers the presenter keeps calm: a running upload (S03) and search unconfirmed under 15 minutes (F02).
    private static func calmAmber(_ name: String, _ p: P) -> Bool {
        let words = p.rows.flatMap(\.strings)
        if name == "search" { return words.contains(Say.f02.text) }
        if name == "backup_offsite" { return p.headline == Say.s03Headline.text || words.contains { $0.hasPrefix(Say.s03.text) } }
        return false
    }

    /// The rows a Needs-you row concerns, by the registry id of its title.
    static func concerns(title: String, thing: Thing?, _ c: BrainBarMenuCapture) -> Set<String> {
        let id = BrainBarMenuPresenter.entry(for: title, c)?.0.id ?? ""
        switch id {
        case "S13", "S13.missing": return ["drive_auth", "backup_offsite", "transcripts"]
        case "S19": return ["integrity"]
        case "F07": return ["embedding"]
        case "S07": return ["backup_local"]
        case "U03": return ["nightly"]
        case _ where id.hasPrefix("S32"): return ["backup_offsite"]
        case "S06":
            return Set((c.verdict?.checks ?? []).filter { ["backup_local", "backup_offsite"].contains($0.name) && $0.state != .green }.map(\.name))
        default:
            // R17's generic row: the non-green checks behind it.
            return Set((c.verdict?.checks ?? []).filter { $0.state == .red || $0.state == .amber }.map(\.name))
        }
    }

    /// L4: where a gauge already carries the Drive copy's age, the fault's body drops that sentence.
    static func withoutGaugeAges(_ strings: [String], _ c: BrainBarMenuCapture) -> [String] {
        strings.compactMap { string in
            let kept = BrainBarMenuPresenter.sentences(string).filter { BrainBarMenuPresenter.entry(for: $0, c)?.0.id != "R15" }
            return kept.isEmpty ? nil : kept.joined(separator: " ")
        }
    }

    /// The presentation as the new surfaces show it: its headline, count and icon, its rows, and a row per runtime warning
    /// that needs him (F03). "Everything else is fine." goes wherever the surfaces drop it.
    static func shown(_ p: P, headline: String, count: Int, icon: P.Icon, faults: [Fault], extraRows: Int,
                      everythingElse: Bool) -> P {
        var rows = p.rows.filter { everythingElse || $0.kind != .everythingElse }
        if extraRows > 0 {
            let extra = faults.suffix(extraRows).map { P.Row(kind: .needsYou, strings: [$0.title] + $0.facts, action: $0.action,
                                                             mark: .alertCircle, dividerBefore: true) }
            let at = (rows.lastIndex { $0.kind == .needsYou || $0.kind == .factLine }).map { $0 + 1 }
                ?? (rows.firstIndex { $0.kind == .headline }.map { $0 + 1 } ?? 0)
            rows.insert(contentsOf: extra, at: at)
        }
        if headline != p.headline {
            // The shown headline replaced the presenter's (a runtime warning): "All good"'s own detail goes with it.
            rows = rows.compactMap { row in
                var row = row
                if row.kind == .headline { row.strings = [headline]; row.mark = icon == .marked ? .marked : .wait }
                if row.kind == .headlineDetail, p.headline == Say.allGood.text { return nil }
                return row
            }
        }
        return P(form: icon == .marked ? .needs : p.form, headline: headline, count: count, icon: icon, rows: rows)
    }

    static func driveSetting(_ areas: [Area], _ faults: [Fault]) -> DriveSetting {
        let row = areas.first { $0.thing == .safe }?.rows.first { $0.id == "drive_auth" }
        let state: String = switch row?.number.value {
        case "connected"?: "Connected"
        case "disconnected"?: "Disconnected"
        case "not connected"?: "Not connected"
        default: "—"
        }
        let action = faults.first { $0.concerns.contains("drive_auth") }?.action
        return DriveSetting(state: state, action: action.flatMap { [.reconnectDrive, .connectDrive].contains($0.kind) ? $0 : nil })
    }

    static func checkedLine(_ c: BrainBarMenuCapture) -> String? {
        guard let v = c.verdict else { return nil }
        let age = max(0, c.now.timeIntervalSince(v.evaluatedAt))
        let words = NDFormat.ago(age)
        return (0...900).contains(c.verdictAwakeAgeS) ? "Checked \(words) ago" : "Last checked \(words) ago"
    }

    // MARK: rows

    /// A gauge row from one verdict check: unknown when the verdict is not fresh or the check is missing or unknown.
    static func gaugeRow(_ ctx: Context, id: String, name: String, check: String, areaFault: Bool) -> Row {
        guard let k = ctx.check(check), k.state != .unknown, let age = k.ageS, age.isFinite, age >= 0,
              let limit = k.thresholdS, limit > 0 else {
            return Row(id: id, name: name, visual: .unknown, number: .unknown, accessibility: "\(name), not known")
        }
        return ageRow(ctx, id: id, name: name, age: age, limit: limit, concerned: ctx.concerned(check), areaFault: areaFault)
    }

    static func ageRow(_ ctx: Context, id: String, name: String, age: Double, limit: Double, concerned: Bool, areaFault: Bool) -> Row {
        let fraction = age / limit
        // ≥ 100 % is drawn in Fault only when a fault is counted for it; otherwise Warn (never red without a counted fault).
        let tone: Tone = concerned || (fraction >= 1 && areaFault) ? .fault : fraction >= 0.75 ? .warn : .fine
        let n = NDFormat.age(age)
        return Row(id: id, name: name, visual: .gauge(Gauge(fraction: fraction, tone: tone, concerned: concerned)),
                   number: Number(value: n.value, unit: n.unit),
                   strong: concerned,
                   accessibility: id == "ingest" ? "\(name), last reported \(n.spoken) ago, limit \(NDFormat.age(limit).spoken)"
                       : "\(name), \(n.spoken) old, limit \(NDFormat.age(limit).spoken)")
    }

    static func remembering(_ ctx: Context, _ mark: Mark) -> Area {
        // The reader of new conversations: the ingest check's heartbeat age; with a stale verdict, the same heartbeat as the
        // app reads it itself from watcher-health.json (R20), against the last known limit.
        // F07: the ingest check's age is the conversation reader's heartbeat (health_verdict.py), not the newest
        // conversation's age, which the app does not have. The row is named for what it measures.
        var ingest = gaugeRow(ctx, id: "ingest", name: Self.readerRow, check: "ingest", areaFault: mark == .fault)
        if ingest.visual == .unknown, let age = ctx.i.watcherHeartbeatAgeS, age >= 0,
           let limit = ctx.c.verdict?.checks.first(where: { $0.name == "ingest" })?.thresholdS, limit > 0 {
            ingest = ageRow(ctx, id: "ingest", name: Self.readerRow, age: age, limit: limit,
                            concerned: ctx.concerned("ingest"), areaFault: mark == .fault)
        }
        // Saves waiting: the queue the app reads itself; an empty queue is a measured zero (N7).
        let saves: Row
        // Saves waiting rests on the verdict's stores check (§ 4): with no fresh, known check it is unknown, whatever the
        // app's own queue read says.
        let stores = ctx.check("stores")
        if stores == nil || stores?.state == .unknown {
            saves = Row(id: "stores", name: "Saves waiting", visual: .unknown, number: .unknown, accessibility: "Saves waiting, not known")
        } else if let depth = ctx.i.savesWaiting, depth == 0 {
            saves = Row(id: "stores", name: "Saves waiting", visual: .emptyTrack, number: Number(value: "none", secondary: true),
                        accessibility: "Saves waiting, none")
        } else if let oldest = ctx.i.oldestSaveQueuedAt, ctx.i.savesWaiting != nil,
                  let limit = ctx.c.verdict?.checks.first(where: { $0.name == "stores" })?.thresholdS, limit > 0 {
            saves = ageRow(ctx, id: "stores", name: "Saves waiting", age: max(0, ctx.c.now.timeIntervalSince(oldest)), limit: limit,
                           concerned: ctx.concerned("stores"), areaFault: mark == .fault)
        } else {
            saves = gaugeRow(ctx, id: "stores", name: "Saves waiting", check: "stores", areaFault: mark == .fault)
        }
        return Area(thing: .remembering, name: "Remembering", mark: mark, rows: [ingest, saves])
    }

    static func finding(_ ctx: Context, _ mark: Mark) -> Area {
        let search: Row
        if let k = ctx.check("search"), k.state == .green {
            search = Row(id: "search", name: "Search", visual: .none, number: Number(value: "working", secondary: true),
                         stateWord: true, accessibility: "Search, working")
        } else if let k = ctx.check("search"), let fails = k.fails, fails > 0, k.state != .unknown {
            let concerned = ctx.concerned("search") || mark == .fault
            search = Row(id: "search", name: "Search", visual: .streak(fails),
                         number: Number(value: String(fails), tone: concerned ? .fault : .warn, strong: concerned), strong: concerned,
                         accessibility: "Search failed \(fails) times in a row")
        } else {
            search = Row(id: "search", name: "Search", visual: .unknown, number: .unknown, accessibility: "Search, not known")
        }
        let meaning = gaugeRow(ctx, id: "embedding", name: "Search by meaning", check: "embedding", areaFault: mark == .fault)
        let searchable: Row = ctx.i.searchable.map { share in
            let n = NDFormat.percent(share)
            return Row(id: "searchable", name: "Searchable", visual: .fill(share), number: Number(value: n, unit: "%"),
                       accessibility: "Searchable by meaning, \(n) percent")
        } ?? Row(id: "searchable", name: "Searchable", visual: .unknown, number: .unknown, accessibility: "Searchable, not known")
        return Area(thing: .finding, name: "Finding", mark: mark, rows: [search, meaning, searchable])
    }

    static func agentsArea(_ ctx: Context) -> Area {
        let scale = max(1, ctx.i.agents.agents.compactMap(\.saves).max() ?? 1)
        let rows = ctx.i.agents.agents.map { a -> Row in
            let number = a.saves.map { Number(value: NDFormat.integer($0)) } ?? .unknown
            let sessions = a.sessions.map { $0 == 1 ? "1 session now" : "\($0) sessions now" } ?? "sessions not known"
            // L5 ruling: the count is every memory attributed to the agent's family in 24 h, not its saves alone.
            let memories = a.saves.map { "\(NDFormat.integer($0)) memories in the last 24 hours" } ?? "memories not known"
            return Row(id: a.family.rawValue, name: NDProvider.name(a.family), visual: .agent(sessions: a.sessions, saves: a.saves, scale: scale),
                       number: number, accessibility: "\(NDProvider.name(a.family)): \(sessions), \(memories)")
        }
        let unknown = ctx.i.agents.agents.contains { $0.sessions == nil || $0.saves == nil }
        return Area(thing: nil, name: "Agents", mark: unknown ? .unknown : .none, rows: rows, limitLine: false,
                    aside: BrainBarMenuPresenter.SurfaceSay.memoriesIn24h)
    }

    static func backups(_ ctx: Context, _ mark: Mark) -> Area {
        let fault = mark == .fault
        var rows = [
            gaugeRow(ctx, id: "backup_local", name: BrainBarMenuPresenter.SurfaceSay.localCopy, check: "backup_local", areaFault: fault),
            gaugeRow(ctx, id: "backup_offsite", name: "Copy in Google Drive", check: "backup_offsite", areaFault: fault),
            gaugeRow(ctx, id: "transcripts", name: "Conversations in Drive", check: "transcripts", areaFault: fault),
        ]
        // A copy whose age is not measured this moment (backups not measured, R30 streaks) is not shown as known.
        if !ctx.c.backupsMeasured {
            rows = rows.map { Row(id: $0.id, name: $0.name, visual: .unknown, number: .unknown, accessibility: "\($0.name), not known") }
        }
        func word(_ id: String, _ name: String, _ check: String, _ words: [BrainBarHealthVerdict.State: String]) -> Row {
            guard let k = ctx.check(check), let w = words[k.state] else {
                return Row(id: id, name: name, visual: .unknown, number: .unknown, accessibility: "\(name), not known")
            }
            let concerned = ctx.concerned(check) || k.state == .red
            let tone: Tone = k.state == .red ? .fault : k.state == .amber ? .warn : .fine
            // A failed database check draws the fault square; VoiceOver hears the registry's words for it (N04).
            let spoken = w == Number.faultMark.value ? Say.s19.text : w
            return Row(id: id, name: name, visual: .none,
                       number: Number(value: w, tone: tone, strong: concerned && tone == .fault, secondary: tone == .fine),
                       strong: concerned && tone == .fault, stateWord: true, accessibility: "\(name), \(spoken)")
        }
        let missing = ctx.c.driveAuthModel == .missing
        rows.append(word("drive_auth", "Google Drive", "drive_auth",
                         [.green: "connected", .amber: "connected", .red: missing ? "not connected" : "disconnected"]))
        // The database check reads "passed"; on failure the fault row above says what failed, and the cell shows the mark.
        rows.append(word("integrity", "Database check", "integrity", [.green: "passed", .red: "■"]))
        rows.append(word("disk", "Room for backups", "disk", [.green: "enough", .amber: "tight", .red: "full"]))
        return Area(thing: .safe, name: "Backups", mark: mark, rows: rows)
    }

    static func upkeep(_ ctx: Context, _ mark: Mark) -> Area {
        func run(_ id: String, _ name: String, _ last: NDRun, scale: Double, streak: Int?) -> Row {
            if let streak {
                return Row(id: id, name: name, visual: .streak(streak),
                           number: Number(value: String(streak), unit: "nights"), strong: true,
                           accessibility: "Nightly upkeep failed \(streak) nights in a row")
            }
            guard case let .ran(at) = last, ctx.c.maintenanceNightFailures != nil || id == "weekly" else {
                return Row(id: id, name: name, visual: .unknown, number: .unknown, accessibility: "\(name), not known")
            }
            let age = max(0, ctx.c.now.timeIntervalSince(at))
            // L3: the scale places the notch; the tone never comes from it.
            let n = NDFormat.age(age)
            return Row(id: id, name: name, visual: .gauge(Gauge(fraction: age / scale, tone: .fine)),
                       number: Number(value: n.value, unit: n.unit), accessibility: "\(name), last ran \(n.spoken) ago")
        }
        let failing = ctx.c.maintenanceNightFailures.flatMap { $0 >= 3 ? $0 : nil }
        return Area(thing: .upkeep, name: "Upkeep", mark: mark, rows: [
            run("nightly", "Nightly", ctx.i.nightlyLastRun, scale: nightlyScaleS, streak: failing),
            run("weekly", "Weekly", ctx.i.weeklyLastRun, scale: weeklyScaleS, streak: nil),
        ])
    }

    // MARK: the dropdown's four rows

    static func menu(_ ctx: Context, areas: [Area], marks: [Thing: Mark]) -> [MenuArea] {
        func area(_ t: Thing) -> Area? { areas.first { $0.thing == t } }
        var out: [MenuArea] = []
        // Remembering: the waveform, and the newest conversation's age.
        let ingest = area(.remembering)?.rows.first { $0.id == "ingest" }
        let waveKnown = ctx.i.menuWave?.readable == true
        out.append(MenuArea(thing: .remembering, name: "Remembering", mark: marks[.remembering] ?? .none,
                            visual: waveKnown ? .wave : .unknown, number: ingest?.number ?? .unknown,
                            accessibility: waveAccessibility(ctx.i.menuWave, minutes: 60)))
        // Finding: the searchable share (the app's own read), floored so it never overstates.
        let share = ctx.i.searchable
        out.append(MenuArea(thing: .finding, name: "Finding", mark: marks[.finding] ?? .none,
                            visual: share.map { .fill($0) } ?? .unknown,
                            number: share.map { Number(value: String(Int(($0 * 100).rounded(.down))), unit: "%") } ?? .unknown,
                            accessibility: share.map { "Searchable by meaning, \(NDFormat.percent($0)) percent" } ?? "Finding, not known"))
        // Backups: the copy closest to its limit; a copy the fault concerns speaks first.
        let copies = (area(.safe)?.rows ?? []).filter { ["backup_local", "backup_offsite", "transcripts"].contains($0.id) }
        let gauges = copies.compactMap { row -> (Row, Gauge)? in if case let .gauge(g) = row.visual { return (row, g) } else { return nil } }
        let pick = gauges.first(where: { $0.1.concerned }) ?? gauges.max { $0.1.fraction < $1.1.fraction }
        let backupsVisual: MenuArea.Visual = gauges.count == copies.count && !copies.isEmpty ? pick.map { .gauge($0.1) } ?? .unknown : .unknown
        out.append(MenuArea(thing: .safe, name: "Backups", mark: marks[.safe] ?? .none, visual: backupsVisual,
                            number: backupsVisual == .unknown ? .unknown : pick?.0.number ?? .unknown,
                            accessibility: backupsVisual == .unknown ? "Backups, not known" : pick?.0.accessibility ?? "Backups"))
        // Upkeep: the nightly run.
        let nightly = area(.upkeep)?.rows.first { $0.id == "nightly" }
        let upkeepVisual: MenuArea.Visual = switch nightly?.visual {
        case let .gauge(g)?: .gauge(g)
        case let .streak(n)?: .streak(n)
        default: .unknown
        }
        out.append(MenuArea(thing: .upkeep, name: "Upkeep", mark: marks[.upkeep] ?? .none, visual: upkeepVisual,
                            number: nightly?.number ?? .unknown, accessibility: nightly?.accessibility ?? "Upkeep, not known"))
        return out.map { area in
            var area = area
            if area.mark == .unknown, area.visual != .unknown { area.unknownReason = unconfirmed }
            return area
        }
    }

    static func waveAccessibility(_ wave: NDWave?, minutes: Int) -> String {
        guard let wave, wave.readable else { return "Remembering, not known" }
        return BrainBarMenuPresenter.SurfaceSay.wave(span: minutes, saves: wave.savesTotal, reads: wave.readsTotal)
    }
}

// MARK: - Words for numbers

enum NDFormat {
    struct Age: Equatable { var value: String, unit: String, spoken: String }

    /// "48 s", "7 min", "21 h", "4 d"; the unit is drawn in the secondary ink.
    static func age(_ seconds: Double) -> Age {
        let s = max(0, Int(seconds.rounded(.down)))
        switch s {
        case ..<60: return Age(value: String(s), unit: "s", spoken: "\(s) seconds")
        case ..<3600: return Age(value: String(s / 60), unit: "min", spoken: "\(s / 60) minutes")
        case ..<(48 * 3600): return Age(value: String(s / 3600), unit: "h", spoken: "\(s / 3600) hours")
        default: return Age(value: String(s / 86_400), unit: "d", spoken: "\(s / 86_400) days")
        }
    }

    /// "1 min", "18 min", "3 h": a checked-ago phrase.
    static func ago(_ seconds: Double) -> String {
        let a = age(seconds)
        return a.value + " " + a.unit
    }

    /// "in 4 min", "in 19 h", "in 3 d".
    static func until(_ seconds: Double) -> String {
        let a = age(max(0, seconds))
        return "in " + a.value + " " + a.unit
    }

    static func integer(_ value: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.locale = Locale(identifier: "en_US")
        return f.string(from: NSNumber(value: value)) ?? String(value)
    }

    /// One decimal, floored: 99.46 % is "99.4", never rounded up to a share it has not reached.
    static func percent(_ share: Double) -> String {
        let tenths = (max(0, min(1, share)) * 1000).rounded(.down) / 10
        return tenths == tenths.rounded() ? String(format: "%.0f", tenths) : String(format: "%.1f", tenths)
    }
}
