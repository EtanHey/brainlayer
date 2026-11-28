import Foundation

// MARK: - The graph (L18: provisional, declared once; R1.3: the loop)

/// One drawn pipe of the loop (R1.3 changes 1 to 6, R1.3.1 L1 to L5). The agents' group sends sessions to Read and saves
/// to the Save queue; Memory feeds Search, whose answers return to the agents; the copies and Jobs hang off Memory.
enum NDPipe: String, CaseIterable, Sendable {
    case sessions
    case readMemory = "read_mem"
    case saves
    case queueMemory = "queue_mem"
    case memorySearch = "mem_search"
    case answers
    case memoryMac = "mem_mac"
    case macDrive = "mac_drive"
    case rail
    case jobsMemory = "jobs"
    /// R1.10 (L49): the weekly clean's own two-way link to Memory.
    case weeklyMemory = "weekly"
}

/// The eight things you can point at (R1.3): the agents' group, six stations and the Memory hub. Search is one stage;
/// the word index is not drawn (its missing check is a row in Search's details).
/// R1.10 (L49): the jobs are two stations: `jobs` is the nightly check-up (its name since 1.10), `weekly` the weekly clean.
enum NDStation: String, CaseIterable, Sendable, Identifiable {
    case agents, read, queue, memory, jobs, weekly, search, mac, drive
    var id: String { rawValue }
}

enum NDBreakSide: String, Sendable { case source, target }
enum NDSeverity: Sendable { case red, orange }

/// The draft's graph: which connection each check guards.
enum NDPipelineGraph {
    struct Connection: Sendable {
        let id: String
        let from: String
        let to: String
        let pipes: [NDPipe]
    }

    /// c1 to c12 of the draft, redrawn as the loop. c6 (Memory to the word index) has no check and no drawn pipe: Search
    /// is one stage. c7 and c8 (Search, then the answers to the agents) are one drawn return.
    static let connections: [Connection] = [
        Connection(id: "c1", from: "Agents' sessions", to: "Read", pipes: [.sessions]),
        Connection(id: "c2", from: "Read", to: "Memory", pipes: [.readMemory]),
        Connection(id: "c3", from: "Agent tools", to: "Save queue", pipes: [.saves]),
        Connection(id: "c4", from: "Save queue", to: "Memory", pipes: [.queueMemory]),
        Connection(id: "c5", from: "Memory", to: "Search (by meaning)", pipes: [.memorySearch]),
        Connection(id: "c6", from: "Memory", to: "Search (by words)", pipes: []),
        Connection(id: "c7", from: "Search", to: "Answers", pipes: [.answers]),
        Connection(id: "c8", from: "Answers", to: "Your agents", pipes: [.answers]),
        Connection(id: "c9", from: "Memory", to: "Local copy", pipes: [.memoryMac]),
        Connection(id: "c10", from: "Local copy", to: "Database in Google Drive", pipes: [.macDrive]),
        Connection(id: "c11", from: "Agents' sessions", to: "Session files in Google Drive", pipes: [.rail]),
        Connection(id: "c12", from: "Nightly check-up", to: "Memory", pipes: [.jobsMemory]),
        Connection(id: "c13", from: "Weekly clean", to: "Memory", pipes: [.weeklyMemory]),
    ]

    /// What a check guards: connections (by id), a stage, or nothing because it is the observer.
    enum Guard: Equatable, Sendable {
        case connections([String])
        case station(NDStation)
        case observer
    }

    /// Every health check maps once (`drive_auth` is the one deliberate exception: one cause, two connections).
    static let checks: [String: Guard] = [
        "ingest": .connections(["c1", "c2"]),
        "stores": .connections(["c3", "c4"]),
        "embedding": .connections(["c5"]),
        "search": .connections(["c7", "c8"]),
        "backup_local": .connections(["c9"]),
        "backup_offsite": .connections(["c10"]),
        "transcripts": .connections(["c11"]),
        "drive_auth": .connections(["c10", "c11"]),
        "integrity": .station(.mac),
        "disk": .station(.mac),
        "monitor": .observer,
    ]

    /// Connections guarded by a presenter fact rather than one of the eleven checks: c12 by the job alerts' failed
    /// streaks (`maintenanceNightFailures`).
    static let factGuards: [String: String] = ["c12": "job alerts (maintenanceNightFailures)",
                                                "c13": "the weekly job's own failure streak (maintenanceWeekly)"]

    /// Connections no check guards today (draft "what it cannot show yet"): not drawn; said in Search's details.
    static let unguarded: Set<String> = ["c6"]

    static func connection(_ id: String) -> Connection? { connections.first { $0.id == id } }
    static func pipes(for check: String) -> [NDPipe] {
        guard case let .connections(ids)? = checks[check] else { return [] }
        return Array(Set(ids.compactMap(connection).flatMap(\.pipes))).sorted { $0.rawValue < $1.rawValue }
    }

    /// Which stage each pipe leaves and reaches (`nil` is the agents' group or the hub, which never break a pipe).
    static let ends: [NDPipe: (from: NDStation, to: NDStation)] = [
        .sessions: (.agents, .read), .readMemory: (.read, .memory), .saves: (.agents, .queue), .queueMemory: (.queue, .memory),
        .memorySearch: (.memory, .search), .answers: (.search, .agents), .memoryMac: (.memory, .mac), .macDrive: (.mac, .drive),
        .rail: (.agents, .drive), .jobsMemory: (.jobs, .memory), .weeklyMemory: (.weekly, .memory),
    ]
}

// MARK: - The drawn state, derived from the verdict's facts (L5; R1.3 changes 9 to 12)

struct NDPipelineState: Equatable, Sendable {
    /// A station's one status line: caption `t2`; under a fault red Semibold, under a notice amber Semibold.
    struct Status: Equatable, Sendable {
        var text: String
        var severity: NDSeverity? = nil
    }

    /// One look per station, the same on the chart, in the dropdown rows and in the icon (L29).
    enum Treatment: Equatable, Sendable { case calm, fault, notice, notConfirmed }

    /// Red stations: ring, badge, red name and status, and the break on the pipe out of the stage (change 9).
    var faults: Set<NDStation> = []
    /// Amber stations: ring, triangle badge, amber status, and an amber stretch on the pipe into the stage (change 10).
    var notices: Set<NDStation> = []
    /// Stations whose fault is drawn where the pipe arrives (Google Drive, which nothing leaves; Search when only its
    /// meaning index is stalled, because answers still flow by words).
    var arrivalBreaks: Set<NDStation> = []
    /// An agent whose sessions are not read: its row in the group is the station (badge left of the name).
    var agentFaults: [AgentFamily: NDSeverity] = [:]
    /// w307: running agents per harness, from BrainBar's own agent monitor (`AgentActivitySnapshot.running`: live sessions
    /// only, through `NDInputs.agents[].running`). A harness missing here has unknown liveness. The count is not the verdict's, so a stale verdict
    /// does not hide it.
    var agentsRunning: [AgentFamily: Int] = [:]
    /// Not confirmed for one check (its pipes dashed).
    var unconfirmed: Set<NDPipe> = []
    var status: [NDStation: Status] = [:]
    /// The observer is stale: every pipe and ring dashed, one caption (change 11).
    var allUnconfirmed = false
    var unconfirmedSince: Date?
    /// "14:02": the last confirmed evaluation's time, in the capture's zone (never the render time; R1.2 change 12).
    var lastKnownAt: String?

    static let calm = NDPipelineState()

    func treatment(_ station: NDStation) -> Treatment {
        if allUnconfirmed { return .notConfirmed }
        if station == .agents {
            if agentFaults.values.contains(.red) { return .fault }
            return agentFaults.isEmpty ? .calm : .notice
        }
        if faults.contains(station) { return .fault }
        if notices.contains(station) { return .notice }
        return .calm
    }

    /// The red things on the chart (stations and agent rows): the icon's count and the headline's count follow it.
    var faultCount: Int { faults.count + agentFaults.values.filter { $0 == .red }.count }

    /// How a stage reads for the dropdown rows: the same treatment, plus a dash where one check cannot say.
    enum Condition: Equatable, Sendable {
        case fine
        case notConfirmed
        case issue(String?, NDSeverity)
    }

    func condition(_ station: NDStation) -> Condition {
        switch treatment(station) {
        case .notConfirmed: return .notConfirmed
        case .fault: return .issue(status[station]?.text, .red)
        case .notice: return .issue(status[station]?.text, .orange)
        case .calm:
            let touching = NDPipelineGraph.ends.filter { $0.value.to == station || $0.value.from == station }.map(\.key)
            return touching.contains(where: unconfirmed.contains) ? .notConfirmed : .fine
        }
    }

    /// How one pipe is drawn (R1.4 change 16, L36): the pipe after a faulty stage is faint (28 %); nothing else on a pipe
    /// changes for a fault or a notice. Dashed when not confirmed, otherwise flowing.
    enum PipeLook: Equatable, Sendable { case flowing, dashed, faint }

    func look(_ pipe: NDPipe) -> PipeLook {
        if allUnconfirmed { return .dashed }
        guard let ends = NDPipelineGraph.ends[pipe] else { return .flowing }
        if faults.contains(ends.from), !arrivalBreaks.contains(ends.from) { return .faint }
        if unconfirmed.contains(pipe) { return .dashed }
        return .flowing
    }
}

extension NDPipelineState {
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    /// The chart for a moment: the verdict's checks mapped through `NDPipelineGraph`, plus the presenter facts the
    /// draft names (`maintenanceNightFailures`, the pending saves). It never draws a fault the checks cannot tell apart.
    static func make(_ i: NDInputs) -> NDPipelineState {
        let c = i.capture
        var s = NDPipelineState()
        s.status = statusLines(i)
        for agent in i.agents.agents {
            if let running = agent.running { s.agentsRunning[agent.family] = max(0, running) }
        }
        guard let verdict = c.verdict else {
            s.allUnconfirmed = true
            return s
        }
        let fresh = (0...900).contains(c.verdictAwakeAgeS)
        let monitor = verdict.checks.first { $0.name == "monitor" }
        if !fresh || monitor?.state == .red {
            s.allUnconfirmed = true
            s.unconfirmedSince = verdict.evaluatedAt
            s.lastKnownAt = NDR1.hhmm(verdict.evaluatedAt, c)
            s.status = lastKnownLines(i, s.status, at: verdict.evaluatedAt)
            return s
        }
        func check(_ name: String) -> BrainBarHealthVerdict.Check? { verdict.checks.first { $0.name == name } }
        func severity(_ k: BrainBarHealthVerdict.Check) -> NDSeverity? {
            switch k.state {
            case .red: .red
            case .amber: .orange
            default: nil
            }
        }
        func mark(_ station: NDStation, _ sev: NDSeverity) {
            if sev == .red { s.faults.insert(station) } else { s.notices.insert(station) }
        }
        for k in verdict.checks where k.state == .unknown {
            for pipe in NDPipelineGraph.pipes(for: k.name) { s.unconfirmed.insert(pipe) }
        }

        // c10 and c11: Drive refuses (one cause, two connections); its status says so.
        let driveRefuses = check("drive_auth")?.state == .red
        if driveRefuses {
            s.faults.insert(.drive)
            // R1.10.1: after Reconnect, the station waits like every other surface (one source, NDDriveWaiting).
            // Still a fault (Drive is signed out until the browser approves), so the line stays red.
            s.status[.drive] = Status(text: NDDriveWaiting.shows(i, for: nil) ? Say.stationWaiting : Say.signedOut, severity: .red)
        }

        // c1: one agent's sessions not read (its row in the group), or the reader itself stalled (the Read station).
        if let ingest = check("ingest"), var sev = severity(ingest) {
            // L55: a red reader that is only the lag draining by itself is drawn as the notice the menu says (the one
            // switch's decision), never as a fault.
            if sev == .red, BrainBarMenuPresenter.lagIsNotice(c) { sev = .orange }
            let agents = agentCodes(ingest)
            if agents.isEmpty {
                mark(.read, sev)
                // R1.13: the lag's own words, as the menu's: "catching up" while BrainLayer measures it draining, else how
                // far behind ("117 MB behind") once it stopped; never "not being read" for a reader that is reading.
                if BrainBarMenuPresenter.Say.lagDraining(c) {
                    s.status[.read] = Status(text: Say.catchingUp, severity: sev)
                } else if BrainBarMenuPresenter.Say.producerCatchingUp(c) {
                    s.status[.read] = Status(text: BrainBarMenuPresenter.Say.lagSize(c).map(Say.behind) ?? Say.catchingUp, severity: sev)
                } else if BrainBarMenuPresenter.Say.producerStalled(c), let size = BrainBarMenuPresenter.Say.lagSize(c) {
                    s.status[.read] = Status(text: Say.behind(size), severity: sev)
                } else {
                    // D6: amber is late, not stopped: the last-read time in amber; red says it is not being read.
                    s.status[.read] = sev == .red ? Status(text: Say.notBeingRead, severity: .red)
                        : s.status[.read].map { Status(text: $0.text, severity: .orange) }
                }
            } else {
                // R1.4 change 3 (L38): only the providers the check names; Read's status names them. R1.5 change 6 (L41):
                // in plain words, the normal status colour: the fault is marked once, on the agent's icon.
                for family in agents { s.agentFaults[family] = sev }
                let names = agents.map(NDProvider.name).joined(separator: ", ")
                // R1.12 (E4): the status reads as the fault it is, red Semibold like every fault status; Read still gets no
                // ring and no badge (the provider's icon keeps the one fault mark, L41).
                s.status[.read] = Status(text: Say.providerNotRead(names), severity: sev)
            }
        }

        // c3 / c4: saves that do not reach memory.
        if let stores = check("stores"), let sev = severity(stores) {
            mark(.queue, sev)
            if sev == .red, let queued = i.oldestSaveQueuedAt {
                // R1.12 (F): the fault's tail, "stuck {t}", under "Save queue".
                s.status[.queue] = Status(text: Say.stuckFor(NDFormat.ageText(max(0, c.now.timeIntervalSince(queued)))), severity: sev)
            } else if let text = s.status[.queue]?.text {
                s.status[.queue] = Status(text: text, severity: sev)
            }
        }

        // c5: search by meaning behind (a notice) or stalled (a fault drawn where Memory's pipe arrives, since answers still
        // flow by words). The status is the embedding check's own backlog age (L33).
        if let embedding = check("embedding"), let sev = severity(embedding) {
            mark(.search, sev)
            if sev == .red { s.arrivalBreaks.insert(.search) }
            // R1.12 (F, § C): a notice says "catching up", the tail of "Search by meaning is catching up"; a stall keeps
            // its backlog age.
            if sev == .orange {
                s.status[.search] = Status(text: Say.catchingUp, severity: sev)
            } else if let age = embedding.ageS, age.isFinite, age >= 0 {
                s.status[.search] = Status(text: Say.byMeaningBehind(NDFormat.ageText(age)), severity: sev)
            }
        }

        // c7 / c8: one canary for both; a red one breaks the answers out of Search.
        if let search = check("search"), let sev = severity(search) {
            mark(.search, sev)
            if sev == .red {
                s.arrivalBreaks.remove(.search)
                // R1.12 (F): "failed {n} checks" when the canary counts them.
                if let fails = search.fails, fails >= 2 {
                    s.status[.search] = Status(text: Say.failedChecks(fails), severity: .red)
                } else if s.status[.search]?.severity == nil { s.status[.search] = nil }
            }
        }

        // c9: the copy on this Mac is late; integrity and disk are states of that copy.
        if let local = check("backup_local"), let sev = severity(local) {
            mark(.mac, sev)
            if let text = s.status[.mac]?.text { s.status[.mac] = Status(text: text, severity: sev) }
        }
        if check("integrity")?.state == .red || check("disk")?.state == .red {
            s.faults.insert(.mac)
            if let text = s.status[.mac]?.text { s.status[.mac] = Status(text: text, severity: .red) }
        }
        if !driveRefuses {
            for name in ["backup_offsite", "transcripts"] {
                if let k = check(name), let sev = severity(k) {
                    mark(.drive, sev)
                    if let text = s.status[.drive]?.text { s.status[.drive] = Status(text: text, severity: s.faults.contains(.drive) ? .red : sev) }
                }
            }
        }
        // R1.12 (E3, replaces R1.7 item 6): when the local copy is late, Drive's older upload is its consequence, not a
        // second problem: its line stays in `t2` with no ring (nothing new was made to upload).
        s.arrivalBreaks.insert(.drive)

        // c12: a scheduled job failing; the Jobs status names it (change 13).
        if let nights = c.maintenanceNightFailures, nights >= 3 {
            s.faults.insert(.jobs)
            // R2 item 10: the fault said once, short, on the secondary line. R1.10: the station is named "Nightly
            // check-up" alone, so its line is "failed 3 nights" again.
            s.status[.jobs] = Status(text: Say.failedNights(nights), severity: .red)
        }
        // R1.10 (L49, L51): the weekly clean faults on its own job's failure streak only.
        if let weekly = i.jobs?.first(where: { $0.job == .maintenanceWeekly }), let weeks = Self.weeklyFailures(weekly) {
            s.faults.insert(.weekly)
            s.status[.weekly] = Status(text: Say.failedWeeks(weeks), severity: .red)
        }
        // A fault outranks a notice on the same station.
        s.notices.subtract(s.faults)
        for station in s.faults { if let text = s.status[station]?.text { s.status[station] = Status(text: text, severity: .red) } }
        for (pipe, _) in NDPipelineGraph.ends where s.look(pipe) != .dashed { s.unconfirmed.remove(pipe) }
        return s
    }

    /// R2A item 6: under "values last known at {time}" no relative age: the reader's last read as a clock time; every
    /// other status a dash, as the dropdown shows. R1.7 item 5: the copy on this Mac says "—" like the others.
    /// R1.12 (E2): under "values last known at {time}" each station keeps its last-known value, as a clock time (with its
    /// weekday when not today), never an age: "last read 06:59", "3 kept, newest 03:16", "last upload 04:20", "last run
    /// 04:00", "last run Sun 04:00". A value with no time stays a dash. Nothing later than the caption's time is shown.
    static func lastKnownLines(_ i: NDInputs, _ lines: [NDStation: Status], at evaluated: Date) -> [NDStation: Status] {
        let c = i.capture
        var out: [NDStation: Status] = [:]
        for station in lines.keys { out[station] = Status(text: NDModel.Number.unknown.value) }
        func known(_ date: Date?) -> String? { date.flatMap { $0 <= evaluated ? NDR1.clockOn($0, c) : nil } }
        // Only a successful-read timestamp supports "last read"; a poll heartbeat does not.
        if let at = known(c.watcher?.lastReadAt) {
            out[.read] = Status(text: Say.lastReadClock(at))
        }
        let checks = c.verdict?.checks ?? []
        func copied(_ name: String) -> Date? {
            checks.first { $0.name == name }?.ageS.flatMap { $0.isFinite && $0 >= 0 ? evaluated.addingTimeInterval(-$0) : nil }
        }
        if let at = known(i.newestLocalCopyAt ?? copied("backup_local")) {
            out[.mac] = Status(text: i.localCopiesKept.map { Say.keptNewestClock($0, at) } ?? Say.backupAt(at))
        }
        if let at = known(copied("backup_offsite")) { out[.drive] = Status(text: Say.lastUploadClock(at)) }
        if case let .ran(at) = i.nightlyLastRun, let when = known(at) { out[.jobs] = Status(text: Say.lastRunClock(when)) }
        if let weekly = i.jobs?.first(where: { $0.job == .maintenanceWeekly }), weekly.lastOutcome == .ok, let when = known(weekly.lastRun) {
            out[.weekly] = Status(text: Say.lastRunClock(when))
        }
        return out
    }

    /// The calm status lines (change 2, sources § 5). No measured value means no line.
    static func statusLines(_ i: NDInputs) -> [NDStation: Status] {
        let c = i.capture
        let checks = c.verdict?.checks ?? []
        func age(_ name: String) -> Double? { checks.first { $0.name == name }?.ageS.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil } }
        var out: [NDStation: Status] = [:]
        if let read = c.watcher?.lastReadAt, read <= c.now, read.timeIntervalSince1970.isFinite {
            out[.read] = Status(text: Say.lastRead(NDFormat.ageText(c.now.timeIntervalSince(read))))
        }
        // R1.12 (F): Save queue says "empty", or "{n} waiting" while under its threshold (a fault says "stuck {t}").
        if let waiting = i.savesWaiting {
            out[.queue] = Status(text: waiting > 0 ? Say.nWaiting(waiting) : Say.empty)
        }
        // R1.12 (F): Search says how much of memory is searchable ("99% searchable", the dropdown's value too).
        if let share = i.searchable, share.isFinite, share >= 0 {
            out[.search] = Status(text: Say.searchablePercent(Int((share * 100).rounded(.down))))
        }
        // R1.5 change 8 (L45): "{n} kept, newest {t} ago", n the copies actually found; without a count, the newest's age.
        if let mac = age("backup_local") {
            out[.mac] = Status(text: i.localCopiesKept.map { Say.keptNewest($0, NDFormat.ageText(mac)) } ?? Say.lastBackup(NDFormat.ageText(mac)))
        }
        // Council r2a finding 2: one measure for Drive everywhere: the newest Drive upload of either kind (R1.2 change
        // 11, "Last Drive upload" and "Drive … ago"); the Backups page shows each copy, and this is the newer of the two.
        // R2A item 3: Drive's status names what it measures: an upload. (The copy it uploads is the Mac's, whose age the
        // Mac station shows; one age per copy.)
        // R2 item 4: Drive's status is the newest DATABASE copy's upload (what its panel says Drive receives); the session
        // files' upload is a row in its panel.
        if let db = age("backup_offsite") { out[.drive] = Status(text: Say.lastUpload(NDFormat.ageText(db))) }
        // R1.10: each job station says its last run ("last run {t} ago"); the weekly clean only for a run that did its
        // work (a run its safety gates skipped is never shown as done).
        if case let .ran(at) = i.nightlyLastRun {
            out[.jobs] = Status(text: Say.lastRunAgo(NDFormat.ageText(max(0, c.now.timeIntervalSince(at)))))
        }
        if let weekly = i.jobs?.first(where: { $0.job == .maintenanceWeekly }), weekly.lastOutcome == .ok, let at = weekly.lastRun {
            out[.weekly] = Status(text: Say.lastRunAgo(NDFormat.ageText(max(0, c.now.timeIntervalSince(at)))))
        }
        return out
    }

    /// R1.10: the weekly clean is failing when its last run failed and its own streak counts the weeks.
    static func weeklyFailures(_ weekly: NDJobInput) -> Int? {
        guard weekly.lastOutcome == .failed, let n = weekly.failureStreak, n >= 1 else { return nil }
        return n
    }

    /// The agents an `ingest` check names: the growth check (`growth_not_indexed:<agent>`) or the census
    /// (`unindexed_agents:<agent>`).
    static func agentCodes(_ k: BrainBarHealthVerdict.Check) -> [AgentFamily] {
        let codes = (k.contextCodes ?? []) + (k.blindCodes ?? [])
        var out: [AgentFamily] = []
        for code in codes {
            let parts = code.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2, ["growth_not_indexed", "unindexed_agents", "unindexed_agent"].contains(parts[0]),
                  let family = AgentFamily.allCases.first(where: { $0.rawValue == parts[1].lowercased() }),
                  !out.contains(family) else { continue }
            out.append(family)
        }
        return out
    }
}

extension NDFormat {
    /// "2 h", "38 min": an age as words in a phrase (the number and its unit, one unit).
    static func ageText(_ seconds: Double) -> String {
        let a = age(seconds)
        return a.value + "\u{00A0}" + a.unit
    }
}
