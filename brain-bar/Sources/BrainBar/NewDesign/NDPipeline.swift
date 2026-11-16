import Foundation

// MARK: - The graph (L18: provisional, declared once, from PIPELINE-GRAPH-DRAFT.md)

/// One drawn pipe of the pipeline chart. A connection of the draft is drawn as one pipe, or several (c1 is four agent
/// stubs and the junction into Read). Swapping in BrainLayer's own graph file later changes `NDPipelineGraph`, not this
/// drawing's grammar.
enum NDPipe: String, CaseIterable, Sendable {
    case agentClaude = "a_claude", agentCodex = "a_codex", agentCursor = "a_cursor", agentGemini = "a_gemini"
    case junctionRead = "j_read"
    case conversationsCopy = "rail"
    case readMemory = "read_mem"
    case inSaves = "in_saves"
    case queueMemory = "q_mem"
    case memoryMac = "mem_mac"
    case macDrive = "mac_drive"
    case memoryMeaning = "mem_meaning"
    case meaningSearch = "meaning_search"
    case words
    case out
    case jobsMemory = "jobs_mem"

    static func agent(_ family: AgentFamily) -> NDPipe {
        switch family {
        case .claude: .agentClaude
        case .codex: .agentCodex
        case .cursor: .agentCursor
        case .gemini: .agentGemini
        }
    }
}

/// The eight stages on screen (Memory is the hub).
enum NDStation: String, CaseIterable, Sendable, Identifiable {
    case read, queue, memory, mac, drive, meaning, search, jobs
    var id: String { rawValue }
}

enum NDBreakSide: String, Sendable { case source, target }
enum NDSeverity: Sendable { case red, orange }

/// The draft's graph: which connection each check guards, and on which side a break is drawn.
enum NDPipelineGraph {
    struct Connection: Sendable {
        let id: String
        let from: String
        let to: String
        let pipes: [NDPipe]
    }

    /// c1 to c12 of the draft. c6 (the word index) has no check: it is drawn dashed, "by words, not checked", always.
    static let connections: [Connection] = [
        Connection(id: "c1", from: "Agents' sessions", to: "Read",
                   pipes: [.agentClaude, .agentCodex, .agentCursor, .agentGemini, .junctionRead]),
        Connection(id: "c2", from: "Read", to: "Memory", pipes: [.readMemory]),
        Connection(id: "c3", from: "Agent tools", to: "Save queue", pipes: [.inSaves]),
        Connection(id: "c4", from: "Save queue", to: "Memory", pipes: [.queueMemory]),
        Connection(id: "c5", from: "Memory", to: "Meaning index", pipes: [.memoryMeaning]),
        Connection(id: "c6", from: "Memory", to: "Word index", pipes: [.words]),
        Connection(id: "c7", from: "Meaning index", to: "Search", pipes: [.meaningSearch]),
        Connection(id: "c8", from: "Search", to: "Answers to agents", pipes: [.out]),
        Connection(id: "c9", from: "Memory", to: "Copy on this Mac", pipes: [.memoryMac]),
        Connection(id: "c10", from: "Copy on this Mac", to: "Copy in Google Drive", pipes: [.macDrive]),
        Connection(id: "c11", from: "Agents' sessions", to: "Conversations in Google Drive", pipes: [.conversationsCopy]),
        Connection(id: "c12", from: "Jobs", to: "Memory", pipes: [.jobsMemory]),
    ]

    /// What a check guards: connections (by id), a stage (`s10`), or nothing because it is the observer.
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
    static let factGuards: [String: String] = ["c12": "job alerts (maintenanceNightFailures)"]

    /// Connections no check guards today (draft "what it cannot show yet"): drawn dashed and labelled.
    static let unguarded: Set<String> = ["c6"]

    static func connection(_ id: String) -> Connection? { connections.first { $0.id == id } }
    static func pipes(for check: String) -> [NDPipe] {
        guard case let .connections(ids)? = checks[check] else { return [] }
        return ids.compactMap(connection).flatMap(\.pipes)
    }
}

// MARK: - The drawn state, derived from the verdict's facts (L5)

struct NDPipelineState: Equatable, Sendable {
    struct Break: Equatable, Sendable {
        var side: NDBreakSide
        var severity: NDSeverity
    }
    /// A station's one status line (R1.1 change 2): caption `t2`; in a fault Semibold `red-t` or `orange-t`.
    struct Status: Equatable, Sendable {
        var text: String
        var severity: NDSeverity? = nil
    }
    /// A phrase that is not a station's (the agents' "not being read", under the agent list).
    struct Phrase: Equatable, Sendable {
        enum Anchor: Sendable { case start, middle, end }
        var text: String
        var x: CGFloat
        var y: CGFloat
        var anchor: Anchor
        var severity: NDSeverity
    }

    var breaks: [NDPipe: Break] = [:]
    /// Downstream of a break: solid at 28 % (change 8), with "delayed" or "stopped" set in the line when it is long enough.
    var quiet: [NDPipe: NDSeverity] = [:]
    /// Not confirmed: the check guarding it cannot say this moment (dashed, the only meaning of a dash).
    var unconfirmed: Set<NDPipe> = []
    /// The stage itself refuses (red ring, symbol and name), and the pipes into it end in red.
    var faultStations: Set<NDStation> = []
    var stationPipes: Set<NDPipe> = []
    var redAgents: Set<AgentFamily> = []
    var status: [NDStation: Status] = [:]
    var phrases: [Phrase] = []
    /// The observer is stale: every pipe dashed, no red, no marks; status lines at their last values, faded.
    var allUnconfirmed = false
    var unconfirmedSince: Date?

    static let calm = NDPipelineState()

    func isBroken(_ pipe: NDPipe) -> Bool { breaks[pipe] != nil || stationPipes.contains(pipe) }
}

extension NDPipelineState {
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    /// The chart for a moment: the verdict's checks mapped through `NDPipelineGraph`, plus the presenter facts the
    /// draft names (`maintenanceNightFailures`, the pending saves). It never draws a break the checks cannot tell apart.
    static func make(_ i: NDInputs) -> NDPipelineState {
        let c = i.capture
        var s = NDPipelineState()
        s.status = statusLines(i)
        guard let verdict = c.verdict else {
            s.allUnconfirmed = true
            return s
        }
        let fresh = (0...900).contains(c.verdictAwakeAgeS)
        let monitor = verdict.checks.first { $0.name == "monitor" }
        if !fresh || monitor?.state == .red {
            s.allUnconfirmed = true
            s.unconfirmedSince = verdict.evaluatedAt
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
        for k in verdict.checks where k.state == .unknown {
            for pipe in NDPipelineGraph.pipes(for: k.name) where pipe != .words { s.unconfirmed.insert(pipe) }
        }

        // c10 and c11: Drive refuses (one cause, two connections), drawn on the station; its status says so.
        let driveRefuses = check("drive_auth")?.state == .red
        if driveRefuses {
            s.faultStations.insert(.drive)
            s.stationPipes.formUnion([.macDrive, .conversationsCopy])
            s.status[.drive] = Status(text: Say.signedOut, severity: .red)
        }

        // c1: one agent's sessions not read (source, per agent), or the reader itself stalled (junction → Read, target).
        if let ingest = check("ingest"), let sev = severity(ingest) {
            let agents = agentCodes(ingest)
            if agents.isEmpty {
                s.breaks[.junctionRead] = Break(side: .target, severity: sev)
                s.status[.read] = Status(text: Say.notBeingRead, severity: sev)
            } else {
                for family in agents {
                    s.breaks[NDPipe.agent(family)] = Break(side: .source, severity: sev)
                    s.quiet[NDPipe.agent(family)] = sev
                    s.redAgents.insert(family)
                }
                s.phrases.append(Phrase(text: Say.notBeingRead, x: 0, y: 116, anchor: .start, severity: sev))
            }
        }

        // c3 / c4: a save that does not round-trip breaks the way in; saves waiting break the way into Memory.
        if let stores = check("stores"), let sev = severity(stores) {
            if let queued = i.oldestSaveQueuedAt {
                s.breaks[.queueMemory] = Break(side: .target, severity: sev)
                s.status[.queue] = Status(text: Say.waiting(NDFormat.ageText(max(0, c.now.timeIntervalSince(queued)))), severity: sev)
            } else {
                s.breaks[.inSaves] = Break(side: .source, severity: sev)
                s.quiet[.inSaves] = sev
            }
        }

        // c5: search by meaning behind (orange, late) or stalled (red); the onward pipe is downstream ("delayed" / "stopped").
        if let embedding = check("embedding"), let sev = severity(embedding) {
            s.breaks[.memoryMeaning] = Break(side: .target, severity: sev)
            s.quiet[.meaningSearch] = sev
            if let age = embedding.ageS, age.isFinite, age >= 0 {
                s.status[.meaning] = Status(text: Say.behind(NDFormat.ageText(age)), severity: sev)
            }
        }

        // c7 / c8: one canary for both; drawn on the way out of Search.
        if let search = check("search"), let sev = severity(search) {
            s.breaks[.out] = Break(side: .source, severity: sev)
            s.quiet[.out] = sev
            s.status[.search] = nil
        }

        // c9: the copy on this Mac is late; integrity and disk are states of that copy (a station ring, not a break).
        if let local = check("backup_local"), let sev = severity(local) {
            s.breaks[.memoryMac] = Break(side: .target, severity: sev)
            if let text = s.status[.mac]?.text { s.status[.mac] = Status(text: text, severity: sev) }
        }
        if check("integrity")?.state == .red || check("disk")?.state == .red {
            s.faultStations.insert(.mac)
        }
        if !driveRefuses {
            if let offsite = check("backup_offsite"), let sev = severity(offsite) {
                s.breaks[.macDrive] = Break(side: .target, severity: sev)
                if let text = s.status[.drive]?.text { s.status[.drive] = Status(text: text, severity: sev) }
            }
            if let transcripts = check("transcripts"), let sev = severity(transcripts) {
                s.breaks[.conversationsCopy] = Break(side: .target, severity: sev)
            }
        }

        // c12: a scheduled job failing; the Jobs status names it (change 13).
        if let nights = c.maintenanceNightFailures, nights >= 3 {
            s.breaks[.jobsMemory] = Break(side: .source, severity: .red)
            s.quiet[.jobsMemory] = .red
            s.status[.jobs] = Status(text: Say.nightlyFailedShort(nights), severity: .red)
        }
        s.unconfirmed.subtract(s.breaks.keys)
        s.unconfirmed.subtract(s.stationPipes)
        s.unconfirmed.subtract(s.quiet.keys)
        return s
    }

    /// The calm status lines (change 2, sources § 5). No measured value means no line.
    static func statusLines(_ i: NDInputs) -> [NDStation: Status] {
        let c = i.capture
        let checks = c.verdict?.checks ?? []
        func age(_ name: String) -> Double? { checks.first { $0.name == name }?.ageS.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil } }
        var out: [NDStation: Status] = [:]
        if let heartbeat = i.watcherHeartbeatAgeS, heartbeat >= 0 {
            out[.read] = Status(text: Say.ago(NDFormat.ageText(heartbeat)))
        }
        if let waiting = i.savesWaiting {
            if waiting == 0 {
                out[.queue] = Status(text: Say.empty)
            } else if let queued = i.oldestSaveQueuedAt {
                out[.queue] = Status(text: Say.waiting(NDFormat.ageText(max(0, c.now.timeIntervalSince(queued)))))
            }
        }
        if let mac = age("backup_local") { out[.mac] = Status(text: Say.ago(NDFormat.ageText(mac))) }
        let drive = [age("backup_offsite"), age("transcripts")].compactMap { $0 }
        if drive.count == 2, let older = drive.max() { out[.drive] = Status(text: Say.ago(NDFormat.ageText(older))) }
        if let share = i.searchable { out[.meaning] = Status(text: Say.percentReady(Int((share * 100).rounded(.down)))) }
        if checks.first(where: { $0.name == "search" })?.state == .green { out[.search] = Status(text: Say.working) }
        if case let .ran(at) = i.nightlyLastRun {
            out[.jobs] = Status(text: Say.nightlyAgo(NDFormat.ageText(max(0, c.now.timeIntervalSince(at)))))
        }
        return out
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
