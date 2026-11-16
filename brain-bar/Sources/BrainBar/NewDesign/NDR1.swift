import Foundation

/// What round 1 of the design (DESIGN-SPEC-R1) draws on top of `NDModel`: the headline items with their pipe strips and
/// one action each, the dropdown's four rows, the window footer and the pipeline. Pure: one value from `NDInputs` and
/// the `NDModel` made from them. It never says less than the presenter: every presenter fault the pipeline does not
/// explain is kept as its own item, in the presenter's words.
struct NDR1: Equatable, Sendable {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    typealias P = BrainBarMenuPresentation

    /// The broken pipe cut to its two stages, for the dropdown (§ 3.2, "In the dropdown").
    struct Strip: Equatable, Sendable {
        var from: String
        var to: String
        var side: NDBreakSide
        var severity: NDSeverity
    }

    struct Item: Equatable, Sendable, Identifiable {
        var id: String
        var severity: NDSeverity
        var title: String
        var detail: String?
        var action: P.Action?
        var strip: Strip?
        /// The dropdown's shorter title, when the deck gives one ("Nightly check-up failed 3 nights").
        var menuTitle: String? = nil
        /// The dropdown's one line under the title (R1.1 change 28; the strips are gone).
        var menuDetail: String? = nil
        /// An amber state the app is handling itself (catching up on conversations or search by meaning): L23 keeps
        /// the resting icon for these.
        var catchingUp: Bool { severity == .orange && (id == "meaning" || id == "reader") }
        /// The page that owns it (the dropdown's row values turn red when concerned).
        var concerns: Set<String>
        /// R1.3: the station that carries it on the loop (its treatment, its popover, its action at rest).
        var station: NDStation? = nil
    }

    /// The station a check belongs to on the loop.
    static let stationForCheck: [String: NDStation] = [
        "ingest": .read, "stores": .queue, "embedding": .search, "search": .search, "backup_local": .mac, "integrity": .mac,
        "disk": .mac, "backup_offsite": .drive, "transcripts": .drive, "drive_auth": .drive,
        "maintenanceNightly": .jobs, "nightly": .jobs, "weekly": .jobs,
    ]

    struct MenuRow: Equatable, Sendable, Identifiable {
        enum Kind: String, Sendable { case activity, search, backups, jobs }
        var kind: Kind
        var id: String { kind.rawValue }
        var label: String
        var value: String
        var tone: NDModel.Tone
        var help: String?
    }

    var pipeline: NDPipelineState
    /// "All good", one item's title, "N things need you", or "Not confirmed".
    var headline: String
    var headlineSymbol: String?
    var headlineTone: NDModel.Tone
    /// The line under the title for one amber notice or not confirmed (a single red item uses its own detail).
    var headlineDetail: String?
    /// The dropdown's blocks: something red, or two or more amber notices (one amber notice says itself in the headline).
    var showsProblemRows: Bool
    var items: [Item]
    /// Not confirmed: the stale verdict's time and age.
    var unconfirmedSince: String?
    var unconfirmedDetail: String?
    var menuRows: [MenuRow]
    var footerLeft: String
    var footerRight: String?
    var calm: Bool { items.isEmpty && unconfirmedSince == nil }
    var redCount: Int { items.filter { $0.severity == .red }.count }
    /// R1.3: each station's item (red before amber), for its action at rest and its popover.
    var stationItems: [NDStation: Item] {
        var out: [NDStation: Item] = [:]
        for item in items.sorted(by: { ($0.severity == .red ? 0 : 1) < ($1.severity == .red ? 0 : 1) }) {
            guard let station = item.station, out[station] == nil else { continue }
            out[station] = item
        }
        return out
    }
    /// L23: the amber dot on the icon iff an amber notice that is not catching up, and nothing red.
    var amberDot: Bool { redCount == 0 && items.contains { $0.severity == .orange && !$0.catchingUp } }
    /// Tab badges (D6): red items only, by the page that owns them.
    var backupsBadge: Int {
        items.filter { $0.severity == .red && !$0.concerns.isDisjoint(with: NDR1.backupChecks) }.count
    }
    var jobsBadge: Int {
        items.filter { $0.severity == .red && ($0.id == "nightly" || $0.concerns.contains("nightly")) }.count
    }
    static let backupChecks: Set<String> = ["drive_auth", "backup_local", "backup_offsite", "transcripts", "integrity", "disk"]

    static func make(_ i: NDInputs, _ m: NDModel) -> NDR1 {
        let c = i.capture
        let pipeline = NDPipelineState.make(i)
        let verdict = c.verdict
        func check(_ name: String) -> BrainBarHealthVerdict.Check? { verdict?.checks.first { $0.name == name } }
        var items: [Item] = []
        var covered: Set<String> = []
        let copyReport = P.Action(kind: .copyReport, label: BrainBarMenuPresenter.Say.copyReport.text)

        if !pipeline.allUnconfirmed {
            // Google Drive refuses: one cause, two connections (c10, c11).
            if check("drive_auth")?.state == .red {
                let action = m.faults.first { $0.action.map { [.reconnectDrive, .connectDrive].contains($0.kind) } == true }?.action
                    ?? P.Action(kind: .reconnectDrive, label: BrainBarMenuPresenter.Say.reconnect.text)
                items.append(Item(id: "drive", severity: .red, title: Say.driveSignedOut,
                                  detail: c.driveAuthModel == .missing ? BrainBarMenuPresenter.Say.s13DetailMissing.text
                                      : BrainBarMenuPresenter.Say.s13Detail.text,
                                  action: action, strip: Strip(from: Say.stageMac, to: Say.stageDrive, side: .target, severity: .red),
                                  menuDetail: driveAges(verdict).min().map { Say.lastDriveUpload(NDFormat.ageText($0)) },
                                  concerns: ["drive_auth", "backup_offsite", "transcripts"], station: .drive))
                covered.formUnion(["drive_auth", "backup_offsite", "transcripts"])
            }
            // One agent's sessions not read, or the reader itself.
            if let ingest = check("ingest"), let sev = sev(ingest) {
                let agents = NDPipelineState.agentCodes(ingest)
                if let first = agents.first {
                    items.append(Item(id: "agent", severity: sev, title: Say.agentNotRead(first.label), detail: Say.agentNotReadDetail,
                                      action: sev == .red ? copyReport : nil, strip: Strip(from: first.label, to: Say.stageRead, side: .source, severity: sev),
                                      concerns: ["ingest"], station: .agents))
                } else {
                    // D6: amber is late, not stopped: "Catching up on new conversations", no "needs you", no button.
                    items.append(Item(id: "reader", severity: sev,
                                      title: sev == .red ? Say.readerNotReading : BrainBarMenuPresenter.Say.r02Headline.text,
                                      detail: sev == .red ? BrainBarMenuPresenter.Say.cantFix.text : nil, action: sev == .red ? copyReport : nil,
                                      strip: Strip(from: Say.yourAgents, to: Say.stageRead, side: .target, severity: sev), concerns: ["ingest"],
                                      station: .read))
                }
                covered.insert("ingest")
            }
            // Saves stuck in the queue (c4, target).
            if let stores = check("stores"), let sev = sev(stores), let queued = i.oldestSaveQueuedAt {
                let age = NDFormat.ageText(max(0, c.now.timeIntervalSince(queued)))
                items.append(Item(id: "saves", severity: sev, title: Say.savesStuck,
                                  detail: Say.savesWaited(i.savesWaiting.flatMap { $0 > 0 ? $0 : nil }, age), action: sev == .red ? copyReport : nil,
                                  strip: Strip(from: Say.stageQueue, to: Say.stageMemory, side: .target, severity: sev), concerns: ["stores"],
                                  station: .queue))
                covered.insert("stores")
            }
            // The nightly check-up failing (c12, source).
            if let nights = c.maintenanceNightFailures, nights >= 3 {
                let since = lastNightly(i).map { Say.notCheckedSince(weekday($0, c)) }
                items.append(Item(id: "nightly", severity: .red, title: Say.nightlyFailed(nights),
                                  detail: since ?? BrainBarMenuPresenter.Say.cantFix.text, action: copyReport,
                                  strip: Strip(from: Say.jobs, to: Say.stageMemory, side: .source, severity: .red),
                                  menuTitle: Say.nightlyFailedShort(nights),
                                  menuDetail: lastNightly(i).map { Say.notCheckedSinceShort(weekday($0, c)) },
                                  concerns: ["maintenanceNightly", "nightly"], station: .jobs))
                covered.formUnion(["maintenanceNightly", "nightly"])
            }
        }
        // Search by meaning behind is drawn below as its own (orange or red) item.
        if !pipeline.allUnconfirmed, check("embedding").flatMap(sev) != nil { covered.insert("embedding") }
        // Every presenter fault the pipeline did not explain, in the presenter's own words.
        for f in m.faults {
            let explained = !f.concerns.isDisjoint(with: covered)
                || (covered.contains("maintenanceNightly") && (f.jobs.contains(.maintenanceNightly) || f.thing == .upkeep))
                || (covered.contains("embedding") && f.thing == .finding)
            guard !explained else { continue }
            if f.thing == .finding, check("embedding").flatMap(sev) != nil { continue }  // drawn as "behind" below
            // D6: severity comes from the verdict: red only when a check it concerns is red (or it is not a check's: a job
            // alert or a runtime warning); a fault over amber checks only is late, so amber, and no report button.
            let concerned = (verdict?.checks ?? []).filter { f.concerns.contains($0.name) }
            let red = concerned.isEmpty || concerned.contains { $0.state == .red }
            let action = red ? f.action : f.action.flatMap { NDButtonPolicy.filled($0) ? $0 : nil }
            let station = (!f.jobs.isEmpty || f.thing == .upkeep) ? NDStation.jobs
                : f.concerns.sorted().compactMap { stationForCheck[$0] }.first
            items.append(Item(id: "p." + f.title, severity: red ? .red : .orange, title: f.title, detail: f.shownFacts.joined(separator: " "),
                              action: action, strip: nil, concerns: f.concerns, station: station))
        }
        // Search by meaning behind (orange, late) or stalled (red): no button, there is nothing to do.
        if !pipeline.allUnconfirmed, let embedding = check("embedding"), let s = sev(embedding) {
            let age = NDFormat.ageText(max(0, embedding.ageS ?? 0))
            items.append(Item(id: "meaning", severity: s, title: s == .red ? Say.meaningBehind(age) : Say.meaningCatchingUp,
                              detail: Say.meaningBehindDetail, action: nil,
                              strip: Strip(from: Say.stageMemory, to: Say.search, side: .target, severity: s), concerns: ["embedding"],
                              station: .search))
        }
        // One filled button per surface: the first item's action; the copy-report button is never the filled one when a
        // reconnect exists (kept as data: the views read `filledActionIndex`).

        // Headline (R1.2 change 8 and L24; changes 4 to 7 skipped by the lead for revision 1.3: red keeps 6d779bc1's headline).
        var unconfirmedSince: String?
        var unconfirmedDetail: String?
        var headline = BrainBarMenuPresenter.Say.allGood.text
        var detail: String?
        var symbol: String?
        var tone: NDModel.Tone = .fine
        var rows = false
        let red = items.filter { $0.severity == .red }.count
        if pipeline.allUnconfirmed, items.isEmpty {
            headline = Say.notConfirmed
            symbol = "questionmark.circle"
            if let at = verdict?.evaluatedAt {
                unconfirmedSince = Say.notConfirmedSince(hhmm(at, c))
                headline = unconfirmedSince ?? headline
                unconfirmedDetail = Say.nothingKnownBroken(NDFormat.ageText(max(0, c.now.timeIntervalSince(at))))
                detail = unconfirmedDetail
            }
        } else if red > 0 {
            // L27: one line. The single fault's own sentence, or "N things need you"; each problem is carried by its station.
            if red == 1, let only = items.first(where: { $0.severity == .red }) {
                headline = only.title
            } else {
                headline = Say.thingsNeedYou(red)
            }
            symbol = "exclamationmark.circle.fill"
            tone = .fault
            rows = true
        } else if items.count == 1, let only = items.first {
            // Amber only, one: a catching-up state names itself; any other notice is "Check soon" with its sentence.
            symbol = "exclamationmark.triangle.fill"
            tone = .warn
            if only.catchingUp {
                headline = only.title
                detail = only.detail
            } else {
                headline = Say.checkSoon
                detail = only.title
            }
        } else if items.count > 1 {
            headline = Say.checkSoon
            symbol = "exclamationmark.triangle.fill"
            tone = .warn
            rows = true
        }

        // Footer (finding 8).
        var left = "", right: String?
        if let at = verdict?.evaluatedAt {
            let age = NDFormat.ageText(max(60, c.now.timeIntervalSince(at)))
            if pipeline.allUnconfirmed {
                left = Say.waitingForNextCheck
                _ = age
            } else {
                // R1.2 change 14: "status" check, so it never collides with the nightly check-up.
                left = Say.statusChecked(age)
                let next = at.addingTimeInterval(300).timeIntervalSince(c.now)
                if next > 0 { right = Say.nextStatusCheck(NDFormat.ageText(max(60, next))) }
            }
        }

        return NDR1(pipeline: pipeline, headline: headline, headlineSymbol: symbol, headlineTone: tone, headlineDetail: detail,
                    showsProblemRows: rows, items: items, unconfirmedSince: unconfirmedSince, unconfirmedDetail: unconfirmedDetail,
                    menuRows: menuRows(i, m, items: items, pipeline: pipeline), footerLeft: left, footerRight: right)
    }

    /// The Drive copies' ages (database and conversations), measured ones only.
    static func driveAges(_ verdict: BrainBarHealthVerdict?) -> [Double] {
        (verdict?.checks ?? []).filter { ["backup_offsite", "transcripts"].contains($0.name) }
            .compactMap(\.ageS).filter { $0.isFinite && $0 >= 0 }
    }

    /// The last nightly check-up that ran (the maintenance run record).
    static func lastNightly(_ i: NDInputs) -> Date? {
        if case let .ran(at) = i.nightlyLastRun { return at }
        return nil
    }

    /// "Thursday": the day a past date fell on, in the capture's zone.
    static func weekday(_ date: Date, _ c: BrainBarMenuCapture) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = c.timeZone
        f.dateFormat = "EEEE"
        return f.string(from: date)
    }

    private static func sev(_ k: BrainBarHealthVerdict.Check) -> NDSeverity? {
        switch k.state {
        case .red: .red
        case .amber: .orange
        default: nil
        }
    }

    static func hhmm(_ date: Date, _ c: BrainBarMenuCapture) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = c.timeZone
        f.dateFormat = "HH:mm"
        return f.string(from: date)
    }

    /// The dropdown's four rows (§ 2.2): what each area is doing, in one value.
    static func menuRows(_ i: NDInputs, _ m: NDModel, items: [Item], pipeline: NDPipelineState) -> [MenuRow] {
        let c = i.capture
        let fresh = !pipeline.allUnconfirmed
        let checks = c.verdict?.checks ?? []
        // Activity: the last hour, as the help tag says it.
        let hour = i.menuWave
        let activityHelp = hour.map { w in
            Say.lastHour(saved: NDFormat.integer(w.savesTotal), read: NDFormat.integer(w.readsTotal))
        }
        // D2: a row may show a fine value only when its stages read fine on the Overview; otherwise it leads with what is
        // wrong (the stage's status words), or "—" when nothing is confirmed.
        func worst(_ stations: [NDStation]) -> NDPipelineState.Condition {
            let all = stations.map(pipeline.condition)
            if let red = all.first(where: { if case .issue(_, .red) = $0 { true } else { false } }) { return red }
            if let issue = all.first(where: { if case .issue = $0 { true } else { false } }) { return issue }
            return all.contains(.notConfirmed) ? .notConfirmed : .fine
        }
        func tone(_ s: NDSeverity) -> NDModel.Tone { s == .red ? .fault : .warn }
        // Search: the share searchable by meaning, when search by meaning and search read fine.
        let percent = i.searchable.map { Int(($0 * 100).rounded(.down)) }
        var searchValue = percent.map { Say.percentReady($0) } ?? NDModel.Number.unknown.value
        var searchTone: NDModel.Tone = .fine
        switch worst([.search]) {
        case .fine: break
        case .notConfirmed: searchValue = NDModel.Number.unknown.value
        case let .issue(_, sev):
            // R1.2 change 9: the row says how far behind ("2 h behind"); the station says "by meaning 2 h behind".
            let embedding = checks.first { $0.name == "embedding" && $0.state != .green }
            if let age = embedding?.ageS, age.isFinite, age >= 0 {
                searchValue = Say.behind(NDFormat.ageText(age))
            } else {
                searchValue = sev == .red ? Say.failed : Say.late
            }
            searchTone = tone(sev)
        }
        let searchHelp = i.searchable.map { Say.searchable(String(format: "%.1f", $0 * 100)) }
        // Backups: the oldest of the three newest copies, so the number is the one closest to late.
        let copyNames = ["backup_local", "backup_offsite", "transcripts"]
        let ages = copyNames.compactMap { name in checks.first { $0.name == name }?.ageS }.filter { $0.isFinite && $0 >= 0 }
        // L29: one treatment: a red concerned item makes the row red; an amber one makes it amber.
        let concerned = items.filter { !$0.concerns.isDisjoint(with: Set(copyNames + ["drive_auth", "integrity", "disk"])) }
        var backupsValue: String
        var backupsTone: NDModel.Tone = concerned.contains { $0.severity == .red } ? .fault : concerned.isEmpty ? .fine : .warn
        if fresh, ages.count == copyNames.count, let oldest = ages.max() {
            backupsValue = Say.ago(NDFormat.ageText(oldest))
        } else {
            backupsValue = NDModel.Number.unknown.value
        }
        switch worst([.mac, .drive]) {
        case .fine: break
        case .notConfirmed: backupsValue = NDModel.Number.unknown.value
        case let .issue(_, sev):
            if backupsTone == .fine { backupsTone = tone(sev) }
            // D7.3: name the copy that is stale, with one number for one fact. Drive: the last time anything reached it
            // (the same age as "Nothing has uploaded for …"); otherwise the copy on this Mac.
            if pipeline.condition(.drive) != .fine, let newest = driveAges(c.verdict).min() {
                backupsValue = Say.driveAgo(NDFormat.ageText(newest))
            } else if let mac = checks.first(where: { $0.name == "backup_local" })?.ageS, mac.isFinite, mac >= 0 {
                backupsValue = Say.macAgo(NDFormat.ageText(mac))
            }
        }
        let backupsHelp: String? = fresh && ages.count == copyNames.count ? zip([Say.laneMac, Say.laneDrive, Say.laneConversations], ages)
            .map { "\($0.0): " + Say.ago(NDFormat.ageText($0.1)) }.joined(separator: ", ") : nil
        // Jobs: fine, failed (red) or late (orange).
        let jobsFailed = items.contains { $0.id == "nightly" } || m.faults.contains { !$0.jobs.isEmpty || $0.thing == .upkeep }
        let upkeep = m.areas.first { $0.thing == .upkeep }
        let jobsLate = upkeep?.rows.contains { $0.number.tone == .warn } == true
        let jobsValue: String
        let jobsTone: NDModel.Tone
        let jobsStage = worst([.jobs])
        if !fresh || jobsStage == .notConfirmed {
            jobsValue = NDModel.Number.unknown.value
            jobsTone = .fine
        } else if case let .issue(_, sev) = jobsStage {
            jobsValue = sev == .red ? Say.failed : Say.late
            jobsTone = tone(sev)
        } else if jobsFailed {
            jobsValue = Say.failed
            jobsTone = .fault
        } else if jobsLate {
            jobsValue = Say.late
            jobsTone = .warn
        } else {
            jobsValue = Say.fine
            jobsTone = .fine
        }
        let nextJob = (i.jobs ?? []).compactMap { j -> Date? in j.enabled == false ? nil : j.nextRun }.filter { $0 > c.now }.min()
        let jobsHelp = nextJob.map { Say.nextRun + " " + Say.inAge(NDFormat.ageText($0.timeIntervalSince(c.now))) }
        return [
            MenuRow(kind: .activity, label: Say.activity, value: "", tone: .fine, help: activityHelp),
            MenuRow(kind: .search, label: Say.search, value: searchValue, tone: searchTone, help: searchHelp),
            MenuRow(kind: .backups, label: Say.backups, value: backupsValue, tone: backupsTone, help: backupsHelp),
            MenuRow(kind: .jobs, label: Say.jobs, value: jobsValue, tone: jobsTone, help: jobsHelp),
        ]
    }
}
