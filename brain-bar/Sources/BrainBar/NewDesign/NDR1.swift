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
        /// An amber state the app is handling itself (catching up on conversations or search by meaning). L47 gives it
        /// the icon's amber dot like any amber state (L23 kept the resting icon for it).
        var catchingUp: Bool { severity == .orange && (id == "meaning" || id == "reader") }
        /// The page that owns it (the dropdown's row values turn red when concerned).
        var concerns: Set<String>
        /// R1.3: the station that carries it on the loop (its treatment, its popover, its action at rest).
        var station: NDStation? = nil
        /// R1.10 (L48): the problem row's few words, when the deck gives them ("Google Drive signed out").
        var rowTitle: String? = nil
    }

    /// The station a check belongs to on the loop.
    static let stationForCheck: [String: NDStation] = [
        "ingest": .read, "stores": .queue, "embedding": .search, "search": .search, "backup_local": .mac, "integrity": .mac,
        "disk": .mac, "backup_offsite": .drive, "transcripts": .drive, "drive_auth": .drive,
        "maintenanceNightly": .jobs, "nightly": .jobs, "maintenanceWeekly": .weekly, "weekly": .weekly,
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
    /// L47: a check in the verdict is amber.
    var amberCheck = false
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
    /// R1.10 (L48): one very short row per red problem, under the headline: its words and its action (the same action as
    /// its station's panel and its dropdown block). Reconnect first, then the pipeline's order; amber gets no row.
    struct ProblemRow: Equatable, Sendable, Identifiable {
        var id: String
        var words: String
        var action: P.Action?
        var station: NDStation?
    }

    static let pipelineOrder: [NDStation] = [.agents, .read, .queue, .memory, .jobs, .weekly, .search, .mac, .drive]

    var problemRows: [ProblemRow] {
        let red = items.filter { $0.severity == .red }
        func rank(_ item: Item) -> Int {
            if item.action.map({ [.reconnectDrive, .connectDrive].contains($0.kind) }) == true { return -1 }
            return item.station.flatMap { Self.pipelineOrder.firstIndex(of: $0) } ?? Self.pipelineOrder.count
        }
        return red.enumerated().sorted { (rank($0.element), $0.offset) < (rank($1.element), $1.offset) }.map(\.element).map { item in
            ProblemRow(id: item.id, words: Self.rowWords(item, pipeline), action: item.action, station: item.station)
        }
    }

    /// The row's few words (CHANGES-R1.10 § A 2); anything longer stays in the station's panel.
    static func rowWords(_ item: Item, _ pipeline: NDPipelineState) -> String { item.rowTitle ?? item.menuTitle ?? item.title }

    /// L47 (replaces L23): the amber dot on the icon whenever any check is amber and none is red, the catching-up states
    /// (new conversations, search by meaning) included.
    var amberDot: Bool { redCount == 0 && (amberCheck || items.contains { $0.severity == .orange }) }
    /// Tab badges (D6): red items only, by the page that owns them.
    var backupsBadge: Int {
        items.filter { $0.severity == .red && !$0.concerns.isDisjoint(with: NDR1.backupChecks) }.count
    }
    var jobsBadge: Int {
        items.filter { item in item.severity == .red && ["nightly", "weekly"].contains { item.id == $0 || item.concerns.contains($0) } }.count
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
                                  concerns: ["drive_auth", "backup_offsite", "transcripts"], station: .drive))  // R1.12: one wording
                covered.formUnion(["drive_auth", "backup_offsite", "transcripts"])
            }
            // One agent's sessions not read, or the reader itself.
            if let ingest = check("ingest"), let sev = sev(ingest) {
                let agents = NDPipelineState.agentCodes(ingest)
                if let first = agents.first {
                    items.append(Item(id: "agent", severity: sev, title: Say.agentNotRead(NDProvider.name(first)), detail: Say.agentNotReadDetail,
                                      action: sev == .red ? copyReport : nil, strip: Strip(from: NDProvider.name(first), to: Say.stageRead, side: .source, severity: sev),
                                      concerns: ["ingest"], station: .agents))  // R1.12: one wording; Read's status is the tail
                } else if sev == .red, BrainBarMenuPresenter.Say.lagShown(c) {
                    // R1.11 (A1): a red reader that is only behind names the lag, says how far, and has no button; its
                    // severity is the presenter's one switch.
                    let lagSeverity: NDSeverity = BrainBarMenuPresenter.lagIsNotice(c) ? .orange : .red
                    items.append(Item(id: "reader", severity: lagSeverity,
                                      title: BrainBarMenuPresenter.Say.lagBehind.text, detail: BrainBarMenuPresenter.lagDetail(c), action: nil,
                                      strip: Strip(from: Say.yourAgents, to: Say.stageRead, side: .target, severity: lagSeverity),
                                      concerns: ["ingest"], station: .read))
                } else {
                    // D6: amber is late, not stopped: "Catching up on new conversations", no "needs you", no button.
                    items.append(Item(id: "reader", severity: sev,
                                      title: sev == .red ? Say.readerNotReading : BrainBarMenuPresenter.Say.r02Headline.text,
                                      detail: sev == .red ? BrainBarMenuPresenter.Say.agent.text : nil, action: sev == .red ? copyReport : nil,
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
                                  station: .queue))  // R1.12: "Saves are stuck" everywhere; the station says "stuck {t}"
                covered.insert("stores")
            }
            // The nightly check-up failing (c12, source).
            if let nights = c.maintenanceNightFailures, nights >= 3 {
                let since = lastNightly(i).map { Say.notCheckedSince(weekday($0, c)) }
                items.append(Item(id: "nightly", severity: .red, title: Say.nightlyFailed(nights),
                                  detail: since ?? BrainBarMenuPresenter.Say.agent.text, action: copyReport,
                                  strip: Strip(from: Say.jobs, to: Say.stageMemory, side: .source, severity: .red),
                                  menuDetail: lastNightly(i).map { Say.notCheckedSinceShort(weekday($0, c)) },
                                  concerns: ["maintenanceNightly", "nightly"], station: .jobs))  // R1.12: one wording
                covered.formUnion(["maintenanceNightly", "nightly"])
            }
            // R1.10 (L49, L51): the weekly clean failing, from its own job's failure streak (c13, source).
            if let weekly = i.jobs?.first(where: { $0.job == .maintenanceWeekly }), let weeks = NDPipelineState.weeklyFailures(weekly) {
                // R1.12: "Weekly clean failed {n} weeks" everywhere; the station says "failed {n} weeks".
                items.append(Item(id: "weekly", severity: .red, title: Say.weeklyFailed(weeks), detail: nil, action: copyReport, strip: nil,
                                  concerns: ["maintenanceWeekly", "weekly"], station: .weekly))
                covered.formUnion(["maintenanceWeekly", "weekly"])
            }
        }
        // Amber notices the pipeline draws on a station with no item yet, in the station's own words (R2A 9), decided
        // from the verdict itself, never from the presenter's classification: main's R31 presenter makes an amber check a
        // notice rather than a fault, and the window must say the same thing on either side of that change.
        if !pipeline.allUnconfirmed {
            for station in [NDStation.queue, .mac, .drive, .jobs] where pipeline.notices.contains(station) {
                guard !items.contains(where: { $0.station == station }), let status = pipeline.status[station]?.text else { continue }
                let here = Set(stationForCheck.filter { $0.value == station }.map(\.key))
                let amber = (verdict?.checks ?? []).filter { here.contains($0.name) && $0.state == .amber }.map(\.name)
                guard !amber.isEmpty else { continue }
                // R2 item 5: a late copy says it is late (only when the local-copy check is amber for age).
                let localAge = (verdict?.checks ?? []).first { $0.name == "backup_local" && $0.state == .amber }?.ageS
                // R1.12 (E3, L53 C4): the local copy running late is the menu's sentence, with its second line: how old the
                // newest is and, when the schedule says, when the next is due. It is one notice; Drive's older upload is its
                // consequence, not a second problem.
                let late = station == .mac && amber.contains("backup_local") && localAge.map({ $0.isFinite && $0 >= 0 }) == true
                let title = late ? BrainBarMenuPresenter.Say.localRunningLate.text : Say.stationSays(NDPipelineLayout.name(station), status)
                let detail = late ? Say.localLateLine(NDFormat.ageText(localAge!), nextDue(c)) : nil
                items.append(Item(id: "notice." + station.rawValue, severity: .orange,
                                  title: title, detail: detail, action: nil,
                                  strip: nil, concerns: Set(amber), station: station))
                covered.formUnion(amber)
            }
        }
        // Search by meaning behind is drawn below as its own (orange or red) item.
        if !pipeline.allUnconfirmed, check("embedding").flatMap(sev) != nil { covered.insert("embedding") }
        // Every presenter fault the pipeline did not explain, in the presenter's own words.
        for f in m.faults {
            let explained = !f.concerns.isDisjoint(with: covered)
                || (covered.contains("maintenanceNightly") && (f.jobs.contains(.maintenanceNightly) || f.thing == .upkeep))
                || (covered.contains("maintenanceWeekly") && f.jobs.contains(.maintenanceWeekly))
                || (covered.contains("embedding") && f.thing == .finding)
            guard !explained else { continue }
            if f.thing == .finding, check("embedding").flatMap(sev) != nil { continue }  // drawn as "behind" below
            // D6: severity comes from the verdict: red only when a check it concerns is red (or it is not a check's: a job
            // alert or a runtime warning); a fault over amber checks only is late, so amber, and no report button.
            let concerned = (verdict?.checks ?? []).filter { f.concerns.contains($0.name) }
            let red = concerned.isEmpty || concerned.contains { $0.state == .red }
            let action = red ? f.action : f.action.flatMap { NDButtonPolicy.filled($0) ? $0 : nil }
            // R1.10: a weekly job's alert belongs to the weekly clean, any other job's to the nightly check-up.
            let station = f.jobs == [.maintenanceWeekly] ? NDStation.weekly
                : (!f.jobs.isEmpty || f.thing == .upkeep) ? NDStation.jobs
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
            // R1.10 (L48, L51): the headline counts and the problem rows name, so words never repeat: "1 thing needs you"
            // with one problem, "N things need you" with more.
            headline = red == 1 ? Say.oneThingNeedsYou : Say.thingsNeedYou(red)
            symbol = nil  // L53, one mark per problem: the count headline has none; each problem row carries its own
            tone = .fault
            rows = true
        } else if items.count == 1, let only = items.first {
            // Amber only, one: a catching-up state names itself; R2A item 9: any other notice says what it is,
            // "{station}: {its status}" from the station's approved words (else its own title), with no line under it.
            symbol = "exclamationmark.triangle.fill"
            tone = .warn
            if only.catchingUp {
                headline = only.title
                detail = only.detail
            } else if only.id.hasPrefix("notice.") {
                headline = only.title
                detail = only.detail  // R1.12 (E3): the late copy's second line, when it has one
            } else if let station = only.station, let status = pipeline.status[station]?.text {
                headline = Say.stationSays(NDPipelineLayout.name(station), status)
            } else {
                headline = only.title
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
                // R1.7 item 5: when the next check is still due (known), the footer says when.
                let next = at.addingTimeInterval(300).timeIntervalSince(c.now)
                if next > 0 { right = Say.nextStatusCheck(NDFormat.ageText(max(60, next))) }
            } else {
                // R1.2 change 14: "status" check, so it never collides with the nightly check-up.
                left = Say.statusChecked(age)
                let next = at.addingTimeInterval(300).timeIntervalSince(c.now)
                if next > 0 { right = Say.nextStatusCheck(NDFormat.ageText(max(60, next))) }
            }
        }

        return NDR1(pipeline: pipeline, headline: headline, headlineSymbol: symbol, headlineTone: tone, headlineDetail: detail,
                    showsProblemRows: rows, items: items, unconfirmedSince: unconfirmedSince, unconfirmedDetail: unconfirmedDetail,
                    menuRows: menuRows(i, m, items: items, pipeline: pipeline), footerLeft: left, footerRight: right,
                    amberCheck: verdict?.checks.contains { $0.state == .amber } == true)
    }

    /// A destination's last backup: Drive's is its newest copy of either kind (measured ones only).
    static func driveNewest(_ verdict: BrainBarHealthVerdict?) -> Double? { driveAges(verdict).min() }

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

    /// R1.10 (L50): a past moment as a clock time on its day, in the capture's zone: "today at 03:16", "yesterday at
    /// 23:40", or "Thursday at 04:00".
    static func dayAt(_ date: Date, _ c: BrainBarMenuCapture) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = c.timeZone
        let time = hhmm(date, c)
        if calendar.isDate(date, inSameDayAs: c.now) { return Say.dayTodayAt(time) }
        if let yesterday = calendar.date(byAdding: .day, value: -1, to: c.now), calendar.isDate(date, inSameDayAs: yesterday) {
            return Say.dayYesterdayAt(time)
        }
        return Say.dayWeekdayAt(weekday(date, c), time)
    }

    static func hhmm(_ date: Date, _ c: BrainBarMenuCapture) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = c.timeZone
        f.dateFormat = "HH:mm"
        return f.string(from: date)
    }

    /// R1.12 (E3): when the next database backup is due, from the job's schedule: "tonight at 03:15" for a run in the
    /// coming night, "today at 14:00" for one later today; nil when the schedule is not read or the run is a day off.
    static func nextDue(_ c: BrainBarMenuCapture) -> String? {
        guard let next = c.backupDailyNextRunAt, next > c.now, next.timeIntervalSince(c.now) < 86_400 else { return nil }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = c.timeZone
        let hour = calendar.component(.hour, from: next)
        if hour < 6 || hour >= 18 { return Say.tonightAt(hhmm(next, c)) }
        return calendar.isDate(next, inSameDayAs: c.now) ? Say.dayTodayAt(hhmm(next, c)) : nil
    }

    /// R1.12 (E2): a last-known moment as a clock time, with its weekday when it is not today ("06:59", "Sun 04:00").
    static func clockOn(_ date: Date, _ c: BrainBarMenuCapture) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = c.timeZone
        if calendar.isDate(date, inSameDayAs: c.now) { return hhmm(date, c) }
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = c.timeZone
        f.dateFormat = "EEE"
        return f.string(from: date) + " " + hhmm(date, c)
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
        // R2 item 14: say what is ready.
        var searchValue = percent.map { Say.searchablePercent($0) } ?? NDModel.Number.unknown.value  // R1.12 (F): one wording
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
        // The area row says the older of the two destinations' last backups: the Mac's or Drive's number, never a third.
        let mac = checks.first { $0.name == "backup_local" }?.ageS.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        let driveDB = checks.first { $0.name == "backup_offsite" }?.ageS.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        if fresh, ages.count == copyNames.count, let mac, let drive = driveDB {
            backupsValue = Say.ago(NDFormat.ageText(max(mac, drive)))
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
            if pipeline.condition(.drive) != .fine, let db = driveDB {
                backupsValue = Say.driveAgo(NDFormat.ageText(db))
            } else if let mac = checks.first(where: { $0.name == "backup_local" })?.ageS, mac.isFinite, mac >= 0 {
                backupsValue = Say.macAgo(NDFormat.ageText(mac))
            }
        }
        let backupsHelp: String? = fresh && ages.count == copyNames.count ? zip([Say.laneMac, Say.laneDrive, Say.laneConversations], ages)
            .map { "\($0.0): " + Say.ago(NDFormat.ageText($0.1)) }.joined(separator: ", ") : nil
        // Jobs: fine, failed (red) or late (orange).
        let jobsFailed = items.contains { $0.id == "nightly" || $0.id == "weekly" } || m.faults.contains { !$0.jobs.isEmpty || $0.thing == .upkeep }
        let upkeep = m.areas.first { $0.thing == .upkeep }
        let jobsLate = upkeep?.rows.contains { $0.number.tone == .warn } == true
        let jobsValue: String
        let jobsTone: NDModel.Tone
        let jobsStage = worst([.jobs, .weekly])
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
